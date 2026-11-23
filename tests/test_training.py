import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from training.agent import Agent, extend_cnn_history, extend_imu_features
from training.betaflight_link import timed_mean_throttle
from training.demonstrations import Clip, Demonstrations
from training.inertial import RelativeAttitude
from training.replay import Replay
from training.sim import CameraRenderer, gate_count, load_sim
from training.telemetry import finish_time_history


class ThrottleTimingTest(unittest.TestCase):
    def test_weights_partial_frames_between_first_and_final_gates(self):
        actions = [[0., -1., 0., 0.], [0., 1., 0., 0.], [0., 0., 0., 0.]]
        self.assertAlmostEqual(timed_mean_throttle(actions, 10, 0.025, 0.25), 0.125 / 0.225)
        self.assertIsNone(timed_mean_throttle(actions, 10, 0.025, 0.31))


class ReplayTest(unittest.TestCase):
    def test_restart_only_truncates_open_episodes(self):
        replay = Replay(24, 2, (1, 1, 3), 2, 2, 0.99, np.random.default_rng(1))
        initial = np.zeros((2, 1, 1, 3), dtype=np.uint8)
        last = np.full_like(initial, 27)
        replay.append(initial, np.zeros((2, 4)), np.zeros(2), np.array([0, 1]), last)
        replay.truncate_open(last)
        np.testing.assert_array_equal(replay.flags[0], [2, 1])
        np.testing.assert_array_equal(replay.next_start, [1, 1])
        np.testing.assert_array_equal(replay.truncated_frames[0, 0], last[0])
        self.assertNotIn((0, 1), replay.truncated_frames)

    def test_timeout_bootstraps_from_terminal_frame_within_same_episode(self):
        replay = Replay(20, 1, (1, 1, 3), 2, 3, 0.5, np.random.default_rng(0))
        for i, (value, flag) in enumerate([(10, 0), (11, 2), (20, 0), (21, 0), (22, 0)]):
            frames = np.full((1, 1, 1, 3), value, np.uint8)
            next_frames = np.full_like(frames, 12)
            replay.append(frames, np.zeros((1, 4)), np.array([i + 1]), np.array([flag]), next_frames)
        obs, _, reward, discount, next_obs = replay.sample(2)
        np.testing.assert_array_equal(obs[:, :, 0, 0, 0], [[10, 11], [10, 11]])
        np.testing.assert_array_equal(next_obs[:, :, 0, 0, 0], [[11, 12], [11, 12]])
        np.testing.assert_array_equal(reward, [2, 2])
        np.testing.assert_array_equal(discount, [0.5, 0.5])

    def test_ring_buffer_does_not_use_overwritten_history_or_cross_resets(self):
        replay = Replay(5, 1, (1, 1, 3), 2, 2, 0.5, np.random.default_rng(1))
        for i in range(12):
            frames = np.full((1, 1, 1, 3), i, np.uint8)
            replay.append(frames, np.zeros((1, 4)), np.ones(1), np.array([i == 8]), frames)
        obs, _, _, discount, _ = replay.sample(20)
        for stack, weight in zip(obs[:, :, 0, 0, 0], discount):
            old, new = stack
            self.assertIn((old, new, weight), [(7, 8, 0.0), (9, 9, 0.25)])


class DemonstrationTest(unittest.TestCase):
    def test_frame_stack_restarts_after_a_camera_gap(self):
        frames = np.broadcast_to(np.arange(4, dtype=np.uint8)[:, None, None, None], (4, 1, 1, 3))
        actions = np.arange(16, dtype=np.float32).reshape(4, 4) / 16
        with TemporaryDirectory() as directory:
            path = Path(directory) / "clip.npz"
            np.savez(path, frames=frames, actions=actions, timestamps=[0., .033, .2, .233],
                     weights=np.ones(4, dtype=np.float32))
            clip = Clip(path, 2, 1, 1)
        image, label, _ = clip.batch(np.array([1, 2, 3]))
        np.testing.assert_array_equal(image[:, :, 0, 0, 0], [[0, 1], [2, 2], [2, 3]])
        np.testing.assert_array_equal(label, actions[1:])
        _, _, _, future = clip.batch(np.arange(4), action_horizon=3)
        np.testing.assert_array_equal(future, actions[[[1, 1], [1, 1], [3, 3], [3, 3]]])


    def test_self_imitation_retains_only_successful_independent_commands(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "clip.npz"
            frames = np.zeros((10, 1, 1, 3), dtype=np.uint8)
            original = np.zeros((10, 4), dtype=np.float32)
            np.savez(path, frames=frames, actions=original, executed=np.ones_like(original),
                     timestamps=np.arange(10) / 30, next_gate=np.arange(10),
                     weights=np.ones(10, dtype=np.float32), independent=True, passed=4)
            clip = Clip(path, 2, 1, 1, preserve_success=True)
            self.assertEqual(clip.preserved_rows, 3)
            np.testing.assert_allclose(clip.actions[:3], .65)
            np.testing.assert_array_equal(clip.actions[3:], original[3:])
            np.savez(path, frames=frames, actions=original, executed=np.ones_like(original),
                     timestamps=np.arange(10) / 30, next_gate=np.arange(10),
                     weights=np.array([.2, .2, 5, 5, 5, 5, 5, 5, 5, 5]), independent=False, passed=5)
            segment = Clip(path, 2, 1, 1, preserve_handoff=True)
            self.assertEqual(segment.preserved_rows, 2)
            np.testing.assert_allclose(segment.actions[2:4], .65)
            np.testing.assert_array_equal(segment.actions[:2], original[:2])

    def test_completion_batch_emphasizes_independent_approaches(self):
        with TemporaryDirectory() as directory:
            paths = [Path(directory) / f"{i}.npz" for i in range(3)]
            for i, path in enumerate(paths):
                frames = np.zeros((120, 1, 1, 3), dtype=np.uint8)
                actions = np.full((120, 4), i / 3, dtype=np.float32)
                np.savez(path, frames=frames, actions=actions, timestamps=np.arange(120) / 30,
                         weights=np.ones(120, dtype=np.float32),
                         next_gate=np.repeat(np.arange(4), 30))
            demos = Demonstrations(paths[:1], [], 2, 1, 1, 2, primary_fraction=.25)
            demos.independent = [Clip(paths[1], 2, 1, 1)]
            demos.assisted = [Clip(paths[2], 2, 1, 1)]
            _, actions, _ = demos.sample(128)
        np.testing.assert_array_equal(np.unique(actions[:, 0], return_counts=True)[1], [32, 64, 32])
        clip = demos.independent[0]
        rows = clip.sample_rows(np.random.default_rng(5), 64, prioritize_approach=True)
        self.assertTrue(np.all(np.isin(rows[:32], clip.approach_rows)))
        self.assertTrue(np.all((rows[:32] >= 60) & (rows[:32] < 117)))


    def test_focus_batches_stay_in_requested_gate_range(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "clip.npz"
            gates = np.repeat(np.arange(8), 10)
            frames = np.zeros((len(gates), 1, 1, 3), dtype=np.uint8)
            actions = np.tile(gates[:, None] / 10, (1, 4)).astype(np.float32)
            np.savez(path, frames=frames, actions=actions, timestamps=np.arange(len(gates)) / 30,
                     weights=np.ones(len(gates), np.float32), next_gate=gates)
            demos = Demonstrations([path], [], 2, 1, 1, 3, primary_fraction=.25,
                                   gate_labels=True, focus_gates=[3, 5])
            demos.independent = [Clip(path, 2, 1, 1)]
            _, _, _, labels = demos.sample(80)
        self.assertEqual(demos.focus_samples, 24)
        self.assertTrue(np.all(np.isin(labels[20:44], [2, 3, 4])))


class TelemetryTest(unittest.TestCase):
    def test_finish_history_keeps_unfinished_evaluations_as_gaps(self):
        series = finish_time_history([{"steps": 0, "official_start_seconds": 111.7},
                                      {"steps": 8000},
                                      {"steps": 16000, "official_mean_seconds": 110.2}], speed=True)
        self.assertEqual(series[1], {"at": 8000, "standard_s": None, "comparison_s": None})
        self.assertEqual(series[2], {"at": 16000, "standard_s": None, "comparison_s": 110.2})
        self.assertEqual(finish_time_history([{"round": 9, "best_two_lap_s": 113.5}], speed=False),
                         [{"at": 9, "standard_s": None, "comparison_s": 113.5}])


class SharedCameraTest(unittest.TestCase):
    def test_external_pose_uses_the_native_2026_camera(self):
        with patch.dict("os.environ", {"A2RL_COURSE": "2026-practice"}):
            lib = load_sim()
            env = CameraRenderer(lib, 96, 72)
            shifted = CameraRenderer(lib, 96, 72, mount_pitch=60.)
            try:
                self.assertEqual(gate_count(lib), 12)
                state = [8., 22., .205, 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]
                expected = env.render_pose(state, 0.).copy()
                self.assertEqual(expected.shape, (72, 96, 3))
                self.assertEqual(env.mount_pitch, 70.)
                chase = env.render_pose(state, 0., chase=True)
                self.assertFalse(np.array_equal(chase, expected))
                self.assertTrue(np.any((chase[..., 0] > 180) & (chase[..., 1] > 100) & (chase[..., 2] < 70)))
                self.assertFalse(np.array_equal(shifted.render_pose(state, 0.), expected))
                state[8:10] = [.7071, .7071]
                self.assertFalse(np.array_equal(env.render_pose(state, 0.), expected))
            finally:
                env.close()
                shifted.close()


class QualifiedPolicyTest(unittest.TestCase):
    def test_gyro_attitude_and_imu_weight_extension(self):
        estimate = RelativeAttitude()
        angular_rate = np.array([0, 0, np.pi / 2, 0, 0, 0], np.float32)
        for tick in range(31):
            features = estimate.augment(angular_rate, tick / 30)
        np.testing.assert_allclose(features[6:], [0, 0, np.sqrt(.5), np.sqrt(.5)], atol=1e-6)
        original = Agent(2, "cpu", 0., with_imu=True)
        expanded = Agent(2, "cpu", 0., with_imu=True, imu_channels=10)
        expanded.set_weights(extend_imu_features(original.weights(), 10))
        images = np.zeros((1, 2, 16, 12, 3), np.uint8)
        imus = np.repeat(features[None, None], 2, axis=1)
        np.testing.assert_allclose(expanded.action(images, imu=imus),
                                   original.action(images, imu=imus[..., :6]), atol=1e-6)

    def test_extended_history_preserves_initial_cnn_policy(self):
        old = Agent(2, "cpu", 0., with_imu=True)
        new = Agent(6, "cpu", 0., with_imu=True)
        new.set_weights(extend_cnn_history(old.weights(), 6))
        rng = np.random.default_rng(2)
        frames = rng.integers(0, 256, (2, 6, 16, 12, 3), dtype=np.uint8)
        inertial = rng.normal(size=(2, 6, 6)).astype(np.float32)
        expected = old.action(frames[:, -2:].copy(), imu=inertial[:, -2:])
        np.testing.assert_allclose(new.action(frames, imu=inertial), expected, atol=1e-6)

    def test_rl_updates_keep_the_reference_and_frozen_features_and_allow_rollback(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "qualified.pt"
            policy = Agent(2, "cpu", 0.)
            policy.save(path)
            policy.configure_warm_start()
            policy.protect_qualified_policy(path)
            frames = np.random.default_rng(1).integers(0, 256, (2, 2, 16, 12, 3), dtype=np.uint8)
            initial = policy.action(frames)
            observations = (frames, initial, np.ones(2, np.float32), np.full(2, .95, np.float32), frames)
            with torch.no_grad():
                policy.actor[-1].bias.add_(.1)
            policy.update(observations, .02)
            self.assertFalse(np.array_equal(policy.action(frames), initial))
            np.testing.assert_array_equal(policy.action(frames, reference=True), initial)
            self.assertTrue(all(weight.grad is None for weight in policy.encoder.parameters()))
            policy.restore_actor(path)
            np.testing.assert_array_equal(policy.action(frames), initial)


if __name__ == "__main__":
    unittest.main()
