import unittest
from unittest.mock import patch
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from training.agent import Agent
from training.betaflight_link import timed_mean_throttle
from training.demonstrations import Clip
from training.racing_school import matches_teacher_pace
from training.replay import Replay
from training.sim import Environment, gate_count, load_sim
from training.telemetry import FramePublisher, finish_time_history, read_packet
from training.train import beats_record


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


class TelemetryTest(unittest.TestCase):
    def test_live_frame_preserves_driver_and_rgb(self):
        with TemporaryDirectory() as directory:
            publisher = FramePublisher(directory)
            image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
            publisher.show(image, gates=2, step=96, phase="IMITATION", status=0, driver="Student")
            info, pixels = read_packet(BytesIO(publisher.path.read_bytes()))
        self.assertEqual((info["width"], info["height"], info["driver"], info["gates"]), (3, 2, "Student", 2))
        self.assertEqual(pixels, image.tobytes())

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
            env = Environment(lib, 96, 72)
            try:
                self.assertEqual(gate_count(lib), 12)
                expected = env.official_start().copy()
                state = env.rigid_state()
                np.testing.assert_array_equal(env.render_pose(state, 0.), expected)
                state[8:10] = [0., 1.]
                self.assertFalse(np.array_equal(env.render_pose(state, 0.), expected))
                self.assertEqual(env.lap_stats()[0], 0)
            finally:
                env.close()


class RecordTargetTest(unittest.TestCase):
    def test_course_qualification_requires_continuous_pace(self):
        expert = {"two_lap_rate": 1., "worst_two_lap_s": 35.,
                  "longest_completed_run_below_2_m_s": .35, "mean_gate_crossing_speed_m_s": 5.2}
        self.assertTrue(matches_teacher_pace(expert | {"worst_two_lap_s": 39.}, expert))
        self.assertFalse(matches_teacher_pace(expert | {"two_lap_rate": 2 / 3}, expert))
        self.assertFalse(matches_teacher_pace(expert | {"worst_two_lap_s": 45.}, expert))
        self.assertFalse(matches_teacher_pace(expert | {"longest_completed_run_below_2_m_s": 3.}, expert))
        self.assertFalse(matches_teacher_pace(expert | {"mean_gate_crossing_speed_m_s": 2.8}, expert))

    def test_strict_time_and_clean_finishes_from_every_start(self):
        result = {"official_start_seconds": 16.55, "official_finishes": 3, "crash_rate": 0.}
        self.assertTrue(beats_record(result, 16.56, 3))
        self.assertFalse(beats_record(result | {"official_start_seconds": 16.56}, 16.56, 3))
        self.assertFalse(beats_record(result | {"official_finishes": 2}, 16.56, 3))
        self.assertFalse(beats_record(result | {"crash_rate": 1 / 3}, 16.56, 3))
        self.assertFalse(beats_record(result | {"official_start_seconds": None}, 16.56, 3))


class QualifiedPolicyTest(unittest.TestCase):
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


class EnvironmentTest(unittest.TestCase):
    def test_logged_route_expert_finishes_two_official_laps(self):
        env = Environment(load_sim(), 16, 12)
        try:
            env.official_start()
            self.assertIsNone(env.official_seconds())
            for _ in range(180 * 30):
                _, _, status = env.advance(env.expert_action())
                if status:
                    break
            self.assertEqual(status, 4)
            self.assertEqual(env.lap_stats()[0], 22)
            self.assertGreater(env.official_seconds(), 0.)
        finally:
            env.close()

    def test_gate_exit_starts_with_an_uncounted_gate_two_approach(self):
        env = Environment(load_sim(), 96, 72)
        try:
            before = env.ground_start().copy()
            after = env.after_gate(speed=4.).copy()
            self.assertGreater(np.mean(before != after), 0.1)
            self.assertEqual(env.lap_stats(), (0, None))
            _, _, terminated, truncated = env.step([0, env.lib.sim_hover_stick(), 0, 0])
            self.assertFalse(terminated or truncated)
        finally:
            env.close()

    def test_native_interface_returns_pixels_and_reward_without_telemetry(self):
        env = Environment(load_sim(), 96, 72)
        try:
            first = env.reset().copy()
            self.assertEqual(first.shape, (72, 96, 3))
            self.assertEqual(first.dtype, np.uint8)
            for _ in range(8):
                frame, reward, terminated, truncated = env.step([0, env.lib.sim_hover_stick(), 0, 0.8])
            self.assertFalse(terminated or truncated)
            self.assertTrue(np.isfinite(reward))
            self.assertGreater(np.mean(first != frame), 0.25)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
