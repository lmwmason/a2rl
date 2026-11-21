import unittest

import numpy as np

from training.replay import Replay
from training.sim import Environment, load_sim


class ReplayTest(unittest.TestCase):
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


class EnvironmentTest(unittest.TestCase):
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
