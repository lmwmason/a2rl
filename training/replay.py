"""Store each RGB frame once and reconstruct stacks without crossing resets."""
import numpy as np


class Replay:
    def __init__(self, capacity, envs, image_shape, stack, n_step, gamma, rng):
        self.rows = capacity // envs
        if self.rows < stack + n_step + 1:
            raise ValueError("Replay capacity is too small for the frame stack and n-step return.")
        self.envs, self.stack, self.n_step, self.gamma, self.rng = envs, stack, n_step, gamma, rng
        self.frames = np.empty((self.rows, envs, *image_shape), dtype=np.uint8)
        self.actions = np.empty((self.rows, envs, 4), dtype=np.float32)
        self.rewards = np.empty((self.rows, envs), dtype=np.float32)
        self.flags = np.empty((self.rows, envs), dtype=np.uint8)
        self.episode_starts = np.empty((self.rows, envs), dtype=np.int64)
        self.next_start = np.zeros(envs, dtype=np.int64)
        self.truncated_frames = {}
        self.count = 0

    def append(self, frames, actions, rewards, flags, next_frames):
        row = self.count % self.rows
        self.frames[row] = frames
        self.actions[row] = actions
        self.rewards[row] = rewards
        self.flags[row] = flags
        self.episode_starts[row] = self.next_start
        for env in range(self.envs):
            self.truncated_frames.pop((self.count - self.rows, env), None)
            if flags[env] == 2:
                self.truncated_frames[self.count, env] = next_frames[env].copy()
        self.count += 1
        self.next_start = np.where(flags, self.count, self.next_start)

    def ready(self):
        return self.count >= self.stack + self.n_step

    def truncate_open(self, next_frames):
        if self.count == 0:
            return
        row = (self.count - 1) % self.rows
        for env in np.flatnonzero(self.flags[row] == 0):
            self.flags[row, env] = 2
            self.truncated_frames[self.count - 1, env] = next_frames[env].copy()
        self.next_start.fill(self.count)

    def _stack(self, rows, envs, start):
        offsets = np.arange(1 - self.stack, 1)
        rows = np.maximum(rows[:, None] + offsets, start[:, None])
        return self.frames[rows % self.rows, envs[:, None]]

    def sample(self, batch):
        low = max(0, self.count - self.rows) + self.stack - 1
        rows = self.rng.integers(low, self.count - self.n_step, size=batch)
        envs = self.rng.integers(0, self.envs, size=batch)
        start = self.episode_starts[rows % self.rows, envs]
        obs = self._stack(rows, envs, start)
        returns = np.zeros(batch, np.float32)
        discount = np.ones(batch, np.float32)
        active = np.ones(batch, bool)
        end = rows + self.n_step
        end_flag = np.zeros(batch, np.uint8)
        for offset in range(self.n_step):
            index = (rows + offset) % self.rows
            returns += discount * self.rewards[index, envs] * active
            discount[active] *= self.gamma
            flags = self.flags[index, envs]
            stopped = active & (flags != 0)
            end[stopped] = rows[stopped] + offset + 1
            end_flag[stopped] = flags[stopped]
            active &= ~stopped
        # The final camera frame on a timeout belongs to the old episode and still bootstraps.
        discount[end_flag == 1] = 0
        next_obs = self._stack(end, envs, start)
        for idx in np.flatnonzero(end_flag == 2):
            next_obs[idx, -1] = self.truncated_frames[end[idx] - 1, envs[idx]]
        return obs, self.actions[rows % self.rows, envs], returns, discount, next_obs
