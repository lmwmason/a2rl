"""Frame-stacked demonstrations for the same RGB-only Mode 2 policy as online RL."""
from pathlib import Path

import numpy as np


class Clip:
    def __init__(self, path, stack, height, width, with_imu=False):
        path = Path(path)
        shape = (height, width, 3)
        if path.is_dir():
            actions = np.fromfile(path / "actions.f32", dtype="<f4").reshape(-1, 4)
            self.frames = np.memmap(path / "frames.rgb", mode="r", dtype=np.uint8).reshape(-1, *shape)
            gaps = np.zeros(len(actions), dtype=bool)
            self.weights = np.ones(len(actions), np.float32)
        else:
            with np.load(path, allow_pickle=False) as data:
                self.frames = data["frames"]
                actions = data["actions"].astype(np.float32)
                times = data["timestamps"]
                gaps = np.r_[False, np.diff(times) > .075]
                self.weights = data["weights"].astype(np.float32)
        if len(actions) < stack or self.frames.shape != (len(actions), *shape):
            raise ValueError(f"The RGB frame count or resolution is invalid in {path}.")
        if (actions.shape != (len(self.frames), 4) or not np.isfinite(actions).all()
                or self.weights.shape != (len(actions),) or not np.isfinite(self.weights).all()
                or np.any(self.weights <= 0)):
            raise ValueError(f"Malformed Mode 2 labels or weights in {path}.")
        self.actions = actions
        self.imu = None
        if with_imu:
            self.imu = np.load(path.with_suffix(".imu.npy"), allow_pickle=False)
            if self.imu.shape != (len(actions), 6) or not np.isfinite(self.imu).all():
                raise ValueError(f"Malformed synchronized IMU samples for {path}.")
        self.offsets = np.arange(1 - stack, 1)
        self.start = np.maximum.accumulate(np.where(gaps, np.arange(len(gaps)), 0))

    def __len__(self):
        return len(self.actions)

    def batch(self, rows):
        indices = np.maximum(rows[:, None] + self.offsets, self.start[rows, None])
        batch = self.frames[indices], self.actions[rows], self.weights[rows]
        return (*batch, self.imu[indices]) if self.imu is not None else batch


class Demonstrations:
    def __init__(self, real_paths, sim_paths, stack, height, width, seed, with_imu=False):
        self.real = [Clip(path, stack, height, width, with_imu) for path in real_paths]
        self.sim = [Clip(path, stack, height, width, with_imu) for path in sim_paths]
        self.rng = np.random.default_rng(seed)
        if not self.real and not self.sim:
            raise ValueError("At least one demonstration is required.")

    def _sample_group(self, group, count):
        ids = self.rng.integers(len(group), size=count)
        return [clip.batch(self.rng.integers(len(clip), size=int(sum(ids == i))))
                for i, clip in enumerate(group) if np.any(ids == i)]

    def sample(self, count):
        real_count = (max(1, round(.40 * count)) if self.sim else count) if self.real else 0
        batches = []
        if real_count:
            batches += self._sample_group(self.real, real_count)
        if count > real_count:
            batches += self._sample_group(self.sim, count - real_count)
        return tuple(np.concatenate(parts) for parts in zip(*batches))
