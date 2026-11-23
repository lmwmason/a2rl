"""Frame-stacked demonstrations for the RGB and IMU Mode 2 policy."""
from pathlib import Path

import numpy as np


class Clip:
    def __init__(self, path, stack, height, width, with_imu=False):
        path = Path(path)
        shape = (height, width, 3)
        self.approach_rows = None
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
                if "next_gate" in data:
                    gates = data["next_gate"]
                    if gates.shape == (len(actions),) and len(gates) >= stack:
                        # Learn the approach before contact, where steering still has time to work.
                        end = max(1, len(gates) - 3)
                        rows = np.arange(max(0, end - 60), end)
                        self.approach_rows = rows[gates[rows] >= gates[-1] - 1]
                        if not len(self.approach_rows):
                            self.approach_rows = None
        if not len(actions) or self.frames.shape != (len(actions), *shape):
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

    def sample_rows(self, rng, count, prioritize_approach):
        uniform = rng.integers(len(self), size=count)
        if prioritize_approach and self.approach_rows is not None:
            focus_count = count // 2
            uniform[:focus_count] = rng.choice(self.approach_rows, size=focus_count)
        return uniform

    def batch(self, rows):
        indices = np.maximum(rows[:, None] + self.offsets, self.start[rows, None])
        batch = self.frames[indices], self.actions[rows], self.weights[rows]
        return (*batch, self.imu[indices]) if self.imu is not None else batch


class Demonstrations:
    def __init__(self, real_paths, sim_paths, stack, height, width, seed, with_imu=False, primary_fraction=.40):
        self.real = [Clip(path, stack, height, width, with_imu) for path in real_paths]
        self.sim = [Clip(path, stack, height, width, with_imu) for path in sim_paths]
        self.rng = np.random.default_rng(seed)
        self.primary_fraction = primary_fraction
        self.independent = []
        self.assisted = []
        self.handoff = []
        if not self.real and not self.sim:
            raise ValueError("At least one demonstration is required.")

    def _sample_group(self, group, count, *, prioritize_approach=False):
        counts = np.bincount(self.rng.integers(len(group), size=count), minlength=len(group))
        return [clip.batch(clip.sample_rows(self.rng, rows, prioritize_approach))
                for clip, rows in zip(group, counts) if rows]

    def sample(self, count):
        has_corrections = self.independent or self.assisted or self.handoff or self.sim
        real_count = (max(1, round(self.primary_fraction * count)) if has_corrections else count) if self.real else 0
        batches = self._sample_group(self.real, real_count) if real_count else []
        remaining = count - real_count
        if remaining and self.handoff:
            segment_count = round(remaining * .45) if self.independent or self.assisted else remaining
            batches += self._sample_group(self.handoff, segment_count, prioritize_approach=True)
            remaining -= segment_count
        if remaining and self.independent and self.assisted:
            independent_count = round(remaining * 2 / 3)
            batches += self._sample_group(self.independent, independent_count, prioritize_approach=True)
            batches += self._sample_group(self.assisted, remaining - independent_count)
        elif remaining and self.independent:
            batches += self._sample_group(self.independent, remaining, prioritize_approach=True)
        elif remaining:
            batches += self._sample_group(self.assisted or self.sim, remaining)
        return tuple(np.concatenate(parts) for parts in zip(*batches))
