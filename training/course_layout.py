"""Generate repeatable installation offsets shared by stacked gate frames."""

import numpy as np


def varied_gate_poses(poses, *, seed, xy_m, z_m, yaw_deg):
    bounds = np.asarray([xy_m, xy_m, z_m, yaw_deg], dtype=np.float64)
    if not np.isfinite(bounds).all() or np.any(bounds < 0):
        raise ValueError("Course offsets must have finite nonnegative bounds.")
    poses = np.asarray(poses, dtype=np.float64)
    # Matching horizontal centers denote apertures in one physical assembly.
    _, group = np.unique(poses[:, :2], axis=0, return_inverse=True)
    rng = np.random.default_rng(seed)
    displacement = rng.uniform(-bounds, bounds, size=(int(group.max()) + 1, 4))
    return poses + displacement[group]
