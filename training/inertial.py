"""Inertial attitude relative to the first observation, using only gyro samples."""
import numpy as np


class RelativeAttitude:
    def __init__(self):
        self.previous = None
        self.orientation = np.array([0., 0., 0., 1.])

    def augment(self, sample, elapsed):
        if self.previous is not None:
            time, previous_rate = self.previous
            dt = np.clip(elapsed - time, 0., .075)
            displacement = .5 * (previous_rate + sample[:3]) * dt
            angle = np.linalg.norm(displacement)
            if angle > 1e-8:
                delta = displacement * (np.sin(.5 * angle) / angle)
                scalar = np.cos(.5 * angle)
                q = self.orientation
                v = scalar * q[:3] + q[3] * delta + np.cross(q[:3], delta)
                w = scalar * q[3] - q[:3] @ delta
                self.orientation = np.r_[v, w]
                self.orientation /= np.linalg.norm(self.orientation)
        self.previous = (elapsed, sample[:3].copy())
        return np.concatenate((sample, self.orientation)).astype(np.float32)
