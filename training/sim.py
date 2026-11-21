"""Native simulation bindings. The policy gets HWC uint8 camera pixels."""
import ctypes as ct
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HOVER = 0.966 * 9.80665 / (4 * 4.8e-7 * 5100 ** 2)


def load_sim():
    if sys.platform == "darwin":
        filename = "liba2rl_sim.dylib"
    elif os.name == "nt":
        filename = "a2rl_sim.dll"
    else:
        filename = "liba2rl_sim.so"
    path = Path(os.environ.get("A2RL_SIM_LIB", ROOT / "target" / "release" / filename))
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing. Run cargo build --release --lib first.")
    lib = ct.CDLL(str(path.resolve()))
    lib.sim_create.argtypes = [ct.c_size_t, ct.c_size_t]
    lib.sim_create.restype = ct.c_void_p
    lib.sim_destroy.argtypes = [ct.c_void_p]
    lib.sim_destroy.restype = None
    lib.sim_reset.argtypes = [ct.c_void_p, ct.c_int, ct.c_float, ct.c_float, ct.c_float, ct.c_void_p]
    lib.sim_reset.restype = None
    lib.sim_step.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_float)]
    lib.sim_step.restype = ct.c_int
    lib.sim_lap_stats.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_float)]
    lib.sim_lap_stats.restype = None
    return lib


class Environment:
    def __init__(self, lib, width, height):
        self.lib = lib
        self.handle = lib.sim_create(width, height)
        if not self.handle:
            raise ValueError("Unsupported camera resolution.")
        self.image = np.empty((height, width, 3), dtype=np.uint8)
        self.reward = ct.c_float()

    def reset(self, gate=-1, side=0., height=0., exposure=1.):
        self.lib.sim_reset(self.handle, gate, side, height, exposure, self.image.ctypes.data)
        return self.image

    def step(self, action):
        image, reward, status = self.advance(action)
        return image, reward, status in (1, 3), status == 2

    def advance(self, action):
        native = np.asarray(action, dtype=np.float32).copy()
        if native.shape != (4,):
            raise ValueError("Expected four policy action values.")
        # Keep neutral near hover without restricting the available thrust range.
        native[0] = HOVER + native[0] * (HOVER if native[0] < 0 else 1 - HOVER)
        status = self.lib.sim_step(self.handle, native.ctypes.data, self.image.ctypes.data, ct.byref(self.reward))
        if status < 0:
            raise ValueError("Non-finite action or step after episode end.")
        return self.image, self.reward.value, status

    def lap_stats(self):
        gates, best = ct.c_uint32(), ct.c_float()
        self.lib.sim_lap_stats(self.handle, ct.byref(gates), ct.byref(best))
        return gates.value, None if np.isnan(best.value) else best.value

    def close(self):
        if self.handle:
            self.lib.sim_destroy(self.handle)
            self.handle = None
