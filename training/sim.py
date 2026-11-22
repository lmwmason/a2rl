"""Native simulation bindings. The policy gets HWC uint8 camera pixels."""
import ctypes as ct
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_sim():
    if sys.platform == "darwin":
        filename = "liba2rl_sim.dylib"
    elif os.name == "nt":
        filename = "a2rl_sim.dll"
    else:
        filename = "liba2rl_sim.so"
    path = Path(os.environ.get("A2RL_SIM_LIB", ROOT / "target" / "release" / filename))
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing. Run cargo build --release --lib --no-default-features first.")
    lib = ct.CDLL(str(path.resolve()))
    lib.sim_gate_count.argtypes = []
    lib.sim_gate_count.restype = ct.c_size_t
    lib.sim_hover_stick.argtypes = []
    lib.sim_hover_stick.restype = ct.c_float
    lib.sim_create.argtypes = [ct.c_size_t, ct.c_size_t]
    lib.sim_create.restype = ct.c_void_p
    lib.sim_destroy.argtypes = [ct.c_void_p]
    lib.sim_destroy.restype = None
    lib.sim_reset.argtypes = [ct.c_void_p, ct.c_int, ct.c_float, ct.c_float, ct.c_float, ct.c_float, ct.c_uint32, ct.c_void_p]
    lib.sim_reset.restype = None
    lib.sim_reset_after_gate.argtypes = [
        ct.c_void_p, ct.c_size_t, ct.c_float, ct.c_float, ct.c_float, ct.c_float,
        ct.c_float, ct.c_uint32, ct.c_void_p,
    ]
    lib.sim_reset_after_gate.restype = None
    lib.sim_expert_action.argtypes = [ct.c_void_p, ct.c_void_p]
    lib.sim_expert_action.restype = None
    lib.sim_expert_features.argtypes = [ct.c_void_p, ct.c_void_p]
    lib.sim_expert_features.restype = None
    lib.sim_step.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_float)]
    lib.sim_step.restype = ct.c_int
    lib.sim_lap_stats.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_float)]
    lib.sim_lap_stats.restype = None
    lib.sim_official_seconds.argtypes = [ct.c_void_p]
    lib.sim_official_seconds.restype = ct.c_float
    return lib


class Environment:
    def __init__(self, lib, width, height):
        self.lib = lib
        self.handle = lib.sim_create(width, height)
        if not self.handle:
            raise ValueError("Unsupported camera resolution.")
        self.image = np.empty((height, width, 3), dtype=np.uint8)
        self.reward = ct.c_float()

    def reset(self, gate=-1, side=0., height=0., exposure=1., breakup=0., video_seed=1):
        self.lib.sim_reset(self.handle, gate, side, height, exposure, breakup, video_seed, self.image.ctypes.data)
        return self.image

    def ground_start(self, side=0., exposure=1., breakup=0., video_seed=1):
        return self.reset(-2, side, exposure=exposure, breakup=breakup, video_seed=video_seed)

    def official_start(self, side=0., exposure=1.):
        return self.reset(-3, side, exposure=exposure)

    def official_seconds(self):
        value = self.lib.sim_official_seconds(self.handle)
        return None if np.isnan(value) else value

    def after_gate(self, gate=0, side=0., height=0., speed=0., exposure=1., breakup=0., video_seed=1):
        self.lib.sim_reset_after_gate(self.handle, gate, side, height, speed, exposure,
                                      breakup, video_seed, self.image.ctypes.data)
        return self.image

    def expert_action(self):
        action = np.empty(4, dtype=np.float32)
        self.lib.sim_expert_action(self.handle, action.ctypes.data)
        return action

    def expert_features(self):
        targets = np.empty(12, dtype=np.float32)
        self.lib.sim_expert_features(self.handle, targets.ctypes.data)
        return targets

    def step(self, action):
        image, reward, status = self.advance(action)
        return image, reward, status in (1, 3, 4), status == 2

    def advance(self, action):
        native = np.ascontiguousarray(action, dtype=np.float32)
        if native.shape != (4,):
            raise ValueError("Expected four policy action values.")
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
