"""Native simulation bindings. The policy gets HWC uint8 camera pixels."""
import ctypes as ct
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FLOAT_POINTER = ct.POINTER(ct.c_float)
LEGACY_PHYSICS = {"engine": "rust", "model_version": 1}


def physics_identity(lib):
    engine = os.environ.get("A2RL_PHYSICS_ENGINE", "rust")
    if engine != "rust":
        raise ValueError("Batch environments use Rust. Run tools/elodin_a2rl.py for Elodin validation.")
    physics = {"engine": engine, "model_version": lib.sim_physics_model_version()}
    if course_profile() != "2025":
        physics["course"] = course_profile()
        physics["layout_revision"] = 3
    return physics


def course_profile():
    name = os.environ.get("A2RL_COURSE", "2025")
    if name not in ("2025", "2026-practice"):
        raise ValueError(f"Unknown course: {name}")
    return name


def reward_profile():
    profile = os.environ.get("A2RL_REWARD_PROFILE", "standard")
    if profile not in ("standard", "pace"):
        raise ValueError(f"Unknown reward profile: {profile}")
    return profile


def teacher_profile():
    profile = os.environ.get("A2RL_TEACHER_PROFILE", "standard")
    if profile not in ("standard", "flow"):
        raise ValueError(f"Unknown teacher profile: {profile}")
    return profile

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
    if hasattr(lib, "sim_course_gate_count"):
        lib.sim_course_gate_count.argtypes = [ct.c_bool]
        lib.sim_course_gate_count.restype = ct.c_size_t
    lib.sim_hover_stick.argtypes = []
    lib.sim_hover_stick.restype = ct.c_float
    lib.sim_physics_model_version.argtypes = []
    lib.sim_physics_model_version.restype = ct.c_uint32
    lib.sim_reward_model_version.argtypes = []
    lib.sim_reward_model_version.restype = ct.c_uint32
    lib.sim_pace_reward_version.argtypes = []
    lib.sim_pace_reward_version.restype = ct.c_uint32
    lib.sim_set_reward_profile.argtypes = [ct.c_void_p, ct.c_bool]
    lib.sim_set_reward_profile.restype = None
    lib.sim_flow_stats.argtypes = [ct.c_void_p, FLOAT_POINTER]
    lib.sim_flow_stats.restype = None
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
    lib.sim_set_flow_teacher.argtypes = [ct.c_void_p, ct.c_bool]
    lib.sim_set_flow_teacher.restype = None
    if hasattr(lib, "sim_set_season2_course"):
        lib.sim_set_season2_course.argtypes = [ct.c_void_p, ct.c_bool]
        lib.sim_set_season2_course.restype = None
    lib.sim_sprint_window.argtypes = [ct.c_void_p]
    lib.sim_sprint_window.restype = ct.c_bool
    lib.sim_expert_features.argtypes = [ct.c_void_p, ct.c_void_p]
    lib.sim_expert_features.restype = None
    lib.sim_step.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_float)]
    lib.sim_step.restype = ct.c_int
    lib.sim_rigid_state.argtypes = [ct.c_void_p, FLOAT_POINTER]
    lib.sim_rigid_state.restype = None
    if hasattr(lib, "sim_render_pose"):
        lib.sim_render_pose.argtypes = [ct.c_void_p, FLOAT_POINTER, ct.c_double, ct.c_void_p]
        lib.sim_render_pose.restype = ct.c_bool
    lib.sim_lap_stats.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_float)]
    lib.sim_lap_stats.restype = None
    lib.sim_official_seconds.argtypes = [ct.c_void_p]
    lib.sim_official_seconds.restype = ct.c_float
    return lib


def gate_count(lib):
    if course_profile() == "2025":
        return lib.sim_gate_count()
    if not hasattr(lib, "sim_course_gate_count"):
        raise ValueError("Rebuild the simulator library for the 2026 course.")
    return lib.sim_course_gate_count(True)


class Environment:
    def __init__(self, lib, width, height):
        self.lib = lib
        profile = reward_profile()
        teacher = teacher_profile()
        physics_identity(lib)
        self.handle = lib.sim_create(width, height)
        if not self.handle:
            raise ValueError("Unsupported camera resolution.")
        lib.sim_set_reward_profile(self.handle, profile == "pace")
        lib.sim_set_flow_teacher(self.handle, teacher == "flow")
        if course_profile() == "2026-practice":
            if not hasattr(lib, "sim_set_season2_course"):
                self.close()
                raise ValueError("Build the simulator library with support for the 2026 course.")
            lib.sim_set_season2_course(self.handle, True)
        self.image = np.empty((height, width, 3), dtype=np.uint8)
        self.reward = ct.c_float()
        self._imu = None

    def reset(self, gate=-1, side=0., height=0., exposure=1., breakup=0., video_seed=1):
        self.lib.sim_reset(self.handle, gate, side, height, exposure, breakup, video_seed, self.image.ctypes.data)
        self._imu = None
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
        self._imu = None
        return self.image

    def expert_action(self):
        action = np.empty(4, dtype=np.float32)
        self.lib.sim_expert_action(self.handle, action.ctypes.data)
        return action

    def sprint_window(self):
        return bool(self.lib.sim_sprint_window(self.handle))

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
        previous = self.rigid_state() if self._imu is not None else None
        status = self.lib.sim_step(self.handle, native.ctypes.data, self.image.ctypes.data, ct.byref(self.reward))
        if status < 0:
            raise ValueError("Non-finite action or step after episode end.")
        if previous is not None:
            state = self.rigid_state()
            specific_force = (state[3:6] - previous[3:6]) * 30.
            specific_force[2] += 9.80665
            self._imu[:-1] = self._imu[1:]
            self._imu[-1] = self._body_imu(state, specific_force)
        return self.image, self.reward.value, status

    @staticmethod
    def _body_imu(state, specific_force):
        # Undo the body-to-world quaternion to express specific force in sensor axes.
        axis, scalar = -state[6:9], state[9]
        twice_cross = 2. * np.cross(axis, specific_force)
        force = specific_force + scalar * twice_cross + np.cross(axis, twice_cross)
        return np.concatenate((state[10:13], force))

    def imu_window(self, stack):
        if self._imu is None or len(self._imu) != stack:
            initial = self._body_imu(self.rigid_state(), np.array([0., 0., 9.80665], dtype=np.float32))
            self._imu = np.repeat(initial[None], stack, axis=0)
        return self._imu

    def rigid_state(self):
        state = (ct.c_float * 13)()
        self.lib.sim_rigid_state(self.handle, state)
        return np.asarray(state).copy()

    def render_pose(self, state, elapsed):
        pose = np.ascontiguousarray(state, dtype=np.float32)
        if pose.shape != (13,) or not self.lib.sim_render_pose(
                self.handle, pose.ctypes.data_as(FLOAT_POINTER), elapsed, self.image.ctypes.data):
            raise ValueError("Expected a finite position, velocity, quaternion and body rates.")
        return self.image

    def lap_stats(self):
        gates, best = ct.c_uint32(), ct.c_float()
        self.lib.sim_lap_stats(self.handle, ct.byref(gates), ct.byref(best))
        return gates.value, None if np.isnan(best.value) else best.value

    def flow_stats(self):
        values = (ct.c_float * 2)()
        self.lib.sim_flow_stats(self.handle, values)
        return values[0], values[1]

    def close(self):
        if self.handle:
            self.lib.sim_destroy(self.handle)
            self.handle = None
