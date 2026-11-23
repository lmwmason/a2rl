"""Course and camera bindings for state supplied by Elodin."""
import ctypes as ct
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FLOAT_POINTER = ct.POINTER(ct.c_float)


def course_profile():
    name = os.environ.get("A2RL_COURSE", "2025")
    if name not in ("2025", "2026-practice"):
        raise ValueError(f"Unknown course: {name}")
    return name


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
    if hasattr(lib, "sim_course_camera_tilt"):
        lib.sim_course_camera_tilt.argtypes = [ct.c_bool]
        lib.sim_course_camera_tilt.restype = ct.c_float
    lib.sim_create.argtypes = [ct.c_size_t, ct.c_size_t]
    lib.sim_create.restype = ct.c_void_p
    lib.sim_destroy.argtypes = [ct.c_void_p]
    lib.sim_destroy.restype = None
    lib.sim_configure_scene.argtypes = [ct.c_void_p, ct.c_float, ct.c_float, ct.c_float, ct.c_uint32]
    lib.sim_configure_scene.restype = None
    lib.sim_set_flow_teacher.argtypes = [ct.c_void_p, ct.c_bool]
    lib.sim_set_flow_teacher.restype = None
    if hasattr(lib, "sim_set_season2_course"):
        lib.sim_set_season2_course.argtypes = [ct.c_void_p, ct.c_bool]
        lib.sim_set_season2_course.restype = None
    if hasattr(lib, "sim_render_pose"):
        lib.sim_render_pose.argtypes = [ct.c_void_p, FLOAT_POINTER, ct.c_double, ct.c_void_p]
        lib.sim_render_pose.restype = ct.c_bool
    for render in (lib.sim_render_heat_pose, lib.sim_render_chase_pose):
        render.argtypes = [ct.c_void_p, FLOAT_POINTER, ct.c_double, FLOAT_POINTER, ct.c_size_t, ct.c_void_p]
        render.restype = ct.c_bool
    return lib


def gate_count(lib):
    if course_profile() == "2025":
        return lib.sim_gate_count()
    if not hasattr(lib, "sim_course_gate_count"):
        raise ValueError("Rebuild the simulator library for the 2026 course.")
    return lib.sim_course_gate_count(True)


class CameraRenderer:
    def __init__(self, lib, width, height, *, exposure=1., breakup=0., video_seed=1, mount_pitch=None):
        season2 = course_profile() == "2026-practice"
        self.mount_pitch = lib.sim_course_camera_tilt(season2) if mount_pitch is None else mount_pitch
        if not math.isfinite(self.mount_pitch) or not 0. <= self.mount_pitch < 90.:
            raise ValueError("Camera mount pitch must be finite and in [0, 90) degrees.")
        self.image = np.empty((height, width, 3), dtype=np.uint8)
        self.lib = lib
        self.handle = lib.sim_create(width, height)
        if not self.handle:
            raise ValueError("Unsupported camera resolution.")
        lib.sim_set_season2_course(self.handle, season2)
        lib.sim_configure_scene(self.handle, self.mount_pitch, exposure, breakup, video_seed)

    def place_gates(self, poses):
        function = self.lib.sim_set_gate_poses
        function.argtypes = [ct.c_void_p, FLOAT_POINTER, ct.c_size_t]
        function.restype = ct.c_bool
        poses = np.ascontiguousarray(poses, dtype=np.float32)
        if poses.ndim != 2 or poses.shape[1] != 4 or not function(
                self.handle, poses.ctypes.data_as(FLOAT_POINTER), len(poses)):
            raise ValueError("Invalid gate poses for the camera scene.")

    def render_pose(self, state, elapsed, peers=(), *, chase=False):
        pose = np.ascontiguousarray(state, dtype=np.float32)
        if pose.shape != (13,):
            raise ValueError("Expected a finite position, velocity, quaternion and body rates.")
        pointer = pose.ctypes.data_as(FLOAT_POINTER)
        if chase or len(peers):
            opponents = np.ascontiguousarray(peers, dtype=np.float32).reshape(-1, 3)
            render = self.lib.sim_render_chase_pose if chase else self.lib.sim_render_heat_pose
            valid = render(self.handle, pointer, elapsed, opponents.ctypes.data_as(FLOAT_POINTER),
                           len(opponents), self.image.ctypes.data)
        else:
            valid = self.lib.sim_render_pose(self.handle, pointer, elapsed, self.image.ctypes.data)
        if not valid:
            raise ValueError("Expected finite pose and opponent positions.")
        return self.image

    def close(self):
        if self.handle:
            self.lib.sim_destroy(self.handle)
            self.handle = None
