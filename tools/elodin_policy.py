"""Bind the project's RGB+IMU policy to Elodin's Betaflight RC channels."""
import atexit
import ctypes as ct
import os
from pathlib import Path
from struct import unpack
import subprocess

import numpy as np
from solver.api import RCCommand

from training.betaflight_link import encode_rc


contact = lambda: False
hover = .17
pilot = None
frames = imu = None
held = RCCommand()
expert = None
camera = None


def rigid_state(update):
    return (*update.world_pos[4:], *update.world_vel[3:], *update.world_pos[:4], *update.gyro)


class CameraFeed:
    """Render the Rust camera from Elodin state, without stepping the Rust plant."""
    def __init__(self, lib, path, fps, breakup, exposure, seed):
        from training.sim import Environment

        self.width, self.height = 480, 360
        self.env = Environment(lib, self.width, self.height)
        self.env.reset(-3, exposure=exposure, breakup=breakup, video_seed=seed)
        self.path, self.fps = path, fps
        self.encoder = None
        self.frame_number = -1
        atexit.register(self.close)

    def update(self, sensors):
        number = int(sensors.t * self.fps + 1e-6)
        if number == self.frame_number:
            return None
        self.frame_number = number
        state = rigid_state(sensors)
        if not np.isfinite(state).all():
            return None
        image = self.env.render_pose(state, sensors.t)
        if self.path:
            if self.encoder is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.encoder = subprocess.Popen([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo",
                    "-pix_fmt", "rgb24", "-s", f"{self.width}x{self.height}", "-r", str(self.fps), "-i", "pipe:0",
                    "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path),
                ], stdin=subprocess.PIPE)
            self.encoder.stdin.write(image.tobytes())
        return image

    def close(self):
        if self.encoder is not None:
            encoder, self.encoder = self.encoder, None
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError("FFmpeg could not write the Elodin FPV replay.")
            print(f"[A2RL] FPV replay: {self.frame_number + 1} frames written to {self.path}", flush=True)
        self.env.close()


def configure_expert(lib, season2):
    global expert
    handle = lib.sim_create(1, 1)
    lib.sim_set_season2_course(handle, season2)
    lib.sim_set_flow_teacher(handle, True)
    pixel = (ct.c_ubyte * 3)()
    lib.sim_reset(handle, -3, 0., 0., 1., 0., 1, ct.addressof(pixel))
    lib.sim_external_expert_action.argtypes = [
        ct.c_void_p, ct.POINTER(ct.c_float), ct.c_size_t, ct.POINTER(ct.c_float)]
    lib.sim_external_expert_action.restype = ct.c_bool
    state, command = (ct.c_float * 13)(), (ct.c_float * 4)()

    def action(update):
        state[:] = rigid_state(update)
        if not lib.sim_external_expert_action(handle, state, update.last_gate_passed + 1, command):
            return RCCommand()
        return rc_command(command)

    expert = action


def rc_command(sticks):
    roll, pitch, throttle, yaw = unpack("<4H", encode_rc(sticks)[5:-1])
    return RCCommand(throttle=throttle, roll=roll, pitch=pitch, yaw=yaw, arm=1800)


def autopilot(update):
    global pilot, frames, imu, held
    image = camera.update(update) if camera is not None else None
    if contact() or update.t < .5 or update.next_gate_index < 0:
        return RCCommand()
    if update.t < .75:
        held = RCCommand(arm=1800)
        return held
    if expert is not None:
        if update.tick % 10 == 0:
            held = expert(update)
        return held
    path = os.environ.get("A2RL_ELODIN_POLICY")
    if not path:
        # Short, explicit integration smoke test. No racing score is expected.
        throttle = hover + .065 if update.t < 1.5 else hover
        return RCCommand(arm=1800, throttle=round(1000 + throttle * 1000))
    if image is None:
        return held
    import cv2

    if pilot is None:
        import torch
        from training.sim import load_sim
        from training.watch import replay_agent

        torch.set_num_threads(2)
        pilot, _ = replay_agent(Path(path).read_bytes(), load_sim())
        if pilot.imu_encoder is None:
            raise ValueError("Elodin requires an RGB+IMU policy checkpoint.")
        print(f"[A2RL] RGB+IMU policy loaded ({pilot.stack} observations per action)", flush=True)
    image = cv2.resize(image, (96, 72), interpolation=cv2.INTER_AREA)
    sensor = np.concatenate((update.gyro, update.accel)).astype(np.float32)
    if frames is None:
        frames = np.repeat(image[None], pilot.stack, axis=0)
        imu = np.repeat(sensor[None], pilot.stack, axis=0)
    else:
        frames[:-1], frames[-1] = frames[1:], image
        imu[:-1], imu[-1] = imu[1:], sensor
    sticks = pilot.action(frames[None], imu=imu[None])[0]
    held = rc_command(sticks)
    return held
