"""Bind the project's RGB+IMU policy to Elodin's Betaflight RC channels."""
import atexit
import ctypes as ct
import json
import os
from pathlib import Path
from struct import unpack
import subprocess

import numpy as np
from solver.api import RCCommand

from training.betaflight_link import encode_rc


contact = lambda: False
hover = .17
control_rate = 30
last_expert_sample = -1
pilot = None
frames = imu = None
held = RCCommand()
last_command = held
expert = None
camera = None


def rigid_state(update):
    return (*update.world_pos[4:], *update.world_vel[3:], *update.world_pos[:4], *update.gyro)


class CameraFeed:
    """Render the Rust camera from Elodin state, without stepping the Rust plant."""
    def __init__(self, lib, path, fps, breakup, exposure, seed, dataset=None):
        from training.sim import Environment

        self.width, self.height = 480, 360
        self.env = Environment(lib, self.width, self.height)
        self.env.reset(-3, exposure=exposure, breakup=breakup, video_seed=seed)
        self.path, self.fps = path, fps
        self.encoder = None
        self.frame_number = -1
        self.actions = []
        self.dataset = dataset
        self.samples = []
        self.closed = False
        atexit.register(self.close)

    def update(self, sensors):
        if self.closed:
            return None
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

    def record_command(self, command):
        if self.path:
            self.actions.append(mode2_sticks(command))

    def record_demonstration(self, image, update, command):
        if self.dataset and update.t >= .75 and command.arm >= 1700:
            frame, inertial = policy_observation(image, update)
            self.samples.append((frame, inertial, update.t, mode2_sticks(command)))

    def save_demonstration(self):
        if self.dataset:
            frames, inertial, times, actions = zip(*self.samples)
            self.dataset.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.dataset, frames=np.asarray(frames),
                                actions=np.asarray(actions, dtype=np.float32),
                                timestamps=np.asarray(times),
                                weights=np.ones(len(times), dtype=np.float32))
            np.save(self.dataset.with_suffix(".imu.npy"), np.asarray(inertial, dtype=np.float32))
            print(f"[A2RL] Saved {len(times)} RGB/IMU/Mode 2 samples to {self.dataset}", flush=True)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.encoder is not None:
            encoder, self.encoder = self.encoder, None
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError("FFmpeg could not write the Elodin FPV replay.")
            self.path.with_suffix(".controls.json").write_text(json.dumps({
                "sample_rate": self.fps, "actions": self.actions,
            }, separators=(",", ":")) + "\n")
            print(f"[A2RL] FPV replay: {len(self.actions)} frames written to {self.path}", flush=True)
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
        # The firmware and rotor lag carry thrust through the launch command
        # transition. Limit the initial pulse for the low first gate.
        if update.last_gate_passed < 0 and state[2] < .55:
            command[1] = min(command[1], .1)
        return rc_command(command)

    expert = action


def rc_command(sticks):
    roll, pitch, throttle, yaw = unpack("<4H", encode_rc(sticks)[5:-1])
    return RCCommand(throttle=throttle, roll=roll, pitch=pitch, yaw=yaw, arm=1800)


def mode2_sticks(command):
    return [(channel - 1500) / 500 for channel in
            (command.yaw, command.throttle, command.roll, command.pitch)]


def policy_observation(image, update):
    import cv2

    frame = cv2.resize(image, (96, 72), interpolation=cv2.INTER_AREA)
    inertial = np.concatenate((update.gyro, update.accel)).astype(np.float32)
    return frame, inertial


def autopilot(update):
    global last_command
    ended = contact() or update.next_gate_index < 0
    image = camera.update(update) if camera is not None else None
    if image is not None:
        # The harness sends the previous command before sampling this tick.
        camera.record_command(last_command)
    if ended and camera is not None:
        camera.close()
    last_command = choose_command(update, image, ended)
    if image is not None and expert is not None:
        camera.record_demonstration(image, update, last_command)
    return last_command


def choose_command(update, image, ended):
    global pilot, frames, imu, held, last_expert_sample
    if ended or update.t < .5:
        return RCCommand()
    if update.t < .75:
        held = RCCommand(arm=1800)
        return held
    if expert is not None:
        sample = int(update.t * control_rate + 1e-6)
        if sample != last_expert_sample:
            held = expert(update)
            last_expert_sample = sample
        return held
    path = os.environ.get("A2RL_ELODIN_POLICY")
    if not path:
        # Short, explicit integration smoke test. No racing score is expected.
        throttle = hover + .065 if update.t < 1.5 else hover
        return RCCommand(arm=1800, throttle=round(1000 + throttle * 1000))
    if image is None:
        return held
    if pilot is None:
        import torch
        from training.sim import load_sim
        from training.watch import replay_agent

        torch.set_num_threads(2)
        pilot, _ = replay_agent(Path(path).read_bytes(), load_sim())
        if pilot.imu_encoder is None:
            raise ValueError("Elodin requires an RGB+IMU policy checkpoint.")
        print(f"[A2RL] RGB+IMU policy loaded ({pilot.stack} observations per action)", flush=True)
    image, sensor = policy_observation(image, update)
    if frames is None:
        frames = np.repeat(image[None], pilot.stack, axis=0)
        imu = np.repeat(sensor[None], pilot.stack, axis=0)
    else:
        frames[:-1], frames[-1] = frames[1:], image
        imu[:-1], imu[-1] = imu[1:], sensor
    sticks = pilot.action(frames[None], imu=imu[None])[0]
    held = rc_command(sticks)
    return held
