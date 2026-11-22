"""Bind the project's RGB+IMU policy to Elodin's Betaflight RC channels."""
import os
import ctypes as ct
from pathlib import Path
from struct import unpack

import numpy as np
from solver.api import RCCommand

from training.betaflight_link import encode_rc


contact = lambda: False
hover = .17
pilot = None
frames = imu = None
held = RCCommand()
expert = None


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
        state[:] = (*update.world_pos[4:], *update.world_vel[3:], *update.world_pos[:4], *update.gyro)
        if not lib.sim_external_expert_action(handle, state, update.last_gate_passed + 1, command):
            return RCCommand()
        return rc_command(command)

    expert = action


def rc_command(sticks):
    roll, pitch, throttle, yaw = unpack("<4H", encode_rc(sticks)[5:-1])
    return RCCommand(throttle=throttle, roll=roll, pitch=pitch, yaw=yaw, arm=1800)


def autopilot(update):
    global pilot, frames, imu, held
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
    if update.frame_rgba is None or not update.frame_fresh:
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
    image = cv2.resize(update.frame_rgba[..., :3], (96, 72), interpolation=cv2.INTER_AREA)
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
