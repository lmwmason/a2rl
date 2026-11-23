"""Bind the project's RGB+IMU policy to Elodin's Betaflight RC channels."""
import atexit
from collections import deque
import ctypes as ct
import json
import os
from pathlib import Path
from queue import SimpleQueue
from struct import unpack
import subprocess
from threading import Thread

import numpy as np
from solver.api import RCCommand

from training.betaflight_link import encode_rc
from training.inertial import RelativeAttitude


contact = lambda: False
hover = .17
control_rate = 30
last_expert_sample = -1
pilot = None
policy_steps = 0
relative_attitude = RelativeAttitude()
forecasts = deque()
use_late_controller = False
boundary_ticks = 0
frames = imu = None
held = RCCommand()
last_command = held
expert = None
shadow_expert = False
record_full_rollout = False
assisted_rollout = False
completion_profile = {}
handoff = None
handoff_frames = 0
handoff_ready_since = None
intervention_ticks = 0
reference_command = None
student_influence_frames = 0
reduced_assistance_frames = 0
camera = None
heat = None
heat_sample = -1
peer_positions = []
peer_contact = False
heat_failure = None
heat_airborne = False
heat_arm_time = .5
heat_epoch = None


def connect_heat(path, index):
    from tools.elodin_heat import HeatClient

    global heat, heat_arm_time
    heat = HeatClient(path, index)
    heat_arm_time = 3. + float(os.environ.get("A2RL_HEAT_LAUNCH_DELAY", "0"))


def rigid_state(update):
    return (*update.world_pos[4:], *update.world_vel[3:], *update.world_pos[:4], *update.gyro)


class CameraFeed:
    """Render FPV camera frames from Elodin state."""
    def __init__(self, lib, path, fps, breakup, exposure, seed, dataset=None, *, mount_pitch=None, chase=False):
        from training.sim import CameraRenderer

        self.width, self.height = 480, 360
        self.env = CameraRenderer(lib, self.width, self.height,
                                  exposure=exposure, breakup=breakup, video_seed=seed, mount_pitch=mount_pitch)
        self.path, self.fps = path, fps
        self.encoder = None
        self.chase = CameraRenderer(lib, self.width, self.height) if path and chase else None
        self.chase_path = path.with_name(path.stem + "-chase.mp4") if self.chase else None
        self.chase_queue = SimpleQueue() if self.chase is not None else None
        self.chase_worker = None
        self.chase_failure = None
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
        image = self.env.render_pose(state, sensors.t, peer_positions)
        if self.path:
            if self.encoder is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.encoder = self.open_encoder(self.path)
                if self.chase is not None:
                    self.chase_worker = Thread(target=self.write_chase, daemon=True)
                    self.chase_worker.start()
            self.encoder.stdin.write(image.tobytes())
            if self.chase_queue is not None:
                self.chase_queue.put((state, sensors.t, tuple(peer_positions)))
        return image

    def open_encoder(self, path):
        return subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{self.width}x{self.height}", "-r", str(self.fps), "-i", "pipe:0",
            "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
        ], stdin=subprocess.PIPE)

    def write_chase(self):
        try:
            encoder = self.open_encoder(self.chase_path)
            try:
                while (sample := self.chase_queue.get()) is not None:
                    state, elapsed, peers = sample
                    frame = self.chase.render_pose(state, elapsed, peers, chase=True)
                    encoder.stdin.write(frame.tobytes())
            finally:
                encoder.stdin.close()
                if encoder.wait():
                    raise RuntimeError("FFmpeg could not write the third-person replay.")
        except (OSError, RuntimeError, ValueError) as error:
            self.chase_failure = error

    def record_command(self, command):
        if self.path:
            self.actions.append(mode2_sticks(command))

    def record_demonstration(self, image, update, command, weight=1., executed=None):
        if self.dataset and update.t >= .75 and command.arm >= 1700:
            frame, inertial = policy_observation(image, update)
            self.samples.append((frame, inertial, update.t, mode2_sticks(command), weight, update.next_gate_index,
                                 mode2_sticks(command if executed is None else executed)))

    def save_demonstration(self, passed=0, independent=False):
        if self.dataset and self.samples:
            frames, inertial, times, actions, weights, gates, executed = zip(*self.samples)
            self.dataset.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.dataset.with_suffix(".imu.npy"), np.asarray(inertial, dtype=np.float32))
            temporary = self.dataset.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, frames=np.asarray(frames),
                                actions=np.asarray(actions, dtype=np.float32),
                                timestamps=np.asarray(times),
                                weights=np.asarray(weights, dtype=np.float32), next_gate=np.asarray(gates, dtype=np.int16),
                                executed=np.asarray(executed, dtype=np.float32),
                                independent=bool(independent), passed=int(passed))
            temporary.replace(self.dataset)
            print(f"[A2RL] Saved {len(times)} RGB/IMU/Mode 2 samples to {self.dataset}", flush=True)

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.chase_worker is not None:
                self.chase_queue.put(None)
            if self.encoder is not None:
                self.encoder.stdin.close()
                if self.encoder.wait():
                    raise RuntimeError("FFmpeg could not write the FPV replay.")
            if self.chase_worker is not None:
                self.chase_worker.join()
                if self.chase_failure is not None:
                    raise RuntimeError("Could not record the third-person replay.") from self.chase_failure
            if self.encoder is not None:
                self.path.with_suffix(".controls.json").write_text(json.dumps({
                    "sample_rate": self.fps, "actions": self.actions,
                }, separators=(",", ":")) + "\n")
                print(f"[A2RL] FPV replay: {len(self.actions)} frames written to {self.path}", flush=True)
        finally:
            if self.chase_worker is not None:
                self.chase_worker.join()
            self.env.close()
            if self.chase is not None:
                self.chase.close()


def configure_expert(lib, season2):
    global expert
    handle = lib.sim_create(1, 1)
    lib.sim_set_season2_course(handle, season2)
    lib.sim_set_flow_teacher(handle, True)
    lib.sim_configure_scene(handle, lib.sim_course_camera_tilt(season2), 1., 0., 1)
    atexit.register(lib.sim_destroy, handle)
    if pace := os.environ.get("A2RL_ELODIN_TEACHER_PACE"):
        lib.sim_set_teacher_pace.argtypes = [ct.c_void_p, ct.c_float]
        lib.sim_set_teacher_pace.restype = ct.c_bool
        if not lib.sim_set_teacher_pace(handle, float(pace)):
            raise ValueError("Teacher pace must be between 0.60 and 1.00.")

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


def mode2_sticks(command):
    return [(channel - 1500) / 500 for channel in
            (command.yaw, command.throttle, command.roll, command.pitch)]


def policy_observation(image, update):
    import cv2

    frame = cv2.resize(image, (96, 72), interpolation=cv2.INTER_AREA)
    inertial = np.concatenate((update.gyro, update.accel)).astype(np.float32)
    return frame, inertial


def autopilot(update):
    global last_command, heat_sample, peer_positions, peer_contact, heat_failure, heat_airborne, heat_epoch
    if heat is not None:
        sample = int(update.t * control_rate + 1e-6)
        if sample != heat_sample:
            try:
                peer_positions, peer_contact, heat_epoch = heat.update(sample, update.world_pos[4:].tolist())
                heat_sample = sample
                heat_airborne |= update.world_pos[6] > .45
            except (OSError, ValueError) as error:
                heat_failure = error
                raise
    ended = contact() or peer_contact or update.next_gate_index < 0
    image = camera.update(update) if camera is not None else None
    if image is not None:
        # The harness sends the previous command before sampling this tick.
        camera.record_command(last_command)
    if ended and camera is not None:
        camera.close()
    last_command = choose_command(update, image, ended)
    if image is not None and expert is not None and not ended:
        if not shadow_expert:
            camera.record_demonstration(image, update, last_command)
        elif update.t >= .75 and (record_full_rollout or (update.t <= 3.75 and update.last_gate_passed < 0)):
            weight = 2. if record_full_rollout and update.last_gate_passed >= 0 else 1.
            if update.next_gate_index == completion_profile.get("focus_gate"):
                weight *= 2.
            if handoff is not None:
                weight = 5. if handoff_frames > 0 else .20
            if completion_profile.get("intervention") and intervention_ticks:
                weight = 4.
            target = reference_command if assisted_rollout else expert(update)
            camera.record_demonstration(image, update, target, weight, executed=last_command)
    return last_command


def choose_command(update, image, ended):
    global pilot, policy_steps, frames, imu, held, last_expert_sample, reference_command
    global student_influence_frames, reduced_assistance_frames, handoff_frames, intervention_ticks
    global handoff_ready_since, use_late_controller, boundary_ticks
    if ended or update.t < heat_arm_time:
        return RCCommand()
    if update.t < heat_arm_time + .25:
        held = RCCommand(arm=1800)
        return held
    if expert is not None and not shadow_expert:
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
        from training.agent import load_policy

        torch.set_num_threads(2)
        pilot = load_policy(Path(path).read_bytes())
        if pilot.imu_encoder is None:
            raise ValueError("Elodin requires an RGB+IMU policy checkpoint.")
        print(f"[A2RL] RGB+IMU policy loaded ({pilot.stack} observations per action)", flush=True)
    image, sensor = policy_observation(image, update)
    if pilot.imu_channels == 10:
        sensor = relative_attitude.augment(sensor, update.t)
    if frames is None:
        frames = np.repeat(image[None], pilot.stack, axis=0)
        imu = np.repeat(sensor[None], pilot.stack, axis=0)
    else:
        frames[:-1], frames[-1] = frames[1:], image
        imu[:-1], imu[-1] = imu[1:], sensor
    if handoff is not None:
        reference_command = expert(update)
        if update.next_gate_index >= handoff["after_gates"] and handoff_ready_since is None:
            handoff_ready_since = update.t
        if handoff_ready_since is None or update.t < handoff_ready_since + handoff.get("delay_s", 0.):
            held = reference_command
            return held
        handoff_frames += 1
    policy_steps += 1
    if pilot.action_horizon > 1:
        if pilot.protected_prefix:
            prediction, progress = pilot.forecast_with_boundary(frames[None], imu[None], use_late_controller)
            if not use_late_controller:
                boundary_ticks = boundary_ticks + 1 if progress >= .90 else 0
                if boundary_ticks >= 4:
                    use_late_controller = True
                    print(f"[A2RL] Learned route transition at t={update.t:.3f} s", flush=True)
                    forecasts.clear()
                    prediction, _ = pilot.forecast_with_boundary(frames[None], imu[None], True)
        else:
            prediction = pilot.forecast(frames[None], imu[None])
        forecasts.appendleft(prediction[0])
        if len(forecasts) > pilot.action_horizon:
            forecasts.pop()
        sticks = np.average([prediction[age] for age, prediction in enumerate(forecasts)],
                            axis=0, weights=pilot.forecast_decay ** np.arange(len(forecasts)))
    else:
        sticks = pilot.action(frames[None], imu=imu[None])[0]
    if handoff is not None:
        held = rc_command(sticks)
        return held
    if assisted_rollout:
        reference_command = expert(update)
        held = reference_command
        if completion_profile.get("intervention"):
            if update.next_gate_index < completion_profile.get("intervention_after", 0):
                return held
            reference = np.asarray(mode2_sticks(reference_command))
            tolerance = np.asarray([.16, .075, .18, .18])
            if np.any(np.abs(sticks - reference) > tolerance):
                intervention_ticks = 5
            if intervention_ticks:
                intervention_ticks -= 1
            else:
                held = rc_command(sticks)
                student_influence_frames += 1
                reduced_assistance_frames += 1
            return held
        # Give the learner brief control perturbations after takeoff while the
        # expert keeps the rollout close enough to reach the rest of the course.
        sample = int(update.t * control_rate + 1e-6)
        reliable = update.next_gate_index < completion_profile.get("reliable_gates", 0)
        relieved = reliable and update.t >= 1.2
        if (update.last_gate_passed >= 0 or relieved) and sample % (3 if relieved else 6) == 0:
            reference = np.asarray(mode2_sticks(reference_command))
            limit = np.asarray([.05, .03, .07, .07])
            delta = sticks - reference
            if relieved and np.all(np.abs(delta) <= 2 * limit):
                held = rc_command(sticks)
                reduced_assistance_frames += 1
            else:
                held = rc_command(reference + np.clip(delta, -limit, limit))
            student_influence_frames += 1
        return held
    held = rc_command(sticks)
    return held
