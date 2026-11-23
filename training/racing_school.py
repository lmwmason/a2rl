"""Train an FPV pilot with simulator expert corrections before speed RL."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from itertools import count
import json
import hashlib
from pathlib import Path
import time
from zipfile import ZipFile

import numpy as np
import torch

from . import checkpoint
from .agent import Agent, pixel_tensor, random_shift, stick_loss
from .betaflight_link import decode_rc, encode_rc
from .demonstrations import Demonstrations
from .sim import Environment, LEGACY_PHYSICS, course_profile, gate_count as course_gate_count, load_sim, physics_identity, teacher_profile
from .telemetry import FramePublisher
from .train import fresh_stack


class ExpertReplay:
    """Keep each RGB image once and reconstruct short histories inside episodes."""

    def __init__(self, rows, envs, height, width, stack, with_imu=False):
        self.frames = np.empty((rows, envs, height, width, 3), np.uint8)
        self.sticks = np.empty((rows, envs, 4), np.float32)
        self.features = np.empty((rows, envs, 12), np.float32)
        self.weights = np.empty((rows, envs), np.float32)
        self.starts = np.empty((rows, envs), np.int64)
        self.imu = np.empty((rows, envs, 6), np.float32) if with_imu else None
        self.rows, self.envs, self.stack = rows, envs, stack
        self.next_start = np.zeros(envs, np.int64)
        self.count = 0

    def append(self, frame, labels, features, confidence, ended, imu=None):
        index = self.count % self.rows
        self.frames[index] = frame
        self.sticks[index] = labels
        self.features[index] = features
        self.weights[index] = confidence
        self.starts[index] = self.next_start
        if self.imu is not None:
            self.imu[index] = imu
        self.count += 1
        self.next_start[ended] = self.count

    def sample(self, rng, batch):
        first = max(0, self.count - self.rows)
        row = rng.integers(first, self.count, size=batch)
        env = rng.integers(self.envs, size=batch)
        indexes = row[:, None] + np.arange(1 - self.stack, 1)
        indexes = np.maximum(indexes, np.maximum(first, self.starts[row % self.rows, env])[:, None])
        batch = (self.frames[indexes % self.rows, env[:, None]],
                self.sticks[row % self.rows, env], self.features[row % self.rows, env],
                self.weights[row % self.rows, env])
        return (*batch, self.imu[indexes % self.rows, env[:, None]]) if self.imu is not None else batch


REPLAY_FIELDS = ("frames", "sticks", "features", "weights", "starts")
EVALUATION_SEED = 9127


def save_school(path, agent, aux, optimizer, buffer, rng, progress):
    payload = {
        "agent": agent.weights(), "aux": aux.state_dict(), "optimizer": optimizer.state_dict(),
        "progress": progress, "count": buffer.count, "layout": (tuple(buffer.frames.shape), buffer.stack),
        "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if agent.device == "cuda" else [],
    }
    with checkpoint.atomic_archive(path) as archive:
        data = BytesIO()
        torch.save(payload, data)
        archive.writestr("school.pt", data.getbuffer())
        for field in REPLAY_FIELDS + (("imu",) if buffer.imu is not None else ()):
            checkpoint.write_array(archive, field, getattr(buffer, field)[:min(buffer.count, buffer.rows)])


def restore_school(path, agent, aux, optimizer, buffer, rng, physics):
    with ZipFile(path) as archive:
        payload = torch.load(BytesIO(archive.read("school.pt")), map_location="cpu", weights_only=True)
        if payload["progress"].get("physics", LEGACY_PHYSICS) != physics:
            raise ValueError("Use the original physics to resume, or initialize a new run from weights.")
        if payload["layout"] != (tuple(buffer.frames.shape), buffer.stack):
            raise ValueError("Use the original environment, replay and frame-stack settings to resume.")
        buffer.count = payload["count"]
        if ("imu.npy" in archive.namelist()) != (buffer.imu is not None):
            raise ValueError("Use the original IMU setting to resume.")
        for field in REPLAY_FIELDS + (("imu",) if buffer.imu is not None else ()):
            getattr(buffer, field)[:min(buffer.count, buffer.rows)] = checkpoint.read_array(archive, field)
        buffer.next_start.fill(buffer.count)
    agent.set_weights(payload["agent"])
    aux.load_state_dict(payload["aux"])
    optimizer.load_state_dict(payload["optimizer"])
    rng.bit_generator.state = payload["numpy_rng"]
    torch.set_rng_state(payload["torch_rng"])
    if agent.device == "cuda" and payload["cuda_rng"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload["progress"]


def evaluate_student(agent, env, stack, seed, episodes, teacher=False):
    """Use held-out ground starts. No intervention is permitted in student evaluations."""
    n_gates = course_gate_count(env.lib) * 2
    passes, crashes, times = [], 0, []
    approach_delay, crossing_speed = 0., 0.
    longest_slow_seconds = 0.
    rng = np.random.default_rng(seed)
    for i in range(episodes):
        if i == 0:
            frame = env.official_start()
        else:
            frame = env.official_start(side=float(rng.uniform(-.22, .22)),
                                     exposure=float(rng.uniform(.90, 1.10)))
        obs = fresh_stack(frame, stack)
        slow_ticks = 0
        for tick in range(1, 180 * 30 + 1):
            action = decode_rc(encode_rc(env.expert_action())) if teacher else agent.command(obs, env)
            frame, _, status = env.advance(action)
            gates, _ = env.lap_stats()
            if gates > 0:
                velocity = env.rigid_state()[3:6]
                slow_ticks += int(velocity @ velocity < 4.)
            if status or gates >= n_gates:
                passes.append(gates)
                delay, speed_sum = env.flow_stats()
                approach_delay += delay
                crossing_speed += speed_sum
                if status == 4 and gates == n_gates:
                    times.append(env.official_seconds())
                    longest_slow_seconds = max(longest_slow_seconds, slow_ticks / 30.)
                crashes += int(status == 1)
                break
            obs[:-1] = obs[1:]
            obs[-1] = frame
    return {
        "gates": passes,
        "gates_per_race": n_gates,
        "gate_one_rate": sum(x >= 1 for x in passes) / episodes,
        "gate_two_rate": sum(x >= 2 for x in passes) / episodes,
        "two_lap_rate": len(times) / episodes,
        "crash_rate": crashes / episodes,
        "best_two_lap_s": min(times, default=None),
        "worst_two_lap_s": max(times, default=None),
        "longest_completed_run_below_2_m_s": longest_slow_seconds if times else None,
        "mean_gate_approach_delay_seconds": approach_delay / episodes,
        "mean_gate_crossing_speed_m_s": crossing_speed / sum(passes) if sum(passes) else None,
    }


def evaluation_rank(result):
    lap_time = result["best_two_lap_s"]
    crossing_speed = result["mean_gate_crossing_speed_m_s"]
    return (result["two_lap_rate"], min(result["gates"]),
            float(np.mean(result["gates"])), -result["crash_rate"],
            -lap_time if lap_time is not None else -float("inf"),
            -result["mean_gate_approach_delay_seconds"],
            crossing_speed if crossing_speed is not None else 0.)


def matches_teacher_pace(result, expert):
    return (result["two_lap_rate"] == 1.
            and result["worst_two_lap_s"] <= 1.20 * expert["worst_two_lap_s"]
            and result["longest_completed_run_below_2_m_s"]
            <= expert["longest_completed_run_below_2_m_s"] + 1.
            and result["mean_gate_crossing_speed_m_s"] >= .80 * expert["mean_gate_crossing_speed_m_s"])


def demonstration_loss(agent, demonstrations, batch):
    frames, controls, confidence, *imu = demonstrations.sample(batch)
    pixels = random_shift(pixel_tensor(frames, agent.device).float() / 255. - .5, pad=1)
    # Preserve brief high-throttle examples in the real-pilot demonstration set.
    emphasis = 1. + np.maximum(0., controls[:, 1] - .5)
    return stick_loss(agent.actor(agent.policy_features(pixels, imu[0] if imu else None)).tanh(),
                      controls, confidence * emphasis)


def demonstration_error(agent, sample):
    frames, controls, _, *imu = sample
    return float(np.abs(agent.action(frames, imu=imu[0] if imu else None) - controls).mean())


def training_reset(env, index, rng, target, continuous=False):
    breakup = .18 if rng.random() < .25 else 0.
    video_seed = int(rng.integers(1 << 32))
    exposure = float(rng.uniform(.82, 1.18))
    if continuous:
        return env.official_start(side=rng.uniform(-.22, .22), exposure=exposure,
                                  breakup=breakup, video_seed=video_seed)
    if index % 4 < 2:
        return env.ground_start(side=rng.uniform(-.25, .25), exposure=exposure,
                                breakup=breakup, video_seed=video_seed)
    gate = (target - 1) % course_gate_count(env.lib)
    if index % 4 == 2 and target > 1:
        gate = (gate - 1) % course_gate_count(env.lib)
        return env.after_gate(gate, side=rng.uniform(-.20, .20), height=rng.uniform(-.17, .17),
                              speed=rng.uniform(.5, 3.), exposure=exposure, breakup=breakup, video_seed=video_seed)
    return env.reset(gate, side=rng.uniform(-.38, .38),
                     height=rng.uniform(-.26, .26), exposure=exposure, breakup=breakup, video_seed=video_seed)


def collect(envs, workers, agent, buffer, obs, rng, ticks, teacher_probability, target, publisher=None,
            continuous_envs=(), full_course=False):
    ended = np.zeros(len(envs), dtype=bool)
    authority = np.ones(len(envs), dtype=bool)
    gate_count = np.zeros(len(envs), dtype=np.int32)
    delay = np.zeros(len(envs), dtype=np.float32)
    crossing_ticks = np.zeros(len(envs), dtype=np.int32)
    pace_assist_ticks = np.zeros(len(envs), dtype=np.int32)
    clean_indexes = list(continuous_envs)
    pace_training = bool(continuous_envs) or full_course
    if pace_training:
        for i, env in enumerate(envs):
            gate_count[i], _ = env.lap_stats()
            delay[i], _ = env.flow_stats()
    accidents = pace_samples = course_finishes = teacher_finishes = 0
    for tick in range(ticks):
        imu = np.stack([env.imu_window(buffer.stack) for env in envs]) if buffer.imu is not None else None
        labels = np.asarray([env.expert_action() for env in envs], dtype=np.float32)
        features = np.asarray([env.expert_features() for env in envs], dtype=np.float32)
        if teacher_probability >= 1.:
            assisted = np.ones(len(envs), dtype=bool)
            action = labels.copy()
        else:
            if tick % 6 == 0:
                authority = rng.random(len(envs)) < teacher_probability
            student = agent.action(obs, imu=imu) if imu is not None else agent.action(obs)
            errors = np.abs(student - labels).max(axis=1)
            intervene = (errors > .48) & (rng.random(len(envs)) < .55)
            assisted = authority | intervene | (pace_assist_ticks > 0)
            action = np.where(assisted[:, None], labels, student)
        # Small disturbances create new camera states for exact corrective labels.
        if tick % 7 in (0, 1):
            action = (action + rng.normal(0., [.025, .014, .04, .04], action.shape)).clip(-1, 1).astype(np.float32)
        if clean_indexes:
            assisted[clean_indexes] = True
            action[clean_indexes] = labels[clean_indexes]
        if imu is not None:
            action = [decode_rc(encode_rc(command)) for command in action]
        following = np.empty_like(obs[:, -1])
        emphasis = 1. + 1.25 * np.abs(labels[:, [0, 2, 3]]).max(axis=1)
        if full_course:
            emphasis += np.maximum(0., labels[:, 1] - .5)
        tasks = [workers.submit(env.advance, command) for env, command in zip(envs, action)]
        for i, task in enumerate(tasks):
            image, _, status = task.result()
            gates, _ = envs[i].lap_stats()
            threshold = target if i % 4 < 2 else 2
            continuous = i in continuous_envs
            ended[i] = bool(status) or (not full_course and not continuous and gates >= threshold)
            accidents += int(status == 1)
            course_finishes += int(status == 4)
            teacher_finishes += int(status == 4 and continuous)
            if pace_training:
                approach_delay, _ = envs[i].flow_stats()
                slow = approach_delay - delay[i] > .006
                crossed = gates > gate_count[i]
                # Learn the corrective sticks during a loss of pace and the first half-second after a gate.
                crossing_ticks[i] = 15 if crossed else max(0, crossing_ticks[i] - 1)
                if slow or crossing_ticks[i]:
                    emphasis[i] *= 2.5
                    pace_samples += 1
                pace_assist_ticks[i] = 6 if slow else max(0, pace_assist_ticks[i] - 1)
                delay[i] = 0. if ended[i] else approach_delay
                gate_count[i] = 0 if ended[i] else gates
                if ended[i]:
                    crossing_ticks[i] = 0
                    pace_assist_ticks[i] = 0
            if i == 0 and publisher is not None:
                publisher.show(image, gates=gates, step=buffer.count * len(envs),
                               phase="IMITATION", status=status, driver="Teacher" if assisted[0] else "Student")
            following[i] = training_reset(envs[i], i, rng, target, full_course or continuous) if ended[i] else image
        buffer.append(obs[:, -1], labels, features, emphasis, ended, None if imu is None else imu[:, -1])
        obs[:, :-1] = obs[:, 1:]
        obs[:, -1] = following
        obs[ended] = following[ended, None]
    return accidents, pace_samples, course_finishes, teacher_finishes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--rounds", type=int, default=0, help="Limit rounds for a smoke test. Zero runs until qualified.")
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--collect-ticks", type=int, default=600)
    parser.add_argument("--updates", type=int, default=360)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--capacity", type=int, default=56_000)
    parser.add_argument("--stack", type=int, default=8)
    parser.add_argument("--imu", action="store_true", help="Use RGB and simulated gyro/acceleration history.")
    parser.add_argument("--backbone", choices=("cnn", "mobilenet-v3-small"), default="cnn")
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--initialize", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--consolidation-rounds", type=int, default=6)
    parser.add_argument("--teacher-warmup-rounds", type=int, default=0)
    parser.add_argument("--full-course", action="store_true",
                        help="Train complete two-lap episodes from takeoff without gate-count resets.")
    parser.add_argument("--teacher-only", action="store_true")
    parser.add_argument("--external-qualification", action="store_true",
                        help="Keep training until the requested round, leaving qualification to firmware evaluation.")
    parser.add_argument("--human-demos", nargs="+", type=Path, default=[],
                        help="Synchronized RGB and real Mode 2 inputs used alongside simulator expert labels.")
    parser.add_argument("--human-held-out", nargs="+", type=Path, default=[],
                        help="Separate real-pilot flights used only to report action error.")
    parser.add_argument("--firmware-launch-demos", type=Path,
                        help="Directory containing completed RGB+IMU launch corrections from Betaflight SITL.")
    parser.add_argument("--monorace-demos", nargs="+", type=Path, default=[],
                        help="MonoRace FPV with estimated controls from the official autonomous flight logs.")
    parser.add_argument("--monorace-held-out", nargs="+", type=Path, default=[],
                        help="Separate MonoRace flights used only to report estimated control error.")
    args = parser.parse_args()
    if min(args.envs, args.collect_ticks, args.updates, args.batch, args.stack, args.eval_episodes,
           args.save_every) <= 0 or min(args.rounds, args.consolidation_rounds, args.teacher_warmup_rounds) < 0:
        parser.error("Training counts must be positive.")
    if args.initialize and args.resume:
        parser.error("Choose initialization or resumption.")
    if args.firmware_launch_demos and not args.imu:
        parser.error("Firmware launch corrections require an IMU policy.")
    if args.capacity < args.envs * args.stack:
        parser.error("Replay capacity must fit one observation stack per environment.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable.")
    torch.set_num_threads(min(4, args.envs))
    torch.manual_seed(74)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True
    rng = np.random.default_rng(74)
    lib = load_sim()
    physics = physics_identity(lib)
    teacher = teacher_profile()
    line_teacher = course_profile() == "2026-practice" and teacher == "flow"
    qualify_pace = args.full_course and line_teacher
    teacher_revision = 6 if line_teacher else 1
    height, width = 72, 96
    human_demo = (Demonstrations(args.human_demos, [], args.stack, height, width, 213, args.imu)
                  if args.human_demos else None)
    human_validation = (Demonstrations(args.human_held_out, [], args.stack, height, width, 917, args.imu).sample(192)
                        if args.human_held_out else None)
    monorace_demo = (Demonstrations(args.monorace_demos, [], args.stack, height, width, 614, args.imu)
                     if args.monorace_demos else None)
    monorace_validation = (Demonstrations(args.monorace_held_out, [], args.stack, height, width, 918, args.imu).sample(192)
                           if args.monorace_held_out else None)
    demo_ids = {}
    train_paths = args.human_demos + args.monorace_demos
    validation_paths = args.human_held_out + args.monorace_held_out
    for path in train_paths + validation_paths:
        with path.open("rb") as stream:
            demo_ids[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
        if args.imu:
            with path.with_suffix(".imu.npy").open("rb") as stream:
                demo_ids[str(path)] += ":" + hashlib.file_digest(stream, "sha256").hexdigest()
    if {demo_ids[str(path)] for path in train_paths} & {demo_ids[str(path)] for path in validation_paths}:
        parser.error("Keep demonstration and held-out flights separate.")
    human_ids = {str(path): demo_ids[str(path)] for path in args.human_demos + args.human_held_out}
    monorace_ids = {str(path): demo_ids[str(path)] for path in args.monorace_demos + args.monorace_held_out}
    test_env = Environment(lib, width, height)
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        expert = evaluate_student(None, test_env, args.stack, EVALUATION_SEED, args.eval_episodes, teacher=True)
        print(json.dumps({"expert_validation": expert}), flush=True)
        if expert["two_lap_rate"] < 1.:
            raise RuntimeError("The demonstration teacher failed the two-lap acceptance test.")
        if args.teacher_only:
            return

        agent = Agent(args.stack, args.device, lib.sim_hover_stick(), args.imu, backbone=args.backbone,
                      pretrained=args.backbone != "cnn" and not (args.initialize or args.resume))
        auxiliary = torch.nn.Linear(160 if args.imu else 128, 12).to(args.device)
        parameters = [*agent.encoder.parameters(), *agent.actor.parameters()]
        if agent.imu_encoder is not None:
            parameters.extend(agent.imu_encoder.parameters())
        optimizer = torch.optim.AdamW(
            [*parameters, *auxiliary.parameters()],
            lr=1.0e-4 if args.initialize else 1.5e-4,
        )
        buffer = ExpertReplay(args.capacity // args.envs, args.envs, height, width, args.stack, args.imu)
        best = first_full = None
        qualified = first_round = 0
        n_gates = course_gate_count(lib) * 2
        target = n_gates if args.full_course else 1
        if args.initialize:
            agent.load(args.initialize)
        if args.resume:
            progress = restore_school(args.resume, agent, auxiliary, optimizer, buffer, rng, physics)
            if (progress.get("human_ids", {}) != human_ids or progress.get("monorace_ids", {}) != monorace_ids
                    or progress.get("teacher_profile", "standard") != teacher
                    or progress.get("teacher_revision", 1) != teacher_revision
                    or progress.get("teacher_warmup_rounds", 0) != args.teacher_warmup_rounds):
                raise ValueError("Resume with the original teacher profile and flight datasets.")
            if progress.get("full_course", False) != args.full_course:
                raise ValueError("Use a new run initialized from model weights to change the curriculum.")
            if human_demo is not None:
                human_demo.rng.bit_generator.state = progress["human_rng"]
            if monorace_demo is not None:
                monorace_demo.rng.bit_generator.state = progress["monorace_rng"]
            best, first_full = progress["best"], progress["first_full"]
            qualified, first_round, target = progress["qualified"], progress["round"], progress["target"]
            seed_changed = progress.get("evaluation_seed") != EVALUATION_SEED
            if seed_changed or (best is not None and len(best) != 7):
                if seed_changed:
                    first_full, qualified = None, 0
                saved_best = args.out / "best.pt"
                if saved_best.is_file():
                    previous = Agent(args.stack, args.device, lib.sim_hover_stick(), args.imu, backbone=args.backbone)
                    previous.load(saved_best)
                    best = evaluation_rank(evaluate_student(previous, test_env, args.stack,
                                                            EVALUATION_SEED, args.eval_episodes))
                    del previous
                else:
                    best = None
        envs = [Environment(lib, width, height) for _ in range(args.envs)]
        # Keep clean teacher races alongside mixed student/teacher rollouts.
        if args.full_course:
            continuous_envs = tuple(range(1, len(envs), 4))
        else:
            continuous_envs = (1,) if teacher == "flow" and len(envs) > 1 else ()
        obs = np.asarray([fresh_stack(training_reset(env, i, rng, target,
                                                    args.full_course or i in continuous_envs), args.stack)
                          for i, env in enumerate(envs)])
        publisher = FramePublisher(args.out) if args.live else None
        launch_paths = []
        if args.firmware_launch_demos:
            complete = (path for path in args.firmware_launch_demos.glob("*.npz")
                        if not path.name.endswith(".tmp.npz"))
            launch_paths = sorted(complete, key=lambda path: path.stat().st_mtime)[-12:]
        launch_demo = Demonstrations([], launch_paths, args.stack, height, width, 623, args.imu) if launch_paths else None
        started = time.monotonic()
        try:
            with ThreadPoolExecutor(max_workers=args.envs) as workers, (
                args.out / "metrics.jsonl"
            ).open("a", encoding="utf-8", buffering=1) as log:
                for round_no in count(first_round + 1):
                    if args.rounds and round_no > args.rounds:
                        break
                    warmup = args.teacher_warmup_rounds
                    teacher_probability = 1. if round_no <= warmup else max(.19, .57 - .012 * (round_no - warmup - 1))
                    accidents, pace_samples, course_finishes, teacher_finishes = collect(
                        envs, workers, agent, buffer, obs, rng, args.collect_ticks,
                        teacher_probability, target, publisher, continuous_envs, args.full_course,
                    )
                    for _ in range(args.updates):
                        frames, labels, features, weights, *imu = buffer.sample(rng, args.batch)
                        pixels = random_shift(pixel_tensor(frames, args.device).float() / 255. - .5, pad=1)
                        exposure = torch.empty((args.batch, 1, 1, 1), device=args.device).uniform_(.92, 1.08)
                        embedding = agent.policy_features((pixels * exposure).clamp(-.5, .5), imu[0] if imu else None)
                        action = agent.actor(embedding).tanh()
                        pose_loss = torch.nn.functional.smooth_l1_loss(
                            auxiliary(embedding), torch.from_numpy(features).to(args.device))
                        loss = stick_loss(action, labels, weights) + .4 * pose_loss
                        human_loss = None
                        if human_demo is not None:
                            human_loss = demonstration_loss(agent, human_demo, max(1, args.batch // 4))
                            loss = loss + .35 * human_loss
                        monorace_loss = None
                        if monorace_demo is not None:
                            monorace_loss = demonstration_loss(agent, monorace_demo, max(1, args.batch // 8))
                            loss = loss + .10 * monorace_loss
                        launch_loss = None
                        if launch_demo is not None:
                            launch_loss = demonstration_loss(agent, launch_demo, max(1, args.batch // 2))
                            loss = loss + 1.2 * launch_loss
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(parameters, 5.)
                        optimizer.step()

                    result = evaluate_student(agent, test_env, args.stack, EVALUATION_SEED, args.eval_episodes)
                    robustness = None
                    if result["two_lap_rate"] == 1.:
                        robustness = evaluate_student(agent, test_env, args.stack,
                                                      EVALUATION_SEED + round_no, args.eval_episodes)
                    ready = robustness is not None and robustness["two_lap_rate"] == 1.
                    if ready and qualify_pace:
                        ready = matches_teacher_pace(result, expert) and matches_teacher_pace(robustness, expert)
                    result.update(round=round_no, transitions=buffer.count * args.envs,
                                  target_gate=target,
                                  evaluation_seed=EVALUATION_SEED,
                                  holdout_two_lap_rate=robustness["two_lap_rate"] if robustness else None,
                                  meets_teacher_pace=matches_teacher_pace(result, expert) if qualify_pace else None,
                                  teacher_probability=teacher_probability, collection_crashes=accidents,
                                  pace_training_samples=pace_samples,
                                  collection_two_lap_finishes=course_finishes,
                                  continuous_teacher_finishes=teacher_finishes,
                                  full_course=args.full_course,
                                  continuous_teacher_gates=max((envs[i].lap_stats()[0] for i in continuous_envs),
                                                               default=None),
                                  imitation_loss=float(loss.detach()), seconds=round(time.monotonic() - started, 1))
                    if human_loss is not None:
                        result["human_training_loss"] = float(human_loss.detach())
                    if human_validation is not None:
                        result["human_heldout_stick_mae"] = demonstration_error(agent, human_validation)
                    if monorace_loss is not None:
                        result["monorace_training_loss"] = float(monorace_loss.detach())
                    if launch_loss is not None:
                        result["firmware_launch_loss"] = float(launch_loss.detach())
                        result["firmware_launch_clips"] = len(launch_paths)
                    if monorace_validation is not None:
                        result["monorace_heldout_proxy_mae"] = demonstration_error(agent, monorace_validation)
                    print(json.dumps(result), flush=True)
                    log.write(json.dumps(result) + "\n")
                    agent.save(args.out / "latest.pt", round=round_no, steps=result["transitions"])
                    rank = evaluation_rank(result)
                    if best is None or rank > best:
                        agent.save(args.out / "best.pt")
                        best = rank
                    qualified = qualified + 1 if ready else 0
                    if ready and first_full is None:
                        first_full = round_no
                        print(json.dumps({"first_full_round": first_full,
                                          "consolidation_rounds": args.consolidation_rounds}), flush=True)
                    if min(result["gates"]) >= target and target < n_gates:
                        target = min(n_gates, min(result["gates"]) + 1)
                        for i, env in enumerate(envs):
                            if i not in continuous_envs:
                                obs[i] = fresh_stack(training_reset(env, i, rng, target), args.stack)
                                buffer.next_start[i] = buffer.count
                    if round_no == 1 or round_no % args.save_every == 0 or round_no == args.rounds:
                        save_school(args.out / "imitation-state.zip", agent, auxiliary, optimizer, buffer, rng,
                                    {"round": round_no, "target": target, "best": best,
                                     "first_full": first_full, "qualified": qualified, "physics": physics,
                                     "evaluation_seed": EVALUATION_SEED, "human_ids": human_ids,
                                     "teacher_warmup_rounds": args.teacher_warmup_rounds, "full_course": args.full_course,
                                     "teacher_profile": teacher, "teacher_revision": teacher_revision,
                                     "monorace_ids": monorace_ids,
                                     "monorace_rng": monorace_demo.rng.bit_generator.state if monorace_demo else None,
                                     "human_rng": human_demo.rng.bit_generator.state if human_demo else None})
                    if not args.external_qualification and qualified >= 2 and round_no - first_full >= args.consolidation_rounds:
                        agent.save(args.out / "qualified.pt")
                        print(json.dumps({"qualified": True, "verified_two_lap_evaluations": 4 * args.eval_episodes}), flush=True)
                        break
        finally:
            for env in envs:
                env.close()
    finally:
        test_env.close()


if __name__ == "__main__":
    main()
