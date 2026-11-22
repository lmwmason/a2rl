"""Train an RGB-only pilot with exact simulator expert corrections before speed RL."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from itertools import count
import json
from pathlib import Path
import time
from zipfile import ZipFile

import numpy as np
import torch

from . import checkpoint
from .agent import Agent, pixel_tensor, random_shift, stick_loss
from .sim import Environment, LEGACY_PHYSICS, load_sim, physics_identity
from .telemetry import FramePublisher
from .train import fresh_stack


class ExpertReplay:
    """Keep each RGB image once and reconstruct short histories inside episodes."""

    def __init__(self, rows, envs, height, width, stack):
        self.frames = np.empty((rows, envs, height, width, 3), np.uint8)
        self.sticks = np.empty((rows, envs, 4), np.float32)
        self.features = np.empty((rows, envs, 12), np.float32)
        self.weights = np.empty((rows, envs), np.float32)
        self.starts = np.empty((rows, envs), np.int64)
        self.rows, self.envs, self.stack = rows, envs, stack
        self.next_start = np.zeros(envs, np.int64)
        self.count = 0

    def append(self, frame, labels, features, confidence, ended):
        index = self.count % self.rows
        self.frames[index] = frame
        self.sticks[index] = labels
        self.features[index] = features
        self.weights[index] = confidence
        self.starts[index] = self.next_start
        self.count += 1
        self.next_start[ended] = self.count

    def sample(self, rng, batch):
        first = max(0, self.count - self.rows)
        row = rng.integers(first, self.count, size=batch)
        env = rng.integers(self.envs, size=batch)
        indexes = row[:, None] + np.arange(1 - self.stack, 1)
        indexes = np.maximum(indexes, np.maximum(first, self.starts[row % self.rows, env])[:, None])
        return (self.frames[indexes % self.rows, env[:, None]],
                self.sticks[row % self.rows, env], self.features[row % self.rows, env],
                self.weights[row % self.rows, env])


REPLAY_FIELDS = ("frames", "sticks", "features", "weights", "starts")


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
        for field in REPLAY_FIELDS:
            checkpoint.write_array(archive, field, getattr(buffer, field)[:min(buffer.count, buffer.rows)])


def restore_school(path, agent, aux, optimizer, buffer, rng, physics):
    with ZipFile(path) as archive:
        payload = torch.load(BytesIO(archive.read("school.pt")), map_location="cpu", weights_only=True)
        if payload["progress"].get("physics", LEGACY_PHYSICS) != physics:
            raise ValueError("Use the original physics to resume, or initialize a new run from weights.")
        if payload["layout"] != (tuple(buffer.frames.shape), buffer.stack):
            raise ValueError("Use the original environment, replay and frame-stack settings to resume.")
        buffer.count = payload["count"]
        for field in REPLAY_FIELDS:
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


def evaluate_student(agent, env, stack, round_no, episodes, teacher=False):
    """Use held-out ground starts. No intervention is permitted in student evaluations."""
    n_gates = env.lib.sim_gate_count() * 2
    passes, crashes, times = [], 0, []
    rng = np.random.default_rng(9127 + round_no)
    for i in range(episodes):
        if i == 0:
            frame = env.official_start()
        else:
            frame = env.official_start(side=float(rng.uniform(-.22, .22)),
                                     exposure=float(rng.uniform(.90, 1.10)))
        obs = fresh_stack(frame, stack)
        for tick in range(1, 180 * 30 + 1):
            action = env.expert_action() if teacher else agent.action(obs[None])[0]
            frame, _, status = env.advance(action)
            gates, _ = env.lap_stats()
            if status or gates >= n_gates:
                passes.append(gates)
                if status == 4 and gates == n_gates:
                    times.append(env.official_seconds())
                crashes += int(status == 1)
                break
            obs[:-1] = obs[1:]
            obs[-1] = frame
    return {
        "gates": passes,
        "gate_one_rate": sum(x >= 1 for x in passes) / episodes,
        "gate_two_rate": sum(x >= 2 for x in passes) / episodes,
        "two_lap_rate": len(times) / episodes,
        "crash_rate": crashes / episodes,
        "best_two_lap_s": min(times, default=None),
    }


def training_reset(env, index, rng, target):
    breakup = .18 if rng.random() < .25 else 0.
    video_seed = int(rng.integers(1 << 32))
    exposure = float(rng.uniform(.82, 1.18))
    if index % 4 < 2:
        return env.ground_start(side=rng.uniform(-.25, .25), exposure=exposure,
                                breakup=breakup, video_seed=video_seed)
    gate = (target - 1) % env.lib.sim_gate_count()
    if index % 4 == 2 and target > 1:
        gate = (gate - 1) % env.lib.sim_gate_count()
        return env.after_gate(gate, side=rng.uniform(-.20, .20), height=rng.uniform(-.17, .17),
                              speed=rng.uniform(.5, 3.), exposure=exposure, breakup=breakup, video_seed=video_seed)
    return env.reset(gate, side=rng.uniform(-.38, .38),
                     height=rng.uniform(-.26, .26), exposure=exposure, breakup=breakup, video_seed=video_seed)


def collect(envs, workers, agent, buffer, obs, rng, ticks, teacher_probability, target, publisher=None):
    ended = np.zeros(len(envs), dtype=bool)
    authority = np.ones(len(envs), dtype=bool)
    accidents = 0
    for tick in range(ticks):
        labels = np.asarray([env.expert_action() for env in envs], dtype=np.float32)
        features = np.asarray([env.expert_features() for env in envs], dtype=np.float32)
        if teacher_probability >= 1.:
            assisted = np.ones(len(envs), dtype=bool)
            action = labels.copy()
        else:
            if tick % 6 == 0:
                authority = rng.random(len(envs)) < teacher_probability
            student = agent.action(obs)
            errors = np.abs(student - labels).max(axis=1)
            intervene = (errors > .48) & (rng.random(len(envs)) < .55)
            assisted = authority | intervene
            action = np.where(assisted[:, None], labels, student)
        # Small disturbances create new camera states for exact corrective labels.
        if tick % 7 in (0, 1):
            action = (action + rng.normal(0., [.025, .014, .04, .04], action.shape)).clip(-1, 1).astype(np.float32)
        following = np.empty_like(obs[:, -1])
        tasks = [workers.submit(env.advance, command) for env, command in zip(envs, action)]
        for i, task in enumerate(tasks):
            image, _, status = task.result()
            gates, _ = envs[i].lap_stats()
            threshold = target if i % 4 < 2 else 2
            ended[i] = bool(status) or gates >= threshold
            accidents += int(status == 1)
            if i == 0 and publisher is not None:
                publisher.show(image, gates=gates, step=buffer.count * len(envs),
                               phase="IMITATION", status=status, driver="Teacher" if assisted[0] else "Student")
            following[i] = training_reset(envs[i], i, rng, target) if ended[i] else image
        emphasis = 1. + 1.25 * np.abs(labels[:, [0, 2, 3]]).max(axis=1)
        buffer.append(obs[:, -1], labels, features, emphasis, ended)
        obs[:, :-1] = obs[:, 1:]
        obs[:, -1] = following
        obs[ended] = following[ended, None]
    return accidents


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
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--initialize", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--consolidation-rounds", type=int, default=6)
    parser.add_argument("--teacher-only", action="store_true")
    args = parser.parse_args()
    if min(args.envs, args.collect_ticks, args.updates, args.batch, args.stack, args.eval_episodes,
           args.save_every) <= 0 or min(args.rounds, args.consolidation_rounds) < 0:
        parser.error("Training counts must be positive.")
    if args.initialize and args.resume:
        parser.error("Choose initialization or resumption.")
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
    height, width = 72, 96
    test_env = Environment(lib, width, height)
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        expert = evaluate_student(None, test_env, args.stack, 0, args.eval_episodes, teacher=True)
        print(json.dumps({"expert_validation": expert}), flush=True)
        if expert["two_lap_rate"] < 1.:
            raise RuntimeError("The demonstration teacher failed the two-lap acceptance test.")
        if args.teacher_only:
            return

        agent = Agent(args.stack, args.device, lib.sim_hover_stick())
        auxiliary = torch.nn.Linear(128, 12).to(args.device)
        optimizer = torch.optim.AdamW(
            [*agent.encoder.parameters(), *agent.actor.parameters(), *auxiliary.parameters()],
            lr=1.0e-4 if args.initialize else 1.5e-4,
        )
        buffer = ExpertReplay(args.capacity // args.envs, args.envs, height, width, args.stack)
        best = first_full = None
        qualified = first_round = 0
        target = 1
        if args.initialize:
            agent.load(args.initialize)
        if args.resume:
            progress = restore_school(args.resume, agent, auxiliary, optimizer, buffer, rng, physics)
            best, first_full = progress["best"], progress["first_full"]
            qualified, first_round, target = progress["qualified"], progress["round"], progress["target"]
        envs = [Environment(lib, width, height) for _ in range(args.envs)]
        obs = np.asarray([fresh_stack(training_reset(env, i, rng, target), args.stack) for i, env in enumerate(envs)])
        publisher = FramePublisher(args.out) if args.live else None
        n_gates = lib.sim_gate_count() * 2
        started = time.monotonic()
        try:
            with ThreadPoolExecutor(max_workers=args.envs) as workers, (
                args.out / "metrics.jsonl"
            ).open("a", encoding="utf-8", buffering=1) as log:
                for round_no in count(first_round + 1):
                    if args.rounds and round_no > args.rounds:
                        break
                    teacher_probability = max(.19, .57 - .012 * (round_no - 1))
                    accidents = collect(envs, workers, agent, buffer, obs, rng, args.collect_ticks,
                                        teacher_probability, target, publisher)
                    for _ in range(args.updates):
                        frames, labels, features, weights = buffer.sample(rng, args.batch)
                        pixels = random_shift(pixel_tensor(frames, args.device).float() / 255. - .5, pad=1)
                        exposure = torch.empty((args.batch, 1, 1, 1), device=args.device).uniform_(.92, 1.08)
                        embedding = agent.encoder((pixels * exposure).clamp(-.5, .5))
                        action = agent.actor(embedding).tanh()
                        pose_loss = torch.nn.functional.smooth_l1_loss(
                            auxiliary(embedding), torch.from_numpy(features).to(args.device))
                        loss = stick_loss(action, labels, weights) + .4 * pose_loss
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_([*agent.encoder.parameters(), *agent.actor.parameters()], 5.)
                        optimizer.step()

                    result = evaluate_student(agent, test_env, args.stack, round_no, args.eval_episodes)
                    result.update(round=round_no, transitions=buffer.count * args.envs,
                                  target_gate=target,
                                  teacher_probability=teacher_probability, collection_crashes=accidents,
                                  imitation_loss=float(loss.detach()), seconds=round(time.monotonic() - started, 1))
                    print(json.dumps(result), flush=True)
                    log.write(json.dumps(result) + "\n")
                    agent.save(args.out / "latest.pt")
                    rank = (result["two_lap_rate"], min(result["gates"]), float(np.mean(result["gates"])), -result["crash_rate"])
                    if best is None or rank > best:
                        agent.save(args.out / "best.pt")
                        best = rank
                    qualified = qualified + 1 if result["two_lap_rate"] == 1. else 0
                    if result["two_lap_rate"] == 1. and first_full is None:
                        first_full = round_no
                        print(json.dumps({"first_full_round": first_full,
                                          "consolidation_rounds": args.consolidation_rounds}), flush=True)
                    if min(result["gates"]) >= target and target < n_gates:
                        target = min(n_gates, min(result["gates"]) + 1)
                        for i, env in enumerate(envs):
                            obs[i] = fresh_stack(training_reset(env, i, rng, target), args.stack)
                        buffer.next_start.fill(buffer.count)
                    if round_no == 1 or round_no % args.save_every == 0 or round_no == args.rounds:
                        save_school(args.out / "imitation-state.zip", agent, auxiliary, optimizer, buffer, rng,
                                    {"round": round_no, "target": target, "best": best,
                                     "first_full": first_full, "qualified": qualified, "physics": physics})
                    if qualified >= 2 and round_no - first_full >= args.consolidation_rounds:
                        agent.save(args.out / "qualified.pt")
                        print(json.dumps({"qualified": True, "verified_two_lap_evaluations": 2 * args.eval_episodes}), flush=True)
                        break
        finally:
            for env in envs:
                env.close()
    finally:
        test_env.close()


if __name__ == "__main__":
    main()
