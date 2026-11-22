"""Train an RGB-only pilot with exact simulator expert corrections before speed RL."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import torch

from .agent import Agent, pixel_tensor, random_shift, stick_loss
from .sim import Environment, load_sim
from .train import fresh_stack


class ExpertReplay:
    """Keep each RGB image once and reconstruct short histories inside episodes."""

    def __init__(self, rows, envs, height, width, stack):
        self.frames = np.empty((rows, envs, height, width, 3), np.uint8)
        self.sticks = np.empty((rows, envs, 4), np.float32)
        self.weights = np.empty((rows, envs), np.float32)
        self.starts = np.empty((rows, envs), np.int64)
        self.rows, self.envs, self.stack = rows, envs, stack
        self.next_start = np.zeros(envs, np.int64)
        self.count = 0

    def append(self, frame, labels, confidence, ended):
        index = self.count % self.rows
        self.frames[index] = frame
        self.sticks[index] = labels
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
                self.sticks[row % self.rows, env], self.weights[row % self.rows, env])


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


def training_reset(env, index, rng):
    breakup = .18 if rng.random() < .25 else 0.
    video_seed = int(rng.integers(1 << 32))
    exposure = float(rng.uniform(.82, 1.18))
    if index == 0 or rng.random() < .25:
        return env.ground_start(side=rng.uniform(-.25, .25), exposure=exposure,
                                breakup=breakup, video_seed=video_seed)
    if rng.random() < .45:
        gate = int(rng.choice([0, 3, 4, 5, 6, 7, 8, 9, 10]))
        return env.after_gate(gate, side=rng.uniform(-.20, .20), height=rng.uniform(-.17, .17),
                              speed=rng.uniform(.5, 3.), exposure=exposure, breakup=breakup, video_seed=video_seed)
    return env.reset(int(rng.integers(env.lib.sim_gate_count())), side=rng.uniform(-.38, .38),
                     height=rng.uniform(-.26, .26), exposure=exposure, breakup=breakup, video_seed=video_seed)


def collect(envs, workers, agent, buffer, obs, rng, ticks, teacher_probability):
    n_gates = envs[0].lib.sim_gate_count() * 2
    ended = np.zeros(len(envs), dtype=bool)
    authority = np.ones(len(envs), dtype=bool)
    accidents = 0
    for tick in range(ticks):
        labels = np.asarray([env.expert_action() for env in envs], dtype=np.float32)
        if teacher_probability >= 1.:
            action = labels.copy()
        else:
            if tick % 6 == 0:
                authority = rng.random(len(envs)) < teacher_probability
            student = agent.action(obs)
            errors = np.abs(student - labels).max(axis=1)
            intervene = (errors > .48) & (rng.random(len(envs)) < .72)
            action = np.where((authority | intervene)[:, None], labels, student)
        # Small disturbances create new camera states for exact corrective labels.
        if tick % 7 in (0, 1):
            action = (action + rng.normal(0., [.025, .014, .04, .04], action.shape)).clip(-1, 1).astype(np.float32)
        following = np.empty_like(obs[:, -1])
        tasks = [workers.submit(env.advance, command) for env, command in zip(envs, action)]
        for i, task in enumerate(tasks):
            image, _, status = task.result()
            gates, _ = envs[i].lap_stats()
            ended[i] = bool(status) or gates >= n_gates
            accidents += int(status == 1)
            following[i] = training_reset(envs[i], i, rng) if ended[i] else image
        emphasis = 1. + .65 * np.abs(labels[:, [0, 2, 3]]).max(axis=1)
        buffer.append(obs[:, -1], labels, emphasis, ended)
        obs[:, :-1] = obs[:, 1:]
        obs[:, -1] = following
        obs[ended] = following[ended, None]
    return accidents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--rounds", type=int, default=48)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--collect-ticks", type=int, default=600)
    parser.add_argument("--updates", type=int, default=360)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--capacity", type=int, default=56_000)
    parser.add_argument("--stack", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--teacher-only", action="store_true")
    args = parser.parse_args()
    if min(args.rounds, args.envs, args.collect_ticks, args.updates, args.batch, args.stack, args.eval_episodes) <= 0:
        parser.error("Training counts must be positive.")
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
        optimizer = torch.optim.AdamW([*agent.encoder.parameters(), *agent.actor.parameters()], lr=1.5e-4)
        buffer = ExpertReplay(args.capacity // args.envs, args.envs, height, width, args.stack)
        envs = [Environment(lib, width, height) for _ in range(args.envs)]
        obs = np.asarray([fresh_stack(training_reset(env, i, rng), args.stack) for i, env in enumerate(envs)])
        best = None
        qualified = 0
        started = time.monotonic()
        try:
            with ThreadPoolExecutor(max_workers=args.envs) as workers, (
                args.out / "metrics.jsonl"
            ).open("a", encoding="utf-8", buffering=1) as log:
                for round_no in range(1, args.rounds + 1):
                    teacher_probability = 1. if round_no <= 2 else max(.24, .96 - .022 * (round_no - 2))
                    accidents = collect(envs, workers, agent, buffer, obs, rng, args.collect_ticks, teacher_probability)
                    for _ in range(args.updates):
                        frames, labels, weights = buffer.sample(rng, args.batch)
                        pixels = random_shift(pixel_tensor(frames, args.device).float() / 255. - .5, pad=1)
                        exposure = torch.empty((args.batch, 1, 1, 1), device=args.device).uniform_(.92, 1.08)
                        action = agent.actor(agent.encoder((pixels * exposure).clamp(-.5, .5))).tanh()
                        loss = stick_loss(action, labels, weights)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_([*agent.encoder.parameters(), *agent.actor.parameters()], 5.)
                        optimizer.step()

                    result = evaluate_student(agent, test_env, args.stack, round_no, args.eval_episodes)
                    result.update(round=round_no, transitions=buffer.count * args.envs,
                                  teacher_probability=teacher_probability, collection_crashes=accidents,
                                  imitation_loss=float(loss.detach()), seconds=round(time.monotonic() - started, 1))
                    print(json.dumps(result), flush=True)
                    log.write(json.dumps(result) + "\n")
                    agent.save(args.out / "latest.pt")
                    rank = (result["two_lap_rate"], min(result["gates"]), np.mean(result["gates"]), -result["crash_rate"])
                    if best is None or rank > best:
                        agent.save(args.out / "best.pt")
                        best = rank
                    qualified = qualified + 1 if result["two_lap_rate"] == 1. else 0
                    if qualified >= 2:
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
