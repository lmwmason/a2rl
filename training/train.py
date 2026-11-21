"""Train or evaluate an RGB-only policy against the Rust race environment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import torch

from .agent import Agent
from .replay import Replay
from .sim import Environment, load_sim

def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--stack", type=int, default=4)
    parser.add_argument("--replay", type=int, default=40_000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=8_000)
    parser.add_argument("--update-every", type=int, default=8, help="Environment transitions per optimizer step.")
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--n-step", type=int, default=5)
    parser.add_argument("--hdzero-max", type=float, default=0.45, help="Maximum pixel breakup strength.")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/drq"))
    parser.add_argument("--evaluate", type=Path, help="Evaluate this checkpoint on the official simulator start.")
    args = parser.parse_args()
    positive = (args.steps, args.envs, args.width, args.height, args.stack, args.replay, args.batch,
                args.update_every, args.n_step, args.eval_every, args.eval_episodes)
    if min(positive) <= 0 or args.warmup < 0 or not 0 < args.gamma <= 1 or not 0 <= args.hdzero_max <= 1:
        parser.error("Counts must be positive, warmup nonnegative and gamma and HDZero strength must be in range.")
    return args


def fresh_stack(frame, stack):
    return np.repeat(frame[None], stack, axis=0)


def evaluate(agent, env, stack, episodes):
    returns, passes, records, crashes = [], [], [], 0
    stalls = 0
    rng = np.random.default_rng(71)
    for index in range(episodes):
        frame = env.ground_start() if index == 0 else env.ground_start(
            side=rng.uniform(-0.35, 0.35), exposure=rng.uniform(0.85, 1.15))
        obs = fresh_stack(frame, stack)
        reward_total = 0.
        while True:
            frame, reward, status = env.advance(agent.action(obs[None])[0])
            reward_total += reward
            obs[:-1] = obs[1:]
            obs[-1] = frame
            if status:
                crashes += int(status == 1)
                stalls += int(status == 3)
                break
        gates, best = env.lap_stats()
        passes.append(gates)
        if best is not None:
            records.append(best)
        returns.append(reward_total)
    gates_per_lap = env.lib.sim_gate_count()
    return {"return": float(np.mean(returns)), "gates_mean": float(np.mean(passes)),
            "laps_total": sum(gate // gates_per_lap for gate in passes),
            "best_lap_seconds": min(records, default=None), "crash_rate": crashes / episodes,
            "stall_rate": stalls / episodes,
            "launch_to_gate_one_rate": sum(gate >= 1 for gate in passes) / episodes,
            "split_s_rate": sum(gate >= 6 for gate in passes) / episodes,
            "slalom_rate": sum(gate >= 10 for gate in passes) / episodes,
            "lap_rate": sum(gate >= gates_per_lap for gate in passes) / episodes,
            "two_lap_rate": sum(gate >= 2 * gates_per_lap for gate in passes) / episodes}


def initial_exploration(rng, previous):
    # Temporally correlated commands explore smoother maneuvers than independent full-stick samples.
    commands = previous * 0.93 + rng.normal(0., 0.075, previous.shape)
    return np.clip(commands, -0.72, 0.72).astype(np.float32)


def train(args, agent, lib):
    rng = np.random.default_rng(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "config.json").write_text(
        json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8")
    shape = (args.height, args.width, 3)
    replay = Replay(args.replay, args.envs, shape, args.stack, args.n_step, args.gamma, rng)
    envs = [Environment(lib, args.width, args.height) for _ in range(args.envs)]
    test_env = Environment(lib, args.width, args.height)
    rewards = np.zeros(args.envs, np.float32)
    flags = np.zeros(args.envs, np.uint8)
    next_frames = np.empty((args.envs, *shape), np.uint8)
    exploration = np.zeros((args.envs, 4), np.float32)
    trim = np.array([0., lib.sim_hover_stick(), 0., 0.], dtype=np.float32)
    starts = np.zeros((args.envs, args.stack, *shape), np.uint8)
    gate_practice = np.linspace(0.80, 0.25, 101)
    gate_weights = np.ones(lib.sim_gate_count())
    gate_weights[[4, 6, 10]] = 2.  # Split-S, slalom and hairpin.
    gate_weights /= gate_weights.sum()
    started = time.monotonic()
    last_report, last_eval = started, 0
    last_report_steps = 0
    updates, steps = 0, 0
    best_rank = None
    losses = None

    def reset(env, step):
        breakup = rng.uniform(0.05 * args.hdzero_max, args.hdzero_max) if rng.random() < 0.4 else 0.
        video_seed = int(rng.integers(1 << 32))
        probability = gate_practice[min(100, round(step * 100 / args.steps))]
        if rng.random() > probability:
            return env.ground_start(rng.uniform(-0.3, 0.3), rng.uniform(0.8, 1.2), breakup, video_seed)
        return env.reset(int(rng.choice(len(gate_weights), p=gate_weights)), rng.uniform(-0.6, 0.6),
                         rng.uniform(-0.35, 0.35), rng.uniform(0.75, 1.25), breakup, video_seed)

    try:
        for i, env in enumerate(envs):
            starts[i] = fresh_stack(reset(env, 0), args.stack)
        with ThreadPoolExecutor(max_workers=args.envs) as workers, (
            args.run_dir / "metrics.jsonl"
        ).open("a", encoding="utf-8", buffering=1) as log:
            while steps < args.steps:
                noise = max(0.10, 0.32 - 0.22 * steps / max(1, args.steps // 2))
                if steps < args.warmup:
                    exploration = initial_exploration(rng, exploration)
                    actions = np.clip(exploration + trim, -1., 1.)
                else:
                    actions = agent.action(starts, noise)
                tasks = [workers.submit(env.step, action) for env, action in zip(envs, actions)]
                for idx, task in enumerate(tasks):
                    frame, reward, crash, timeout = task.result()
                    next_frames[idx] = frame
                    rewards[idx], flags[idx] = reward, 1 if crash else 2 if timeout else 0
                replay.append(starts[:, -1], actions, rewards, flags, next_frames)
                for idx in np.flatnonzero(flags):
                    next_frames[idx] = reset(envs[idx], steps)
                    starts[idx] = next_frames[idx]
                    exploration[idx] = 0
                starts[:, :-1] = starts[:, 1:]
                starts[:, -1] = next_frames
                previous_steps = steps
                steps += args.envs
                if steps >= args.warmup and replay.ready():
                    for _ in range(steps // args.update_every - previous_steps // args.update_every):
                        losses = agent.update(replay.sample(args.batch), noise)
                        updates += 1
                now = time.monotonic()
                if now - last_report > 10:
                    record = {"steps": steps, "updates": updates, "fps": round(steps / (now - started), 1),
                              "window_fps": round((steps - last_report_steps) / (now - last_report), 1)}
                    if losses is not None:
                        record["critic_loss"], record["actor_loss"] = (float(loss) for loss in losses)
                    print(json.dumps(record), flush=True)
                    last_report = now
                    last_report_steps = steps
                if steps - last_eval >= args.eval_every or steps >= args.steps:
                    result = evaluate(agent, test_env, args.stack, args.eval_episodes)
                    result.update(steps=steps, updates=updates)
                    print(json.dumps(result), flush=True)
                    log.write(json.dumps(result) + "\n")
                    agent.save(args.run_dir / "latest.pt")
                    fastest = result["best_lap_seconds"]
                    rank = (result["two_lap_rate"], result["lap_rate"], -fastest if fastest is not None else 0.,
                            result["gates_mean"], -result["crash_rate"], result["return"])
                    if best_rank is None or rank > best_rank:
                        agent.save(args.run_dir / "best.pt")
                        best_rank = rank
                    last_eval = steps
        elapsed = time.monotonic() - started
        print(json.dumps({"steps": steps, "updates": updates, "seconds": round(elapsed, 2),
                          "fps": round(steps / elapsed, 1)}), flush=True)
    finally:
        for env in [*envs, test_env]:
            env.close()


def main():
    args = options()
    torch.manual_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this Python environment.")
    torch.set_num_threads(min(4, args.envs))
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    lib = load_sim()
    agent = Agent(args.stack, device, lib.sim_hover_stick())
    if args.evaluate:
        agent.load(args.evaluate)
        env = Environment(lib, args.width, args.height)
        try:
            print(json.dumps(evaluate(agent, env, args.stack, args.eval_episodes)))
        finally:
            env.close()
    else:
        train(args, agent, lib)


if __name__ == "__main__":
    main()
