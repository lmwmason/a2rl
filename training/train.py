"""Train or evaluate a camera policy against the Rust race environment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from . import checkpoint
from .agent import Agent
from .demonstrations import Demonstrations
from .replay import Replay
from .sim import Environment, LEGACY_PHYSICS, gate_count as course_gate_count, load_sim, physics_identity, reward_profile
from .telemetry import FramePublisher

def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--steps", type=int, default=1_000_000,
                        help="Training budget, or schedule length in record-target mode.")
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--stack", type=int, default=4)
    parser.add_argument("--imu", action="store_true", help="Train with synchronized gyro and accelerometer history.")
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
    parser.add_argument("--initialize", type=Path, help="Initialize online training from a Mode 2 checkpoint.")
    parser.add_argument("--resume", type=Path, help="Restore a saved training state, including optimizer and replay.")
    parser.add_argument("--protect-qualified", type=Path,
                        help="Retain a qualified RGB policy as the reference during speed fine-tuning.")
    parser.add_argument("--sprint-exploration-steps", type=int, default=0,
                        help="Try short full-throttle bursts on aligned straights, fading out by this step.")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Training steps between resumable snapshots.")
    parser.add_argument("--focus-gate-two", action="store_true", help="Practice the turn from gate 1 to gate 2.")
    parser.add_argument("--official-eval", action="store_true", help="Rank speed by the official two-lap time.")
    parser.add_argument("--target-two-lap-seconds", type=float,
                        help="Train without a step cap until the official start beats this time and every evaluation finishes.")
    parser.add_argument("--live", action="store_true", help="Publish sampled live training FPV frames.")
    parser.add_argument("--demo-sim", type=Path, nargs="*", default=[],
                        help="Simulator demonstrations used as a decaying actor regularizer.")
    args = parser.parse_args()
    if sum(value is not None for value in (args.evaluate, args.initialize, args.resume)) > 1:
        parser.error("Choose evaluation, weight initialization or training-state resume.")
    positive = (args.steps, args.envs, args.width, args.height, args.stack, args.replay, args.batch,
                args.update_every, args.n_step, args.eval_every, args.eval_episodes)
    if (min(positive) <= 0 or min(args.warmup, args.checkpoint_every, args.sprint_exploration_steps) < 0
            or not 0 < args.gamma <= 1 or not 0 <= args.hdzero_max <= 1):
        parser.error("Counts must be positive, warmup nonnegative and gamma and HDZero strength must be in range.")
    if args.target_two_lap_seconds is not None and (
        not math.isfinite(args.target_two_lap_seconds) or args.target_two_lap_seconds <= 0
        or not args.official_eval or args.evaluate
    ):
        parser.error("A positive record target requires training with --official-eval.")
    if args.protect_qualified is not None and (not args.official_eval or not (args.initialize or args.resume)):
        parser.error("Qualified-policy protection requires official evaluation and initialization or resume.")
    return args


def fresh_stack(frame, stack):
    return np.repeat(frame[None], stack, axis=0)


def evaluate(agent, env, stack, episodes, official=False):
    returns, passes, records, crashes = [], [], [], 0
    official_times = []
    official_start_seconds = None
    stalls = 0
    approach_delay = 0.
    gate_speed_sum = 0.
    gates_per_lap = course_gate_count(env.lib)
    rng = np.random.default_rng(71)
    for index in range(episodes):
        reset = env.official_start if official else env.ground_start
        frame = reset() if index == 0 else reset(
            side=rng.uniform(-0.35, 0.35), exposure=rng.uniform(0.85, 1.15))
        obs = fresh_stack(frame, stack)
        reward_total = 0.
        while True:
            frame, reward, status = env.advance(agent.command(obs, env))
            reward_total += reward
            obs[:-1] = obs[1:]
            obs[-1] = frame
            if status:
                crashes += int(status == 1)
                stalls += int(status == 3)
                break
        gates, best = env.lap_stats()
        delay, speed_sum = env.flow_stats()
        approach_delay += delay
        gate_speed_sum += speed_sum
        if official and status == 4 and gates == 2 * gates_per_lap:
            seconds = env.official_seconds()
            if seconds is not None and math.isfinite(seconds) and seconds > 0:
                official_times.append(seconds)
                if index == 0:
                    official_start_seconds = seconds
        passes.append(gates)
        if best is not None:
            records.append(best)
        returns.append(reward_total)
    return {"return": float(np.mean(returns)), "gates_mean": float(np.mean(passes)),
            "gates_per_race": 2 * gates_per_lap,
            "laps_total": sum(gate // gates_per_lap for gate in passes),
            "best_lap_seconds": min(records, default=None), "crash_rate": crashes / episodes,
            "official_mean_seconds": float(np.mean(official_times)) if official_times else None,
            "official_start_seconds": official_start_seconds,
            "official_best_seconds": min(official_times, default=None),
            "official_finishes": len(official_times),
            "mean_gate_approach_delay_seconds": approach_delay / episodes,
            "mean_gate_crossing_speed_m_s": gate_speed_sum / sum(passes) if any(passes) else None,
            "stall_rate": stalls / episodes,
            "launch_to_gate_one_rate": sum(gate >= 1 for gate in passes) / episodes,
            "gate_two_rate": sum(gate >= 2 for gate in passes) / episodes,
            "split_s_rate": sum(gate >= (8 if gates_per_lap == 12 else 6) for gate in passes) / episodes,
            "slalom_rate": sum(gate >= 10 for gate in passes) / episodes if gates_per_lap == 11 else None,
            "lap_rate": sum(gate >= gates_per_lap for gate in passes) / episodes,
            "two_lap_rate": sum(gate >= 2 * gates_per_lap for gate in passes) / episodes}


def evaluate_gate_exit(agent, env, stack):
    passed = 0
    crashes = 0
    starts = [(2.5, 0.), (5., -0.18), (7.5, 0.18)]
    for speed, side in starts:
        obs = fresh_stack(env.after_gate(side=side, speed=speed), stack)
        while True:
            frame, _, status = env.advance(agent.action(obs[None])[0])
            gates, _ = env.lap_stats()
            if gates or status:
                passed += int(gates > 0)
                crashes += int(status == 1)
                break
            obs[:-1] = obs[1:]
            obs[-1] = frame
    return {"gate_exit_to_two_rate": passed / len(starts), "gate_exit_crash_rate": crashes / len(starts)}


def policy_rank(result, official):
    if official:
        time_seconds = result["official_mean_seconds"]
        return (result["two_lap_rate"], -result["crash_rate"],
                -time_seconds if time_seconds is not None else float("-inf"), result["gates_mean"])
    fastest = result["best_lap_seconds"]
    return (result["two_lap_rate"], result["lap_rate"], -fastest if fastest is not None else 0.,
            result["gate_two_rate"], result["gates_mean"], -result["crash_rate"], result["return"])


def beats_record(result, target, episodes):
    seconds = result["official_start_seconds"]
    return (target is not None and seconds is not None and 0 < seconds < target
            and result["official_finishes"] == episodes and result["crash_rate"] == 0.)


def initial_exploration(rng, previous):
    # Temporally correlated commands explore smoother maneuvers than independent full-stick samples.
    commands = previous * 0.93 + rng.normal(0., 0.075, previous.shape)
    return np.clip(commands, -0.72, 0.72).astype(np.float32)


def train(args, agent, lib):
    rng = np.random.default_rng(args.seed)
    physics = physics_identity(lib)
    reward_name = reward_profile()
    reward_version = lib.sim_pace_reward_version() if reward_name == "pace" else lib.sim_reward_model_version()
    protected = args.protect_qualified is not None
    anchor_digest = hashlib.sha256(args.protect_qualified.read_bytes()).hexdigest() if protected else None
    args.run_dir.mkdir(parents=True, exist_ok=True)
    shape = (args.height, args.width, 3)
    replay = Replay(args.replay, args.envs, shape, args.stack, args.n_step, args.gamma, rng, with_imu=args.imu)
    demonstrations = (Demonstrations([], args.demo_sim, args.stack, args.height, args.width, args.seed, with_imu=args.imu)
                      if args.demo_sim else None)
    rewards = np.zeros(args.envs, np.float32)
    flags = np.zeros(args.envs, np.uint8)
    next_frames = np.empty((args.envs, *shape), np.uint8)
    exploration = np.zeros((args.envs, 4), np.float32)
    sprint_cooldown = np.zeros(args.envs, np.int32)
    sprint_remaining = np.zeros(args.envs, np.int32)
    full_throttle_steps = 0
    trim = np.array([0., lib.sim_hover_stick(), 0., 0.], dtype=np.float32)
    starts = np.zeros((args.envs, args.stack, *shape), np.uint8)
    gate_practice = np.linspace(0.80, 0.25, 101)
    if protected:
        gate_practice = np.linspace(0.20, 0.35, 101)
    gate_weights = np.ones(course_gate_count(lib))
    difficult = [5, 6, 7, 10, 11] if len(gate_weights) == 12 else [4, 6, 10]
    gate_weights[difficult] = 2.
    gate_weights /= gate_weights.sum()
    started = time.monotonic()
    last_report, last_eval = started, 0
    updates, steps = 0, 0
    best_rank = None
    losses = None
    warm_started = bool(args.initialize)
    if args.resume:
        progress = checkpoint.load(args.resume, agent, replay, demonstrations)
        if progress.get("physics", LEGACY_PHYSICS) != physics:
            raise ValueError("Use the original physics to resume, or initialize a new run from weights.")
        if progress.get("reward_version", 1) != reward_version:
            raise ValueError("Use the original reward to resume, or initialize a new run from weights.")
        if progress.get("anchor_digest") != anchor_digest:
            raise ValueError("Resume with the original qualified checkpoint or initialize a new run.")
        if progress.get("sprint_exploration_steps", 0) != args.sprint_exploration_steps:
            raise ValueError("Resume with the original sprint exploration schedule.")
        full_throttle_steps = progress.get("full_throttle_steps", 0)
        steps, updates = progress["steps"], progress["updates"]
        last_eval, best_rank = progress["last_eval"], progress["best_rank"]
        warm_started = progress["warm_started"]
        print(json.dumps({"restored_steps": steps, "restored_updates": updates,
                          "restored_replay_rows": min(replay.count, replay.rows)}), flush=True)
    (args.run_dir / "config.json").write_text(
        json.dumps(vars(args) | {"physics": physics, "reward_profile": reward_name, "reward_version": reward_version},
                   default=str, indent=2) + "\n", encoding="utf-8")
    last_report_steps = initial_steps = last_save = steps
    envs = [Environment(lib, args.width, args.height) for _ in range(args.envs)]
    test_env = Environment(lib, args.width, args.height)
    publisher = FramePublisher(args.run_dir) if args.live else None
    target = args.target_two_lap_seconds
    achieved = False
    reference_count = min(2, args.envs - 1) if protected else 0
    reference_rng = np.random.default_rng(71)
    reference_starts = [(0., 1.), (reference_rng.uniform(-.35, .35), reference_rng.uniform(.85, 1.15))]

    def record_evaluation(result, log):
        result.update(steps=steps, updates=updates)
        if target is not None:
            result.update(target_two_lap_seconds=target,
                          target_beaten=beats_record(result, target, args.eval_episodes))
        print(json.dumps(result), flush=True)
        log.write(json.dumps(result) + "\n")
        if not result.get("target_beaten"):
            return False
        agent.save(args.run_dir / "record.pt")
        path = args.run_dir / "record.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        return True

    def reset(index, step):
        env = envs[index]
        if index < reference_count:
            side, exposure = reference_starts[index]
            return env.official_start(side, exposure)
        breakup = rng.uniform(0.05 * args.hdzero_max, args.hdzero_max) if rng.random() < 0.4 else 0.
        video_seed = int(rng.integers(1 << 32))
        if args.focus_gate_two:
            progress = min(1., step / args.steps)
            if rng.random() < .53 - .33 * progress:
                return env.after_gate(side=rng.uniform(-.35, .35), height=rng.uniform(-.22, .22),
                                      speed=rng.uniform(2., 6. + 4. * progress),
                                      exposure=rng.uniform(.8, 1.2), breakup=breakup, video_seed=video_seed)
        probability = gate_practice[min(100, round(step * 100 / args.steps))]
        if rng.random() > probability:
            return env.ground_start(rng.uniform(-0.3, 0.3), rng.uniform(0.8, 1.2), breakup, video_seed)
        return env.reset(int(rng.choice(len(gate_weights), p=gate_weights)), rng.uniform(-0.6, 0.6),
                         rng.uniform(-0.35, 0.35), rng.uniform(0.75, 1.25), breakup, video_seed)

    try:
        for i in range(len(envs)):
            starts[i] = fresh_stack(reset(i, steps), args.stack)
        with ThreadPoolExecutor(max_workers=args.envs) as workers, (
            args.run_dir / "metrics.jsonl"
        ).open("a", encoding="utf-8", buffering=1) as log:
            if args.official_eval and args.initialize:
                result = evaluate(agent, test_env, args.stack, args.eval_episodes, official=True)
                result["initializer"] = True
                best_rank = policy_rank(result, True)
                agent.save(args.run_dir / "best.pt")
                agent.save(args.run_dir / "latest.pt")
                achieved = record_evaluation(result, log)
            while not achieved and (target is not None or steps < args.steps):
                if protected:
                    noise = min(.06, .02 + .04 * steps / 500_000)
                elif warm_started:
                    noise = max(.055, .13 - .075 * steps / max(1, args.steps // 2))
                else:
                    noise = max(.10, .32 - .22 * steps / max(1, args.steps // 2))
                imu = np.stack([env.imu_window(args.stack) for env in envs]) if args.imu else None
                if steps < args.warmup and not warm_started:
                    exploration = initial_exploration(rng, exploration)
                    actions = np.clip(exploration + trim, -1., 1.)
                else:
                    actions = agent.action(starts, noise, imu=imu)
                if reference_count:
                    actions[:reference_count] = agent.action(starts[:reference_count], reference=True,
                                                             imu=imu[:reference_count] if args.imu else None)
                if steps < args.sprint_exploration_steps:
                    probability = .16 * (1. - steps / args.sprint_exploration_steps)
                    for idx in range(reference_count, args.envs):
                        if sprint_cooldown[idx] > 0:
                            sprint_cooldown[idx] -= 1
                            continue
                        starting = sprint_remaining[idx] == 0
                        if starting and rng.random() >= probability:
                            continue
                        if envs[idx].sprint_window():
                            # Recheck the available space before each full-throttle command.
                            sprint_remaining[idx] = 3 if starting else sprint_remaining[idx]
                            actions[idx, 1] = 1.
                            sprint_remaining[idx] -= 1
                            full_throttle_steps += 1
                            if sprint_remaining[idx] == 0:
                                sprint_cooldown[idx] = 10
                        elif not starting:
                            sprint_remaining[idx] = 0
                            sprint_cooldown[idx] = 10
                tasks = [workers.submit(env.step, action) for env, action in zip(envs, actions)]
                for idx, task in enumerate(tasks):
                    frame, reward, crash, timeout = task.result()
                    next_frames[idx] = frame
                    rewards[idx], flags[idx] = reward, 1 if crash else 2 if timeout else 0
                    if idx == reference_count and publisher is not None:
                        publisher.show(frame, gates=envs[idx].lap_stats()[0], step=steps,
                                       phase="SPEED RL", status=1 if crash else 2 if timeout else 0)
                next_imu = np.stack([env.imu_window(1)[0] for env in envs]) if args.imu else None
                replay.append(starts[:, -1], actions, rewards, flags, next_frames,
                              imu[:, -1] if args.imu else None, next_imu)
                for idx in np.flatnonzero(flags):
                    next_frames[idx] = reset(idx, steps)
                    starts[idx] = next_frames[idx]
                    exploration[idx] = 0
                    sprint_cooldown[idx] = 0
                    sprint_remaining[idx] = 0
                starts[:, :-1] = starts[:, 1:]
                starts[:, -1] = next_frames
                previous_steps = steps
                steps += args.envs
                if steps >= args.warmup and replay.ready():
                    actor_ready = not warm_started or steps >= args.warmup + (16_000 if protected else 8_000)
                    imitation_weight = max(0., 1. - steps / 200_000.)
                    for _ in range(steps // args.update_every - previous_steps // args.update_every):
                        update_actor = actor_ready and (not protected or updates % 4 == 0)
                        demo = (demonstrations.sample(args.batch // 2 or 1)
                                if demonstrations and update_actor and imitation_weight else None)
                        losses = agent.update(replay.sample(args.batch), noise, update_actor=update_actor,
                                              demonstration=demo, imitation_weight=imitation_weight)
                        updates += 1
                now = time.monotonic()
                if now - last_report > 10:
                    record = {"steps": steps, "updates": updates, "fps": round((steps - initial_steps) / (now - started), 1),
                              "window_fps": round((steps - last_report_steps) / (now - last_report), 1)}
                    if args.sprint_exploration_steps:
                        record["full_throttle_steps"] = full_throttle_steps
                    if losses is not None:
                        record["critic_loss"], record["actor_loss"] = (float(loss) for loss in losses)
                    print(json.dumps(record), flush=True)
                    last_report = now
                    last_report_steps = steps
                budget_reached = target is None and steps >= args.steps
                if steps - last_eval >= args.eval_every or budget_reached:
                    result = evaluate(agent, test_env, args.stack, args.eval_episodes, official=args.official_eval)
                    if args.focus_gate_two:
                        result.update(evaluate_gate_exit(agent, test_env, args.stack))
                    rank = policy_rank(result, args.official_eval)
                    if protected and best_rank is not None and rank[:2] < best_rank[:2]:
                        agent.restore_actor(args.run_dir / "best.pt")
                        result["policy_restored"] = True
                    if best_rank is None or rank > best_rank:
                        agent.save(args.run_dir / "best.pt")
                        best_rank = rank
                    agent.save(args.run_dir / "latest.pt")
                    last_eval = steps
                    achieved = record_evaluation(result, log)
                first_checkpoint = args.eval_every if protected else 25_000
                save_after = args.checkpoint_every if last_save else min(first_checkpoint, args.checkpoint_every)
                if args.checkpoint_every and (steps - last_save >= save_after or budget_reached or achieved):
                    path = args.run_dir / "training-state.zip"
                    checkpoint.save(path, agent, replay,
                                    {"steps": steps, "updates": updates, "last_eval": last_eval,
                                     "best_rank": best_rank, "warm_started": warm_started, "physics": physics,
                                     "reward_version": reward_version, "anchor_digest": anchor_digest,
                                     "sprint_exploration_steps": args.sprint_exploration_steps,
                                     "full_throttle_steps": full_throttle_steps},
                                    starts[:, -1], demonstrations,
                                    np.stack([env.imu_window(1)[0] for env in envs]) if args.imu else None)
                    last_save = steps
                    print(json.dumps({"saved_training_state": str(path), "steps": steps}), flush=True)
        elapsed = time.monotonic() - started
        print(json.dumps({"steps": steps, "updates": updates, "seconds": round(elapsed, 2),
                          "fps": round((steps - initial_steps) / elapsed, 1)}), flush=True)
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
    agent = Agent(args.stack, device, lib.sim_hover_stick(), with_imu=args.imu)
    if args.initialize:
        agent.load(args.initialize)
        # Preserve visual features while the critic adapts to the new replay.
        agent.configure_warm_start()
    if args.protect_qualified:
        agent.protect_qualified_policy(args.protect_qualified)
    if args.evaluate:
        agent.load(args.evaluate)
        env = Environment(lib, args.width, args.height)
        try:
            result = evaluate(agent, env, args.stack, args.eval_episodes, official=args.official_eval)
            if args.focus_gate_two:
                result.update(evaluate_gate_exit(agent, env, args.stack))
            print(json.dumps(result))
        finally:
            env.close()
    else:
        train(args, agent, lib)


if __name__ == "__main__":
    main()
