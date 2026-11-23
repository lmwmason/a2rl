"""Pretrain the Mode 2 actor on recorded FPV demonstrations."""
import argparse
from itertools import count
import json
from pathlib import Path

import numpy as np
import torch

from .agent import Agent, pixel_tensor, random_shift, stick_loss
from .demonstrations import Clip, Demonstrations
from .sim import Environment, load_sim
from .train import evaluate


@torch.no_grad()
def held_out_mae(agent, clips, batch_size):
    errors = []
    for clip in clips:
        for first in range(0, len(clip), batch_size):
            frames, actions, _, *imu = clip.batch(np.arange(first, min(len(clip), first + batch_size)))
            errors.append(np.abs(agent.action(frames, imu=imu[0] if imu else None) - actions))
    return np.concatenate(errors).mean(axis=0).tolist() if errors else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, nargs="+", required=True)
    parser.add_argument("--sim", type=Path, nargs="*", default=[])
    parser.add_argument("--held-out", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--initialize", type=Path, help="Start from an existing policy checkpoint.")
    parser.add_argument("--until", type=Path, help="Keep training offline until the evaluator writes this file.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--updates", type=int, default=1400)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--primary-fraction", type=float, default=.40,
                        help="Batch fraction drawn from --real when both dataset groups are supplied.")
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--stack", type=int, default=4, help="Number of consecutive RGB frames per observation.")
    parser.add_argument("--official-eval", action="store_true")
    parser.add_argument("--offline", action="store_true",
                        help="Select checkpoints by held-out stick error or loss, without a Rust rollout.")
    parser.add_argument("--imu", action="store_true", help="Fuse synchronized gyro and accelerometer samples with RGB.")
    parser.add_argument("--backbone", choices=("cnn", "mobilenet-v3-small"), default="cnn")
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=3)
    args = parser.parse_args()
    if min(args.batch, args.stack, args.eval_every, args.eval_episodes) <= 0 or args.updates < 0:
        parser.error("Counts must be positive.")
    if args.updates == 0 and not args.until:
        parser.error("Unlimited training requires an evaluator marker.")
    if (args.until and not args.offline) or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Use a positive learning rate and offline mode for evaluator-controlled training.")
    if not 0 < args.primary_fraction <= 1:
        parser.error("The primary dataset fraction must be greater than zero and at most one.")
    if args.offline and args.official_eval:
        parser.error("Official Rust evaluation is unavailable in offline mode.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this Python environment.")
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True

    stack, height, width = args.stack, 72, 96
    latest = args.out / "latest.pt"
    resumed = bool(args.until and latest.is_file())
    source = latest if resumed else args.initialize
    weights = torch.load(source, map_location="cpu", weights_only=True) if source else None
    lib = load_sim() if not args.offline or weights is None else None
    agent = Agent(stack, args.device, lib.sim_hover_stick() if lib else 0., with_imu=args.imu,
                  backbone=weights.get("vision_backbone", "cnn") if weights else args.backbone,
                  pretrained=weights is None and args.backbone != "cnn")
    if weights is not None:
        agent.set_weights(weights)
    progress = weights.get("training_progress", {}) if resumed else {}
    first_round, first_step, first_samples = (progress.get(key, 0) for key in ("round", "updates", "steps"))
    demos = Demonstrations(args.real, args.sim, stack, height, width, args.seed, with_imu=args.imu,
                           primary_fraction=args.primary_fraction)
    if args.until:
        print(json.dumps({"imitation_datasets": [
            {"path": str(path), "frames": len(clip)}
            for path, clip in zip(args.real + args.sim, demos.real + demos.sim)
        ], "batch": args.batch, "primary_fraction": args.primary_fraction, "device": args.device}), flush=True)
    if {path.resolve() for path in args.real + args.sim} & {path.resolve() for path in args.held_out}:
        parser.error("Hold out complete flights from training.")
    held_out = [Clip(path, stack, height, width, args.imu) for path in args.held_out]
    parameters = [*agent.encoder.parameters(), *agent.actor.parameters()]
    if agent.imu_encoder is not None:
        parameters.extend(agent.imu_encoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    run = None if args.offline else Environment(lib, width, height)
    args.out.mkdir(parents=True, exist_ok=True)
    best_rank = None
    try:
        with (args.out / "metrics.jsonl").open("a" if resumed else "w", encoding="utf-8", buffering=1) as log:
            for step in range(1, args.updates + 1) if args.updates else count(1):
                if args.until and args.until.is_file():
                    break
                frames, labels, weight, *imu = demos.sample(args.batch)
                obs = random_shift(pixel_tensor(frames, args.device).float() / 255. - 0.5, pad=3)
                scale = torch.empty((len(frames), 1, 1, 1), device=args.device).uniform_(.85, 1.15)
                offset = torch.empty_like(scale).uniform_(-.07, .07)
                features = agent.policy_features((obs * scale + offset).clamp(-.5, .5), imu[0] if imu else None)
                action = agent.actor(features).tanh()
                loss = stick_loss(action, labels, weight)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 5.)
                optimizer.step()
                if step % args.eval_every and step != args.updates:
                    continue
                real_error = held_out_mae(agent, held_out, args.batch)
                if run is None:
                    metrics = {}
                    rank = (-float(np.mean(real_error)) if real_error is not None else -float(loss.detach()),)
                else:
                    metrics = evaluate(agent, run, stack, args.eval_episodes, official=args.official_eval)
                    metrics["best_two_lap_s"] = metrics["official_best_seconds"]
                    record = metrics["best_lap_seconds"]
                    rank = (metrics["two_lap_rate"], metrics["lap_rate"], metrics["gates_mean"],
                            -metrics["crash_rate"], -record if record else 0., metrics["return"])
                    if not args.sim and real_error is not None:
                        rank = (-float(np.mean(real_error)),)
                round_no = first_round + (step + args.eval_every - 1) // args.eval_every
                updates = first_step + step
                steps = first_samples + step * args.batch if args.until else step
                metrics.update(update=updates, steps=steps, round=round_no,
                               bc_loss=float(loss.detach()), real_holdout_mae=real_error, offline=args.offline)
                print(json.dumps(metrics), flush=True)
                log.write(json.dumps(metrics) + "\n")
                agent.save(latest, round=round_no, steps=steps, updates=updates)
                if best_rank is None or rank > best_rank:
                    agent.save(args.out / "best.pt")
                    best_rank = rank
    finally:
        if run is not None:
            run.close()


if __name__ == "__main__":
    main()
