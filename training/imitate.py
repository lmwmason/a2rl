"""Pretrain the RGB-only Mode 2 actor on actual and simulator FPV demonstrations."""
import argparse
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
            frames, actions, _ = clip.batch(np.arange(first, min(len(clip), first + batch_size)))
            errors.append(np.abs(agent.action(frames) - actions))
    return np.concatenate(errors).mean(axis=0).tolist() if errors else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, nargs="+", required=True)
    parser.add_argument("--sim", type=Path, nargs="*", default=[])
    parser.add_argument("--held-out", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--updates", type=int, default=1400)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--stack", type=int, default=4, help="Number of consecutive RGB frames per observation.")
    parser.add_argument("--official-eval", action="store_true")
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=3)
    args = parser.parse_args()
    if min(args.updates, args.batch, args.stack, args.eval_every, args.eval_episodes) <= 0:
        parser.error("Counts must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this Python environment.")
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True

    stack, height, width = args.stack, 72, 96
    lib = load_sim()
    agent = Agent(stack, args.device, lib.sim_hover_stick())
    demos = Demonstrations(args.real, args.sim, stack, height, width, args.seed)
    if {path.resolve() for path in args.real} & {path.resolve() for path in args.held_out}:
        parser.error("Hold out complete flights from training.")
    held_out = [Clip(path, stack, height, width) for path in args.held_out]
    optimizer = torch.optim.AdamW([*agent.encoder.parameters(), *agent.actor.parameters()], lr=3e-4)
    run = Environment(lib, width, height)
    args.out.mkdir(parents=True, exist_ok=True)
    best_rank = None
    try:
        with (args.out / "metrics.jsonl").open("w", encoding="utf-8", buffering=1) as log:
            for step in range(1, args.updates + 1):
                frames, labels, weight = demos.sample(args.batch)
                obs = random_shift(pixel_tensor(frames, args.device).float() / 255. - 0.5, pad=3)
                scale = torch.empty((len(frames), 1, 1, 1), device=args.device).uniform_(.85, 1.15)
                offset = torch.empty_like(scale).uniform_(-.07, .07)
                features = agent.encoder((obs * scale + offset).clamp(-.5, .5))
                action = agent.actor(features).tanh()
                loss = stick_loss(action, labels, weight)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([*agent.encoder.parameters(), *agent.actor.parameters()], 5.)
                optimizer.step()
                if step % args.eval_every and step != args.updates:
                    continue
                metrics = evaluate(agent, run, stack, args.eval_episodes, official=args.official_eval)
                real_error = held_out_mae(agent, held_out, args.batch)
                metrics.update(update=step, steps=step, round=(step + args.eval_every - 1) // args.eval_every,
                               bc_loss=float(loss.detach()), real_holdout_mae=real_error,
                               best_two_lap_s=metrics["official_best_seconds"])
                print(json.dumps(metrics), flush=True)
                log.write(json.dumps(metrics) + "\n")
                agent.save(args.out / "latest.pt")
                record = metrics["best_lap_seconds"]
                rank = (metrics["two_lap_rate"], metrics["lap_rate"], metrics["gates_mean"],
                        -metrics["crash_rate"], -record if record else 0., metrics["return"])
                if not args.sim and real_error is not None:
                    rank = (-float(np.mean(real_error)),)
                if best_rank is None or rank > best_rank:
                    agent.save(args.out / "best.pt")
                    best_rank = rank
    finally:
        run.close()


if __name__ == "__main__":
    main()
