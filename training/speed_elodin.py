"""Optimize the Mode 2 policy against Elodin/Betaflight race returns."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
import os
from pathlib import Path
from statistics import mean
import time
from types import SimpleNamespace

import torch

from .curriculum import FINISH_WINDOW, PACE_MARKER, SPEED_MARKER, SPEED_RUN


STARTS = (0., -.03, .03)


def clean_finish(result):
    return result["finished"] and not result["contact"] and result["gates"] == 24


def race_return(result):
    """Reward continued progress and quick crossings, with a distinct clean-finish bonus."""
    crossings = result["gate_times"]
    progress = sum(math.exp(-seconds / 18.) for seconds in crossings)
    if clean_finish(result):
        return progress + 30. - .6 * result["official_seconds"]
    return progress - (4. if result["contact"] else 2.)


def perturb(policy, generator, radius, sign):
    trial = dict(policy)
    for name in ("actor", "future_actor"):
        if name not in policy:
            continue
        layer = policy[name]
        last = max(int(key.split(".")[0]) for key in layer) if name == "actor" else None
        updated = {}
        for key, value in layer.items():
            if name == "future_actor" or key.startswith(f"{last}."):
                noise = torch.randn(value.shape, generator=generator, dtype=value.dtype)
                updated[key] = value.add(noise, alpha=radius * sign)
            else:
                updated[key] = value
        trial[name] = updated
    return trial


def save_checkpoint(path, policy):
    temporary = path.with_suffix(".tmp")
    torch.save(policy, temporary)
    temporary.replace(path)


def train(args):
    from tools.evaluate_rgb_imu_race import evaluate

    qualified = args.run / PACE_MARKER
    output = args.run / SPEED_RUN
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "latest.pt"
    policy = torch.load(latest if latest.exists() else qualified, weights_only=True, map_location="cpu")
    if "imu_encoder" not in policy or policy.get("protected_prefix"):
        raise ValueError("Speed training requires an RGB+IMU checkpoint with a single policy head.")
    state = policy.get("training_progress", {}) if latest.exists() else {}
    iteration = state.get("round", 0)
    steps = state.get("steps", 0)
    episodes = state.get("rollouts", 0)
    radius = state.get("exploration_radius", args.radius)
    settings = SimpleNamespace(harness=args.harness, out=output, public=None, in_process=True,
                               metrics_only=True, rollout_corrections=False)
    os.environ["A2RL_HANDOFF_SITL_EXECUTABLE"] = str(args.firmware)
    while not (args.run / SPEED_MARKER).is_file():
        iteration += 1
        side = STARTS[(iteration - 1) % len(STARTS)]
        experiment = output / "rounds" / str(iteration)
        experiment.mkdir(parents=True, exist_ok=True)
        paths = [experiment / f"policy-{index}.pt" for index in range(1 + 2 * args.directions)]
        save_checkpoint(paths[0], policy)
        for direction in range(args.directions):
            seed = args.seed + args.directions * iteration + direction
            for index, sign in enumerate((1, -1), start=1 + 2 * direction):
                sample = perturb(policy, torch.Generator().manual_seed(seed), radius, sign)
                save_checkpoint(paths[index], sample)

        def flight(path, start, index, name):
            report = evaluate(settings, path, experiment / name, start, seconds=FINISH_WINDOW,
                              port_offset=args.port_offset + 100 * index, max_gate_gap=9.)
            if report["physics_engine"] != "elodin" or report.get("assisted") or report["policy_steps"] <= 0:
                raise RuntimeError("Speed training requires an independent flight with recorded policy actions.")
            return report

        with ThreadPoolExecutor(max_workers=len(paths)) as pool:
            futures = [pool.submit(flight, path, side, i, f"candidate-{i}") for i, path in enumerate(paths)]
            population = [future.result() for future in futures]
            returns = list(map(race_return, population))
            winner = max(range(len(population)),
                         key=lambda i: (clean_finish(population[i]), returns[i]))
            validation = []
            if winner and clean_finish(population[winner]) and (
                not clean_finish(population[0])
                or population[winner]["official_seconds"] <= population[0]["official_seconds"] - .015
            ):
                holdouts = [start for start in STARTS if start != side]
                futures = [pool.submit(flight, paths[winner], start, i, f"holdout-{i}")
                           for i, start in enumerate(holdouts)]
                validation = [future.result() for future in futures]
        accepted = bool(validation) and all(clean_finish(result) for result in validation)
        outcomes = [*population, *validation]
        if accepted:
            policy = torch.load(paths[winner], weights_only=True, map_location="cpu")
        steps += sum(result["policy_steps"] for result in outcomes)
        episodes += len(outcomes)
        radius = min(args.radius * 3., radius * 1.08) if accepted else max(args.radius * .35, radius * .96)
        finished = [result["official_seconds"] for result in outcomes if clean_finish(result)]
        metrics = {
            "physics_engine": "elodin", "curriculum_stage": "speed", "round": iteration,
            "steps": steps, "rollouts": episodes, "algorithm": "Antithetic evolutionary policy search",
            "exploration_radius": radius, "mean_return": mean(map(race_return, outcomes)),
            "accepted": accepted, "gates_mean": mean(result["gates"] for result in outcomes),
            "two_lap_rate": len(finished) / len(outcomes),
            "crash_rate": mean(result["contact"] for result in outcomes),
            "official_start_seconds": population[0]["official_seconds"] if side == 0. else None,
            "official_mean_seconds": mean(finished) if finished else None,
            "best_two_lap_s": min(finished, default=None),
        }
        policy["training_progress"] = metrics
        save_checkpoint(latest, policy)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(metrics) + "\n")
        for path in paths:
            if not accepted or path != paths[winner]:
                path.unlink()
        print(json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--firmware", type=Path, required=True, help="Port-offset Betaflight SITL executable.")
    parser.add_argument("--port-offset", type=int, default=12000)
    parser.add_argument("--directions", type=int, default=2)
    parser.add_argument("--radius", type=float, default=.00035)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.run, args.harness, args.firmware = (path.resolve() for path in (args.run, args.harness, args.firmware))
    if (not 1 <= args.directions <= 8 or not 1 <= args.port_offset <= 50000 - 200 * args.directions
            or not 0 < args.radius <= .01):
        parser.error("Use a valid exploration radius and a non-overlapping SITL port range.")
    if not (args.run / PACE_MARKER).is_file() or not os.access(args.firmware, os.X_OK):
        parser.error("Speed training requires the pace-qualified checkpoint and executable SITL firmware.")
    torch.set_num_threads(1)
    with (args.run / "training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                train(args)
                break
            except (OSError, RuntimeError) as error:
                print(f"Retrying the speed round after a rollout error: {error}", flush=True)
                time.sleep(10)


if __name__ == "__main__":
    main()
