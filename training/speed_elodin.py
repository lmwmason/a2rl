"""Optimize the Mode 2 policy against Elodin/Betaflight race returns."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import fcntl
import json
import os
from pathlib import Path
from statistics import mean
import time
from types import SimpleNamespace

import torch

from .curriculum import FINISH_WINDOW, PACE_MARKER, SPEED_MARKER, SPEED_RUN
from .race_reward import REWARD_VERSION, clean_finish, matched_improvement, reward_terms


STARTS = (0., -.03, .03)
# These are pre-tanh offsets on the Mode 2 throttle channel.
THROTTLE_PROBE_SCALES = (.025, .05, .09)
FINE_THROTTLE_PROBE_SCALES = (.005, .010, .018)
INSTALLATION_STARTS = (-.045, -.015, .015, .045)
THROTTLE_CHANNEL = 1


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


def shift_throttle(policy, delta):
    actor = dict(policy["actor"])
    bias = max((key for key in actor if key.endswith(".bias")),
               key=lambda key: int(key.split(".")[0]))
    actor[bias] = actor[bias].clone()
    # The forecast is relative to the current actor output, so this also shifts its throttle.
    actor[bias][THROTTLE_CHANNEL] += delta
    return {**policy, "actor": actor}


def save_checkpoint(path, policy):
    temporary = path.with_suffix(".tmp")
    torch.save(policy, temporary)
    temporary.replace(path)


def installation_check(candidate, incumbent):
    if len(candidate) != len(INSTALLATION_STARTS) or len(incumbent) != len(candidate):
        return False
    candidate_count = sum(map(clean_finish, candidate))
    incumbent_count = sum(map(clean_finish, incumbent))
    if candidate_count < max(3, incumbent_count):
        return False
    paired_gains = [base["official_seconds"] - trial["official_seconds"]
                    for trial, base in zip(candidate, incumbent)
                    if clean_finish(trial) and clean_finish(base)]
    return (bool(paired_gains) and min(paired_gains) >= -.15
            and (candidate_count > incumbent_count or mean(paired_gains) >= .015))


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
                               metrics_only=True, rollout_corrections=False, reward_trace=True,
                               telemetry=bool(args.exit_progress_weight))
    os.environ["A2RL_HANDOFF_SITL_EXECUTABLE"] = str(args.firmware)
    exit_gates = tuple(sorted(set(args.exit_progress_gates)))
    score = partial(reward_terms, exit_progress_weight=args.exit_progress_weight,
                    exit_progress_gates=exit_gates)
    variant = ""
    if args.exit_progress_weight:
        gates = "-".join(map(str, exit_gates)) or "all"
        variant = f"-gate-exit-v2-{gates}"
    if args.robust_search:
        variant += "-installation-v1"
    scales = FINE_THROTTLE_PROBE_SCALES if args.robust_search else THROTTLE_PROBE_SCALES
    while (not args.stop_after_round or iteration < args.stop_after_round) and not (args.run / SPEED_MARKER).is_file():
        iteration += 1
        side = STARTS[(iteration - 1) % len(STARTS)]
        experiment = output / "rounds" / f"{iteration}-reward-v{REWARD_VERSION}-throttle-v1{variant}"
        experiment.mkdir(parents=True, exist_ok=True)
        random_population = 1 + 2 * args.directions
        paths = [experiment / f"policy-{index}.pt" for index in range(random_population + 2)]
        save_checkpoint(paths[0], policy)
        for direction in range(args.directions):
            seed = args.seed + args.directions * iteration + direction
            for index, sign in enumerate((1, -1), start=1 + 2 * direction):
                sample = perturb(policy, torch.Generator().manual_seed(seed), radius, sign)
                save_checkpoint(paths[index], sample)
        # Sweep the offset independently of the launch offset, which also cycles every three rounds.
        throttle_scale = scales[((iteration - 1) // len(STARTS)) % len(scales)]
        for path, delta in zip(paths[random_population:], (throttle_scale, -throttle_scale)):
            save_checkpoint(path, shift_throttle(policy, delta))

        def flight(path, start, index, name, course_variant=None):
            report = evaluate(settings, path, experiment / name, start, seconds=FINISH_WINDOW,
                              port_offset=args.port_offset + 100 * index, max_gate_gap=9.,
                              course_variant=course_variant)
            if report["physics_engine"] != "elodin" or report.get("assisted") or report["policy_steps"] <= 0:
                raise RuntimeError("Speed training requires an independent flight with recorded policy actions.")
            trace = report.get("reward_trace", {})
            if trace.get("version") != REWARD_VERSION or len(trace["crossing_offsets_m"]) != report["gates"]:
                raise RuntimeError("Missing or incomplete speed reward trace.")
            if args.exit_progress_weight and len(trace.get("exit_progress_m", ())) != report["gates"]:
                raise RuntimeError("Missing or incomplete gate exit trace.")
            if report.get("course_variant") != course_variant:
                raise RuntimeError("The rollout report does not match the scheduled gate installation.")
            return report

        installation = []
        with ThreadPoolExecutor(max_workers=max(len(paths), 8 if args.robust_search else 0)) as pool:
            futures = [pool.submit(flight, path, side, i, f"candidate-{i}") for i, path in enumerate(paths)]
            population = [future.result() for future in futures]
            incumbent = population[0]
            scores = [score(trial, incumbent) for trial in population]
            returns = [sum(terms.values()) for terms in scores]
            eligible = [i for i, trial in enumerate(population[1:], 1) if clean_finish(trial) and (
                not clean_finish(incumbent)
                or trial["official_seconds"] <= incumbent["official_seconds"] - .015
                or (trial["official_seconds"] <= incumbent["official_seconds"] + .05 and returns[i] > returns[0])
            )]
            winner = max(eligible, key=returns.__getitem__, default=None)
            checks = []
            if winner is not None:
                holdouts = [start for start in STARTS if start != side]
                futures = [pool.submit(flight, paths[index], start, 2 * i + j, f"holdout-{i}-{j}")
                           for i, start in enumerate(holdouts) for j, index in enumerate((0, winner))]
                checks = [future.result() for future in futures]
                for base, trial in zip(checks[::2], checks[1::2]):
                    scores.extend((score(base, base), score(trial, base)))
            accepted = bool(checks) and matched_improvement(
                [population[winner], *checks[1::2]], [incumbent, *checks[::2]])
            if accepted and args.robust_search:
                futures = []
                for i, start in enumerate(INSTALLATION_STARTS):
                    layout = dict(seed=args.seed + iteration * len(INSTALLATION_STARTS) + i,
                                  xy_m=.06, z_m=.025, yaw_deg=2.)
                    futures.extend(pool.submit(flight, paths[index], start, 2*i+j, f"installation-{i}-{j}", layout)
                                   for j, index in enumerate((0, winner)))
                installation = [future.result() for future in futures]
                accepted = installation_check(installation[1::2], installation[::2])
        nominal_outcomes = [*population, *checks]
        outcomes = [*nominal_outcomes, *installation]
        if accepted:
            policy = torch.load(paths[winner], weights_only=True, map_location="cpu")
        steps += sum(result["policy_steps"] for result in outcomes)
        episodes += len(outcomes)
        radius = min(args.radius * 3., radius * 1.08) if accepted else max(args.radius * .35, radius * .96)
        finished = [result["official_seconds"] for result in nominal_outcomes if clean_finish(result)]
        metrics = {
            "physics_engine": "elodin", "curriculum_stage": "speed", "round": iteration,
            "steps": steps, "rollouts": episodes, "algorithm": "Antithetic evolutionary policy search",
            "exploration_radius": radius, "reward_version": REWARD_VERSION,
            "exit_progress_weight": args.exit_progress_weight,
            "exit_progress_gates": exit_gates,
            "mean_return": mean(sum(terms.values()) for terms in scores),
            "reward_components": {key: mean(terms[key] for terms in scores) for key in scores[0]},
            "accepted": accepted, "gates_mean": mean(result["gates"] for result in nominal_outcomes),
            "two_lap_rate": len(finished) / len(nominal_outcomes),
            "crash_rate": mean(result["contact"] for result in nominal_outcomes),
            "official_start_seconds": population[0]["official_seconds"] if side == 0. else None,
            "official_mean_seconds": mean(finished) if finished else None,
            "best_two_lap_s": min(finished, default=None),
            "installation_check": {
                "baseline_clean": sum(map(clean_finish, installation[::2])),
                "candidate_clean": sum(map(clean_finish, installation[1::2])),
                "paired_layouts": len(installation) // 2,
            } if installation else None,
            "throttle_probe_logit": throttle_scale,
            "throttle_probe_outcomes": [
                {"delta": delta, "clean_finish": clean_finish(trial),
                 "gates": trial["gates"], "contact": trial["contact"],
                 "time_s": trial["official_seconds"], "throttle_fraction": trial.get("mean_throttle_fraction")}
                for delta, trial in zip((0., throttle_scale, -throttle_scale),
                                        (population[0], *population[random_population:]))
            ],
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
    parser.add_argument("--stop-after-round", type=int, default=0,
                        help="Last round number to execute, or zero for unlimited.")
    parser.add_argument("--exit-progress-weight", type=float, default=0.,
                        help="Optional weight for progress into the following gate after crossing.")
    parser.add_argument("--exit-progress-gates", type=int, nargs="+", default=[],
                        help="Restrict the exit bonus to these 1-based gate numbers on each lap.")
    parser.add_argument("--robust-search", action="store_true",
                        help="Use small throttle probes and check shifted installations before promotion.")
    args = parser.parse_args()
    args.run, args.harness, args.firmware = (path.resolve() for path in (args.run, args.harness, args.firmware))
    last_port_slot = max(2 * args.directions + 2, 7 if args.robust_search else 0)
    if (not 1 <= args.directions <= 8 or not 1 <= args.port_offset <= 50000 - 100 * last_port_slot
            or not 0 < args.radius <= .01):
        parser.error("Use a valid exploration radius and a non-overlapping SITL port range.")
    if not (args.run / PACE_MARKER).is_file() or not os.access(args.firmware, os.X_OK):
        parser.error("Speed training requires the pace-qualified checkpoint and executable SITL firmware.")
    if args.stop_after_round < 0 or not 0. <= args.exit_progress_weight <= 1.:
        parser.error("Use a nonnegative round count and a gate exit weight from zero to one.")
    if any(not 1 <= gate <= 12 for gate in args.exit_progress_gates):
        parser.error("Exit bonus gate numbers must be between 1 and 12.")
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
