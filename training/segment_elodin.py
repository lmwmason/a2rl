"""Search for faster visual-inertial control around selected gate transitions."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
import time

import numpy as np
import torch

from tools.evaluate_rgb_imu_race import evaluate
from .agent import load_policy, pixel_tensor
from .curriculum import FINISH_WINDOW, SPEED_RUN
from .inertial import RelativeAttitude
from .race_reward import clean_finish, matched_improvement
from .speed_elodin import (
    INSTALLATION_STARTS, STARTS, installation_check, save_checkpoint,
)


# Normalized pre-tanh direction families in yaw, throttle, roll, pitch order.
CONTROL_DIRECTIONS = torch.tensor([
    [[0, .5, 0, -.7], [0, 1, 0, -.45]],
    [[.6, .4, .8, -.5], [.6, .8, .8, -.4]],
    [[-.6, .4, -.8, -.5], [-.6, .8, -.8, -.4]],
], dtype=torch.float32)
POLICY_START_SECONDS = .75  # Single-drone arm time plus the warm-up in elodin_policy.


def observations(path, agent, gates):
    """Return hidden activations and smooth, training-only segment labels."""
    with np.load(path, allow_pickle=False) as clip:
        mask = clip["seconds"] >= POLICY_START_SECONDS
        seconds = clip["seconds"][mask]
        gate_index = clip["next_gate_index"][mask]
        rgb, imu = clip["rgb"][mask], clip["imu"][mask]
    if len(seconds) < agent.stack or np.any(np.diff(seconds) <= 0):
        raise ValueError(f"Insufficient synchronized calibration samples in {path}")
    if agent.imu_channels == 10:
        attitude = RelativeAttitude()
        imu = np.stack([attitude.augment(row, time) for row, time in zip(imu, seconds)])
    labels = np.zeros((len(rgb), len(gates)), dtype=np.float32)
    for index in np.unique(gate_index[gate_index >= 0]):
        course_gate = index % 12 + 1
        if course_gate in gates:
            indices = np.flatnonzero(gate_index == index)
            # Advance smoothly into and out of each intervention, including across laps.
            elapsed = seconds[indices] - seconds[indices[0]]
            remaining = seconds[indices[-1]] - seconds[indices]
            labels[indices, gates.index(course_gate)] = np.minimum(
                1., np.minimum(elapsed, remaining) / .18)
    chosen = np.arange(0, len(rgb), 3)
    history = np.arange(1 - agent.stack, 1)
    hidden = []
    with torch.inference_mode():
        for ends in np.array_split(chosen, (len(chosen) + 31) // 32):
            ids = np.maximum(ends[:, None] + history, 0)
            pixels = pixel_tensor(rgb[ids], agent.device).float() / 255. - .5
            features = agent.policy_features(pixels, imu[ids])
            hidden.append(agent.actor[:-1](features).cpu())
    return torch.cat(hidden), torch.from_numpy(labels[chosen])


def fit_segment_directions(paths, checkpoint, device, gates):
    agent = load_policy(checkpoint)
    if agent.imu_encoder is None or agent.protected_prefix:
        raise ValueError("Segment search requires a single-head RGB/IMU policy")
    agent.device = device
    for module in (agent.encoder, agent.imu_encoder, agent.actor):
        module.to(device).eval()
    groups = [observations(path, agent, gates) for path in paths]
    hidden, labels = (torch.cat(group) for group in zip(*groups))
    if not torch.all(labels.amax(dim=0) > .5):
        raise ValueError("Calibration flights must contain every selected gate approach")
    design = torch.cat((hidden, torch.ones_like(hidden[:, :1])), dim=1)
    weights = (1 + 3 * labels.amax(dim=1)).sqrt()[:, None]
    features = design * weights
    penalty = .025 * len(features) * torch.eye(features.shape[1])
    solution = torch.linalg.solve(features.T @ features + penalty, features.T @ (labels * weights))
    response = design @ solution
    envelope = response.abs().sum(dim=1).amax().clamp_min(1.)
    return solution / envelope, float((response - labels).square().mean())


def move_actor(policy, directions, controls):
    """Fold segment-conditioned logit changes into the existing actor layer."""
    actor = dict(policy["actor"])
    last = max(int(name.split(".")[0]) for name in actor)
    update = (directions @ controls).T
    weight, bias = f"{last}.weight", f"{last}.bias"
    actor[weight] = actor[weight] + update[:, :-1].to(actor[weight])
    actor[bias] = actor[bias] + update[:, -1].to(actor[bias])
    return {**policy, "actor": actor}


def search_controls(iteration, rng, gate_count=2):
    # The scale schedule keeps probing beyond small changes after failed iterations.
    scale = (.018, .035, .065, .10)[(iteration - 1) % 4]
    families = CONTROL_DIRECTIONS
    if gate_count == 4:
        # Treat the two gate pairs independently while retaining the shared actor.
        second = families.roll(1 + (iteration - 1) % 2, dims=0)
        families = torch.cat((families, second), dim=1)
    jitter = torch.randn((gate_count, 4), generator=rng).clamp(-2., 2.) * .16
    controls = [*families, families[(iteration - 1) % 3] + jitter]
    return [scale * sign * family for family in controls for sign in (1, -1)], scale


def segment_seconds(report, gates):
    times = report["gate_times"]
    splits = [end - start for start, end in zip([0., *times], times)]
    selected = [value for i, value in enumerate(splits) if i % 12 + 1 in gates]
    return sum(selected) if len(selected) == 2 * len(gates) else None


def run(args):
    output = args.run.resolve() / SPEED_RUN
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "latest.pt"
    history = output / "metrics.jsonl"
    seed = args.initial.read_bytes()
    fingerprint = sha256(seed).hexdigest()
    state = torch.load(latest, map_location="cpu", weights_only=True) if latest.exists() else None
    policy = state if state is not None else torch.load(args.initial, map_location="cpu", weights_only=True)
    progress = state.get("training_progress", {}) if state is not None else {}
    if progress and (tuple(progress["segment_gates"]) != tuple(args.gates)
                     or progress["source_sha256"] != fingerprint):
        raise ValueError("Run directory belongs to a different seed or gate pair")
    if progress:
        last = history.read_text(encoding="utf-8").splitlines()[-1:] if history.exists() else []
        if not last or json.loads(last[0])["round"] < progress["round"]:
            with history.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(progress) + "\n")
    os.environ["A2RL_HANDOFF_SITL_EXECUTABLE"] = str(args.firmware.resolve())
    settings = SimpleNamespace(harness=args.harness.resolve(), out=output, public=None,
                               in_process=True, metrics_only=True, reward_trace=True)
    promoted = progress.get("promoted_round", 0)
    steps, rollouts = progress.get("steps", 0), progress.get("rollouts", 0)
    iteration = progress.get("round", 0)
    calibrated_on = None

    def flight(path, directory, side, slot, *, layout=None, capture=False):
        config = SimpleNamespace(**vars(settings), record_memory=capture)
        report = evaluate(config, path, directory, side, seconds=FINISH_WINDOW,
                          port_offset=args.port_offset + 100 * slot,
                          max_gate_gap=9., course_variant=layout)
        if (report["physics_engine"] != "elodin" or report.get("assisted")
                or report["policy_steps"] <= 0 or report.get("course_variant") != layout):
            raise RuntimeError(f"Invalid independent rollout in {directory}")
        return report

    with ThreadPoolExecutor(max_workers=9) as pool:
        while not args.stop_after_round or iteration < args.stop_after_round:
            iteration += 1
            if calibrated_on != promoted:
                capture = output / "calibration" / str(promoted)
                capture.mkdir(parents=True, exist_ok=True)
                baseline_path = capture / "incumbent.pt"
                save_checkpoint(baseline_path, policy)
                locations = [capture / str(i) for i in range(len(STARTS))]
                references = [
                    pool.submit(flight, baseline_path, directory, side, i, capture=True)
                    for i, (side, directory) in enumerate(zip(STARTS, locations))
                ]
                for result in references:
                    result.result()
                directions, fit_error = fit_segment_directions(
                    [p / "flight-telemetry.npz" for p in locations],
                    baseline_path.read_bytes(), args.device, tuple(args.gates))
                calibrated_on = promoted
            side = STARTS[(iteration - 1) % len(STARTS)]
            round_dir = output / "rounds" / f"{iteration}-segment"
            round_dir.mkdir(parents=True, exist_ok=True)
            controls, scale = search_controls(
                iteration, torch.Generator().manual_seed(args.seed + iteration), len(args.gates))
            trials = [round_dir / f"policy-{i}.pt" for i in range(1 + len(controls))]
            for path, params in zip(trials, [policy, *(move_actor(policy, directions, c) for c in controls)]):
                save_checkpoint(path, params)
            flights = [
                pool.submit(flight, path, round_dir / f"candidate-{i}", side, i) for i, path in enumerate(trials)
            ]
            nominal = [future.result() for future in flights]
            incumbent = nominal[0]
            better = sorted(
                (i for i in range(1, len(nominal)) if clean_finish(nominal[i]) and
                 (not clean_finish(incumbent) or nominal[i]["official_seconds"] <= incumbent["official_seconds"] + .05)),
                key=lambda i: nominal[i]["official_seconds"],
            )[:2]
            checks = []
            accepted = None
            install = None
            for index in better:
                starts = [start for start in STARTS if start != side]
                paired = [job.result() for job in [
                    pool.submit(flight, trials[k], round_dir / f"holdout-{index}-{j}-{k}", start, 2*j + p)
                    for j, start in enumerate(starts) for p, k in enumerate((0, index))
                ]]
                checks.extend(paired)
                if not matched_improvement([nominal[index], *paired[1::2]], [incumbent, *paired[::2]]):
                    continue
                stress = [job.result() for job in [
                    pool.submit(flight, trials[k], round_dir / f"install-{index}-{j}-{k}", start, 2*j + p,
                                layout=dict(seed=args.seed + 4*iteration + j, xy_m=.06, z_m=.025, yaw_deg=2.))
                    for j, start in enumerate(INSTALLATION_STARTS) for p, k in enumerate((0, index))
                ]]
                checks.extend(stress)
                install = {"paired_layouts": len(stress) // 2, "candidate_clean": sum(map(clean_finish, stress[1::2])),
                           "baseline_clean": sum(map(clean_finish, stress[::2]))}
                if installation_check(stress[1::2], stress[::2]):
                    accepted = index
                    policy = torch.load(trials[index], weights_only=True, map_location="cpu")
                    promoted = iteration
                    break
            steps += sum(result["policy_steps"] for result in [*nominal, *checks])
            rollouts += len(nominal) + len(checks)
            times = [row["official_seconds"] for row in nominal if clean_finish(row)]
            metrics = {
                "algorithm": "Observation-conditioned segment policy search", "curriculum_stage": "speed",
                "source_sha256": fingerprint, "physics_engine": "elodin",
                "round": iteration, "steps": steps, "rollouts": rollouts, "segment_gates": args.gates,
                "segment_fit_mse": fit_error, "logit_scale": scale, "accepted": accepted is not None,
                "promoted_round": promoted, "promotion_index": accepted, "installation_check": install,
                "two_lap_rate": len(times) / len(nominal), "crash_rate": mean(row["contact"] for row in nominal),
                "gates_mean": mean(row["gates"] for row in nominal),
                "official_start_seconds": incumbent["official_seconds"] if side == 0 else None,
                "official_mean_seconds": mean(times) if times else None, "best_two_lap_s": min(times, default=None),
                "incumbent_segment_seconds": segment_seconds(incumbent, args.gates),
                "best_candidate_segment_seconds": min((segment_seconds(nominal[i], args.gates)
                                                       for i in better), default=None),
            }
            save_checkpoint(latest, {**policy, "training_progress": metrics})
            with history.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics) + "\n")
            for i, path in enumerate(trials):
                if i != accepted:
                    path.unlink()
            print(json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("initial", "run", "harness", "firmware"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    parser.add_argument("--gates", nargs="+", type=int, default=[5, 6])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--port-offset", type=int, default=35000)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--stop-after-round", type=int, default=0)
    args = parser.parse_args()
    if (len(args.gates) not in (2, 4) or len(set(args.gates)) != len(args.gates)
            or any(not 1 <= gate <= 12 for gate in args.gates)):
        parser.error("--gates requires one or two pairs of distinct 1-based gate numbers")
    if not 1 <= args.port_offset <= 49200 or args.stop_after_round < 0:
        parser.error("Invalid SITL port range or stopping round")
    if not args.initial.is_file() or not os.access(args.firmware, os.X_OK):
        parser.error("Provide a seed checkpoint and an executable SITL firmware")
    torch.set_num_threads(1)
    args.run.mkdir(parents=True, exist_ok=True)
    with (args.run / "segment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                run(args)
                return
            except (OSError, RuntimeError) as error:
                print(f"Retrying after rollout failure: {error}", flush=True)
                time.sleep(10)


if __name__ == "__main__":
    main()
