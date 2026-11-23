"""Evaluate two segment-search actor updates on a frozen main-policy checkpoint."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import torch

from tools.evaluate_rgb_imu_race import evaluate
from training.curriculum import FINISH_WINDOW
from training.race_reward import clean_finish, matched_improvement
from training.segment_elodin import segment_seconds
from training.speed_elodin import INSTALLATION_STARTS, STARTS, installation_check


ROOT = Path(__file__).resolve().parents[1]
GATES = ((5, 6), (11, 12))
COEFFICIENTS = {
    "main": (0., 0.), "gates-5-6": (1., 0.), "gates-11-12": (0., 1.),
    "merge-35": (.35, .35), "merge-65": (.65, .65), "merge-full": (1., 1.),
}


def same(left, right):
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            same(value, right[key]) for key, value in left.items())
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    return left == right


def candidate_policies(seed, main, specialists, seed_digest):
    last = max(int(key.split(".")[0]) for key in seed["actor"])
    head = (f"{last}.weight", f"{last}.bias")
    shared = ("action_layout", "imu_channels", "action_horizon", "forecast_decay",
              "encoder", "imu_encoder")
    for policy in (main, *specialists):
        if (any(not same(seed.get(key), policy.get(key)) for key in shared)
                or seed["actor"].keys() != policy["actor"].keys()
                or any(not same(value, policy["actor"][key])
                       for key, value in seed["actor"].items() if key not in head)):
            raise ValueError("Checkpoint features or actor layout differ from the common seed.")
    for policy, gates in zip(specialists, GATES, strict=True):
        progress = policy["training_progress"]
        if (progress["source_sha256"] != seed_digest or tuple(progress["segment_gates"]) != gates
                or not progress["promoted_round"] or not same(seed.get("future_actor"), policy.get("future_actor"))):
            raise ValueError(f"Invalid specialist provenance for gates {gates}.")
    deltas = [{key: policy["actor"][key] - seed["actor"][key] for key in head}
              for policy in specialists]
    candidates = {}
    for name, (first, second) in COEFFICIENTS.items():
        if name == "main":
            candidates[name] = main
            continue
        actor = dict(main["actor"])
        for key in head:
            actor[key] = actor[key] + first * deltas[0][key] + second * deltas[1][key]
        candidates[name] = {**main, "actor": actor}
    return candidates


def write_json(path, content):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def summary(results):
    completed = [result for result in results if clean_finish(result)]
    return {
        "clean": len(completed), "trials": len(results),
        "contacts": sum(result["contact"] for result in results),
        "mean_seconds_clean_only": mean(r["official_seconds"] for r in completed) if completed else None,
        "segment_seconds_clean_only": {
            f"{a}-{b}": mean(segment_seconds(result, (a, b)) for result in completed) if completed else None
            for a, b in GATES
        },
    }


def run(args):
    inputs = {"seed": args.seed, "main": args.main, "5-6": args.first, "11-12": args.second}
    payloads = {name: path.read_bytes() for name, path in inputs.items()}
    hashes = {name: sha256(payload).hexdigest() for name, payload in payloads.items()}
    states = {name: torch.load(BytesIO(payload), map_location="cpu", weights_only=True)
              for name, payload in payloads.items()}
    candidates = candidate_policies(states["seed"], states["main"], [states["5-6"], states["11-12"]], hashes["seed"])
    args.out.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        (args.out / f"input-{name}.pt").write_bytes(payload)
    paths = {name: args.out / f"{name}.pt" for name in candidates}
    for name, state in candidates.items():
        if name == "main":
            paths[name].write_bytes(payloads["main"])
        else:
            torch.save(state, paths[name])
    manifest = {
        "source_revision": args.revision, "torch_version": str(torch.__version__),
        "firmware_sha256": sha256(args.firmware.read_bytes()).hexdigest(),
        "renderer_sha256": sha256((ROOT / "target/release/liba2rl_sim.so").read_bytes()).hexdigest(),
        "input_sha256": hashes,
        "candidate_sha256": {name: sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        "input_progress": {name: {key: state["training_progress"].get(key) for key in ("round", "promoted_round")}
                           for name, state in states.items()},
        "coefficients": COEFFICIENTS,
        "nominal_starts_m": STARTS, "confirmation_starts_m": STARTS,
        "installation_starts_m": INSTALLATION_STARTS, "installation_seed": args.layout_seed,
        "installation_perturbation": {"xy_m": .06, "z_m": .025, "yaw_deg": 2.},
        "finishing_window_seconds": FINISH_WINDOW, "max_gate_gap_seconds": 9.,
    }
    write_json(args.out / "manifest.json", manifest)
    os.environ["A2RL_HANDOFF_SITL_EXECUTABLE"] = str(args.firmware)

    def flight(stage, name, index, side, slot, layout=None):
        session = args.out / stage / name / str(index)
        settings = SimpleNamespace(harness=args.harness, out=session, public=None, in_process=True, metrics_only=True)
        report = evaluate(settings, paths[name], session, side, seconds=FINISH_WINDOW, max_gate_gap=9.,
                          port_offset=args.port_offset + 100 * slot, course_variant=layout)
        if (report["physics_engine"] != "elodin" or report["course"] != "2026-practice"
                or report.get("assisted") or report.get("course_variant") != layout
                or report["policy_steps"] <= 0 or len(report["gate_times"]) != report["gates"]):
            raise ValueError(f"Invalid evaluation report in {session}")
        return report

    data = {"phase": "screening", "qualified": None}
    write_json(args.out / "summary.json", data)
    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        nominal = {name: [] for name in candidates}
        for index, side in enumerate(STARTS):
            jobs = {name: pool.submit(flight, "nominal", name, index, side, slot)
                    for slot, name in enumerate(candidates)}
            for name, job in jobs.items():
                nominal[name].append(job.result())
            data["nominal"] = {name: summary(rows) for name, rows in nominal.items()}
            write_json(args.out / "summary.json", data)
        eligible = [name for name in candidates if name.startswith("merge-")
                    and matched_improvement(nominal[name], nominal["main"])]
        if eligible:
            winner = min(eligible, key=lambda name: data["nominal"][name]["mean_seconds_clean_only"])
            data.update(phase="confirmation", selected=winner)
            write_json(args.out / "summary.json", data)
            match = ("main", winner)
            tests = (("repeat", STARTS, [None] * len(STARTS)),
                     ("installation", INSTALLATION_STARTS, [
                         {"seed": args.layout_seed + i, **manifest["installation_perturbation"]}
                         for i in range(len(INSTALLATION_STARTS))]))
            verified = True
            for stage, starts, layouts in tests:
                jobs = [(name, pool.submit(flight, stage, name, i, side, 2*i + j, layout))
                        for i, (side, layout) in enumerate(zip(starts, layouts)) for j, name in enumerate(match)]
                results = {name: [job.result() for key, job in jobs if key == name] for name in match}
                check = installation_check if stage == "installation" else matched_improvement
                verified &= check(results[winner], results["main"])
                data[stage] = {name: summary(rows) for name, rows in results.items()}
                write_json(args.out / "summary.json", data)
            data["qualified"] = winner if verified else None
    data["phase"] = "complete"
    write_json(args.out / "summary.json", data)
    print(json.dumps(data), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seed", "main", "first", "second", "harness", "firmware", "out"):
        parser.add_argument(f"--{name}", type=lambda path: Path(path).resolve(), required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--layout-seed", type=int, default=2026112400)
    parser.add_argument("--port-offset", type=int, default=40900)
    args = parser.parse_args()
    if not 1 <= args.port_offset <= 48300:
        parser.error("Invalid SITL port range")
    torch.set_num_threads(1)
    run(args)


if __name__ == "__main__":
    main()
