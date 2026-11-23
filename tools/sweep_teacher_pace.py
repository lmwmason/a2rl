"""Run a closed-loop teacher pace and line sweep on Elodin with Betaflight SITL."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
from statistics import mean
import subprocess
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
# Open portions, technical portions, Ladder speed and Ladder horizontal scale.
PROFILES = {
    "reference": (1., 1., 1., 1.),
    "open-112": (1.12, 1., 1., 1.),
    "open-125": (1.25, 1., 1., 1.),
    "open-142": (1.42, 1., 1., 1.),
    "mixed-118": (1.18, 1.08, 1.10, 1.),
    "mixed-135": (1.35, 1.14, 1.12, 1.),
    "ladder-90": (1., 1., 1.08, .90),
    "ladder-80": (1., 1., 1.08, .80),
    "ladder-70": (1., 1., 1.12, .70),
    "combined-85": (1.25, 1.08, 1.14, .85),
}


def trial(args, name, profile, side, slot):
    folder = args.out / f"{name}-{side:+.2f}"
    folder.mkdir(parents=True, exist_ok=True)
    report = folder / "race.json"
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("A2RL_", "ELODIN_ASSETS_DIR"))}
    env.update(
        A2RL_COURSE="2026-practice", A2RL_ELODIN_ROOT=str(args.harness),
        A2RL_ELODIN_REPORT=str(report), A2RL_ELODIN_FLOW_TEACHER="1",
        A2RL_ELODIN_IN_PROCESS="1", A2RL_ELODIN_SECONDS="45",
        A2RL_ELODIN_MAX_GATE_GAP="9", A2RL_ELODIN_START_SIDE=str(side),
        A2RL_ELODIN_TEACHER_SPEED_PROFILE=json.dumps(profile),
        A2RL_SITL_EXECUTABLE=str(args.firmware),
        A2RL_SITL_PORT_OFFSET=str(args.offset + 100 * slot),
        A2RL_SIM_LIB=str(args.library), ELODIN_ASSETS_DIR=str(args.out / "assets"),
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", UV_NO_SYNC="1",
    )
    if not report.exists():
        command = [str(args.harness / ".venv/bin/python"), str(ROOT / "tools/elodin_a2rl.py")]
        with TemporaryDirectory(prefix="attempt-", dir=folder) as scratch, (folder / "flight.log").open("ab") as log:
            env["A2RL_ELODIN_DB"] = str(Path(scratch) / "flight")
            with subprocess.Popen(command, cwd=args.harness, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True) as process:
                try:
                    return_code = process.wait(timeout=360)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    return_code = "timeout"
        if return_code != 0 or not report.exists():
            log = (folder / "flight.log").read_text(errors="replace")
            return {"profile": name, "inputs": profile, "start_m": side, "complete": False,
                    "failure": "invalid_ladder_line" if "Ladder frame clearance rejected" in log else return_code}
    result = json.loads(report.read_text())
    if result["physics_engine"] != "elodin" or result["course"] != "2026-practice":
        raise ValueError(f"Invalid simulator or course in {report}")
    complete = bool(result["finished"] and result["gates"] == 24 and not result["contact"])
    return {"profile": name, "inputs": profile, "start_m": side, "complete": complete, "gates": result["gates"],
            "contact": result["contact"], "contact_kind": result.get("contact_kind"),
            "seconds": result["official_seconds"], "gate_times": result["gate_times"]}


def batch(pool, args, scenarios):
    jobs = [pool.submit(trial, args, name, profile, side, slot)
            for slot, (name, profile, side) in enumerate(scenarios)]
    results = [job.result() for job in jobs]
    for item in results:
        print(json.dumps(item), flush=True)
    return results


def qualified_profiles(standard, paired):
    confirmed = []
    for result in standard:
        group = [result, *(x for x in paired if x["profile"] == result["profile"])]
        if len(group) == 3 and all(x["complete"] for x in group):
            times = [x["seconds"] for x in group]
            confirmed.append({"profile": result["profile"], "inputs": result["inputs"],
                              "mean_seconds": mean(times), "worst_seconds": max(times)})
    return sorted(confirmed, key=lambda row: row["mean_seconds"])


def refine_profiles(center, seen, iteration):
    bounds = ((.8, 1.8), (.8, 1.8), (.8, 1.8), (.65, 1.15))
    factor = (1., .6, 1.4)[(iteration - 1) % 3]
    for axis, count in enumerate((.1, .065, .07, .08)):
        low, high = bounds[axis]
        for sign in (-1, 1):
            proposal = list(center)
            proposal[axis] = round(max(low, min(high, center[axis] + sign * factor * count)), 4)
            profile = tuple(proposal)
            if profile not in seen:
                seen.add(profile)
                yield f"refine-{iteration}-{axis}-{'up' if sign > 0 else 'down'}", profile


def write_results(args, standard, paired, qualified):
    results = {"target_seconds": 15, "standard": standard, "paired": paired, "qualified": qualified}
    temporary = args.out / "results.tmp.json"
    temporary.write_text(json.dumps(results, indent=2) + "\n")
    temporary.replace(args.out / "results.json")


def sweep(args):
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "assets").mkdir(exist_ok=True)
    profiles = dict(PROFILES)
    seen = set(profiles.values())
    standard, paired = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for iteration in range(args.refine_rounds + 1):
            primary = batch(pool, args, [(name, profile, 0.) for name, profile in profiles.items()])
            standard.extend(primary)
            ordered = sorted((x for x in primary if x["complete"] and x["profile"] != "reference"),
                             key=lambda x: x["seconds"])
            contenders = ["reference", *(x["profile"] for x in ordered[:3])] if iteration == 0 else [
                x["profile"] for x in ordered[:2]]
            paired.extend(batch(pool, args, [(name, profiles[name], side)
                                             for name in contenders for side in (-.03, .03)]))
            qualified = qualified_profiles(standard, paired)
            write_results(args, standard, paired, qualified)
            if qualified and qualified[0]["worst_seconds"] <= 15:
                break
            finishes = sorted((x for x in standard if x["complete"]), key=lambda x: x["seconds"])
            center = (qualified or finishes or [{"inputs": PROFILES["reference"]}])[0]["inputs"]
            profiles = dict(refine_profiles(center, seen, iteration + 1))
            if not profiles:
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("harness", "firmware", "library", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=42000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--refine-rounds", type=int, default=3)
    args = parser.parse_args()
    args.harness, args.firmware, args.library, args.out = (
        path.resolve() for path in (args.harness, args.firmware, args.library, args.out))
    if not 1 <= args.workers <= 10 or not 1 <= args.offset <= 49100 or args.refine_rounds < 0:
        parser.error("Invalid worker count or SITL port range")
    if not ((args.harness / ".venv/bin/python").is_file() and args.firmware.is_file() and args.library.is_file()):
        parser.error("Provide an installed harness, SITL executable and experimental renderer")
    sweep(args)


if __name__ == "__main__":
    main()
