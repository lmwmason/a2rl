"""Run firmware evaluations and authorize the 30 s and 12 s milestones."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
STARTS = (0., -.03, .03)
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12"]


def evaluate(args, weights, session, side):
    session.mkdir(parents=True, exist_ok=True)
    report = session / "replay.result.json"
    if report.is_file():
        return json.loads(report.read_text())
    environment = os.environ | {
        "A2RL_COURSE": "2026-practice",
        "A2RL_ELODIN_ROOT": str(args.harness),
        "A2RL_ELODIN_DB": str(session / "flight"),
        "A2RL_ELODIN_FPV": str(session / "replay.mp4"),
        "A2RL_ELODIN_FLOW_TEACHER": "0",
        "A2RL_ELODIN_POLICY": str(weights),
        "A2RL_ELODIN_START_SIDE": str(side),
        "A2RL_ELODIN_SECONDS": "35",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "UV_NO_SYNC": "1",
        "PATH": str(args.harness / ".venv/bin") + os.pathsep + os.environ["PATH"],
        "ELODIN_ASSETS_DIR": str(args.out / "assets"),
        "A2RL_SIM_LIB": str(ROOT / "target/release/liba2rl_sim.so"),
    }
    (args.out / "assets").mkdir(exist_ok=True)
    with (session / "flight.log").open("wb") as log:
        child = subprocess.Popen([args.elodin, "run", str(ROOT / "tools/elodin_a2rl.py")],
                                 cwd=args.harness, env=environment, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            for _ in range(220):
                if report.is_file():
                    return json.loads(report.read_text())
                if child.poll() is not None:
                    break
                time.sleep(1)
            raise RuntimeError(f"Elodin did not produce {report}. See {session / 'flight.log'}.")
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


def succeeds(result, threshold):
    return (result["physics_engine"] == "elodin" and result["finished"]
            and not result["contact"] and result["gates"] == result["gates_per_race"] == 24
            and 0 < result["official_seconds"] <= threshold)


def remote_path(args, file):
    return shlex.quote(args.remote_run.rstrip("/") + "/" + file)


def fetch(args, file):
    path = remote_path(args, file)
    result = subprocess.run([*SSH, args.host, f"if test -s {path}; then cat {path}; fi"],
                            capture_output=True, check=True, timeout=45)
    return result.stdout


def promote(args, file, weights):
    path = remote_path(args, file)
    temporary = remote_path(args, file + ".tmp")
    subprocess.run([*SSH, args.host, f"cat > {temporary} && mv {temporary} {path}"],
                   input=weights, check=True, timeout=45)


def watch(args):
    for phase, source, marker, target in (("IMITATION", "recovery/latest.pt", "imitation-qualified.pt", 30.),
                                          ("SPEED RL", "speed/best.pt", "speed-qualified.pt", 12.)):
        if fetch(args, marker):
            continue
        examined = set()
        while True:
            try:
                raw = fetch(args, source)
                digest = hashlib.sha256(raw).hexdigest()
                if not raw or digest in examined:
                    time.sleep(30)
                    continue
                weights = args.out / (digest + ".pt")
                weights.write_bytes(raw)
                qualified = True
                for index, side in enumerate(STARTS):
                    session = args.out / digest / str(index)
                    result = evaluate(args, weights, session, side)
                    print(json.dumps({"phase": phase, "checkpoint": digest, "start_side": side,
                                      "target_seconds": target, **result}), flush=True)
                    if not succeeds(result, target):
                        qualified = False
                        break
                if qualified:
                    promote(args, marker, raw)
                    print(json.dumps({"qualified": phase, "checkpoint": digest, "target_seconds": target}),
                          flush=True)
                    break
                examined.add(digest)
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
                print(f"Evaluation will retry: {error}", flush=True)
                time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-run", required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--elodin", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.harness = args.harness.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    watch(args)


if __name__ == "__main__":
    main()
