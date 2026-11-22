"""Run firmware evaluations and authorize the 30 s and 12 s milestones."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
from shutil import copyfile, rmtree
import signal
import subprocess
from tempfile import mkdtemp
import time


ROOT = Path(__file__).resolve().parents[1]
STARTS = (0., -.03, .03)
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12"]


def evaluate(args, weights, session, side):
    session.mkdir(parents=True, exist_ok=True)
    report = session / "replay.result.json"
    if report.is_file():
        return json.loads(report.read_text())
    database = Path(mkdtemp(prefix="attempt-", dir=session))
    environment = os.environ | {
        "A2RL_COURSE": "2026-practice",
        "A2RL_ELODIN_ROOT": str(args.harness),
        "A2RL_ELODIN_DB": str(database / "flight"),
        "A2RL_ELODIN_FPV": str(session / "replay.mp4"),
        "A2RL_ELODIN_FLOW_TEACHER": "0",
        "A2RL_ELODIN_IN_PROCESS": "1" if args.in_process else "0",
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
        script = str(ROOT / "tools/elodin_a2rl.py")
        command = [str(args.harness / ".venv/bin/python"), script] if args.in_process else [args.elodin, "run", script]
        child = subprocess.Popen(command, cwd=args.harness, env=environment, stdout=log,
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
            rmtree(database)


def succeeds(result, threshold):
    return (result["physics_engine"] == "elodin" and result["finished"]
            and not result["contact"] and result["gates"] == result["gates_per_race"] == 24
            and 0 < result["official_seconds"] <= threshold)


def remote_path(args, file):
    return shlex.quote(args.remote_run.rstrip("/") + "/" + file)


def fetch(args, file):
    if args.local_run:
        try:
            return (args.local_run / file).read_bytes()
        except FileNotFoundError:
            return b""
    path = remote_path(args, file)
    result = subprocess.run([*SSH, args.host, f"if test -s {path}; then cat {path}; fi"],
                            capture_output=True, check=True, timeout=45)
    return result.stdout


def promote(args, file, weights):
    if args.local_run:
        path = args.local_run / file
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(weights)
        temporary.replace(path)
        return
    path = remote_path(args, file)
    temporary = remote_path(args, file + ".tmp")
    subprocess.run([*SSH, args.host, f"cat > {temporary} && mv {temporary} {path}"],
                   input=weights, check=True, timeout=45)


def publish_replay(directory, session, phase, result):
    if directory is None:
        return
    report = session / "replay.result.json"
    version = str(report.stat().st_mtime_ns)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "replay.json"
    if manifest.is_file() and int(json.loads(manifest.read_text())["version"]) >= int(version):
        return

    video = f"replay-{version}.mp4"
    controls = f"controls-{version}.json"
    # The result is written after FFmpeg and the RC trace have closed.
    copyfile(session / "replay.mp4", directory / video)
    copyfile(session / "replay.controls.json", directory / controls)
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "version": version, "phase": phase, "kind": "student-evaluation",
        "url": video, "controls_url": controls, "course": result["course"],
        "physics_engine": result["physics_engine"], "gates": result["gates"],
        "gates_per_race": result["gates_per_race"],
        "status": 4 if result["finished"] else 1 if result["contact"] else 2,
        "official_seconds": result["official_seconds"],
    }) + "\n")
    temporary.replace(manifest)
    for prefix, suffix in (("replay", ".mp4"), ("controls", ".json")):
        old = sorted(directory.glob(f"{prefix}-[0-9]*{suffix}"), reverse=True)
        for path in old[3:]:
            path.unlink(missing_ok=True)


def watch(args):
    for phase, source, marker, target in (("IMITATION", "recovery/latest.pt", "imitation-qualified.pt", 30.),
                                          ("SPEED RL", "speed/latest.pt", "speed-qualified.pt", 12.)):
        examined = set()
        while True:
            try:
                if fetch(args, marker):
                    break
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
                    if index == 0:
                        publish_replay(args.public, session, phase, result)
                    if args.public is not None:
                        for name in ("replay.mp4", "replay.controls.json"):
                            (session / name).unlink(missing_ok=True)
                    print(json.dumps({"phase": phase, "checkpoint": digest, "start_side": side,
                                      "target_seconds": target, **result}), flush=True)
                    if not succeeds(result, target):
                        qualified = False
                        break
                weights.unlink(missing_ok=True)
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--host")
    source.add_argument("--local-run", type=Path)
    parser.add_argument("--remote-run")
    parser.add_argument("--public", type=Path, help="Publish rendered student evaluation artifacts here.")
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--elodin")
    parser.add_argument("--in-process", action="store_true", help="Run the Elodin JAX model without a telemetry database.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not (args.elodin or args.in_process):
        parser.error("Select an Elodin executable or --in-process.")
    if args.host and (args.host.startswith("-") or not args.remote_run):
        parser.error("--host requires an SSH alias or hostname and --remote-run.")
    if args.local_run:
        args.local_run = args.local_run.resolve()
    if args.public:
        args.public = args.public.resolve()
    args.harness = args.harness.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, stop)
    watch(args)


def stop(_number, _frame):
    raise SystemExit(0)


if __name__ == "__main__":
    main()
