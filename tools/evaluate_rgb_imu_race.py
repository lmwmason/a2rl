"""Run firmware evaluations and authorize the 30 s and 12 s milestones."""
import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shlex
from shutil import copyfile, rmtree
import signal
import subprocess
from tempfile import mkdtemp
import time

from training.curriculum import FINISH_MARKER, FINISH_WINDOW, PACE_MARKER, PACE_TARGET, SPEED_MARKER, SPEED_RUN, SPEED_TARGET


ROOT = Path(__file__).resolve().parents[1]
STARTS = (0., -.03, .03)
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12"]


def completion_profile(history):
    recent = history[-8:]
    failures = Counter(row["gates"] for row in recent if row["status"] in (1, 2, 3) and row["gates"] < 24)
    focus = min(failures, key=lambda gate: (-failures[gate], gate), default=None)
    reliable = 0
    if len(recent) >= 4:
        reliable = sum(4 * sum(row["gates"] > gate for row in recent) >= 3 * len(recent) for gate in range(24))
    return {"focus_gate": focus, "reliable_gates": reliable, "evaluation_count": len(recent)}


def evaluate(args, weights, session, side, correction=None, *, assisted=False, seconds=35, profile=None):
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
        "A2RL_ELODIN_CHASE": "1" if args.public is not None and side == 0. and not assisted else "0",
        "A2RL_ELODIN_FLOW_TEACHER": "0",
        "A2RL_ELODIN_IN_PROCESS": "1" if args.in_process else "0",
        "A2RL_ELODIN_POLICY": str(weights),
        "A2RL_ELODIN_START_SIDE": str(side),
        "A2RL_ELODIN_SECONDS": str(seconds),
        "A2RL_FPV_BREAKUP": "0",
        "A2RL_FPV_EXPOSURE": "1",
        "A2RL_AIRFRAME": str(ROOT / "assets/a2rl-2025-airframe.toml"),
        "A2RL_ELODIN_ASSISTED": "1" if assisted else "0",
        "A2RL_ELODIN_COMPLETION_PROFILE": json.dumps(profile or {}),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "UV_NO_SYNC": "1",
        "PATH": str(args.harness / ".venv/bin") + os.pathsep + os.environ["PATH"],
        "ELODIN_ASSETS_DIR": str(args.out / "assets"),
        "A2RL_SIM_LIB": str(ROOT / "target/release/liba2rl_sim.so"),
    }
    environment.pop("A2RL_FPV_MOUNT_PITCH", None)
    if correction is not None:
        environment["A2RL_ELODIN_DEMONSTRATION"] = str(correction)
        environment["A2RL_ELODIN_CORRECTION_SCOPE"] = "rollout" if args.rollout_corrections else "launch"
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
            and not result.get("assisted") and not result["contact"]
            and result["gates"] == result["gates_per_race"] == 24
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


def publish_replay(directory, session, phase, result, progress, stage=None):
    if directory is None:
        return
    report = session / "replay.result.json"
    version = str(report.stat().st_mtime_ns)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "replay.json"
    previous = json.loads(manifest.read_text()) if manifest.is_file() else {}
    if int(previous.get("version", 0)) >= int(version):
        return

    video = f"replay-{version}.mp4"
    controls = f"controls-{version}.json"
    # The result is written after FFmpeg and the RC trace have closed.
    copyfile(session / "replay.mp4", directory / video)
    copyfile(session / "replay.controls.json", directory / controls)
    chase = session / "replay-chase.mp4"
    chase_urls = {}
    if chase.is_file():
        chase_name = f"chase-{version}.mp4"
        copyfile(chase, directory / chase_name)
        chase_urls["chase_url"] = chase_name
    evaluation = {
        **{key: progress[key] for key in ("round", "steps") if key in progress},
        "version": version, "phase": phase, "gates": result["gates"],
        "status": 4 if result["finished"] else 1 if result["contact"] else 2,
        "official_seconds": result["official_seconds"],
    }
    if stage:
        evaluation["curriculum_stage"] = stage
    history = previous.get("history", [])
    if not history and previous.get("kind") == "student-evaluation":
        history = [{key: previous[key] for key in evaluation if key in previous}]
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        **evaluation, "kind": "student-evaluation",
        "url": video, **chase_urls, "controls_url": controls, "course": result["course"],
        "physics_engine": result["physics_engine"], "gates_per_race": result["gates_per_race"],
        "history": [*history[-59:], evaluation],
    }) + "\n")
    temporary.replace(manifest)
    for prefix, suffix in (("replay", ".mp4"), ("chase", ".mp4"), ("controls", ".json")):
        old = sorted(directory.glob(f"{prefix}-[0-9]*{suffix}"), reverse=True)
        for path in old[3:]:
            path.unlink(missing_ok=True)


def watch(args):
    import torch

    stages = (
        ("IMITATION", "finish", "recovery/latest.pt", FINISH_MARKER, FINISH_WINDOW),
        ("IMITATION", "pace", "recovery/latest.pt", PACE_MARKER, PACE_TARGET),
        ("SPEED RL", "speed", f"{SPEED_RUN}/latest.pt", SPEED_MARKER, SPEED_TARGET),
    )
    for phase, stage, source, marker, target in stages:
        examined = set()
        while True:
            try:
                if fetch(args, marker) or (stage == "finish" and fetch(args, PACE_MARKER)):
                    break
                raw = fetch(args, source)
                digest = hashlib.sha256(raw).hexdigest()
                if not raw or digest in examined:
                    time.sleep(3)
                    continue
                progress = torch.load(BytesIO(raw), map_location="cpu", weights_only=True).get("training_progress", {})
                if stage == "speed" and progress.get("physics_engine") != "elodin":
                    print(f"Skipping speed checkpoint {digest}: Elodin training metadata is missing.", flush=True)
                    examined.add(digest)
                    continue
                weights = args.out / (digest + ".pt")
                weights.write_bytes(raw)
                trial = args.out / stage / digest
                clip_prefix = digest if stage == "finish" else f"{digest}-{stage}"
                profile = None
                if args.adaptive_completion and phase == "IMITATION":
                    replay = args.public / "replay.json"
                    history = json.loads(replay.read_text()).get("history", []) if replay.is_file() else []
                    profile = completion_profile([row for row in history if row["phase"] == phase])
                if phase == "IMITATION" and args.completion_rollouts:
                    session = trial / "assisted"
                    clip = args.rollout_corrections / f"{clip_prefix}-assisted.npz"
                    result = evaluate(args, weights, session, 0., clip, assisted=True, profile=profile)
                    for name in ("replay.mp4", "replay-chase.mp4", "replay.controls.json"):
                        (session / name).unlink(missing_ok=True)
                    print(json.dumps({"phase": phase, "curriculum_stage": stage, "checkpoint": digest, **result}), flush=True)
                qualified = stage != "pace" or progress.get("curriculum_stage") == "pace"
                for index, side in enumerate(STARTS):
                    session = trial / str(index)
                    correction_folder = args.rollout_corrections or args.launch_corrections
                    correction = (correction_folder / f"{clip_prefix}.npz"
                                  if correction_folder and phase == "IMITATION" and index == 0 else None)
                    seconds = FINISH_WINDOW if phase == "IMITATION" else 35
                    result = evaluate(args, weights, session, side, correction, seconds=seconds, profile=profile)
                    if result.get("assisted"):
                        raise ValueError("Assisted rollouts cannot count toward qualification.")
                    if index == 0:
                        publish_replay(args.public, session, phase, result, progress, stage)
                    if args.public is not None:
                        for name in ("replay.mp4", "replay-chase.mp4", "replay.controls.json"):
                            (session / name).unlink(missing_ok=True)
                    print(json.dumps({"phase": phase, "curriculum_stage": stage, "checkpoint": digest, "start_side": side,
                                      "target_seconds": target, **result}), flush=True)
                    if not succeeds(result, target):
                        qualified = False
                        break
                weights.unlink(missing_ok=True)
                if qualified:
                    promote(args, marker, raw)
                    print(json.dumps({"qualified": phase, "curriculum_stage": stage,
                                      "checkpoint": digest, "target_seconds": target}),
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
    corrections = parser.add_mutually_exclusive_group()
    corrections.add_argument("--launch-corrections", type=Path, help="Record teacher corrections on student launches.")
    corrections.add_argument("--rollout-corrections", type=Path, help="Record teacher labels throughout a student's flight.")
    parser.add_argument("--completion-rollouts", action="store_true",
                        help="Collect teacher-assisted rollouts before independent imitation evaluations.")
    parser.add_argument("--adaptive-completion", action="store_true",
                        help="Target repeated failures and ease coaching on reliably passed gates.")
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
    if args.launch_corrections:
        args.launch_corrections = args.launch_corrections.resolve()
    if args.rollout_corrections:
        args.rollout_corrections = args.rollout_corrections.resolve()
    if args.completion_rollouts and not args.rollout_corrections:
        parser.error("Completion rollouts require a full-rollout correction directory.")
    if args.adaptive_completion and not (args.completion_rollouts and args.public):
        parser.error("Adaptive completion requires completion rollouts and an independent evaluation history.")
    args.harness = args.harness.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, stop)
    watch(args)


def stop(_number, _frame):
    raise SystemExit(0)


if __name__ == "__main__":
    main()
