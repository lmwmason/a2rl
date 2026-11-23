"""Record and publish standard-start re-evaluations of the live experiment policies."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace

import torch

from tools.evaluate_rgb_imu_race import evaluate, publish_replay
from training.curriculum import FINISH_WINDOW, SPEED_RUN


def record_teacher(args, session, inputs):
    model = args.teacher_root
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("A2RL_", "ELODIN_ASSETS_DIR"))}
    video = session / "replay.mp4"
    environment.update(
        A2RL_COURSE="2026-practice", A2RL_ELODIN_ROOT=str(args.harness),
        A2RL_ELODIN_DB=str(session / "flight"), A2RL_ELODIN_REPORT=str(session / "replay.result.json"),
        A2RL_ELODIN_FLOW_TEACHER="1", A2RL_ELODIN_IN_PROCESS="1", A2RL_ELODIN_SECONDS="45",
        A2RL_ELODIN_MAX_GATE_GAP="9", A2RL_ELODIN_START_SIDE="0",
        A2RL_ELODIN_TEACHER_SPEED_PROFILE=json.dumps(inputs),
        A2RL_ELODIN_FPV=str(video), A2RL_ELODIN_CHASE="1",
        A2RL_SITL_EXECUTABLE=str(args.firmware), A2RL_SITL_PORT_OFFSET=str(args.port_offset),
        A2RL_SIM_LIB=str(model / "target/release/liba2rl_sim.so"),
        ELODIN_ASSETS_DIR=str(session / "assets"),
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", UV_NO_SYNC="1",
    )
    subprocess.run([str(args.harness / ".venv/bin/python"), str(model / "tools/elodin_a2rl.py")],
                   cwd=args.harness, env=environment, check=True, timeout=600)
    return json.loads((session / "replay.result.json").read_text())


def publish_current(args, name, slot, previous):
    with TemporaryDirectory(prefix=f"a2rl-{name}-", dir=args.scratch) as folder:
        session = Path(folder)
        progress = {}
        if name == "teacher":
            ledger = args.teacher_root / "runs/pace-screen/results.json"
            candidates = json.loads(ledger.read_text())["qualified"]
            if not candidates:
                return previous
            candidate = candidates[0]
            revision = candidate["profile"]
            if revision == previous:
                return previous
            label = f"Teacher pace search · {revision}"
            result = record_teacher(args, session, candidate["inputs"])
        else:
            run = args.runs / name
            checkpoint = run / SPEED_RUN / "latest.pt"
            if not checkpoint.is_file():
                return previous
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            promoted = state["training_progress"]["promoted_round"]
            revision = (state["training_progress"]["source_sha256"], promoted)
            if revision == previous:
                return previous
            weights = session / "policy.pt"
            torch.save(state, weights)
            label = f"Gates {name.removeprefix('segment-')} · " + (
                f"Promoted round {promoted}" if promoted else "Seed policy")
            if promoted:
                progress["round"] = promoted
            settings = SimpleNamespace(harness=args.harness, out=session, public=args.public, in_process=True)
            result = evaluate(settings, weights, session, 0., seconds=FINISH_WINDOW,
                              port_offset=args.port_offset + 100 * slot, max_gate_gap=9.)
        if (result["physics_engine"] != "elodin" or result["course"] != "2026-practice"
                or result.get("assisted")):
            raise ValueError("Unexpected replay evaluation configuration")
        publish_replay(args.public / name, session, label, result, progress)
        print(f"Published {label}", flush=True)
        return revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("harness", "firmware", "teacher-root", "runs", "public", "scratch"):
        parser.add_argument(f"--{name}", type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--port-offset", type=int, default=47600)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    os.environ["A2RL_HANDOFF_SITL_EXECUTABLE"] = str(args.firmware)
    names = ("teacher", "segment-5-6", "segment-11-12")
    seen = dict.fromkeys(names)
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        while True:
            pending = {name: pool.submit(publish_current, args, name, slot, seen[name])
                       for slot, name in enumerate(names)}
            for name, job in pending.items():
                try:
                    seen[name] = job.result()
                except Exception as error:
                    print(f"Replay {name}: {error}", file=sys.stderr, flush=True)
            time.sleep(20)


if __name__ == "__main__":
    main()
