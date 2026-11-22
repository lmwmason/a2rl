#!/usr/bin/env bash
# Train continuously. The separate Elodin evaluator controls both phase gates.
set -euo pipefail
export A2RL_COURSE=2026-practice A2RL_TEACHER_PROFILE=flow
python=${A2RL_TRAIN_PYTHON:-python3}
device=${A2RL_TRAIN_DEVICE:-cuda}
run=${1:?Usage: run_rgb_imu_race.sh RUN_DIR SEED_WEIGHTS DEMO...}
seed=${2:?Missing RGB+IMU seed weights}
shift 2
if (($# == 0)); then
    echo "Supply at least one synchronized teacher demonstration." >&2
    exit 2
fi
mkdir -p "$run"
exec 9>"$run/training.lock"
flock -n 9
imitation="$run/recovery"
speed="$run/speed"
qualified="$run/imitation-qualified.pt"
recorded="$run/speed-qualified.pt"

while [[ ! -s "$qualified" ]]; do
    checkpoint=(--initialize "$seed")
    if [[ -s "$imitation/imitation-state.zip" ]]; then
        checkpoint=(--resume "$imitation/imitation-state.zip")
    fi
    round=$("$python" - "$imitation/metrics.jsonl" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
print(json.loads(path.read_text().splitlines()[-1])["round"] if path.is_file() and path.stat().st_size else 0)
PY
)
    "$python" -u -m training.racing_school --device "$device" --imu --stack 8 --full-course \
        --external-qualification "${checkpoint[@]}" --human-demos "$@" --out "$imitation" \
        --firmware-launch-demos "$run/firmware-launch" \
        --rounds "$((round + 2))" --envs 8 --collect-ticks 700 --updates 200 --batch 64 \
        --eval-episodes 2 --save-every 2
done

export A2RL_REWARD_PROFILE=pace
while [[ ! -s "$recorded" ]]; do
    checkpoint=(--initialize "$qualified")
    if [[ -s "$speed/training-state.zip" ]]; then
        checkpoint=(--resume "$speed/training-state.zip")
    fi
    steps=$("$python" - "$speed/training-state.zip" <<'PY'
from io import BytesIO
from pathlib import Path
import sys
import zipfile

import torch

path = Path(sys.argv[1])
steps = 0
if path.exists():
    with zipfile.ZipFile(path) as archive:
        state = torch.load(BytesIO(archive.read("training.pt")), map_location="cpu", weights_only=True)
        steps = state["progress"]["steps"]
print(steps)
PY
)
    "$python" -u -m training.train --device "$device" --imu --stack 8 \
        "${checkpoint[@]}" --protect-qualified "$qualified" --demo-sim "$@" --run-dir "$speed" \
        --steps "$((steps + 24000))" --envs 8 --batch 64 --update-every 32 \
        --official-eval --eval-episodes 3 --eval-every 8000 --checkpoint-every 24000
done
