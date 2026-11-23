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

if [[ ! -s "$qualified" ]]; then
    "$python" -u -m training.imitate --device "$device" --imu --stack 8 --offline \
        --initialize "$seed" --real "$@" --out "$imitation" \
        --correction-directory "$run/firmware-recovery" --until "$qualified" \
        --updates 0 --eval-every 150 --batch 128 --learning-rate 5e-5 --primary-fraction .60
fi

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
