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
qualified="$run/imitation-qualified.pt"

if [[ ! -s "$qualified" ]]; then
    "$python" -u -m training.imitate --device "$device" --imu --stack 8 \
        --initialize "$seed" --real "$@" --out "$imitation" \
        --correction-directory "$run/firmware-recovery" --until "$qualified" \
        --completion-marker "$run/completion-qualified.pt" \
        --updates 0 --eval-every 150 --batch 128 --learning-rate 5e-5 --primary-fraction .60
fi

echo "Stage 3 is awaiting an Elodin + Betaflight reinforcement learner."
