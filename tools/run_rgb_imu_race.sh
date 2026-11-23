#!/usr/bin/env bash
# Train continuously. The separate Elodin evaluator controls both phase gates.
set -euo pipefail
export A2RL_COURSE=2026-practice
python=${A2RL_TRAIN_PYTHON:-python3}
device=${A2RL_TRAIN_DEVICE:-cuda}
stack=${A2RL_TRAIN_STACK:-8}
history=()
if [[ "$stack" != 8 ]]; then
    history=(--extend-history)
fi
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
    "$python" -u -m training.imitate --device "$device" --imu --stack "$stack" "${history[@]}" \
        --initialize "$seed" --real "$@" --out "$imitation" \
        --correction-directory "$run/firmware-recovery" --until "$qualified" \
        --completion-marker "$run/completion-qualified.pt" \
        --updates 0 --eval-every 150 --batch 128 --learning-rate 5e-5 --primary-fraction .25
fi

if [[ -n "${A2RL_ELODIN_ROOT:-}" && -n "${A2RL_HANDOFF_SITL_EXECUTABLE:-}" ]]; then
    flock -u 9
    exec "$python" -u -m training.speed_elodin --run "$run" \
        --harness "$A2RL_ELODIN_ROOT" --firmware "$A2RL_HANDOFF_SITL_EXECUTABLE"
fi

echo "Set A2RL_ELODIN_ROOT and A2RL_HANDOFF_SITL_EXECUTABLE to start stage 3."
