#!/usr/bin/env bash
set -euo pipefail

school=${A2RL_SCHOOL_DIR:-runs/racing-school-v2}
speed=${A2RL_SPEED_DIR:-runs/speed-rl-v2}
teacher_checkpoint=${A2RL_INITIAL_POLICY:-runs/racing-school-v1/best.pt}
record_seconds=16.56

if [[ ! -s "$school/qualified.pt" ]]; then
  start=()
  if [[ -s "$school/imitation-state.zip" ]]; then
    start=(--resume "$school/imitation-state.zip")
  elif [[ -s "$teacher_checkpoint" ]]; then
    start=(--initialize "$teacher_checkpoint")
  fi
  echo 'STAGE progressive imitation with post-completion consolidation'
  python3 -u -m training.racing_school --device cuda "${start[@]}" --rounds 0 \
    --out "$school" --live --save-every 10 --consolidation-rounds 6
fi
test -s "$school/qualified.pt"

if [[ -s "$speed/record.json" && -s "$speed/record.pt" ]]; then
  cat "$speed/record.json"
  exit 0
fi
start=(--initialize "$school/qualified.pt")
if [[ -s "$speed/training-state.zip" ]]; then
  start=(--resume "$speed/training-state.zip")
fi
echo 'STAGE speed reinforcement learning'
exec python3 -u -m training.train --device cuda "${start[@]}" --stack 8 --live \
  --steps 5000000 --target-two-lap-seconds "$record_seconds" --envs 8 --batch 96 \
  --official-eval --eval-episodes 3 --eval-every 25000 --checkpoint-every 100000 --run-dir "$speed"
