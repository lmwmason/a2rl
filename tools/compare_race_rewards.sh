#!/usr/bin/env bash
set -euo pipefail

source_policy=${1:?Pass the current best speed policy.}
qualified=${2:?Pass the qualified imitation checkpoint.}
out=${3:?Pass a new directory for the paired pilot.}
mkdir "$out"

# Both trials use the same starting policy and the same fresh critic.
python3 - "$source_policy" "$out/initializer.pt" <<'PY'
import sys

import torch

from training.agent import Agent

torch.manual_seed(21)
policy = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
starter = Agent(8, "cpu", 0.).weights()
starter.update(encoder=policy["encoder"], actor=policy["actor"])
torch.save(starter, sys.argv[2])
PY

for profile in pace standard; do
  printf 'STAGE %s reward pilot\n' "$profile"
  A2RL_REWARD_PROFILE="$profile" python3 -u -m training.train --device cuda \
    --initialize "$out/initializer.pt" --protect-qualified "$qualified" \
    --stack 8 --envs 4 --seed 21 --batch 64 --update-every 32 --warmup 1000 --replay 24000 \
    --steps 24000 --official-eval --eval-episodes 3 --eval-every 8000 \
    --checkpoint-every 8000 --run-dir "$out/$profile"
done

python3 - "$out" <<'PY'
import json
from pathlib import Path
import sys

out = Path(sys.argv[1])
summary = {}
for profile in ("pace", "standard"):
    records = [json.loads(line) for line in (out / profile / "metrics.jsonl").read_text().splitlines()]
    initial, final = records[0], records[-1]
    clean = [row for row in records[1:] if row["two_lap_rate"] == 1. and row["crash_rate"] == 0.]
    best = min(clean, key=lambda row: row["official_mean_seconds"], default=None)
    summary[profile] = {
        "initial_mean_s": initial["official_mean_seconds"],
        "final_mean_s": final["official_mean_seconds"],
        "final_finish_rate": final["two_lap_rate"],
        "final_crash_rate": final["crash_rate"],
        "best_clean_mean_s": best["official_mean_seconds"] if best else None,
        "best_clean_standard_s": best["official_start_seconds"] if best else None,
        "evaluated_steps": final["steps"],
    }
(out / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
