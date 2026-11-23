---
name: a2rl-manuscript-evidence
description: Audit the A2RL firmware-in-the-loop racing project and prepare evidence for its research manuscript or supplemental evaluations.
---

- Input: the A2RL code snapshot, private experiment records or original Elodin reports, trained checkpoint IDs, and the intended claim.
- Separate demonstration, handoff/coaching, selection, and independent evaluation trials. Verify 24 ordered gate crossings, contact status, gate-1-to-gate-24 timing and per-start checkpoint identity. Count failures in completion denominators. Training interactions, evaluation attempts and measured real flights are distinct.
- Verify implementation claims in training/imitate.py, training/speed_elodin.py, training/race_reward.py and tools/evaluate_rgb_imu_race.py. Explain that simulator state may be used for supervision and scoring while the student observes camera and IMU signals. Treat three-start qualification as operational evidence, and reserve causal/cross-seed claims for matched ablations.
- Output a claim-to-file/line ledger, tables with their denominator and conditions, and a list of experimental gaps. Keep private notes, raw logs, and unpublished manuscripts out of public Git. If checkpoint or course provenance is missing, hold the affected claim pending verification.
- Store generated LaTeX PDFs, PDF figures, logs and intermediate build files under the private journal workspace's `paper-pdf/`. Use the configured build script from the manuscript source directory and verify that no PDF or build files were generated alongside the source.
