# A2RL simulator validation

This report checks the 2025 Grand Challenge layout, the Mode 2 RC interface, firmware-in-the-loop takeoff, and acceleration against recorded competition flights.

| Check | Result | Scope |
| --- | --- | --- |
| MSP RC interface | Four 1000–2000 µs AETR channels round-trip to Mode 2 commands with 1 µs resolution | MSP v1 command framing, channel mapping, and checksum |
| Native 2026 practice flow teacher | 22 gates and two laps in each of three runs, fastest 66.11 s, mean crossing speed 5.00 m/s | State-driven teacher tested from three ground starts |
| Elodin with Betaflight SITL, 2026 practice | Reached 0.96 m and exported 41 FPV frames at 820 × 616 | Scripted 1.4 s takeoff, no course completion |
| Elodin flow teacher, 2026 practice | Default controller reached 0/22. Two lower-gain trials reached 1/22 before leaving the flight volume. | The rate controller is sensitive to the plant model. The trials did not qualify either gain set for training. |
| MonoRace Log 8, late segment | Axial specific-force RMSE of 3.41 m/s², bias of +0.45 m/s², and peak error of 14.54 m/s² over 367 windows | Calibrated plant prediction using measured rotor speeds and estimated motion |
| MonoRace Log 5, held-out flight | 1.57 m/s² forward and 1.99 m/s² lateral specific-force RMSE over 897 windows | Horizontal drag prediction using independent flight data |
| MonoRace Log 7, held-out flight | 1.40 m/s² forward and 2.25 m/s² lateral specific-force RMSE over 881 windows | Horizontal drag prediction using independent flight data |

The acceleration tests use 20 ms windows. The early segment of Log 8 provides 319 fitting windows for the horizontal drag parameters. The late segment provides the held-out windows. [MonoRace paper](https://arxiv.org/html/2601.15222) · [4TU competition flight data](https://doi.org/10.4121/039abd82-094c-4c93-baa1-e7446f562e0e)

The simulator runs an MSP-compatible receiver boundary, then the project's Acro rate controller and mixer. The published measurements cover propulsion and drag under the recorded inputs. To establish controller fidelity on the intended airframe, compare firmware-in-the-loop command timing and body-rate response, then compare RPM, full trajectories, and gate clearances on held-out runs from that airframe. Report the learned pilot's completion rate separately from the state-driven teacher.

## Reproduce the measurements

```bash
cargo build --release --lib --no-default-features
python -m pip install Pillow
A2RL_COURSE=2026-practice A2RL_TEACHER_PROFILE=flow python -m training.racing_school --device cpu --teacher-only --out runs/validation/native-flow
python -m tools.prepare_monorace_demos runs/validation/monorace --validation-logs
python -m tools.check_flight_log runs/validation/monorace/log8/imu.csv runs/validation/monorace/log8/ekf.csv
python -m tools.check_lateral_drag \
  --calibration runs/validation/monorace/log8/imu.csv runs/validation/monorace/log8/ekf.csv \
  --flight Log8-holdout runs/validation/monorace/log8/imu.csv runs/validation/monorace/log8/ekf.csv 62.4 69.8 \
  --flight Log5 runs/validation/monorace/log5/imu.csv runs/validation/monorace/log5/ekf.csv 10.85 30 \
  --flight Log7 runs/validation/monorace/log7/imu.csv runs/validation/monorace/log7/ekf.csv 26.45 44.5
```
