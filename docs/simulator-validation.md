# A2RL simulator validation

This report checks the 2025 Grand Challenge layout, the Mode 2 RC interface, firmware-in-the-loop takeoff, and acceleration against recorded competition flights.

| Check | Result | Scope |
| --- | --- | --- |
| MSP RC interface | Four 1000–2000 µs AETR channels round-trip to Mode 2 commands with 1 µs resolution | MSP v1 command framing, channel mapping, and checksum |
| Native 2026 flow teacher | Nine clean two-lap runs with 24 passes each, 40.42–40.45 s, mean crossing speed 4.68 m/s | Both Ladder passes traveled southwest. The runs spent no time below 2 m/s after the first gate. |
| Elodin with Betaflight SITL, 2026 race reconstruction | Clean 24-gate run in 34.201 s, mean gate crossing speed 6.04 m/s | 1 kHz lockstep with 30 Hz synchronized RGB, IMU and Mode 2 commands. The dashboard shows this run. |
| MonoRace Log 8, late segment | Axial specific-force RMSE of 3.41 m/s², bias of +0.45 m/s², and peak error of 14.54 m/s² over 367 windows | Calibrated plant prediction using measured rotor speeds and estimated motion |
| MonoRace Log 5, held-out flight | 1.57 m/s² forward and 1.99 m/s² lateral specific-force RMSE over 897 windows | Horizontal drag prediction using independent flight data |
| MonoRace Log 7, held-out flight | 1.40 m/s² forward and 2.25 m/s² lateral specific-force RMSE over 881 windows | Horizontal drag prediction using independent flight data |

The acceleration tests use 20 ms windows. The early segment of Log 8 provides 319 fitting windows for the horizontal drag parameters. The late segment provides the held-out windows. [MonoRace paper](https://arxiv.org/html/2601.15222) · [4TU competition flight data](https://doi.org/10.4121/039abd82-094c-4c93-baa1-e7446f562e0e)

The 2026 reconstruction uses the horizontal gate centers and plane directions visible in FlyBy's real-flight overlay. It includes two double-aperture frames. The 13.2 cm frame depth comes from the X extent of Berkeley's public gate mesh with its stage set to one meter per unit. The FPV camera, Elodin preview, and collision tests all use that depth. Gate elevations, flight volume, and the launch spot are estimates. The paper's virtual route markers do not become physical gate frames. Venue fidelity still requires a measured layout and official gate CAD. [FlyBy Figures 2 and 5](https://www.imavs.org/pdf/imav.2026.1) · [Berkeley gate mesh](https://github.com/airobotics-ucberkeley/isaac_drone_racer_public/blob/66138b7621f051ce6d31dc418a57edf5027f8381/assets/gate/gate.usd)

The ordered track segments in FlyBy Figure 5 cross the Ladder plane toward the southwest twice. The public replay shows the lower crossings at 20.40 and 40.77 seconds from takeoff. The path goes around the west edge between crossings. The teacher's height profile through that loop is an estimate.

The 2026 teacher limits upward pitch near non-climbing gate approaches and leaves the pitch controller free on climbs. Across the recorded two laps, the highest pitch in the last eight control samples of any non-climbing gate approach was 2.3° below horizontal. This revision crosses gates faster on average than the earlier teacher (4.68 vs. 4.35 m/s) and takes about 1.3 s longer over two laps.

The 2026 simulation camera uses a 70° mount pitch. On the dashboard replay trajectory, the target gate center projects inside the image in 65.8% of sampled approach frames at 70° and all four aperture corners fit in 24.0%. At the old 45° pitch, the same poses give 100% center visibility and 64.4% full-aperture visibility. The projection uses 20 frames per gate at 30 Hz and excludes the last four frames before each crossing. The visibility rates describe geometric projection on this rollout. [KAIST camera setup and gate visibility study](https://onlinelibrary.wiley.com/doi/10.1002/rob.70371)

The dashboard run reached 10.62 m/s and the highest body pitch in the last 0.267 s before a non-climbing gate was 2.11° above horizontal. The flight exported 1,075 synchronized demonstration samples and a 1,098-frame video with one recorded RC sample per frame. Timing is measured from the first gate to the last. The [SITL preset](../assets/a2rl-betaflight-sitl.cli) configures the simulator's Acro rates and PID gains. [Elodin reproduction commands](../README.md#elodin-and-betaflight-flight-validation)

The dashboard run covered 207.85 m at a mean speed of 6.08 m/s. The first full circuit, measured from gate one on lap one to gate one on lap two, took 17.824 s. The upper-to-lower Ladder took about 2.58 s on each lap. The 2026 AI Speed Race runs two consecutive laps and A2RL reports a winning time of 12.032 s for TII Racing. Covering the current teacher's path in that time would require averaging 17.27 m/s. [A2RL results](https://a2rl.io/news/45/A2RL-Drone-Championship-Sets-the-Pace-for-AI-in-Autonomous-Flight) · [FlyBy Speed Racing report](https://informatics.tuwien.ac.at/news/3102)

The teacher removed the fixed-speed window at five open gates, while the backward braking pass still enforces subsequent curve limits. On the straight to gate nine, the Betaflight receiver received four 100% throttle samples over two laps, for 0.133 s total. Mean throttle from gate one to the finish was 22.2%. The braking controller adds throttle when the horizontal thrust component opposes travel and caps vertical lift. On measured braking samples with that orientation and deceleration above 1 m/s², mean throttle rose from 11.4% to 12.1%, and mean deceleration from 6.45 to 6.76 m/s². The open-gate variant without the braking adjustment finished in 34.110 s.

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
