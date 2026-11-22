# A2RL simulator validation

This report checks the 2025 Grand Challenge layout, the Mode 2 RC interface, firmware-in-the-loop takeoff, and acceleration against recorded competition flights.

| Check | Result | Scope |
| --- | --- | --- |
| MSP RC interface | Four 1000–2000 µs AETR channels round-trip to Mode 2 commands with 1 µs resolution | MSP v1 command framing, channel mapping, and checksum |
| Native 2026 flow teacher | Nine clean two-lap runs with 24 passes each, 40.42–40.45 s, mean crossing speed 4.68 m/s | Both Ladder passes traveled southwest. The runs spent no time below 2 m/s after the first gate. |
| Elodin with Betaflight SITL, 2026 race reconstruction | Clean 24-gate run in 31.657 s, mean gate crossing speed 6.92 m/s | 1 kHz lockstep with 30 Hz synchronized RGB, IMU and Mode 2 commands. The dashboard shows this run. |
| MonoRace Log 8, late segment | Axial specific-force RMSE of 3.41 m/s², bias of +0.45 m/s², and peak error of 14.54 m/s² over 367 windows | Calibrated plant prediction using measured rotor speeds and estimated motion |
| MonoRace Log 5, held-out flight | 1.57 m/s² forward and 1.99 m/s² lateral specific-force RMSE over 897 windows | Horizontal drag prediction using independent flight data |
| MonoRace Log 7, held-out flight | 1.40 m/s² forward and 2.25 m/s² lateral specific-force RMSE over 881 windows | Horizontal drag prediction using independent flight data |

The acceleration tests use 20 ms windows. The early segment of Log 8 provides 319 fitting windows for the horizontal drag parameters. The late segment provides the held-out windows. [MonoRace paper](https://arxiv.org/html/2601.15222) · [4TU competition flight data](https://doi.org/10.4121/039abd82-094c-4c93-baa1-e7446f562e0e)

The 2026 reconstruction uses the horizontal gate centers and plane directions visible in FlyBy's real-flight overlay. It includes two double-aperture frames. The 13.2 cm frame depth comes from the X extent of Berkeley's public gate mesh with its stage set to one meter per unit. The FPV camera, Elodin preview, and collision tests all use that depth. Gate elevations, flight volume, and the launch spot are estimates. The paper's virtual route markers do not become physical gate frames. Venue fidelity still requires a measured layout and official gate CAD. [FlyBy Figures 2 and 5](https://www.imavs.org/pdf/imav.2026.1) · [Berkeley gate mesh](https://github.com/airobotics-ucberkeley/isaac_drone_racer_public/blob/66138b7621f051ce6d31dc418a57edf5027f8381/assets/gate/gate.usd)

The ordered track segments in FlyBy Figure 5 cross the Ladder plane toward the southwest twice. The public replay shows the lower crossings at 20.40 and 40.77 seconds from takeoff. The path goes around the west edge between crossings. The teacher's height profile through that loop is an estimate.

The 2026 teacher limits upward pitch near non-climbing gate approaches and leaves the pitch controller free on climbs. Across the recorded two laps, the highest pitch in the last eight control samples of any non-climbing gate approach was 2.3° below horizontal. This revision crosses gates faster on average than the earlier teacher (4.68 vs. 4.35 m/s) and takes about 1.3 s longer over two laps.

The 2026 simulation camera uses a 70° mount pitch. On the dashboard replay trajectory, the target gate center projects inside the image in 69.4% of sampled approach frames at 70° and all four aperture corners fit in 36.3%. At the old 45° pitch, the same poses give 100% center visibility and 69.2% full-aperture visibility. The projection uses 20 frames per gate at 30 Hz and excludes the last four frames before each crossing. The visibility rates describe geometric projection on this rollout. [KAIST camera setup and gate visibility study](https://onlinelibrary.wiley.com/doi/10.1002/rob.70371)

The dashboard run reached 12.36 m/s. Its body pitch reached 10.96° above horizontal in the last 0.267 s before gate ten on lap one. The flight exported 998 synchronized demonstration samples and a 1,021-frame video with one recorded RC sample per frame. Timing is measured from the first gate to the last. [SITL preset](../assets/a2rl-betaflight-sitl.cli) · [Elodin reproduction commands](../README.md#elodin-and-betaflight-flight-validation)

The dashboard run covered 208.20 m at a mean speed of 6.58 m/s. The first full circuit, measured from gate one on lap one to gate one on lap two, took 16.386 s. The upper-to-lower Ladder took about 2.39 s per lap, down from 2.59 s. The 2026 AI Speed Race runs two consecutive laps and A2RL reports a winning time of 12.032 s for TII Racing. [A2RL results](https://a2rl.io/news/45/A2RL-Drone-Championship-Sets-the-Pace-for-AI-in-Autonomous-Flight) · [FlyBy Speed Racing report](https://informatics.tuwien.ac.at/news/3102)

The new controller optimizes crossing positions and tangents on four open approaches, checks the resulting line against gate frames, and corrects predicted lateral error before the gate plane. The first geometry-only search made contact at gate two, so the accepted search fixes the gates with the least clearance in the measured reference flight. The selected setting improved the two-lap time from 32.999 to 31.657 s. Mean throttle from gate one to the finish was 23.5%. Throttle at or above 90% lasted 0.100 s (0.3% of the timed run).

### Predictive SITL teacher

The teacher previews the change in velocity along the line to compensate for attitude lag. It also estimates where the drone will cross the gate and limits excess lift when the predicted crossing is high. The takeoff and the lower Ladder use their dedicated approach. In the preliminary Rust sweep, 1,332 of 6,207 settings completed the course and the fastest scored 22.950 s. When the most aggressive settings moved to Elodin with Betaflight, the drone contacted the early gate frames.

A moderate setting cleared all 24 gates in 29.475 s with a 24.83% time-weighted receiver throttle and 0.133 s at or above 90% throttle. Its first full circuit took 15.291 s. The upper-to-lower Ladder still took 2.344 s and 2.340 s on the respective laps. A repeat with the same gains made contact on lap two before the third gate, so this run establishes a best observed time for that setting. After widening the third gate’s lateral prediction window and lowering that leg’s target speed, the reference start and a 3 cm lateral-offset start both cleared all 24 gates. They finished in 29.400 s and 29.468 s respectively, with mean receiver throttle of 24.94% and 24.86%. The best run spent 0.100 s at or above 90% throttle. The archived dashboard replay shows the 29.400 s run. The comparison uses reconstructed gate geometry and the state-driven teacher.

### Ladder trajectory search

The search evaluated 10,001 settings for the guide positions, tangents, and turn and braking limits between the upper and lower Ladder apertures. Eight screening states came from two prior Elodin flight exports. A 12% increase to the turn limit alone caused contact before the lower aperture in a full Elodin and Betaflight rollout.

The selected path moved the loop inward and aligned the descent sooner. The nominal and 3 cm offset starts cleared all 24 gates in 29.217 s and 29.267 s. The selected nominal run took 2.145 s and 2.137 s through the Ladder on its respective laps, compared with 2.336 s and 2.326 s on the previous best run. A more aggressive path reduced those splits to 2.057 s and 2.049 s, but cost time on the approach and finished in 29.349 s.

Keeping the original speed schedule on the approach and applying the revised path after the upper crossing reduced the best full run to 28.984 s. The 3 cm offset retest finished in 28.986 s. Both runs cleared 24 gates with zero contact. The nominal run crossed the Ladder in 2.091 s and 2.082 s and averaged 24.46% receiver throttle. An earlier run of this configuration disarmed after 22 crossings without reported contact. The release build repeated the result in 28.961 s, with 24.42% mean throttle and Ladder splits of 2.090 s and 2.079 s. The archived dashboard replay uses the release run.

### MCK stick-input reference

Using TII Racing v3 flights 13p, 14p and 15p on trackRATM, the time-weighted means of the logged throttle channels were 48.2%, 49.1% and 49.7%. Each flight reached 100%. The logs spent 2.6%, 3.6% and 4.9% of their respective observed airborne intervals at or above 90%. The calculation uses the interval between adjacent retained camera samples as its weight. These intervals cover 44.767, 44.501 and 34.701 seconds on trackRATM. The reported 11-second A2RL run needs its own synchronized transmitter trace to measure its throttle distribution. [TII Racing v3 logs](https://github.com/tii-racing/drone-racing-dataset/releases/tag/v3.0.0)

### 80% mean throttle experiment

In a deterministic Rust screening sweep, 169 of 1,201 variants completed the course. The highest time-sampled mean throttle among those finishers was 25.38%. Two coupled attitude-and-throttle candidates then completed all 24 crossings in Elodin + Betaflight SITL. Their measured means were 23.62% in 31.700 s and 23.63% in 31.483 s. Holding the receiver at 80% after the first crossing caused contact before the second gate. The 80% target remains unmet. The Elodin result sidecar calculates the time-weighted receiver throttle from the first crossing through the finish, including the portions of the first and last recorded control samples that lie inside that interval.

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
