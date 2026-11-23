# Jetson Nano 4 GB flight inference

The training host exports a batch-one actor as ONNX opset 13. The Nano target is the original 4 GB module with JetPack 4.6.6 and its TensorRT 8.2 runtime. The flight process handles camera acquisition, inertial synchronization, the TensorRT actor, action smoothing and communication with Betaflight.

## Export on the training host

Install ONNX and ONNX Runtime alongside the project's working PyTorch installation on the export host. For archived Elodin data, use the synchronized RGB and IMU telemetry file for the numerical comparison.

```sh
python -m tools.export_jetson_policy \
  --checkpoint path/to/checkpoint.pt \
  --telemetry path/to/flight-telemetry.npz \
  --output path/to/export
```

The manifest records the checkpoint hash, fixed binding shapes, inference parameter count and host parity results. The export also includes matching little-endian float32 input and reference output files. Keep these files with the ONNX artifact.

## Validate on the board

Build the engine on the Nano with the installed JetPack version. A TensorRT FP32 build is the reference for comparing mixed-precision control outputs.

```sh
/usr/src/tensorrt/bin/trtexec --onnx=actor.onnx --workspace=256 --saveEngine=actor-fp32.engine
/usr/src/tensorrt/bin/trtexec --onnx=actor.onnx --workspace=256 --fp16 --saveEngine=actor-fp16.engine
```

Run the supplied reference inputs on each engine and compare all forecast channels against the supplied output before using the engine in flight. Measure peak unified-memory use and latency with the camera, preprocessing, sensor synchronization and Betaflight transport active. At 30 Hz, the full sensing-to-command cycle has a 33.3 ms period. Collect p50, p95, p99 and missed deadlines over sustained operation together with frame age, dropped frames, power mode and temperature. Host ONNX Runtime measurements serve as export verification.

## Flight interface

- The RGB binding contains consecutive normalized RGB frames packed oldest to newest as NCHW. For a 24-frame checkpoint the shape is `[1,72,72,96]`. Resize each frame to 96×72 with the same field of view and image preprocessing as training. Pixel normalization is `value / 255 - 0.5`.
- The IMU binding contains one synchronized sample per camera frame. A 10-channel checkpoint uses gyro, accelerometer and relative attitude in XYZW order. The reference preprocessor is `training/inertial.py`. Normalize axis directions and time stamps to match training.
- The forecast binds `[batch, horizon, yaw/throttle/roll/pitch]`, with values in `[-1,1]`. On each tick, average forecasts for that tick using weights `forecast_decay**age`, as in `tools/elodin_policy.py`. Use the same launch history initialization and Mode 2 RC normalization as the simulator.
- Establish a bench test for the camera-to-IMU time alignment, stale samples and Betaflight command timing before collecting real-flight data. Document the live capture path and OS, driver and firmware versions for the experiment.

[NVIDIA Jetson Nano specifications](https://developer.nvidia.com/embedded/jetson-nano) · [JetPack 4.6.6](https://developer.nvidia.com/jetpack-sdk-466) · [TensorRT 8.2 documentation](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-825/pdf/TensorRT-Developer-Guide.pdf)
