"""Export the fixed-shape racing actor for a TensorRT 8.2 deployment."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from training.agent import load_policy
from training.inertial import RelativeAttitude


IMAGE_HEIGHT = 72
IMAGE_WIDTH = 96


class FixedAveragePool(nn.Module):
    """Exact 9×12 to 5×6 adaptive pool using fixed AveragePool operators."""

    def forward(self, features):
        return torch.cat([F.avg_pool2d(features[:, :, top:bottom], (bottom - top, 2))
                          for top, bottom in ((0, 2), (1, 4), (3, 6), (5, 8), (7, 9))], dim=2)


class JetsonActor(nn.Module):
    def __init__(self, agent):
        super().__init__()
        if agent.backbone != "cnn" or agent.protected_prefix or agent.imu_encoder is None:
            raise ValueError("Export requires the single-head CNN RGB/IMU checkpoint.")
        layers = list(agent.encoder.children())
        self.vision = nn.Sequential(*layers[:6], FixedAveragePool(), *layers[7:])
        self.inertial = agent.imu_encoder
        self.control = agent.actor
        self.future = agent.future_actor
        self.horizon = agent.action_horizon
        self.register_buffer("imu_scale", torch.tensor(
            [10., 10., 10., 30., 30., 30.] + [1.] * (agent.imu_channels - 6)))

    def forward(self, rgb, imu):
        features = torch.cat((self.vision(rgb), self.inertial((imu / self.imu_scale).clamp(-4., 4.))), dim=1)
        hidden = self.control[:-1](features)
        current = self.control[-1](hidden).unsqueeze(1)
        if self.future is None:
            return current.tanh()
        future = current + self.future(hidden).reshape(-1, self.horizon - 1, 4)
        return torch.cat((current, future), dim=1).tanh()


def samples(agent, telemetry):
    rng = np.random.default_rng(2026)
    for _ in range(8):
        rgb = rng.integers(0, 256, size=(1, agent.stack, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        imu = rng.normal(size=(1, agent.stack, agent.imu_channels)).astype(np.float32)
        yield rgb, imu
    if telemetry is None:
        return
    with np.load(telemetry, allow_pickle=False) as data:
        rgb, imu, ticks = data["rgb"], data["imu"], data["seconds"]
    if (len(rgb) < agent.stack or rgb.shape[1:] != (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
            or imu.shape != (len(rgb), 6) or not np.all(np.diff(ticks) > 0)):
        raise ValueError("Telemetry requires native policy RGB, raw six-axis IMU and ordered times.")
    if agent.imu_channels == 10:
        relative = RelativeAttitude()
        imu = np.asarray([relative.augment(sample, tick) for sample, tick in zip(imu, ticks)])
    for end in np.linspace(agent.stack, len(rgb), 16, dtype=int):
        yield rgb[None, end - agent.stack:end], imu[None, end - agent.stack:end]


def measure_macs(module, inputs):
    macs = 0

    def count(layer, _inputs, output):
        nonlocal macs
        if isinstance(layer, nn.Conv2d):
            macs += output.numel() * (layer.in_channels // layer.groups) * int(np.prod(layer.kernel_size))
        else:
            macs += output.numel() * layer.in_features

    hooks = [layer.register_forward_hook(count) for layer in module.modules()
             if isinstance(layer, (nn.Conv2d, nn.Linear))]
    try:
        module(*inputs)
    finally:
        for hook in hooks:
            hook.remove()
    return macs


@torch.no_grad()
def export(checkpoint, output, telemetry):
    import onnx
    import onnxruntime

    source = checkpoint.read_bytes()
    agent = load_policy(source)
    model = JetsonActor(agent).eval()
    rgb = torch.zeros(1, agent.stack * 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    imu = torch.zeros(1, agent.stack, agent.imu_channels)
    args = rgb, imu
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "actor.onnx"
    temporary = output / "actor.tmp.onnx"
    torch.onnx.export(model, args, str(temporary), opset_version=13, dynamo=False,
                      input_names=["rgb", "imu"], output_names=["forecasts"])
    graph = onnx.load(temporary)
    onnx.checker.check_model(graph)
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 2
    runtime = onnxruntime.InferenceSession(str(temporary), options, providers=["CPUExecutionProvider"])
    error = []
    for frames, history in samples(agent, telemetry):
        expected = agent.forecast(frames, history)
        pixels = frames.transpose(0, 1, 4, 2, 3).reshape(rgb.shape).astype(np.float32) / 255. - .5
        inputs = {"rgb": pixels, "imu": history}
        actual, = runtime.run(["forecasts"], inputs)
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-4)
        error.append(float(np.max(np.abs(actual - expected))))
    for _ in range(8):
        runtime.run(None, inputs)
    elapsed = []
    for _ in range(60):
        start = perf_counter()
        runtime.run(None, inputs)
        elapsed.append((perf_counter() - start) * 1000)
    temporary.replace(model_path)
    for name, values in inputs.items():
        values.astype("<f4").tofile(output / f"reference-{name}.f32")
    expected.astype("<f4").tofile(output / "reference-forecasts.f32")
    report = {"checkpoint_sha256": sha256(source).hexdigest(),
              "params": sum(p.numel() for p in model.parameters()),
              "conv_linear_macs_per_inference": measure_macs(model, args),
              "onnx_bytes": model_path.stat().st_size, "opset": 13, "onnx_ir_version": graph.ir_version,
              "validation_histories": len(error), "real_histories": len(error) - 8,
              "max_abs_normalized_stick_error": max(error),
              "host_onnxruntime_cpu_p95_ms": float(np.quantile(elapsed, .95)),
              "rgb_shape": list(rgb.shape), "imu_shape": list(imu.shape),
              "forecast_shape": [1, agent.action_horizon, 4], "forecast_decay": agent.forecast_decay,
              "mode2_channels": ["yaw", "throttle", "roll", "pitch"]}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, help="Optional recorded policy RGB and IMU for parity tests.")
    args = parser.parse_args()
    torch.set_num_threads(2)
    export(args.checkpoint, args.output, args.telemetry)


if __name__ == "__main__":
    main()
