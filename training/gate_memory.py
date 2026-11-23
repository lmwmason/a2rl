"""Train a visual–inertial encoder to remember directions to upcoming gates."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .agent import load_policy, pixel_tensor
from .inertial import RelativeAttitude


def body_directions(displacements, quaternion):
    """Rotate world-frame displacements into the current body frame."""
    quaternion = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
    imaginary, real = quaternion[..., :3], quaternion[..., 3:]
    offset = 2 * np.cross(imaginary, displacements)
    vectors = displacements - real * offset + np.cross(imaginary, offset)
    length = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return (vectors / np.maximum(length, 1e-6)).astype(np.float32)


class History:
    def __init__(self, path, stack, imu_channels):
        with np.load(path, allow_pickle=False) as clip:
            self.rgb, self.imu = clip["rgb"], clip["imu"].astype(np.float32)
            gates, path_xyz = clip["next_gate_index"], clip["course_centers_m"]
            position, quaternion = clip["position_m"], clip["quaternion_xyzw"]
            seconds = clip["seconds"]
        if (self.rgb.shape != (len(seconds), 72, 96, 3)
                or self.imu.shape != (len(seconds), 6) or not np.all(np.diff(seconds) > 0)):
            raise ValueError(f"Invalid synchronized observations in {path}")
        valid = (gates >= 0) & (gates + 1 < len(path_xyz)) & (seconds >= .75)
        valid &= np.arange(len(seconds)) >= stack - 1
        self.valid = np.flatnonzero(valid)
        if not len(self.valid):
            raise ValueError(f"No usable gate-transition history in {path}")
        self.stack = stack
        if imu_channels == 10:
            attitude = RelativeAttitude()
            self.imu = np.asarray([attitude.augment(row, now) for row, now in zip(self.imu, seconds)])
        gate_indexes = np.clip(gates[:, None] + np.arange(2), 0, len(path_xyz) - 1)
        self.directions = body_directions(path_xyz[gate_indexes] - position[:, None, :],
                                          quaternion[:, None, :])

    def sample(self, end):
        first = end + 1 - self.stack
        return self.rgb[first:end + 1], self.imu[first:end + 1], self.directions[end]


def references(clips):
    return [(clip, index) for clip in clips for index in clip.valid]


def batch(samples, device):
    frames, imu, target = zip(*(clip.sample(index) for clip, index in samples))
    return (pixel_tensor(np.stack(frames), device).float() / 255. - .5,
            np.stack(imu), torch.as_tensor(np.stack(target), device=device))


def blank_latest_frames(pixels, frames):
    obscured = pixels.clone()
    obscured[:, -3 * frames:] = 0.
    return obscured


def on_device(checkpoint, device):
    agent = load_policy(checkpoint)
    if agent.imu_encoder is None or agent.protected_prefix or agent.stack < 6:
        raise ValueError("Gate memory requires a single-head RGB/IMU policy with at least six frames.")
    agent.device = device
    for module in (agent.encoder, agent.imu_encoder, agent.actor, agent.future_actor):
        if module is not None:
            module.to(device).eval()
    agent.actor.requires_grad_(False)
    if agent.future_actor is not None:
        agent.future_actor.requires_grad_(False)
    return agent


def train(args):
    fingerprints = [{sha256(path.read_bytes()).hexdigest() for path in group}
                    for group in (args.training, args.validation)]
    if fingerprints[0] & fingerprints[1]:
        raise ValueError("Training and validation must contain separate flights.")
    source = args.initial.read_bytes()
    agent = on_device(source, args.device)
    reference = on_device(source, args.device)
    train_rows, check_rows = (references([History(p, agent.stack, agent.imu_channels) for p in paths])
                              for paths in (args.training, args.validation))
    head = nn.Linear(agent.actor[0].in_features, 6).to(args.device)
    optimizer = torch.optim.Adam([
        {"params": agent.observation_parameters(), "lr": 2e-5},
        {"params": head.parameters(), "lr": 1e-3},
    ])
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        order = rng.permutation(len(train_rows))
        for start in range(0, len(order), args.batch_size):
            pixels, imu, target = batch([train_rows[i] for i in order[start:start + args.batch_size]], args.device)
            masked = blank_latest_frames(pixels, int(rng.integers(1, 6)))
            directions = F.normalize(head(agent.policy_features(masked, imu)).reshape(-1, 2, 3), dim=-1)
            with torch.no_grad():
                original = reference.action_sequence(reference.policy_features(pixels, imu))
            action = agent.action_sequence(agent.policy_features(pixels, imu))
            loss = (1. - (directions * target).sum(dim=-1)).mean() + 20. * F.mse_loss(action, original)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(json.dumps({"epoch": epoch + 1, "last_training_batch_loss": float(loss.detach())}), flush=True)

    angles, stick_errors = [], []
    with torch.no_grad():
        for start in range(0, len(check_rows), args.batch_size):
            pixels, imu, target = batch(check_rows[start:start + args.batch_size], args.device)
            masked = blank_latest_frames(pixels, 5)
            direction = F.normalize(head(agent.policy_features(masked, imu)).reshape(-1, 2, 3), dim=-1)
            angles.extend(torch.rad2deg((direction * target).sum(dim=-1).clamp(-1, 1).acos()).cpu().tolist())
            original = reference.action_sequence(reference.policy_features(pixels, imu))
            action = agent.action_sequence(agent.policy_features(pixels, imu))
            stick_errors.extend((original - action).abs().mean(dim=(1, 2)).cpu().tolist())
    metrics = {"initial_sha256": sha256(source).hexdigest(), "seed": args.seed,
               "training_flights": len(args.training), "validation_flights": len(args.validation),
               "heldout_frames": len(check_rows),
               "median_direction_error_deg": np.median(angles, axis=0).tolist(),
               "mean_normalized_stick_drift": float(np.mean(stick_errors)),
               "validation_masked_recent_frames": 5}
    checkpoint = torch.load(args.initial, map_location="cpu", weights_only=True)
    checkpoint.update(encoder=agent.encoder.cpu().state_dict(),
                      imu_encoder=agent.imu_encoder.cpu().state_dict(),
                      gate_memory_training=metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(args.output)
    print(json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--training", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.initial.resolve() == args.output.resolve():
        parser.error("Use positive training sizes and a separate output checkpoint.")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
