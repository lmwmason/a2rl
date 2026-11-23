"""Pretrain the Mode 2 actor on recorded FPV demonstrations."""
import argparse
from itertools import count
import json
from pathlib import Path
from time import sleep

import numpy as np
import torch

from .agent import Agent, extend_cnn_history, extend_imu_features, pixel_tensor, random_shift, stick_loss
from .demonstrations import Clip, Demonstrations


@torch.no_grad()
def held_out_mae(agent, clips, batch_size):
    errors = []
    for clip in clips:
        for first in range(0, len(clip), batch_size):
            frames, actions, _, *imu = clip.batch(np.arange(first, min(len(clip), first + batch_size)))
            errors.append(np.abs(agent.action(frames, imu=imu[0] if imu else None) - actions))
    return np.concatenate(errors).mean(axis=0).tolist() if errors else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, nargs="+", required=True)
    parser.add_argument("--sim", type=Path, nargs="*", default=[])
    parser.add_argument("--held-out", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--initialize", type=Path, help="Start from an existing policy checkpoint.")
    parser.add_argument("--until", type=Path, help="Keep training offline until the evaluator writes this file.")
    parser.add_argument("--completion-marker", type=Path,
                        help="Use the validated finisher to start 30-second pace imitation.")
    parser.add_argument("--correction-directory", type=Path,
                        help="Train one round for each newly recorded student flight.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--updates", type=int, default=1400)
    parser.add_argument("--preserve-successful-prefixes", action="store_true",
                        help="Retain effective student actions in independently cleared gate approaches.")
    parser.add_argument("--protect-prefix-through", type=int, default=0,
                        help="Keep the early student fixed and train a later racing controller with a learned boundary.")
    parser.add_argument("--anchor-prefix", type=Path,
                        help="Preserve a repeatably cleared early route using a frozen student checkpoint.")
    parser.add_argument("--anchor-gates", type=int, default=6)
    parser.add_argument("--anchor-weight", type=float, default=.65)
    parser.add_argument("--focus-directory", type=Path,
                        help="Keep these gate-handoff recordings available after the rolling buffer advances.")
    parser.add_argument("--focus-gates", type=int, nargs=2, metavar=("FIRST", "LAST"),
                        help="Reserve completion batches for live corrections in this inclusive gate range.")
    parser.add_argument("--secondary-focus-gates", type=int, nargs=2, metavar=("FIRST", "LAST"),
                        help="Reserve a smaller separate portion of completion batches for a later gate range.")
    parser.add_argument("--frontier-sampling", action="store_true",
                        help="Sample independent corrections more often when the student reached later gates.")
    parser.add_argument("--action-horizon", type=int, default=1,
                        help="Predict this many consecutive Mode 2 actions for temporal ensembling.")
    parser.add_argument("--gate-progress-loss", type=float, default=0.,
                        help="Supervise the shared RGB+IMU features on the next gate during training.")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--primary-fraction", type=float, default=.40,
                        help="Batch fraction drawn from --real when both dataset groups are supplied.")
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--stack", type=int, default=4, help="Number of consecutive RGB frames per observation.")
    parser.add_argument("--extend-history", action="store_true",
                        help="Start a larger RGB+IMU window from the latest CNN weights.")
    parser.add_argument("--offline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--relative-attitude", action="store_true", help="Add attitude integrated from the observed gyro.")
    parser.add_argument("--imu", action="store_true", help="Fuse synchronized gyro and accelerometer samples with RGB.")
    parser.add_argument("--backbone", choices=("cnn", "mobilenet-v3-small"), default="cnn")
    parser.add_argument("--eval-every", type=int, default=200)
    args = parser.parse_args()
    if min(args.batch, args.stack, args.eval_every) <= 0 or args.updates < 0:
        parser.error("Counts must be positive.")
    if not 0 <= args.protect_prefix_through < 24:
        parser.error("The protected prefix must end before the last gate.")
    if args.protect_prefix_through and (not args.imu or args.action_horizon < 2 or args.anchor_prefix):
        parser.error("A protected prefix needs RGB+IMU forecasts and its own frozen reference.")
    if args.focus_directory and not (args.focus_gates and args.imu):
        parser.error("A focus directory requires RGB+IMU and an explicit gate range.")
    for focus in (args.focus_gates, args.secondary_focus_gates):
        if focus and not 1 <= focus[0] <= focus[1] <= 24:
            parser.error("Focus gates must be an ordered range within 1–24.")
    if args.anchor_prefix and (not 1 <= args.anchor_gates <= 23 or not np.isfinite(args.anchor_weight)
                               or args.anchor_weight <= 0):
        parser.error("Anchored route prefix and weight must be positive.")
    if not 1 <= args.action_horizon <= 12:
        parser.error("Action horizon must be between 1 and 12 observations.")
    if args.updates == 0 and not args.until:
        parser.error("Unlimited training requires an evaluator marker.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Use a positive learning rate.")
    if not np.isfinite(args.gate_progress_loss) or args.gate_progress_loss < 0:
        parser.error("Gate progress loss must be finite and nonnegative.")
    if not 0 < args.primary_fraction <= 1:
        parser.error("The primary dataset fraction must be greater than zero and at most one.")
    if args.correction_directory and not (args.imu and args.until):
        parser.error("The student correction feed requires evaluator-controlled offline RGB+IMU training.")
    if args.completion_marker and not args.until:
        parser.error("Pace imitation requires evaluator-controlled offline training.")
    if args.relative_attitude and not args.imu:
        parser.error("Relative attitude requires synchronized IMU observations.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this Python environment.")
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True

    stack, height, width = args.stack, 72, 96
    latest = args.out / "latest.pt"
    resumed = bool(args.until and latest.is_file())
    source = latest if resumed else args.initialize
    weights = torch.load(source, map_location="cpu", weights_only=True) if source else None
    channels = 10 if args.relative_attitude else 6

    def adapt_weights(params):
        resized = extend_cnn_history(params, stack) if args.extend_history else params
        return extend_imu_features(resized, channels) if args.relative_attitude else resized

    agent = Agent(stack, args.device, -.66, with_imu=args.imu,
                  backbone=weights.get("vision_backbone", "cnn") if weights else args.backbone,
                  pretrained=weights is None and args.backbone != "cnn",
                  action_horizon=args.action_horizon, imu_channels=channels)
    if weights is not None:
        agent.set_weights(adapt_weights(weights))
    if args.protect_prefix_through:
        if weights is None:
            parser.error("Initialize the protected prefix from an independently evaluated student.")
        agent.configure_protected_prefix(args.protect_prefix_through)
    if agent.protected_prefix != args.protect_prefix_through:
        parser.error("Pass the protected prefix length recorded in the checkpoint.")
    progress = weights.get("training_progress", {}) if resumed else {}
    first_round, first_step, first_samples = (progress.get(key, 0) for key in ("round", "updates", "steps"))
    anchor = None
    if args.anchor_prefix:
        anchor_weights = torch.load(args.anchor_prefix, map_location="cpu", weights_only=True)
        anchor = Agent(stack, args.device, 0., with_imu=args.imu,
                       backbone=anchor_weights.get("vision_backbone", "cnn"),
                       action_horizon=anchor_weights.get("action_horizon", 1), imu_channels=channels)
        anchor.set_weights(adapt_weights(anchor_weights))
    demos = Demonstrations(args.real, args.sim, stack, height, width, args.seed, with_imu=args.imu,
                           primary_fraction=args.primary_fraction, gate_labels=args.gate_progress_loss > 0 or anchor is not None or agent.protected_prefix > 0,
                           action_horizon=args.action_horizon, frontier_sampling=args.frontier_sampling,
                           relative_attitude=args.relative_attitude, focus_gates=args.focus_gates,
                           secondary_focus_gates=args.secondary_focus_gates)
    if args.focus_directory:
        selected = sorted(path for path in args.focus_directory.glob("*.npz") if not path.name.endswith(".tmp.npz"))
        demos.focus_clips = [
            Clip(path, stack, height, width, with_imu=True, relative_attitude=args.relative_attitude,
                 preserve_handoff=args.preserve_successful_prefixes)
            for path in selected
        ]
    if args.until:
        print(json.dumps({"imitation_datasets": [
            {"path": str(path), "frames": len(clip)}
            for path, clip in zip(args.real + args.sim, demos.real + demos.sim)
        ], "batch": args.batch, "primary_fraction": args.primary_fraction, "device": args.device,
           "frame_stack": stack, "action_horizon": args.action_horizon, "imu_channels": channels}), flush=True)
    if {path.resolve() for path in args.real + args.sim} & {path.resolve() for path in args.held_out}:
        parser.error("Hold out complete flights from training.")
    held_out = [Clip(path, stack, height, width, args.imu, relative_attitude=args.relative_attitude) for path in args.held_out]
    parameters = [*agent.encoder.parameters(), *agent.actor.parameters()]
    if agent.imu_encoder is not None:
        parameters.extend(agent.imu_encoder.parameters())
    if agent.future_actor is not None:
        parameters.extend(agent.future_actor.parameters())
    if agent.protected_prefix:
        parameters = [*agent.boundary_head.parameters(), *agent.late_actor.parameters(),
                      *agent.late_future_actor.parameters()]
    gate_head = None
    if args.gate_progress_loss:
        if not args.imu:
            parser.error("The gate progress objective requires RGB and IMU.")
        gate_head = torch.nn.Linear(agent.actor[0].in_features, 12).to(args.device)
        if state := (weights or {}).get("training_progress", {}).get("gate_head"):
            gate_head.load_state_dict(state)
        parameters += list(gate_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    args.out.mkdir(parents=True, exist_ok=True)
    best_rank = None
    correction_clips = {}
    last_correction = progress.get("last_correction")
    stage = (progress.get("curriculum_stage") or "finish") if args.completion_marker else None
    if stage == "pace":
        demos.primary_fraction = .80
        demos.focus_gates = None
        demos.secondary_focus_gates = None
    with (args.out / "metrics.jsonl").open("a" if resumed else "w", encoding="utf-8", buffering=1) as log:
        for step in range(1, args.updates + 1) if args.updates else count(1):
            if args.until and args.until.is_file():
                break
            if args.correction_directory and (step - 1) % args.eval_every == 0:
                while not args.until.is_file():
                    complete = (path for path in args.correction_directory.glob("*.npz")
                                if not path.name.endswith(".tmp.npz"))
                    recent = sorted(complete, key=lambda path: path.stat().st_mtime_ns)
                    if recent and recent[-1].name != last_correction:
                        break
                    sleep(1)
                if args.until.is_file():
                    break
                independent = [path for path in recent if not path.stem.endswith(("-assisted", "-handoff"))][-8:]
                assisted = [path for path in recent if path.stem.endswith("-assisted")][-8:]
                handoff = [path for path in recent if path.stem.endswith("-handoff")][-8:]
                correction_clips = {
                    path: correction_clips.get(path) or Clip(
                        path, stack, height, width, with_imu=True, relative_attitude=args.relative_attitude,
                        preserve_success=args.preserve_successful_prefixes and stage == "finish" and path in independent,
                        preserve_handoff=args.preserve_successful_prefixes and stage == "finish" and path in handoff)
                    for path in (*independent, *assisted, *handoff)
                }
                demos.independent = [correction_clips[path] for path in independent]
                demos.assisted = [correction_clips[path] for path in assisted]
                demos.handoff = [correction_clips[path] for path in handoff] if stage == "finish" else []
                last_correction = recent[-1].name
            if (step - 1) % args.eval_every == 0 and stage == "finish" and args.completion_marker.is_file():
                if args.extend_history or args.relative_attitude:
                    qualified = torch.load(args.completion_marker, map_location="cpu", weights_only=True)
                    agent.set_weights(adapt_weights(qualified))
                else:
                    agent.load(args.completion_marker)
                optimizer.state.clear()
                best_rank = None
                demos.primary_fraction = .80
                demos.handoff = []
                demos.focus_gates = None
                demos.secondary_focus_gates = None
                if args.preserve_successful_prefixes and args.correction_directory:
                    correction_clips.clear()
                    demos.independent = [Clip(path, stack, height, width, with_imu=True, relative_attitude=args.relative_attitude) for path in independent]
                stage = "pace"
                print(json.dumps({"curriculum_stage": stage, "primary_fraction": demos.primary_fraction}),
                      flush=True)
            frames, labels, weight, *extra = demos.sample(args.batch)
            imu = extra[0] if args.imu else None
            gate_loss = None
            anchor_loss = None
            boundary_loss = None
            nominal_completion = stage == "finish"
            obs = random_shift(pixel_tensor(frames, args.device).float() / 255. - 0.5,
                               pad=1 if nominal_completion else 3)
            contrast = .06 if nominal_completion else .15
            brightness = .025 if nominal_completion else .07
            scale = torch.empty((len(frames), 1, 1, 1), device=args.device).uniform_(1 - contrast, 1 + contrast)
            offset = torch.empty_like(scale).uniform_(-brightness, brightness)
            obs = (obs * scale + offset).clamp(-.5, .5)
            features = agent.policy_features(obs, imu)
            if gate_head is not None or anchor is not None or agent.protected_prefix:
                next_gate = torch.as_tensor(extra[int(args.imu)], dtype=torch.long, device=args.device)
            if agent.protected_prefix:
                valid = (next_gate >= 0) & (next_gate < 12)
                later = next_gate >= agent.protected_prefix
                logits = agent.boundary_head(features).flatten()
                errors = torch.nn.functional.binary_cross_entropy_with_logits(logits, later.float(), reduction="none")
                boundary_loss = (errors * valid).sum() / valid.sum().clamp(min=1)
                weight = weight * later.cpu().numpy()
            if agent.future_actor is not None:
                actions = agent.action_sequence(features, late=agent.protected_prefix > 0)
                action = actions[:, 0]
            else:
                action = agent.actor(features).tanh()
            bc_loss = stick_loss(action, labels, weight)
            loss = bc_loss + .04 * boundary_loss if boundary_loss is not None else bc_loss
            if agent.future_actor is not None:
                targets = extra[-1].reshape(-1, 4)
                future_loss = stick_loss(actions[:, 1:].reshape(-1, 4), targets,
                                         np.repeat(weight, args.action_horizon - 1))
                loss = loss + .5 * future_loss
            if anchor is not None and stage != "pace":
                prefix = (next_gate >= 0) & (next_gate < args.anchor_gates)
                with torch.no_grad():
                    reference = anchor.actor(anchor.policy_features(obs, imu)).tanh()
                deviation = torch.nn.functional.smooth_l1_loss(action, reference, reduction="none", beta=.12).mean(dim=1)
                anchor_loss = (deviation * prefix).sum() / prefix.sum().clamp(min=1)
                loss = loss + args.anchor_weight * anchor_loss
            if gate_head is not None:
                valid = next_gate >= 0
                target = torch.where(valid, next_gate.remainder(12), -100)
                per_sample = torch.nn.functional.cross_entropy(gate_head(features), target, reduction="none")
                gate_loss = per_sample.sum() / valid.sum().clamp(min=1)
                loss = loss + args.gate_progress_loss * gate_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.)
            optimizer.step()
            if step % args.eval_every and step != args.updates:
                continue
            real_error = held_out_mae(agent, held_out, args.batch)
            metrics = {}
            rank = (-float(np.mean(real_error)) if real_error is not None else -float(bc_loss.detach()),)
            round_no = first_round + (step + args.eval_every - 1) // args.eval_every
            updates = first_step + step
            steps = first_samples + step * args.batch if args.until else step
            metrics.update(update=updates, steps=steps, round=round_no,
                           bc_loss=float(bc_loss.detach()), real_holdout_mae=real_error, offline=True)
            if boundary_loss is not None:
                metrics["route_boundary_loss"] = float(boundary_loss.detach())
            if anchor_loss is not None:
                metrics["prefix_anchor_loss"] = float(anchor_loss.detach())
            if agent.future_actor is not None:
                metrics["future_action_loss"] = float(future_loss.detach())
            if gate_loss is not None:
                metrics["gate_progress_loss"] = float(gate_loss.detach())
            if args.correction_directory:
                metrics.update(rollout_corrections=len(correction_clips), latest_correction=last_correction,
                               correction_frames=sum(map(len, correction_clips.values())),
                               independent_correction_clips=len(demos.independent),
                               assisted_correction_clips=len(demos.assisted),
                               handoff_correction_clips=len(demos.handoff),
                               preserved_successful_frames=sum(clip.preserved_rows for clip in demos.independent),
                               preserved_handoff_frames=sum(clip.preserved_rows for clip in demos.handoff))
            if demos.focus_gates:
                metrics.update(focus_gates=demos.focus_gates, focused_samples=demos.focus_samples)
            if demos.secondary_focus_gates:
                metrics.update(secondary_focus_gates=demos.secondary_focus_gates,
                               secondary_focused_samples=demos.secondary_focus_samples)
            if stage:
                metrics.update(curriculum_stage=stage, primary_fraction=demos.primary_fraction)
            print(json.dumps(metrics), flush=True)
            log.write(json.dumps(metrics) + "\n")
            agent.save(latest, round=round_no, steps=steps, updates=updates,
                       last_correction=last_correction, curriculum_stage=stage,
                       **({"gate_head": gate_head.state_dict()} if gate_head is not None else {}))
            if best_rank is None or rank > best_rank:
                agent.save(args.out / "best.pt")
                best_rank = rank


if __name__ == "__main__":
    main()
