"""DrQ-v2-style continuous actor critic for stacks of RGB frames."""
from copy import deepcopy
from io import BytesIO
from math import atanh

import torch
from torch import nn
from torch.nn import functional as F



def pixel_tensor(frames, device):
    # Stack shares one spatial crop so augmentation cannot create fake optical flow.
    image = torch.from_numpy(frames).to(device)
    batch, time, height, width, channels = image.shape
    return image.permute(0, 1, 4, 2, 3).reshape(batch, time * channels, height, width)


def random_shift(image, pad=4):
    image = F.pad(image, (pad,) * 4, mode="replicate")
    batch, channels, padded_height, padded_width = image.shape
    height, width = padded_height - 2 * pad, padded_width - 2 * pad
    device = image.device
    x = torch.arange(width, device=device).view(1, 1, width)
    y = torch.arange(height, device=device).view(1, height, 1)
    offsets = torch.randint(2 * pad + 1, (batch, 2, 1, 1), device=device)
    x = x + offsets[:, 0]
    y = y + offsets[:, 1]
    index = (y * padded_width + x).flatten(1)
    return image.flatten(2).gather(2, index[:, None].expand(-1, channels, -1)).reshape(batch, channels, height, width)


def stick_loss(prediction, labels, confidence):
    target = torch.from_numpy(labels).to(prediction.device)
    weights = torch.from_numpy(confidence).to(prediction.device)
    axis_weights = prediction.new_tensor([1., 2., 1.4, 1.4])
    errors = F.smooth_l1_loss(prediction, target, reduction="none", beta=.12)
    return (weights * (errors * axis_weights).mean(dim=1)).sum() / weights.sum()


def mlp(in_features, out_features):
    return nn.Sequential(nn.Linear(in_features, 256), nn.ReLU(), nn.Linear(256, 256),
                         nn.ReLU(), nn.Linear(256, out_features))


class Encoder(nn.Sequential):
    def __init__(self, stack):
        super().__init__(
            nn.Conv2d(stack * 3, 32, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(32, 48, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(48, 64, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 6)), nn.Flatten(), nn.Linear(64 * 5 * 6, 128),
            nn.LayerNorm(128), nn.Tanh(),
        )


class MobileNetEncoder(nn.Module):
    """A pretrained per-frame visual stem with its spatial grid and frame order preserved."""

    def __init__(self, stack, pretrained=False):
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.visual = mobilenet_v3_small(weights=weights).features[:7].eval()
        self.readout = nn.Sequential(nn.AdaptiveAvgPool2d((3, 4)), nn.Flatten())
        frames = min(stack, 4)
        self.projection = nn.Sequential(nn.Linear(frames * 40 * 3 * 4, 128), nn.LayerNorm(128), nn.Tanh())
        self.register_buffer("indexes", torch.linspace(0, stack - 1, frames).round().long(), persistent=False)
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None], persistent=False)
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None], persistent=False)

    def forward(self, image):
        batch, _, height, width = image.shape
        frames = image.reshape(batch, -1, 3, height, width).index_select(1, self.indexes)
        pixels = (frames.reshape(-1, 3, height, width) + .5 - self.mean) / self.std
        features = self.readout(self.visual(pixels)).reshape(batch, -1)
        return self.projection(features)


class Critic(nn.Module):
    def __init__(self, features=128):
        super().__init__()
        self.q1, self.q2 = mlp(features + 4, 1), mlp(features + 4, 1)

    def forward(self, features, action):
        value = torch.cat((features, action), dim=-1)
        return self.q1(value).flatten(), self.q2(value).flatten()


class Agent:
    def __init__(self, stack, device, hover_stick, with_imu=False, backbone="cnn",
                 pretrained=False, action_horizon=1):
        self.device = device
        self.stack = stack
        self.backbone = backbone
        if backbone == "mobilenet-v3-small":
            encoder = MobileNetEncoder(stack, pretrained)
        elif backbone == "cnn":
            encoder = Encoder(stack)
        else:
            raise ValueError(f"Unknown vision backbone: {backbone}")
        self.encoder = encoder.to(device, memory_format=torch.channels_last)
        self.imu_encoder = (nn.Sequential(nn.Flatten(), nn.Linear(stack * 6, 64), nn.ReLU(),
                                          nn.Linear(64, 32), nn.LayerNorm(32)).to(device)
                            if with_imu else None)
        self.actor = mlp(160 if with_imu else 128, 4).to(device)
        with torch.no_grad():
            self.actor[-1].bias[1] = atanh(hover_stick)
        self.critic = Critic(160 if with_imu else 128).to(device)
        self.target = deepcopy(self.critic).requires_grad_(False)
        self.critic_opt = torch.optim.Adam([*self.critic.parameters(), *self.observation_parameters()], lr=1e-4)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.reference_actor = None
        self.action_horizon = action_horizon
        self.future_actor = None
        if action_horizon > 1:
            self.future_actor = nn.Linear(self.actor[-1].in_features, (action_horizon - 1) * 4).to(device)
            nn.init.zeros_(self.future_actor.weight)
            nn.init.zeros_(self.future_actor.bias)

    def observation_parameters(self):
        parameters = list(self.encoder.parameters())
        if self.imu_encoder is not None:
            parameters.extend(self.imu_encoder.parameters())
        return parameters

    def policy_features(self, pixels, imu=None):
        image = self.encoder(pixels)
        if self.imu_encoder is None:
            return image
        if imu is None:
            raise ValueError("This policy requires synchronized gyro and acceleration history.")
        sensor = torch.as_tensor(imu, dtype=torch.float32, device=self.device)
        scale = sensor.new_tensor([10., 10., 10., 30., 30., 30.])
        return torch.cat((image, self.imu_encoder((sensor / scale).clamp(-4., 4.))), dim=1)

    @torch.no_grad()
    def action(self, frames, noise=0., *, reference=False, imu=None):
        obs = pixel_tensor(frames, self.device).float() / 255. - 0.5
        actor = self.reference_actor if reference else self.actor
        action = actor(self.policy_features(obs, imu)).tanh()
        if noise:
            action = (action + torch.randn_like(action) * noise).clamp(-1., 1.)
        return action.cpu().numpy()

    def action_sequence(self, features):
        hidden = self.actor[:-1](features)
        current = self.actor[-1](hidden).unsqueeze(1)
        if self.future_actor is None:
            return current.tanh()
        future = current + self.future_actor(hidden).reshape(-1, self.action_horizon - 1, 4)
        return torch.cat((current, future), dim=1).tanh()

    @torch.no_grad()
    def forecast(self, frames, imu):
        obs = pixel_tensor(frames, self.device).float() / 255. - .5
        return self.action_sequence(self.policy_features(obs, imu)).cpu().numpy()

    def update(self, batch, noise, *, update_actor=True, demonstration=None, imitation_weight=0.):
        obs, action, reward, discount, next_obs, *inertial = batch
        imu, next_imu = inertial if inertial else (None, None)
        clean_obs = pixel_tensor(obs, self.device).float() / 255. - 0.5
        obs = random_shift(clean_obs)
        next_obs = random_shift(pixel_tensor(next_obs, self.device).float() / 255. - 0.5)
        action, reward, discount = [torch.from_numpy(x).to(self.device) for x in (action, reward, discount)]
        with torch.no_grad():
            future = self.policy_features(next_obs, next_imu)
            target_action = self.actor(future).tanh()
            perturb = (torch.randn_like(target_action) * noise).clamp(-0.3, 0.3)
            target_action = (target_action + perturb).clamp(-1., 1.)
            target = reward * 0.1 + discount * torch.minimum(*self.target(future, target_action))

        features = self.policy_features(obs, imu)
        if not update_actor:
            features = features.detach()
        q1, q2 = self.critic(features, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_([*self.observation_parameters(), *self.critic.parameters()], 10.)
        self.critic_opt.step()

        actor_loss = critic_loss.new_zeros(())
        if update_actor:
            self.critic.requires_grad_(False)
            actor_action = self.actor(features.detach()).tanh()
            actor_loss = -torch.minimum(*self.critic(features.detach(), actor_action)).mean()
            if self.reference_actor is not None:
                with torch.no_grad():
                    clean_features = self.policy_features(clean_obs, imu)
                    reference_action = self.reference_actor(clean_features).tanh()
                actor_loss += 40. * F.mse_loss(self.actor(clean_features).tanh(), reference_action)
            if demonstration is not None and imitation_weight > 0:
                frames, labels, weights, *demo_imu = demonstration
                with torch.no_grad():
                    demo_image = pixel_tensor(frames, self.device).float() / 255. - .5
                    demo_features = self.policy_features(random_shift(demo_image, pad=3), demo_imu[0] if demo_imu else None)
                actor_loss += imitation_weight * stick_loss(self.actor(demo_features).tanh(), labels, weights)
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            self.critic.requires_grad_(True)
        with torch.no_grad():
            for destination, source in zip(self.target.parameters(), self.critic.parameters()):
                destination.lerp_(source, 0.01)
        return critic_loss.detach(), actor_loss.detach()

    def weights(self):
        params = {"action_layout": "mode2-acro-v1", "encoder": self.encoder.state_dict(),
                  "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                  "target": self.target.state_dict()}
        if self.imu_encoder is not None:
            params["imu_encoder"] = self.imu_encoder.state_dict()
        if self.future_actor is not None:
            params["future_actor"] = self.future_actor.state_dict()
            params["action_horizon"] = self.action_horizon
        if self.backbone != "cnn":
            params["vision_backbone"] = self.backbone
            params["frame_stack"] = self.stack
        return params

    def set_weights(self, params):
        if params.get("action_layout") != "mode2-acro-v1":
            raise ValueError("Checkpoint action layout does not match Mode 2 Acro.")
        if params.get("vision_backbone", "cnn") != self.backbone:
            raise ValueError("Checkpoint vision backbone does not match this policy.")
        self.encoder.load_state_dict(params["encoder"])
        if ("imu_encoder" in params) != (self.imu_encoder is not None):
            raise ValueError("Checkpoint sensor layout does not match this policy.")
        if self.imu_encoder is not None:
            self.imu_encoder.load_state_dict(params["imu_encoder"])
        self.actor.load_state_dict(params["actor"])
        if "future_actor" in params:
            if self.future_actor is None or params.get("action_horizon") != self.action_horizon:
                raise ValueError("Checkpoint prediction horizon does not match this policy.")
            self.future_actor.load_state_dict(params["future_actor"])
        for model, key in ((self.critic, "critic"), (self.target, "target")):
            state = params[key]
            if self.imu_encoder is not None and state["q1.0.weight"].shape[1] == 132:
                # Older imitation checkpoints saved unused RGB-only critics.
                # Preserve the feature and action columns and insert IMU columns.
                state = dict(state)
                for name in ("q1.0.weight", "q2.0.weight"):
                    weight = state[name]
                    state[name] = torch.cat((weight[:, :128], weight.new_zeros((weight.shape[0], 32)),
                                             weight[:, 128:]), dim=1)
            model.load_state_dict(state)

    def save(self, path, **training_progress):
        temp = path.with_suffix(".tmp")
        params = self.weights()
        if training_progress:
            params["training_progress"] = training_progress
        torch.save(params, temp)
        temp.replace(path)

    def load(self, path):
        self.set_weights(torch.load(path, map_location=self.device, weights_only=True))

    def configure_warm_start(self):
        self.critic_opt = torch.optim.Adam([
            {"params": self.critic.parameters(), "lr": 1e-4},
            {"params": self.observation_parameters(), "lr": 1e-5},
        ])

    def protect_qualified_policy(self, path):
        params = torch.load(path, map_location=self.device, weights_only=True)
        self.reference_actor = deepcopy(self.actor).requires_grad_(False)
        self.reference_actor.load_state_dict(params["actor"])
        self.encoder.load_state_dict(params["encoder"])
        self.encoder.requires_grad_(False)
        if self.imu_encoder is not None:
            self.imu_encoder.load_state_dict(params["imu_encoder"])
            self.imu_encoder.requires_grad_(False)
        self.actor_opt.param_groups[0]["lr"] = 1e-5

    def restore_actor(self, path):
        params = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(params["actor"])
        self.actor_opt.state.clear()


def checkpoint_stack(params):
    if "frame_stack" in params:
        return int(params["frame_stack"])
    return params["encoder"]["0.weight"].shape[1] // 3


def extend_cnn_history(params, stack):
    previous = checkpoint_stack(params)
    if stack == previous:
        return params
    if stack < previous or params.get("vision_backbone", "cnn") != "cnn":
        raise ValueError("History extension requires an RGB CNN checkpoint with fewer frames.")
    extra = stack - previous
    expanded = {**params, "frame_stack": stack}
    rgb = params["encoder"]["0.weight"]
    expanded["encoder"] = {**params["encoder"], "0.weight": F.pad(rgb, (0, 0, 0, 0, 3 * extra, 0))}
    if "imu_encoder" in params:
        imu = params["imu_encoder"]["1.weight"]
        expanded["imu_encoder"] = {**params["imu_encoder"], "1.weight": F.pad(imu, (6 * extra, 0))}
    return expanded


def load_policy(weights):
    params = torch.load(BytesIO(weights), map_location="cpu", weights_only=True)
    stack = checkpoint_stack(params)
    agent = Agent(stack, "cpu", 0., with_imu="imu_encoder" in params,
                  backbone=params.get("vision_backbone", "cnn"),
                  action_horizon=params.get("action_horizon", 1))
    agent.set_weights(params)
    return agent
