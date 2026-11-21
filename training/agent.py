"""DrQ-v2-style continuous actor critic for stacks of RGB frames."""
from copy import deepcopy
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


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1, self.q2 = mlp(132, 1), mlp(132, 1)

    def forward(self, features, action):
        value = torch.cat((features, action), dim=-1)
        return self.q1(value).flatten(), self.q2(value).flatten()


class Agent:
    def __init__(self, stack, device, hover_stick):
        self.device = device
        self.encoder = Encoder(stack).to(device, memory_format=torch.channels_last)
        self.actor = mlp(128, 4).to(device)
        with torch.no_grad():
            self.actor[-1].bias[1] = atanh(hover_stick)
        self.critic = Critic().to(device)
        self.target = deepcopy(self.critic).requires_grad_(False)
        self.critic_opt = torch.optim.Adam([*self.critic.parameters(), *self.encoder.parameters()], lr=1e-4)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

    @torch.no_grad()
    def action(self, frames, noise=0.):
        obs = pixel_tensor(frames, self.device).float() / 255. - 0.5
        action = self.actor(self.encoder(obs)).tanh()
        if noise:
            action = (action + torch.randn_like(action) * noise).clamp(-1., 1.)
        return action.cpu().numpy()

    def update(self, batch, noise, *, update_actor=True, demonstration=None, imitation_weight=0.):
        obs, action, reward, discount, next_obs = batch
        obs = random_shift(pixel_tensor(obs, self.device).float() / 255. - 0.5)
        next_obs = random_shift(pixel_tensor(next_obs, self.device).float() / 255. - 0.5)
        action, reward, discount = [torch.from_numpy(x).to(self.device) for x in (action, reward, discount)]
        with torch.no_grad():
            future = self.encoder(next_obs)
            target_action = self.actor(future).tanh()
            perturb = (torch.randn_like(target_action) * noise).clamp(-0.3, 0.3)
            target_action = (target_action + perturb).clamp(-1., 1.)
            target = reward * 0.1 + discount * torch.minimum(*self.target(future, target_action))

        features = self.encoder(obs)
        if not update_actor:
            features = features.detach()
        q1, q2 = self.critic(features, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_([*self.encoder.parameters(), *self.critic.parameters()], 10.)
        self.critic_opt.step()

        actor_loss = critic_loss.new_zeros(())
        if update_actor:
            self.critic.requires_grad_(False)
            actor_action = self.actor(features.detach()).tanh()
            actor_loss = -torch.minimum(*self.critic(features.detach(), actor_action)).mean()
            if demonstration is not None and imitation_weight > 0:
                frames, labels, weights = demonstration
                with torch.no_grad():
                    demo_image = pixel_tensor(frames, self.device).float() / 255. - .5
                    demo_features = self.encoder(random_shift(demo_image, pad=3))
                actor_loss += imitation_weight * stick_loss(self.actor(demo_features).tanh(), labels, weights)
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            self.critic.requires_grad_(True)
        with torch.no_grad():
            for destination, source in zip(self.target.parameters(), self.critic.parameters()):
                destination.lerp_(source, 0.01)
        return critic_loss.detach(), actor_loss.detach()

    def save(self, path):
        temp = path.with_suffix(".tmp")
        torch.save({"action_layout": "mode2-acro-v1", "encoder": self.encoder.state_dict(), "actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(), "target": self.target.state_dict()}, temp)
        temp.replace(path)

    def load(self, path):
        params = torch.load(path, map_location=self.device, weights_only=True)
        if params.get("action_layout") != "mode2-acro-v1":
            raise ValueError("Checkpoint action layout does not match Mode 2 Acro.")
        self.encoder.load_state_dict(params["encoder"])
        self.actor.load_state_dict(params["actor"])
        self.critic.load_state_dict(params["critic"])
        self.target.load_state_dict(params["target"])
