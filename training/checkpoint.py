"""Atomic resumable checkpoints for the optimizer, RNGs and transition replay."""
from contextlib import contextmanager
from io import BytesIO
import os
from tempfile import NamedTemporaryFile
import zipfile

import numpy as np
import torch

ROW_ARRAYS = ("frames", "actions", "rewards", "flags", "episode_starts")


def layout(replay):
    return (tuple(replay.frames.shape), replay.stack, replay.n_step, replay.gamma)


def write_array(archive, name, array):
    with archive.open(name + ".npy", "w", force_zip64=True) as stream:
        np.save(stream, array, allow_pickle=False)


def read_array(archive, name):
    with archive.open(name + ".npy", "r") as stream:
        return np.load(stream, allow_pickle=False)


@contextmanager
def atomic_archive(path):
    with NamedTemporaryFile(dir=path.parent, prefix=path.name, suffix=".tmp", delete=False) as temp:
        temporary = temp.name
        try:
            with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                yield archive
            temp.flush()
            os.fsync(temp.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)



def save(path, agent, replay, state, last_frames, demonstrations, last_inertial=None):
    payload = {
        "version": 1,
        "agent": agent.weights(),
        "critic_optimizer": agent.critic_opt.state_dict(),
        "actor_optimizer": agent.actor_opt.state_dict(),
        "progress": state,
        "layout": layout(replay),
        "replay_count": replay.count,
        "numpy_rng": replay.rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if agent.device == "cuda" else [],
        "demo_rng": demonstrations.rng.bit_generator.state if demonstrations is not None else None,
    }
    occupied = min(replay.count, replay.rows)
    keys = list(replay.truncated_frames)
    with atomic_archive(path) as archive:
        weights = BytesIO()
        torch.save(payload, weights)
        archive.writestr("training.pt", weights.getbuffer())
        for name in ROW_ARRAYS:
            write_array(archive, name, getattr(replay, name)[:occupied])
        write_array(archive, "next_start", replay.next_start)
        write_array(archive, "last_frames", last_frames)
        if replay.inertial is not None:
            write_array(archive, "inertial", replay.inertial[:occupied])
            write_array(archive, "last_inertial", last_inertial)
            if keys:
                write_array(archive, "truncated_inertial", np.stack([replay.truncated_inertial[key] for key in keys]))
        write_array(archive, "truncated_keys", np.array(keys, np.int64).reshape(-1, 2))
        if keys:
            write_array(archive, "truncated_frames", np.stack([replay.truncated_frames[key] for key in keys]))


def load(path, agent, replay, demonstrations):
    with zipfile.ZipFile(path) as archive:
        payload = torch.load(BytesIO(archive.read("training.pt")), map_location="cpu", weights_only=True)
        if payload["version"] != 1 or payload["layout"] != layout(replay):
            raise ValueError("Checkpoint format or replay configuration does not match.")
        if (payload["demo_rng"] is None) != (demonstrations is None):
            raise ValueError("Resume with the original demonstration configuration.")
        replay.count = payload["replay_count"]
        occupied = min(replay.count, replay.rows)
        for name in ROW_ARRAYS:
            getattr(replay, name)[:occupied] = read_array(archive, name)
        replay.next_start[:] = read_array(archive, "next_start")
        keys = read_array(archive, "truncated_keys")
        if len(keys):
            values = read_array(archive, "truncated_frames")
            replay.truncated_frames = {tuple(key): frame for key, frame in zip(keys, values)}
        # Native simulator instances start fresh. Preserve the old final image for time-limit
        # bootstrapping and prevent the image stack from crossing the restart.
        with_imu = "inertial.npy" in archive.namelist()
        if with_imu != (replay.inertial is not None):
            raise ValueError("Resume with the original replay sensor layout.")
        last_inertial = None
        if with_imu:
            replay.inertial[:occupied] = read_array(archive, "inertial")
            last_inertial = read_array(archive, "last_inertial")
            if len(keys):
                samples = read_array(archive, "truncated_inertial")
                replay.truncated_inertial = dict(zip(map(tuple, keys), samples))
        replay.truncate_open(read_array(archive, "last_frames"), last_inertial)
    progress = payload["progress"]
    if progress["warm_started"]:
        agent.configure_warm_start()
    agent.set_weights(payload["agent"])
    agent.critic_opt.load_state_dict(payload["critic_optimizer"])
    agent.actor_opt.load_state_dict(payload["actor_optimizer"])
    replay.rng.bit_generator.state = payload["numpy_rng"]
    if demonstrations is not None:
        demonstrations.rng.bit_generator.state = payload["demo_rng"]
    torch.set_rng_state(payload["torch_rng"])
    if agent.device == "cuda" and payload["cuda_rng"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return progress
