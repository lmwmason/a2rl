"""Publish summaries and rendered policy replays from the training container."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
from queue import Empty, Full, Queue
from shutil import which
import subprocess
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory
from threading import Thread
import time

import torch

from . import telemetry
from .sim import load_sim
from .web_watch import render_video


def send(endpoint, key, name, body, meta=None):
    timestamp = str(time.time_ns() // 1_000_000)
    digest = hashlib.sha256(body).hexdigest()
    encoded_meta = base64.b64encode(json.dumps(meta).encode()).decode() if meta else ""
    message = "\n".join((name, timestamp, digest, encoded_meta)).encode()
    with NamedTemporaryFile() as signature_input:
        signature_input.write(message)
        signature_input.flush()
        signature = subprocess.run(
            ["openssl", "pkeyutl", "-sign", "-rawin", "-inkey", key, "-in", signature_input.name],
            check=True, stdout=subprocess.PIPE,
        ).stdout
    headers = {
        "X-A2RL-Time": timestamp, "X-A2RL-Sha256": digest,
        "X-A2RL-Signature": base64.b64encode(signature).decode(), "X-A2RL-Meta": encoded_meta,
    }
    command = ["curl", "--silent", "--show-error", "--fail", "--connect-timeout", "5",
               "--max-time", "40" if meta else "8", "-X", "PUT", "--data-binary", "@-"]
    for key, value in headers.items():
        command.extend(("-H", f"{key}: {value}"))
    subprocess.run([*command, endpoint + "/" + name], input=body, check=True)


def offer(queue, value):
    try:
        queue.put_nowait(value)
    except Full:
        try:
            queue.get_nowait()
        except Empty:
            pass
        queue.put_nowait(value)


def publish_status(args, pending):
    while True:
        item = pending.get()
        try:
            item.update(connection="connected", snapshot_at=time.time())
            send(args.endpoint, args.key, "status.json", json.dumps(item).encode())
        except (OSError, subprocess.SubprocessError) as error:
            print(f"Status upload failed ({type(error).__name__})", flush=True)


def publish_replays(args, pending, directory):
    lib = load_sim()
    path = directory / "checkpoint.mp4"
    replay = None
    while True:
        try:
            info, weights = pending.get(timeout=15)
        except Empty:
            if replay:
                current = subprocess.run(
                    ["curl", "-fsS", "--max-time", "8", args.endpoint + "/replay.json"],
                    capture_output=True,
                )
                try:
                    if current.returncode == 0 and json.loads(current.stdout)["version"] == replay["version"]:
                        continue
                except (ValueError, KeyError):
                    pass
        else:
            try:
                candidate = path.with_suffix(".tmp.mp4")
                result = render_video(lib, weights, candidate, args.ffmpeg)
                candidate.replace(path)
                replay = info | result | {"url": f"replay-{info['version']}.mp4"}
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                print(f"Replay render failed: {error}", flush=True)
                continue
        if replay:
            try:
                send(args.endpoint, args.key, replay["url"], path.read_bytes(), replay)
                print(f"Published {replay['url']} with {replay['gates']} gates", flush=True)
            except (OSError, subprocess.SubprocessError) as error:
                print(f"Replay upload failed ({type(error).__name__})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("endpoint")
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--school", default="runs/racing-school-v2")
    parser.add_argument("--speed", default="runs/speed-rl-v2")
    parser.add_argument("--ffmpeg")
    args = parser.parse_args()
    args.endpoint = args.endpoint.rstrip("/")
    if args.ffmpeg is None:
        args.ffmpeg = which("ffmpeg")
        if args.ffmpeg is None:
            import imageio_ffmpeg
            args.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    torch.set_num_threads(2)
    statuses, replays = Queue(maxsize=1), Queue(maxsize=1)
    with TemporaryDirectory(prefix="a2rl-feed-") as folder:
        Thread(target=publish_status, args=(args, statuses), daemon=True).start()
        Thread(target=publish_replays, args=(args, replays, Path(folder)), daemon=True).start()
        command = [sys.executable, "-u", "-m", "training.telemetry", str(Path.cwd()),
                   args.school, args.speed, "replay"]
        with subprocess.Popen(command, stdout=subprocess.PIPE) as source:
            try:
                while message := telemetry.read_packet(source.stdout):
                    info, payload = message
                    if info["type"] == "meta":
                        offer(statuses, info)
                    elif info["type"] == "checkpoint":
                        offer(replays, (info, payload))
            finally:
                source.terminate()


if __name__ == "__main__":
    main()
