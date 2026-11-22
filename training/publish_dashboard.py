"""Publish summaries and rendered policy replays from the training container."""
import argparse
import base64
from contextlib import nullcontext
import hashlib
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Full, Queue
import re
from shutil import which
import subprocess
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory
from threading import Thread
import time

import torch

from . import telemetry
from .sim import load_sim
from .web_watch import render_video, write_json


class ReadOnlyFeed(SimpleHTTPRequestHandler):
    def send_head(self):
        if self.path not in ("/status.json", "/replay.json") and not re.fullmatch(
                r"/replay-[0-9]{1,24}\.mp4", self.path):
            self.send_error(404)
            return None
        return super().send_head()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("Vary", "Origin")
        origin = "https://a2rl-fpv-dashboard.front.cbgpu.com"
        if self.headers.get("Origin") == origin:
            self.send_header("Access-Control-Allow-Origin", origin)
        super().end_headers()

    def log_message(self, format, *args):
        pass


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
    command = ["curl", "-4", "--silent", "--show-error", "--fail", "--connect-timeout", "12",
               "--max-time", "50" if meta else "16", "-X", "PUT", "--data-binary", "@-"]
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
            send(args.endpoint, args.key, "status.json", json.dumps(item).encode())
        except (OSError, subprocess.SubprocessError) as error:
            print(f"Status upload failed ({type(error).__name__})", flush=True)


def publish_replays(args, pending, directory):
    lib = load_sim()
    path = None
    replay = None
    versions = sorted(directory.glob("replay-*.mp4"))
    while True:
        try:
            info, weights = pending.get(timeout=15)
        except Empty:
            if replay:
                current = subprocess.run(
                    ["curl", "-4", "-fsS", "--max-time", "15", args.endpoint + "/replay.json"],
                    capture_output=True,
                )
                try:
                    if current.returncode == 0 and json.loads(current.stdout)["version"] == replay["version"]:
                        continue
                except (ValueError, KeyError):
                    pass
        else:
            try:
                candidate = directory / "encoding.mp4"
                result = render_video(lib, weights, candidate, args.ffmpeg)
                replay = info | result | {"url": f"replay-{info['version']}.mp4"}
                path = directory / replay["url"]
                candidate.replace(path)
                if args.serve_port:
                    write_json(directory / "replay.json", replay)
                if path not in versions:
                    versions.append(path)
                if len(versions) > 3:
                    versions.pop(0).unlink(missing_ok=True)
                print(f"Replay ready for round {replay['round']} ({replay['gates']} gates)", flush=True)
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
    parser.add_argument("--serve-port", type=int, help="Also serve the public read-only feed on loopback.")
    parser.add_argument("--cache-dir", type=Path, help="Keep published replays available across publisher restarts.")
    args = parser.parse_args()
    args.endpoint = args.endpoint.rstrip("/")
    if args.ffmpeg is None:
        args.ffmpeg = which("ffmpeg")
        if args.ffmpeg is None:
            import imageio_ffmpeg
            args.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    torch.set_num_threads(2)
    statuses, replays = Queue(maxsize=1), Queue(maxsize=1)
    cache = nullcontext(args.cache_dir) if args.cache_dir else TemporaryDirectory(prefix="a2rl-feed-")
    with cache as folder:
        directory = Path(folder)
        directory.mkdir(parents=True, exist_ok=True)
        if args.serve_port:
            handler = partial(ReadOnlyFeed, directory=directory)
            server = ThreadingHTTPServer(("127.0.0.1", args.serve_port), handler)
            Thread(target=server.serve_forever, daemon=True).start()
        Thread(target=publish_status, args=(args, statuses), daemon=True).start()
        Thread(target=publish_replays, args=(args, replays, directory), daemon=True).start()
        command = [sys.executable, "-u", "-m", "training.telemetry", str(Path.cwd()),
                   args.school, args.speed, "replay"]
        with subprocess.Popen(command, stdout=subprocess.PIPE) as source:
            try:
                while message := telemetry.read_packet(source.stdout):
                    info, payload = message
                    if info["type"] == "meta":
                        info.update(connection="connected", snapshot_at=time.time())
                        if args.serve_port:
                            write_json(directory / "status.json", info)
                        offer(statuses, info)
                    elif info["type"] == "checkpoint":
                        offer(replays, (info, payload))
            finally:
                source.terminate()


if __name__ == "__main__":
    main()
