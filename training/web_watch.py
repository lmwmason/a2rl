"""Serve the static FPV dashboard and render each new policy checkpoint locally."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from queue import Empty, Full, Queue
from shutil import copyfile, which
import subprocess
from tempfile import TemporaryDirectory
from threading import Thread
import time

import torch

from . import telemetry
from .sim import Environment, load_sim
from .train import fresh_stack
from .watch import VIDEO_HEIGHT, VIDEO_WIDTH, replay_agent, replay_step

class DashboardHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        if self.headers.get("Host", "").partition(":")[0] not in ("127.0.0.1", "localhost"):
            self.send_error(400, "Use a loopback address")
            return None
        return super().send_head()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format, *args):
        if self.path not in ("/status.json", "/replay.json"):
            super().log_message(format, *args)


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    temporary.replace(path)


def mirror(args, path, pending):
    job = {}
    latest = None
    while True:
        write_json(path, job | {"connection": "connecting"})
        child = telemetry.connect(args.host, args.remote_root, args.school, args.speed, mode="replay")
        try:
            while message := telemetry.read_packet(child.stdout):
                info, payload = message
                if info["type"] == "meta":
                    job = info
                    write_json(path, job | {"connection": "connected", "snapshot_at": time.time()})
                elif info["type"] == "checkpoint" and info["version"].isdecimal() and info["version"] != latest:
                    latest = info["version"]
                    try:
                        pending.put_nowait((info, payload))
                    except Full:
                        try:
                            pending.get_nowait()
                        except Empty:
                            pass
                        pending.put_nowait((info, payload))
        except (EOFError, ValueError, OSError) as error:
            print(f"Telemetry connection lost: {error}", flush=True)
        finally:
            if child.poll() is None:
                child.terminate()
            child.wait()
            child.stdout.close()
        write_json(path, job | {"connection": "disconnected", "active": False})
        time.sleep(3)


def render_video(lib, weights, path, ffmpeg="ffmpeg"):
    agent, stack = replay_agent(weights, lib)
    small = Environment(lib, 96, 72)
    camera = Environment(lib, VIDEO_WIDTH, VIDEO_HEIGHT)
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo",
               "-pixel_format", "rgb24", "-video_size", f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}",
               "-framerate", "30", "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2",
               "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    try:
        obs = fresh_stack(small.official_start(), stack)
        image = camera.official_start()
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            try:
                encoder.stdin.write(image.tobytes())
                status = 0
                while not status:
                    image, status = replay_step(agent, obs, small, camera)
                    encoder.stdin.write(image.tobytes())
                for _ in range(18):
                    encoder.stdin.write(image.tobytes())
            finally:
                encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError("FFmpeg could not encode the replay.")
        return {"gates": small.lap_stats()[0], "status": status}
    finally:
        small.close()
        camera.close()


def render_checkpoints(root, pending):
    lib = load_sim()
    published = []
    temporary = root / "encoding.mp4"
    while True:
        info, weights = pending.get()
        path = root / f"replay-{info['version']}.mp4"
        try:
            result = render_video(lib, weights, temporary)
            temporary.replace(path)
            write_json(root / "replay.json", info | result | {"url": path.name})
            published.append(path)
            if len(published) > 3:
                published.pop(0).unlink(missing_ok=True)
            print(f"Replay ready {path.name} ({result['gates']} gates)", flush=True)
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            temporary.unlink(missing_ok=True)
            print(f"Replay could not be rendered: {error}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host")
    parser.add_argument("--remote-root", default="~/a2rl-2025-race-school")
    parser.add_argument("--school", default="runs/racing-school-v2")
    parser.add_argument("--speed", default="runs/speed-rl-v2")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host.startswith("-"):
        parser.error("SSH host must be an alias or hostname.")
    if which("ffmpeg") is None:
        parser.error("FFmpeg is required to encode checkpoint replays.")
    torch.set_num_threads(2)
    assets = Path(__file__).resolve().parents[1] / "web"
    with TemporaryDirectory(prefix="a2rl-dashboard-") as directory:
        root = Path(directory)
        for name in ("index.html", "style.css", "dashboard.js"):
            copyfile(assets / name, root / name)
        handler = partial(DashboardHandler, directory=root)
        with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
            pending = Queue(maxsize=1)
            Thread(target=mirror, args=(args, root / "status.json", pending), daemon=True).start()
            Thread(target=render_checkpoints, args=(root, pending), daemon=True).start()
            print(f"Local dashboard http://127.0.0.1:{server.server_port}/", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
