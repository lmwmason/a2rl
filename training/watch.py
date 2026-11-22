"""Bridge a GPU job over SSH into the native Rust FPV training viewer."""
import argparse
from io import BytesIO
from pathlib import Path
import sys
from threading import Lock, Thread
import time

import torch

from . import telemetry
from .agent import Agent
from .sim import Environment, load_sim
from .train import fresh_stack

VIDEO_WIDTH, VIDEO_HEIGHT = 480, 360


class Source:
    def __init__(self, args):
        self.lock = Lock()
        self.status = {"connection": "Connecting to GPU"}
        self.live = None
        self.checkpoint = None
        self.child = telemetry.connect(args.host, args.remote_root, args.school, args.speed)
        Thread(target=self.read, daemon=True).start()

    def read(self):
        try:
            while message := telemetry.read_packet(self.child.stdout):
                meta, payload = message
                with self.lock:
                    if meta["type"] == "meta":
                        self.status = meta | {"connection": "Connected"}
                    elif meta["type"] == "live":
                        self.live = (time.monotonic(), meta, payload)
                    elif meta["type"] == "checkpoint":
                        self.checkpoint = (meta, payload)
        except (EOFError, ValueError, OSError) as error:
            with self.lock:
                self.status["connection"] = str(error)
        finally:
            result = self.child.wait()
            with self.lock:
                if self.status["connection"] in ("Connected", "Connecting to GPU"):
                    self.status["connection"] = f"SSH disconnected ({result})"
                self.status["active"] = False

    def snapshot(self):
        with self.lock:
            return dict(self.status), self.live, self.checkpoint


def watch(source):
    torch.set_num_threads(2)
    lib = load_sim()
    small = Environment(lib, 96, 72)
    camera = Environment(lib, VIDEO_WIDTH, VIDEO_HEIGHT)
    output = sys.stdout.buffer
    agent = loaded = pending = None
    obs = image = None
    frame = gates = status = 0
    restart = time.monotonic()
    try:
        while True:
            begin = time.monotonic()
            job, live, newest = source.snapshot()
            if live is not None and job.get("active") and begin - live[0] < 8.:
                _, meta, image_bytes = live
                view = "LIVE TRAINING" if begin - live[0] < 1. else "LIVE / OPTIMIZING"
                output.write(telemetry.packet(job | meta | {"view": view}, image_bytes))
            else:
                if newest is not None and newest is not loaded:
                    pending = newest
                if pending is not None and (agent is None or (status and begin >= restart) or frame == 0):
                    metadata, weights = pending
                    params = torch.load(BytesIO(weights), map_location="cpu", weights_only=True)
                    stack = params["encoder"]["0.weight"].shape[1] // 3
                    agent = Agent(stack, "cpu", lib.sim_hover_stick())
                    agent.set_weights(params)
                    loaded, pending = pending, None
                    obs = fresh_stack(small.official_start(), stack)
                    image = camera.official_start()
                    frame = gates = status = 0
                if agent is None:
                    output.write(telemetry.packet(job | {"view": "WAITING FOR A CHECKPOINT"}))
                else:
                    if status and begin >= restart:
                        obs = fresh_stack(small.official_start(), len(obs))
                        image = camera.official_start()
                        frame = gates = status = 0
                    title = "BEST CHECKPOINT REPLAY" if loaded[0]["best"] else "LATEST CHECKPOINT REPLAY"
                    output.write(telemetry.packet(job | {"view": title, "width": VIDEO_WIDTH,
                                                         "height": VIDEO_HEIGHT, "gates": gates,
                                                         "step": frame, "status": status}, image.tobytes()))
                    if not status:
                        command = agent.action(obs[None])[0]
                        rgb, _, status = small.advance(command)
                        image, _, recording_status = camera.advance(command)
                        if recording_status != status:
                            raise RuntimeError("The recording diverged from the canonical evaluation.")
                        obs[:-1] = obs[1:]
                        obs[-1] = rgb
                        frame += 1
                        gates, _ = small.lap_stats()
                        if status:
                            restart = time.monotonic() + 2
            output.flush()
            time.sleep(max(0., 1 / 30 - (time.monotonic() - begin)))
    finally:
        small.close()
        camera.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host")
    parser.add_argument("--remote-root", default="~/a2rl-2025-race-school")
    parser.add_argument("--school", default="runs/racing-school-v2")
    parser.add_argument("--speed", default="runs/speed-rl-v2")
    args = parser.parse_args()
    if args.host.startswith("-"):
        parser.error("SSH host must be an alias or hostname.")
    source = Source(args)
    try:
        watch(source)
    except BrokenPipeError:
        pass
    finally:
        if source.child.poll() is None:
            source.child.terminate()


if __name__ == "__main__":
    main()
