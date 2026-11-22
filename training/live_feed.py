"""Expose live FPV frames, training metrics and completed firmware replays."""
import argparse
import fcntl
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import re
from shutil import copyfileobj
import struct
import time
from urllib.parse import urlsplit
from zlib import compress, crc32

from .telemetry import finish_time_history, read_metrics, read_packet


ORIGIN = "https://a2rl-fpv-dashboard.front.cbgpu.com"
MEDIA = re.compile(r"/(?:replay-[0-9]{1,24}\.mp4|controls-[0-9]{1,24}\.json|replay\.json)")


def training_active(run):
    try:
        with (run / "training.lock").open("rb") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
    except FileNotFoundError:
        pass
    return False


def live_frame(folder):
    path = folder / "live-frame.bin"
    with path.open("rb") as stream:
        frame = path.stat()
        if time.time() - frame.st_mtime > 3.:
            raise FileNotFoundError("No recent live frame")
        result = read_packet(stream)
        if result is None:
            raise ValueError("Empty frame packet")
    return result


def encode_png(metadata, pixels):
    width, height = metadata["width"], metadata["height"]
    stride = width * 3
    if not 0 < width <= 640 or not 0 < height <= 480 or len(pixels) != height * stride:
        raise ValueError("Malformed RGB frame")

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", crc32(body))

    scanlines = b"".join(b"\x00" + pixels[offset:offset + stride] for offset in range(0, len(pixels), stride))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">2I5B", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", compress(scanlines, 2)) + chunk(b"IEND", b""))


def phase_folder(run):
    speed = (run / "imitation-qualified.pt").is_file()
    return speed, run / ("speed" if speed else "recovery")


def status(run, media):
    speed, folder = phase_folder(run)
    metrics = read_metrics(folder)
    latest = metrics[-1] if metrics else {}
    completed = finish_time_history(metrics, speed)
    results = [row["comparison_s"] for row in completed if row["comparison_s"] is not None]
    gates = [row["gates_mean"] if "gates_mean" in row else sum(row["gates"])/len(row["gates"])
             for row in metrics[-60:] if "gates_mean" in row or row.get("gates")]
    snapshot = {
        "active": training_active(run), "connection": "connected", "snapshot_at": time.time(),
        "live_feed_version": 1,
        "course": "2026-practice", "phase": "SPEED RL" if speed else "IMITATION",
        "physics_engine": "rust", "gates_per_race": 24, "full_course": True,
        "round": latest.get("round"), "steps": latest.get("steps", latest.get("transitions")),
        "finish_rate": latest.get("two_lap_rate"), "collision_rate": latest.get("crash_rate"),
        "recent_gates": gates, "recent_finish_times": completed,
        "best_two_lap_seconds": min(results, default=None),
        "mean_gate_crossing_speed_m_s": latest.get("mean_gate_crossing_speed_m_s"),
        "mean_gate_approach_delay_seconds": latest.get("mean_gate_approach_delay_seconds"),
        "target_two_lap_seconds": 12 if speed else 30,
        "target_beaten": (run / "speed-qualified.pt").is_file(),
    }
    try:
        frame, _ = live_frame(folder)
        snapshot.update(live_url="/live.png", live_driver=frame["driver"], live_gates=frame["gates"])
    except (OSError, ValueError):
        pass
    try:
        evaluation = json.loads((media / "replay.json").read_text())
        snapshot["firmware_result_seconds"] = evaluation["official_seconds"]
    except (OSError, ValueError, KeyError):
        pass
    return snapshot


class Feed(BaseHTTPRequestHandler):
    def __init__(self, *args, run, media, **kwargs):
        self.run, self.media = run, media
        super().__init__(*args, **kwargs)

    def do_GET(self):
        path_info = urlsplit(self.path).path
        try:
            if path_info == "/status.json":
                body = json.dumps(status(self.run, self.media)).encode()
                self.send("application/json", BytesIO(body), len(body))
            elif path_info == "/live.png":
                metadata, pixels = live_frame(phase_folder(self.run)[1])
                body = encode_png(metadata, pixels)
                self.send("image/png", BytesIO(body), len(body))
            elif MEDIA.fullmatch(path_info):
                path = self.media / path_info[1:]
                kind = "video/mp4" if path.suffix == ".mp4" else "application/json"
                with path.open("rb") as stream:
                    self.send(kind, stream, path.stat().st_size)
            else:
                self.send_error(404)
        except (FileNotFoundError, ValueError):
            self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send(self, kind, stream, length):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", ORIGIN)
        self.end_headers()
        copyfileobj(stream, self.wfile)

    def log_message(self, *_args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), partial(Feed, run=args.run, media=args.media)) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
