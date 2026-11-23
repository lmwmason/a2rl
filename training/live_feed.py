"""Expose training metrics and completed firmware evaluation replays."""
import argparse
import fcntl
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import re
from shutil import copyfileobj
import time
from urllib.parse import urlsplit

from .curriculum import FINISH_MARKER, PACE_MARKER, PACE_TARGET, SPEED_MARKER, SPEED_RUN, SPEED_TARGET

from .telemetry import finish_time_history, read_metrics


ORIGIN = "https://a2rl-fpv-dashboard.front.cbgpu.com"
MEDIA = re.compile(r"/(?:(?:replay|chase|export-fpv|export-chase)-[0-9]{1,24}\.mp4|controls-[0-9]{1,24}\.json|replay\.json)")


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


def status(run, media):
    speed = (run / PACE_MARKER).is_file()
    active = training_active(run)
    speed_checkpoint = (run / SPEED_RUN / "latest.pt").is_file()
    starting_speed = speed and active and not speed_checkpoint
    waiting_for_speed = speed and not active and not speed_checkpoint
    folder = run / (SPEED_RUN if speed else "recovery")
    metrics = read_metrics(folder)
    latest = metrics[-1] if metrics else {}
    stage = "speed" if speed else "pace" if (run / FINISH_MARKER).is_file() else "finish"
    completed = finish_time_history(metrics, speed)
    results = [row["comparison_s"] for row in completed if row["comparison_s"] is not None]
    individual_best = [row["best_two_lap_s"] for row in metrics[-60:] if row.get("best_two_lap_s") is not None]
    gates = [row["gates_mean"] if "gates_mean" in row else sum(row["gates"])/len(row["gates"])
             for row in metrics[-60:] if "gates_mean" in row or row.get("gates")]
    snapshot = {
        "active": active, "connection": "connected", "snapshot_at": time.time(),
        "live_feed_version": 1,
        "course": "2026-practice", "phase": "SPEED RL" if speed else "IMITATION",
        "curriculum_stage": stage,
        "physics_engine": "elodin",
        "awaiting_speed_trainer": waiting_for_speed,
        "starting_speed_trainer": starting_speed,
        "rollouts": latest.get("rollouts", 0 if starting_speed else None),
        "mean_return": latest.get("mean_return"),
        "reward_version": latest.get("reward_version"),
        "reward_components": latest.get("reward_components"),
        "gates_per_race": 24, "full_course": True,
        "round": latest.get("round"), "steps": latest.get("steps", latest.get("transitions")),
        "finish_rate": latest.get("two_lap_rate"), "collision_rate": latest.get("crash_rate"),
        "recent_gates": gates, "recent_finish_times": completed,
        "best_two_lap_seconds": min(results + individual_best, default=None),
        "mean_gate_crossing_speed_m_s": latest.get("mean_gate_crossing_speed_m_s"),
        "mean_gate_approach_delay_seconds": latest.get("mean_gate_approach_delay_seconds"),
        "target_two_lap_seconds": {"finish": None, "pace": PACE_TARGET, "speed": SPEED_TARGET}[stage],
        "target_beaten": (run / SPEED_RUN / "latest.pt").is_file() and (run / SPEED_MARKER).is_file(),
    }
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
