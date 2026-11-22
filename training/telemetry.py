"""Small binary telemetry packets and an optional nonblocking training frame publisher."""
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import time

HEADER = struct.Struct("<II")


def connect(host, root, school, speed, mode="checkpoints"):
    source = Path(__file__).read_text()
    remote = shlex.join(["python3", "-u", "-c", source, root, school, speed, mode])
    return subprocess.Popen(
        ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
         "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2", host, remote],
        stdout=subprocess.PIPE,
    )


def packet(metadata, payload=b""):
    encoded = json.dumps(metadata, separators=(",", ":")).encode()
    return HEADER.pack(len(encoded), len(payload)) + encoded + payload


def read_packet(stream):
    header = stream.read(HEADER.size)
    if not header:
        return None
    if len(header) != HEADER.size:
        raise EOFError("Short telemetry header")
    meta_size, payload_size = HEADER.unpack(header)
    if meta_size > 128_000 or payload_size > 32_000_000:
        raise ValueError("Invalid telemetry packet size")
    meta, payload = stream.read(meta_size), stream.read(payload_size)
    if len(meta) != meta_size or len(payload) != payload_size:
        raise EOFError("Short telemetry packet")
    return json.loads(meta), payload


class FramePublisher:
    def __init__(self, folder):
        self.path = Path(folder) / "live-frame.bin"
        self.temp = self.path.with_suffix(".tmp")
        self.next_frame = 0.

    def show(self, image, *, gates, step, phase, status, driver="Policy"):
        now = time.monotonic()
        if now < self.next_frame:
            return
        self.next_frame = now + .085
        height, width, _ = image.shape
        metadata = {"type": "live", "width": width, "height": height,
                    "gates": gates, "step": step, "phase": phase, "status": status, "driver": driver}
        try:
            self.temp.write_bytes(packet(metadata, image.tobytes()))
            os.replace(self.temp, self.path)
        except OSError:
            # A spectator should never interrupt training.
            pass


def alive(root):
    try:
        pid = int((root / "run.pid").read_text())
        state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0]
        if state in ("Z", "T", "X"):
            return False
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        for child in children:
            command = Path(f"/proc/{child}/cmdline").read_bytes().split(b"\0")
            if b"training.racing_school" in command:
                child_state = Path(f"/proc/{child}/stat").read_text().split(") ", 1)[1][0]
                return child_state not in ("Z", "T", "X")
        return True
    except (OSError, ValueError, IndexError):
        return False


def read_metrics(run):
    try:
        with (run / "metrics.jsonl").open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 350_000))
            if size > 350_000:
                stream.readline()
            rows = []
            for line in stream:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows
    except OSError:
        return []


def finish_time_history(metrics, speed):
    history = []
    for row in metrics[-60:]:
        index = row.get("steps" if speed else "round")
        if index is None:
            continue
        history.append({
            "at": index,
            "standard_s": row.get("official_start_seconds") if speed else None,
            "comparison_s": row.get("official_mean_seconds" if speed else "best_two_lap_s"),
        })
    return history


def stream_run(root, school, speed, mode):
    out = sys.stdout.buffer
    school, speed = root / school, root / speed
    last_image = None
    last_checkpoint = None
    next_status = 0.
    while True:
        active = alive(root)
        run = speed if (speed / "config.json").exists() else school
        if run == school and not (run / "metrics.jsonl").exists():
            previous = school.with_name("racing-school-v1")
            if (previous / "best.pt").exists() and not (run / "live-frame.bin").exists():
                run, active = previous, False
        if time.monotonic() >= next_status:
            next_status = time.monotonic() + 2
            metrics = read_metrics(run)
            latest = metrics[-1] if metrics else {}
            time_key = "official_start_seconds" if run == speed else "best_two_lap_s"
            times = [row[time_key] for row in metrics if row.get(time_key) is not None]
            values = [row.get("gates_mean", sum(row.get("gates", [])) / max(1, len(row.get("gates", []))))
                      for row in metrics]
            out.write(packet({"type": "meta", "active": active,
                              "phase": "SPEED RL" if run == speed else "IMITATION",
                              "course": os.environ.get("A2RL_COURSE", "2025"),
                              "physics_engine": os.environ.get("A2RL_PHYSICS_ENGINE", "rust"),
                              "full_course": latest.get("full_course", False),
                              "round": latest.get("round", 0),
                              "steps": latest.get("steps", latest.get("transitions", 0)),
                              "target_gate": latest.get("target_gate", 0),
                              "recent_gates": values[-60:], "best_gates": max(values, default=0.),
                              "recent_finish_times": finish_time_history(metrics, run == speed),
                              "finish_rate": latest.get("two_lap_rate", 0.),
                              "collision_rate": latest.get("crash_rate", 0.),
                              "mean_gate_crossing_speed_m_s": latest.get("mean_gate_crossing_speed_m_s"),
                              "mean_gate_approach_delay_seconds": latest.get("mean_gate_approach_delay_seconds"),
                              "best_two_lap_seconds": min(times, default=None),
                              "target_two_lap_seconds": latest.get("target_two_lap_seconds"),
                              "target_beaten": (speed / "record.json").exists()}))
            latest_checkpoint = run / "latest.pt"
            checkpoint = latest_checkpoint if active else run / "best.pt"
            if mode == "replay":
                checkpoint = next((file for file in (run / "record.pt", latest_checkpoint) if file.is_file()),
                                  run / "best.pt")
            try:
                with checkpoint.open("rb") as stream:
                    key = (checkpoint, os.fstat(stream.fileno()).st_mtime_ns)
                    if key != last_checkpoint:
                        out.write(packet({"type": "checkpoint", "best": checkpoint.name == "best.pt",
                                          "phase": "SPEED RL" if run == speed else "IMITATION",
                                          "course": os.environ.get("A2RL_COURSE", "2025"),
                                          "physics_engine": os.environ.get("A2RL_PHYSICS_ENGINE", "rust"),
                                          "round": latest.get("round", 0),
                                          "steps": latest.get("steps", latest.get("transitions", 0)),
                                          "version": str(key[1])}, stream.read()))
                        last_checkpoint = key
            except FileNotFoundError:
                pass
            out.flush()

        live_path = run / "live-frame.bin"
        if active and mode != "replay":
            try:
                stamp = live_path.stat()
                key = (live_path, stamp.st_mtime_ns)
                if key != last_image and time.time() - stamp.st_mtime < 2.:
                    out.write(live_path.read_bytes())
                    out.flush()
                    last_image = key
            except FileNotFoundError:
                pass
        time.sleep(.065)


if __name__ == "__main__":
    try:
        stream_run(Path(sys.argv[1]).expanduser(), sys.argv[2], sys.argv[3], sys.argv[4])
    except BrokenPipeError:
        pass
