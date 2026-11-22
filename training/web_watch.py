"""Serve the static FPV dashboard and mirror GPU telemetry to a local file."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from threading import Thread
import time

from . import telemetry


class DashboardHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format, *args):
        if self.path != "/snapshot.bin":
            super().log_message(format, *args)


def mirror(args, path):
    temporary = path.with_suffix(".tmp")
    job = {}

    def publish(info, pixels=b""):
        meta = info | {"snapshot_at": time.time(), "frame_at": frame_at}
        temporary.write_bytes(telemetry.packet(meta, pixels))
        temporary.replace(path)

    while True:
        frame, pixels, frame_at = {}, b"", None
        publish(job | {"connection": "connecting"})
        child = telemetry.connect(args.host, args.remote_root, args.school, args.speed, checkpoints=False)
        try:
            while message := telemetry.read_packet(child.stdout):
                info, payload = message
                if info["type"] == "meta":
                    job = info
                elif info["type"] == "live":
                    frame, pixels, frame_at = info, payload, time.time()
                publish(frame | job | {"connection": "connected"}, pixels)
        except (EOFError, ValueError, OSError) as error:
            print(f"Telemetry connection lost: {error}", flush=True)
        finally:
            if child.poll() is None:
                child.terminate()
            child.wait()
            child.stdout.close()
        publish(frame | job | {"connection": "disconnected", "active": False}, pixels)
        time.sleep(3)


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
    assets = Path(__file__).resolve().parents[1] / "web"
    with TemporaryDirectory(prefix="a2rl-dashboard-") as directory:
        root = Path(directory)
        for name in ("index.html", "style.css", "dashboard.js"):
            copyfile(assets / name, root / name)
        handler = partial(DashboardHandler, directory=root)
        with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
            Thread(target=mirror, args=(args, root / "snapshot.bin"), daemon=True).start()
            print(f"Local dashboard http://127.0.0.1:{server.server_port}/", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
