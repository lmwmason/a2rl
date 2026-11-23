"""Run three Elodin/Betaflight aircraft on one synchronized practice course."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from math import isfinite
import os
from pathlib import Path
import socket
import signal
import subprocess
from tempfile import TemporaryDirectory
from threading import Barrier, Lock
from time import monotonic


ROOT = Path(__file__).resolve().parents[1]
AIRCRAFT = 3
CONTACT_DISTANCE = .38
MAX_TRAINING_SKEW = 2


def send(stream, value):
    stream.write((json.dumps(value, separators=(",", ":")) + "\n").encode())
    stream.flush()


def receive(stream):
    line = stream.readline(4096)
    if not line:
        raise ConnectionError("An aircraft disconnected from the heat.")
    return json.loads(line)


class HeatClient:
    def __init__(self, path, index):
        self.channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.channel.settimeout(45)
        self.channel.connect(path)
        self.stream = self.channel.makefile("rwb")
        send(self.stream, index)

    def update(self, sample, position):
        send(self.stream, [sample, position])
        return receive(self.stream)


def intersects(a, b, next_a, next_b):
    start = [x - y for x, y in zip(a, b)]
    motion = [x - y - start[i] for i, (x, y) in enumerate(zip(next_a, next_b))]
    speed_squared = sum(axis * axis for axis in motion)
    fraction = max(0., min(1., -sum(x * y for x, y in zip(start, motion)) / speed_squared)) if speed_squared else 0.
    return sum((x + fraction * y) ** 2 for x, y in zip(start, motion)) < CONTACT_DISTANCE ** 2


class HeatTimeline:
    def __init__(self):
        self.histories = [{} for _ in range(AIRCRAFT)]
        self.sample = [-1] * AIRCRAFT
        self.aligned = -1
        self.max_skew = 0
        self.contacts = set()
        self.epoch = None
        self.lock = Lock()
        self.start = Barrier(AIRCRAFT, timeout=45)

    def update(self, aircraft, sample, position):
        if sample != self.sample[aircraft] + 1 or len(position) != 3 or not all(map(isfinite, position)):
            raise ValueError("An aircraft skipped a sample or returned a non-finite pose.")
        with self.lock:
            self.histories[aircraft][sample] = position
            self.sample[aircraft] = sample
        if sample == 0:
            self.start.wait()
        with self.lock:
            if self.epoch is None:
                self.epoch = monotonic() + .25
            earliest = min(self.sample)
            self.max_skew = max(self.max_skew, max(self.sample) - earliest)
            while self.aligned < earliest:
                previous = self.aligned
                self.aligned += 1
                if previous < 0:
                    continue
                for i, history in enumerate(self.histories):
                    for j in range(i + 1, AIRCRAFT):
                        other = self.histories[j]
                        if intersects(history[previous], other[previous], history[self.aligned], other[self.aligned]):
                            if i not in self.contacts or j not in self.contacts:
                                print(f"Aircraft {i + 1} and {j + 1} made contact at FPV sample {self.aligned}.", flush=True)
                            self.contacts.update((i, j))
                    del history[previous]
            peers = [history[min(sample, latest)] for i, (history, latest) in
                     enumerate(zip(self.histories, self.sample)) if i != aircraft]
            return [peers, aircraft in self.contacts, self.epoch]


def synchronize(streams, deadline):
    timeline = HeatTimeline()

    def handle(aircraft, stream):
        while monotonic() < deadline:
            try:
                sample, position = receive(stream)
            except ConnectionError:
                return
            send(stream, timeline.update(aircraft, sample, position))
        raise TimeoutError("The synchronized heat exceeded its wall time.")

    with ThreadPoolExecutor(max_workers=AIRCRAFT) as workers:
        futures = [workers.submit(handle, i, stream) for i, stream in enumerate(streams)]
        for future in futures:
            future.result()
    print(f"Peak FPV timestamp skew: {timeline.max_skew} frames.", flush=True)
    return timeline.max_skew


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=8.)
    parser.add_argument("--policy", type=Path, help="Shared RGB+IMU checkpoint. Defaults to the flow teacher.")
    parser.add_argument("--collect-demonstrations", action="store_true", help="Save synchronized data from clean teacher flights.")
    parser.add_argument("--port-offset", type=int, default=200)
    parser.add_argument("--wall-timeout", type=float, default=240.)
    parser.add_argument("--pace", type=float, default=.15, help="Target simulation seconds per wall-clock second.")
    parser.add_argument("--launch-interval", type=float, default=0., help="Stagger takeoff for clearance practice.")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 120 or not 1 <= args.port_offset <= 49960 or not args.wall_timeout > 0:
        parser.error("Use a duration of 1–120 seconds, an offset of 1–49960 and a positive timeout.")
    if not 0 < args.pace <= 1:
        parser.error("The heat pace must be greater than zero and at most one.")
    if not 0 <= args.launch_interval <= 3:
        parser.error("Use a launch interval of 0–3 seconds.")
    if args.seconds <= 3.25 + 2 * args.launch_interval:
        parser.error("The heat duration must extend past the last aircraft's launch.")
    if args.policy and args.collect_demonstrations:
        parser.error("Demonstrations require the teacher pilot.")
    harness = args.harness.resolve(strict=True)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    children, logs, channels = [], [], []
    with TemporaryDirectory(prefix="a2rl-heat-") as temp, socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        path = str(Path(temp) / "positions")
        server.bind(path)
        server.listen(AIRCRAFT)
        server.settimeout(45)
        try:
            for i, (lane, order) in enumerate(((-.46, 1), (0., 0), (.46, 2))):
                store = out / f"aircraft-{i + 1}"
                store.mkdir(exist_ok=False)
                environment = os.environ | {
                    "A2RL_ELODIN_ROOT": str(harness), "A2RL_COURSE": "2026-practice",
                    "A2RL_ELODIN_FLOW_TEACHER": "0" if args.policy else "1",
                    "A2RL_ELODIN_POLICY": str(args.policy.resolve(strict=True)) if args.policy else "",
                    "A2RL_ELODIN_DEMONSTRATION": str(store / ".candidate.npz") if args.collect_demonstrations else "",
                    "A2RL_ELODIN_IN_PROCESS": "1", "A2RL_ELODIN_DB": str(store / "db"),
                    "A2RL_ELODIN_START_SIDE": str(0. if args.launch_interval else lane),
                    "A2RL_HEAT_START_FORWARD": str(lane if args.launch_interval else 0.),
                    "A2RL_ELODIN_FPV": str(store / "replay.mp4"), "A2RL_FPV_BREAKUP": "0",
                    "A2RL_ELODIN_SECONDS": str(args.seconds), "ELODIN_ASSETS_DIR": str(store / "assets"),
                    "A2RL_SITL_PORT_OFFSET": str(args.port_offset + i * 20),
                    "A2RL_HEAT_SOCKET": path, "A2RL_HEAT_PILOT": str(i),
                    "A2RL_HEAT_PACE": str(args.pace),
                    "A2RL_HEAT_LAUNCH_DELAY": str(order * args.launch_interval),
                    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                    "PATH": str(harness / ".venv/bin") + os.pathsep + os.environ["PATH"],
                }
                log = (store / "rollout.log").open("wb")
                logs.append(log)
                children.append(subprocess.Popen([str(harness / ".venv/bin/python"),
                                                  str(ROOT / "tools/elodin_a2rl.py")],
                                                 env=environment, stdout=log, stderr=subprocess.STDOUT,
                                                 start_new_session=True))
            streams = [None] * AIRCRAFT
            for _ in range(AIRCRAFT):
                channel, _ = server.accept()
                channel.settimeout(45)
                stream = channel.makefile("rwb")
                channels.append((channel, stream))
                index = receive(stream)
                if not 0 <= index < AIRCRAFT or streams[index] is not None:
                    raise ValueError("Invalid or duplicate aircraft number.")
                streams[index] = stream
            skew = synchronize(streams, monotonic() + args.wall_timeout)
            for child in children:
                if child.wait(timeout=30):
                    raise RuntimeError(f"An aircraft failed. See {out} for the rollout logs.")
            for index in range(1, AIRCRAFT + 1):
                report = out / f"aircraft-{index}" / "replay.result.json"
                result = json.loads(report.read_text())
                if not result["took_off"]:
                    raise RuntimeError(f"Aircraft {index} did not take off. See {out} for the rollout logs.")
                print(f"Aircraft {index} passed {result['gates']} gates, aircraft contact: {result['aircraft_contact']}.")
            if args.collect_demonstrations:
                if skew > MAX_TRAINING_SKEW:
                    raise RuntimeError(f"The training clips were withheld because FPV timestamps differed by {skew} frames.")
                candidates = sorted(out.glob("aircraft-*/.candidate.npz"))
                if not candidates:
                    raise RuntimeError("The heat produced no clean full-race training clips.")
                for candidate in candidates:
                    candidate.with_suffix(".imu.npy").replace(candidate.with_name("demonstration.imu.npy"))
                    candidate.replace(candidate.with_name("demonstration.npz"))
        finally:
            for channel, stream in channels:
                stream.close()
                channel.close()
            for child in children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
            for log in logs:
                log.close()


if __name__ == "__main__":
    main()
