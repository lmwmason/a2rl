"""Build RGB / inferred Mode-2 datasets from official MonoRace Log 5 and Log 7."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import struct
import time
from urllib.request import Request, urlopen
import zlib

import numpy as np
from PIL import Image

DATASET = "https://doi.org/10.4121/039abd82-094c-4c93-baa1-e7446f562e0e"
ARCHIVE = ("https://data.4tu.nl/file/039abd82-094c-4c93-baa1-e7446f562e0e/"
           "95df1bb5-ef53-4935-bc52-e68a35211b8d")
CENTRAL_START, ARCHIVE_SIZE = 3_213_161_494, 3_220_294_430
LOGS = {
    "log5": ("Log5_2025-04-12_Grand_Challenge_M17", 10.85, 30.0),
    "log7": ("Log7_2025-04-12_AI_vs_Human_Killian_M16", 26.45, 44.5),
}
CENTRAL = struct.Struct("<4s6H3L5H2L")
LOCAL = struct.Struct("<4s5H3L2H")


def download(start, end):
    for attempt in range(3):
        try:
            message = Request(f"{ARCHIVE}?range={start}-{end}", headers={"Range": f"bytes={start}-{end}"})
            with urlopen(message, timeout=40) as response:
                if response.status != 206 or not response.headers["Content-Range"].startswith(f"bytes {start}-{end}/"):
                    raise ValueError("The data server did not return the requested ZIP range.")
                payload = response.read(end - start + 1)
                if len(payload) != end - start + 1:
                    raise IOError("Incomplete ZIP range.")
                return payload
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError("Range fetch failed.")


def directory(data):
    entries = {}
    pos = 0
    while data[pos:pos+4] == b"PK\x01\x02":
        fields = CENTRAL.unpack_from(data, pos)
        length, extra, comment = fields[10:13]
        name = data[pos + CENTRAL.size:pos + CENTRAL.size + length].decode()
        entries[name] = (fields[-1], fields[8], fields[4])
        pos += CENTRAL.size + length + extra + comment
    return entries


def unpack_member(data, entry, first):
    offset, size, method = entry
    index = offset - first
    header = LOCAL.unpack_from(data, index)
    if header[0] != b"PK\x03\x04":
        raise ValueError("Invalid local ZIP member header.")
    payload = index + LOCAL.size + header[-1] + header[-2]
    compressed = data[payload:payload + size]
    if len(compressed) != size:
        raise ValueError("Incomplete ZIP member.")
    if method == 8:
        return zlib.decompress(compressed, -15)
    if method == 0:
        return compressed
    raise ValueError(f"Unsupported ZIP method {method}.")


def read_member(entry):
    offset, size, _ = entry
    return unpack_member(download(offset, offset + size + 511), entry, offset)


def closest(times, query):
    right = np.searchsorted(times, query).clip(1, len(times) - 1)
    left = right - 1
    return np.where(abs(query - times[left]) <= abs(times[right] - query), left, right)


def proxy_actions(imu, stamps):
    """Use short future rotor/rate averages as effective setpoints, not measured stick inputs."""
    timeline = imu[:, 0]
    order = np.argsort(timeline, kind="stable")
    timeline, imu = timeline[order], imu[order]

    def average_at(stamp, lead, fields):
        a, b = np.searchsorted(timeline, [stamp + lead - .007, stamp + lead + .007])
        valid = fields[a:b]
        valid = valid[np.isfinite(valid).all(axis=1)]
        return np.median(valid, axis=0) if len(valid) >= 3 else None

    rows = []
    rpm = imu[:, 8:12].copy()
    rpm[(rpm < 0) | (rpm > 4_500)] = np.nan
    for stamp in stamps:
        rates = average_at(stamp, .045, imu[:, 2:5])
        motors = average_at(stamp, .030, rpm)
        if rates is None or motors is None:
            rows.append([np.nan] * 4)
            continue
        p, q, r = rates  # Original body axes forward-right-down.
        throttle = np.mean(motors ** 2) / 3100.0**2
        rows.append([r / 3.4, 2. * throttle - 1., p / 5.5, -q / 5.5])
    return np.asarray(rows, np.float32)


def chunks(entries):
    """Group neighboring members to keep the number of range requests bounded."""
    group, start, end = [], 0, 0
    for job in sorted(entries, key=lambda job: job[1][0]):
        offset, size, _ = job[1]
        if group and (offset - end > 256_000 or offset + size - start > 8_000_000):
            yield start, end, group
            group = []
        if not group:
            start = offset
        end = offset + size + 511
        group.append(job)
    if group:
        yield start, end, group


def make_log(entries, root, alias, width, height, workers):
    log, start, end = LOGS[alias]
    prefix = f"data/{log}/"
    (root / alias).mkdir(parents=True, exist_ok=True)
    imu_path = root / alias / "imu.csv"
    images_path = root / alias / "images_processed.csv"
    for path in (imu_path, images_path):
        if not path.is_file():
            path.write_bytes(read_member(entries[prefix + path.name]))
    imu = np.loadtxt(imu_path, delimiter=",", skiprows=1, usecols=range(13))
    image_times = np.loadtxt(images_path, delimiter=",", skiprows=1, usecols=(0,))
    image_times.sort()
    names = sorted(
        [(float(name.rsplit("frame_", 1)[1].removesuffix(".jpeg")), info)
         for name, info in entries.items()
         if name.startswith(prefix + "camera/frame_") and name.endswith(".jpeg")],
        key=lambda row: row[0],
    )
    stamps = np.array([pair[0] for pair in names])
    eligible = image_times[(image_times >= start) & (image_times <= end)]
    nearest_processed = eligible[closest(eligible, np.arange(start, end, 1/30))]
    selected = np.unique(closest(stamps, nearest_processed))
    camera_times = stamps[selected]
    labels = proxy_actions(imu, camera_times)
    usable = (np.abs(camera_times - image_times[closest(image_times, camera_times)]) < .024)
    usable &= np.isfinite(labels).all(axis=1)
    chosen = selected[usable]
    labels = labels[usable]
    if len(chosen) < 300:
        raise ValueError(f"Only {len(chosen)} synchronized frames in {alias}.")
    frames = np.empty((len(chosen), height, width, 3), np.uint8)
    output = root / f"{alias}.npz"

    def load_chunk(task):
        first, end, group = task
        payload = download(first, end)
        for destination, entry in group:
            raw = unpack_member(payload, entry, first)
            with Image.open(BytesIO(raw)) as image:
                frames[destination] = np.asarray(image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR))
        return len(group)

    jobs = list(chunks([(index, names[frame][1]) for index, frame in enumerate(chosen)]))
    print(f"{alias}: source {log}, fetching {len(chosen)} aligned frames in {len(jobs)} ranges", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for batch, loaded in enumerate(pool.map(load_chunk, jobs), 1):
            if batch % 25 == 0:
                print(f"{alias}: {batch}/{len(jobs)} ranges", flush=True)
    confidence = np.where((np.abs(labels) > 1).any(axis=1), .35, .72).astype(np.float32)
    np.savez_compressed(output, frames=frames, actions=np.clip(labels, -1, 1),
                        timestamps=camera_times[usable], weights=confidence)
    print(json.dumps({"source": DATASET, "log": log, "frames": len(frames),
                      "saturated_label_fraction": float((confidence < .5).mean()), "file": str(output)}),
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--logs", nargs="+", choices=LOGS, default=list(LOGS))
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--central-cache", type=Path)
    args = parser.parse_args()
    if min(args.width, args.height, args.workers) <= 0:
        parser.error("Resolution and workers must be positive.")
    args.output.mkdir(parents=True, exist_ok=True)
    cached = args.central_cache or (args.output / "zip-directory.bin")
    if not cached.is_file():
        cached.write_bytes(download(CENTRAL_START, ARCHIVE_SIZE - 1))
    entries = directory(cached.read_bytes())
    for alias in args.logs:
        make_log(entries, args.output, alias, args.width, args.height, args.workers)


if __name__ == "__main__":
    main()
