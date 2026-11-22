"""Extract synchronized FPV and stick samples from TII Racing's CC BY 4.0 v3 release."""
import argparse
from bisect import bisect_right
import csv
import io
from itertools import accumulate
import json
from pathlib import Path, PurePosixPath
import shutil
from tempfile import TemporaryFile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
from PIL import Image

RELEASE = "https://github.com/tii-racing/drone-racing-dataset/releases/download/v3.0.0"
CHUNK_SIZES = [1_992_294_400] * 6 + [193_406_873]
BLOCK_SIZE = 32 * 1024**2


class PilotedArchive(io.RawIOBase):
    """Seek through the concatenated release chunks without downloading the full dataset."""
    def __init__(self):
        self.offsets = list(accumulate(CHUNK_SIZES, initial=0))
        self.position = 0
        self.cache_key = None
        self.cache = b""

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        base = (0, self.position, self.offsets[-1])[whence]
        if base + offset < 0:
            raise ValueError("Invalid archive offset")
        self.position = base + offset
        return self.position

    def read(self, size=-1):
        remaining = max(0, self.offsets[-1] - self.position)
        remaining = remaining if size < 0 else min(size, remaining)
        pieces = []
        while remaining:
            part = bisect_right(self.offsets, self.position) - 1
            start = self.position - self.offsets[part]
            block_start = (start // BLOCK_SIZE) * BLOCK_SIZE
            key = part, block_start
            if self.cache_key != key:
                self.cache = self.download(part, block_start)
                self.cache_key = key
            offset = start - block_start
            requested = min(remaining, len(self.cache) - offset)
            pieces.append(self.cache[offset:offset + requested])
            self.position += requested
            remaining -= requested
        return b"".join(pieces)

    @staticmethod
    def download(part, start):
        end = min(start + BLOCK_SIZE, CHUNK_SIZES[part]) - 1
        url = f"{RELEASE}/piloted_zipchunk{part + 1:02d}?range={start}-{end}"
        request = Request(url, headers={"Range": f"bytes={start}-{end}"})
        for attempt in range(3):
            try:
                with urlopen(request, timeout=50) as result:
                    expected = f"bytes {start}-{end}/"
                    if result.status != 206 or not result.headers.get("Content-Range", "").startswith(expected):
                        raise ValueError("The release server did not honor the requested ZIP range")
                    payload = result.read(end - start + 2)
                if len(payload) != end - start + 1:
                    raise IOError("The ZIP range was truncated")
                return payload
            except HTTPError as error:
                if error.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                    raise
            except OSError:
                if attempt == 2:
                    raise
            time.sleep(2 ** (attempt + 1))


def select_rows(stream):
    rows = list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig")))
    number = lambda value: float(value) if value and value.strip() else np.nan
    times = np.asarray([float(row["elapsed_time"]) for row in rows])
    channels = np.asarray([[number(row["channels_" + axis]) for axis in ("yaw", "thrust", "roll", "pitch")]
                           for row in rows])
    gyro = np.asarray([[number(row["gyro_" + axis]) for axis in ("z", "x", "y")] for row in rows])
    names = np.asarray([Path(row["img_filename"]).name for row in rows])
    valid = (names != "") & np.isfinite(times) & np.isfinite(channels).all(axis=1) & np.isfinite(gyro).all(axis=1)
    valid &= np.logical_and(channels >= 950, channels <= 2050).all(axis=1)
    valid &= channels[:, 1] > 1050
    times, channels, gyro, names = (column[valid] for column in (times, channels, gyro, names))
    order = np.argsort(times, kind="stable")
    times, channels, gyro, names = (column[order] for column in (times, channels, gyro, names))
    distinct = np.r_[True, np.diff(times) > 0.]
    times, channels, gyro, names = (column[distinct] for column in (times, channels, gyro, names))
    if len(times) < 30:
        raise ValueError("Not enough airborne footage and aligned channels")

    # The simulator uses forward/left/up and stick order yaw, throttle, roll, pitch.
    # Determine radio reversals against logged body rates before using samples as labels.
    # The dataset's IMU uses forward/left/up.
    rate_to_stick = [-1, 1, 1]
    channel_indexes = [0, 2, 3]
    lags = np.searchsorted(times, times + .05).clip(0, len(times) - 1)
    correlations = [np.corrcoef(channels[:, index], gyro[lags, axis])[0, 1]
                    for axis, index in enumerate(channel_indexes)]
    if not all(np.isfinite(correlations)) or min(abs(value) for value in correlations) < .2:
        raise ValueError(f"Cannot verify transmitter axis signs: {correlations}")
    signs = np.ones(4)
    signs[channel_indexes] = np.sign(correlations) * rate_to_stick
    wanted = np.arange(times[0], times[-1], 1 / 30.)
    indexes = np.searchsorted(times, wanted).clip(1, len(times) - 1)
    left = indexes - 1
    indexes = np.unique(np.where(abs(times[left] - wanted) < abs(times[indexes] - wanted), left, indexes))
    actions = np.clip((channels[indexes] - 1500) / 500 * signs, -1., 1.).astype(np.float32)
    return times[indexes], names[indexes], actions, signs.tolist(), correlations


def camera_zip(archive, directory):
    candidates = [name for name in archive.namelist()
                  if PurePosixPath(name).parent == directory and PurePosixPath(name).name.startswith("camera")
                  and name.casefold().endswith(".zip")]
    if len(candidates) != 1:
        raise ValueError(f"Expected one camera ZIP in {directory}; found {candidates}")
    return candidates[0]


def convert(archive, csv_path, output):
    directory = PurePosixPath(csv_path).parent
    print(f"Reading synchronized channels for {directory.name}", flush=True)
    with archive.open(csv_path) as csv_stream:
        times, images, actions, signs, correlations = select_rows(csv_stream)
    print(f"Loading {len(times)} camera frames at 30 Hz for {directory.name}", flush=True)
    with TemporaryFile() as camera, archive.open(camera_zip(archive, directory)) as stream:
        shutil.copyfileobj(stream, camera, length=BLOCK_SIZE)
        with ZipFile(camera) as images_zip:
            members = {PurePosixPath(name).name: name for name in images_zip.namelist()}
            frames = np.empty((len(times), 72, 96, 3), dtype=np.uint8)
            for index, name in enumerate(images):
                with images_zip.open(members[name]) as jpeg, Image.open(jpeg) as image:
                    frames[index] = np.asarray(image.convert("RGB").resize((96, 72), Image.Resampling.BILINEAR))
    destination = output / (directory.name + ".npz")
    np.savez_compressed(destination, frames=frames, timestamps=times, actions=actions,
                        weights=np.ones(len(times), dtype=np.float32))
    print(json.dumps({"flight": directory.name, "frames": len(times), "stick_signs": signs,
                      "gyro_correlations": correlations, "peak_throttle_stick": float(actions[:, 1].max()),
                      "path": str(destination)}), flush=True)


def export_imu(archive, csv_path, output):
    """Join sensor readings to the previously exported RGB timestamps without fetching images again."""
    destination = output / (PurePosixPath(csv_path).parent.name + ".npz")
    with np.load(destination, allow_pickle=False) as demo:
        timestamps = demo["timestamps"]
    columns = [f"{sensor}_{axis}" for sensor in ("gyro", "accel") for axis in "xyz"]
    needed = set(timestamps)
    with archive.open(csv_path) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig") as stream:
        samples = {stamp: [float(row[column]) for column in columns]
                   for row in csv.DictReader(stream)
                   if (stamp := float(row["elapsed_time"])) in needed}
    imu = np.asarray([samples[stamp] for stamp in timestamps], dtype=np.float32)
    if not np.isfinite(imu).all():
        raise ValueError(f"Nonfinite IMU samples for {destination}")
    np.save(destination.with_suffix(".imu.npy"), imu, allow_pickle=False)
    print(f"{destination.stem}: {len(imu)} gyro and accelerometer samples", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List camera-synchronized piloted flights.")
    parser.add_argument("--imu-only", action="store_true", help="Add synchronized IMU samples to existing demonstrations.")
    parser.add_argument("--out", type=Path, default=Path("runs/mck"))
    parser.add_argument("--flight", action="append", default=[], help="Exact flight directory name from --list.")
    args = parser.parse_args()
    with ZipFile(PilotedArchive()) as archive:
        clips = {PurePosixPath(name).parent.name: name for name in archive.namelist()
                 if name.endswith("_cam_ts_sync.csv") and not name.startswith("__MACOSX/")}
        if args.list:
            print("\n".join(sorted(clips)))
            return
        if not args.flight:
            parser.error("Select a flight from --list.")
        args.out.mkdir(parents=True, exist_ok=True)
        for name in args.flight:
            if name not in clips:
                parser.error(f"Unknown flight directory {name}")
            if not args.imu_only:
                convert(archive, clips[name], args.out)
            export_imu(archive, clips[name], args.out)


if __name__ == "__main__":
    main()
