"""Render the race HUD and synthesized cues into a standalone MP4 download."""

from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
import math
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FPS, RATE = 30, 24000
WIDTH, HEIGHT = 480, 360
PREROLL = 3
INK, PAPER, GREEN, MUTED = "#101c26", "#edf2f5", "#8cf0bd", "#c1ccd2"


@lru_cache
def font(size, bold=False):
    query = "monospace:style=Bold" if bold else "monospace"
    path = subprocess.check_output(["fc-match", "-f", "%{file}", query], text=True)
    return ImageFont.truetype(path, size)


def race_summary(replay, playback):
    timing = replay["timing"]
    ends = timing["lap_end_seconds"]
    first, second = (ends + [None, None])[:2]
    start = timing["start_seconds"]
    stop = timing["stop_seconds"]
    timestamp = stop if stop is not None else playback + timing["video_start_seconds"]
    finished = replay["status"] == 4
    status = {1: "Crash", 2: "Timed out", 3: "Stalled"}.get(replay["status"], "Race ended")
    title = "Race complete" if finished else f"{status} · {replay['gates']} / {replay['gates_per_race']} gates"

    def split(end, begin):
        return f"{max(0., end - begin):.3f} s" if end is not None and begin is not None else "—"

    official = replay.get("official_seconds")
    total = f"{official:.3f} s" if finished and official is not None else split(timestamp, start)
    first_split = split(first, start)
    return title, "Finish time" if finished else "Elapsed time", total, (
        ("Lap", "Split", "Elapsed"), ("1", first_split, first_split),
        ("2", split(second, first), total if finished else split(second, start)),
    )


def overlay(image, replay, actions, sample_rate, playback, countdown=None, results=False):
    draw = ImageDraw.Draw(image)
    timing = replay["timing"]
    start = timing["start_seconds"]
    ends = timing["lap_end_seconds"]
    stop = timing["stop_seconds"]
    timestamp = min(playback + timing["video_start_seconds"], stop if stop is not None else math.inf)
    lap = 2 if ends and timestamp >= ends[0] else 1
    clock = lambda since: f"{max(0., timestamp - since):.3f}" if since is not None else "0.000"
    passed = bisect_right(timing.get("gate_seconds", ()), timestamp)
    gates = replay["gates_per_race"]
    draw.rounded_rectangle((335, 9, 470, 72), 5, fill=INK, outline="#50616b")
    for index, (label, value) in enumerate(((f"Lap {lap} / 2", clock(ends[0] if lap == 2 else start)),
                                            ("Total", clock(start)), ("Gates", f"{passed} / {gates}"))):
        y = 15 + 15 * index
        draw.text((342, y), label, font=font(11), fill=MUTED)
        draw.text((463, y), value, font=font(11, True), fill=GREEN if index == 1 else PAPER, anchor="rt")
    draw.rectangle((342, 64, 463, 66), fill="#35454d")
    if passed:
        draw.rectangle((342, 64, 342 + round(121 * passed / gates), 66), fill=GREEN)

    if len(actions):
        yaw, throttle, roll, pitch = actions[min(len(actions) - 1, int(playback * sample_rate))]
        for label, x, horizontal, vertical in (("Yaw / Throttle", 10, yaw, throttle),
                                                ("Roll / Pitch", 394, roll, pitch)):
            draw.rounded_rectangle((x, 274, x + 76, 350), 5, fill=INK, outline="#50616b")
            draw.text((x + 38, 280), label, font=font(8), fill=PAPER, anchor="mt")
            cx, cy = x + 38, 311
            draw.rectangle((cx - 20, cy - 17, cx + 20, cy + 17), outline="#697982")
            draw.line((cx - 20, cy, cx + 20, cy), fill="#697982")
            draw.line((cx, cy - 17, cx, cy + 17), fill="#697982")
            dx, dy = cx + 18 * horizontal, cy - 15 * vertical
            draw.ellipse((dx - 3, dy - 3, dx + 3, dy + 3), fill=GREEN)
            if x == 10:
                draw.text((cx, 334), f"{(throttle + 1) * 50:.0f}%", font=font(9, True), fill=PAPER, anchor="mt")

    if countdown is not None:
        draw.rounded_rectangle((167, 128, 313, 224), 8, fill=INK, outline="#60818d")
        draw.text((240, 173), countdown, font=font(53, True),
                  fill=GREEN if countdown == "GO" else PAPER, anchor="mm")
    if results:
        title, total_label, total, rows = race_summary(replay, playback)
        draw.rounded_rectangle((69, 88, 411, 264), 8, fill=INK, outline="#50616b")
        draw.line((77, 89, 403, 89), fill=GREEN, width=3)
        draw.text((86, 104), title, font=font(15, True), fill=PAPER)
        draw.text((86, 137), total_label, font=font(12), fill=MUTED)
        draw.text((394, 128), total, font=font(28, True), fill=GREEN, anchor="rt")
        for index, values in enumerate(rows):
            y = 169 + index * 26
            draw.line((86, y, 394, y), fill="#41515a")
            for x, value in zip((87, 234, 394), values):
                draw.text((x, y + 8), value, font=font(12, index > 0), fill=MUTED if not index else PAPER,
                          anchor="lt" if x == 87 else "rt")
    return image.tobytes()


def write_audio(path, replay, actions, sample_rate, frames, hold):
    video_samples = round(frames / FPS * RATE)
    audio = np.zeros((PREROLL + hold) * RATE + video_samples)
    time = np.arange(video_samples) / RATE
    if len(actions):
        command = actions[np.minimum(len(actions) - 1, (time * sample_rate).astype(int))]
        throttle = (command[:, 1] + 1) / 2
        active = throttle > .005
        amount = np.where(active, .018 + .14 * throttle, 0.)
        amount[time < .6] = 0.
        stop = replay["timing"]["stop_seconds"]
        if stop is not None:
            amount[time >= stop - replay["timing"]["video_start_seconds"]] = 0.
        pitch = 74 + 380 * np.sqrt(throttle)
        motors = np.zeros_like(time)
        for index in range(3):
            phase = np.cumsum(pitch * (1 + (index - 1) * .055 + .14 * command[:, 2 if index % 2 else 3])) / RATE
            if index:
                motors += 2 * np.abs(2 * (phase % 1) - 1) - 1
            else:
                motors += np.sin(2 * np.pi * phase) + .27 * np.sin(4 * np.pi * phase)
        audio[PREROLL * RATE:PREROLL * RATE + video_samples] = amount * motors * .6

    def tone(at, frequency, duration, level):
        offset = max(0, round(at * RATE))
        n = min(round(duration * RATE), len(audio) - offset)
        if n <= 0:
            return
        times = np.arange(n) / RATE
        envelope = np.minimum(times / .008, 1.) * np.exp(-4.5 * times / duration)
        audio[offset:offset + n] += level * envelope * np.sin(2 * np.pi * frequency * times)

    for second in range(4):
        tone(second, 920 if second == 3 else 490, .17, .16)
    timing = replay["timing"]
    for index, crossing in enumerate(timing.get("gate_seconds", ())):
        at = PREROLL + crossing - timing["video_start_seconds"]
        if replay["status"] == 4 and index + 1 == replay["gates_per_race"]:
            for note, frequency in enumerate((523.25, 659.25, 783.99)):
                tone(at + .09 * note, frequency, .17, .11)
            for frequency in (523.25, 783.99, 1046.5):
                tone(at + .27, frequency, .44, .065)
        else:
            tone(at, 660, .085, .095)
            tone(at + .045, 880, .12, .07)
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        stream.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def render(source, destination, replay, controls):
    actions = np.clip(np.asarray(controls["actions"], dtype=float), -1., 1.)
    sample_rate = controls["sample_rate"]
    size = WIDTH * HEIGHT * 3
    hold = 5
    duration = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(source)], text=True))
    ffmpeg = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "1"]
    temporary = destination.with_suffix(".part.mp4")
    try:
        with TemporaryDirectory(prefix="a2rl-replay-") as directory:
            sound = Path(directory) / "effects.wav"
            write_audio(sound, replay, actions, sample_rate, round(duration * FPS), hold)
            with subprocess.Popen([*ffmpeg, "-i", str(source), "-vf", f"scale={WIDTH}:{HEIGHT},fps={FPS}",
                                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-threads:v", "1", "pipe:1"],
                                  stdout=subprocess.PIPE) as decoder, \
                 subprocess.Popen([*ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "pipe:0", "-i", str(sound),
                                   "-c:v", "libx264", "-threads:v", "2", "-preset", "veryfast", "-pix_fmt",
                                   "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(temporary)],
                                  stdin=subprocess.PIPE) as encoder:
                raw = decoder.stdout.read(size)
                if len(raw) != size:
                    raise ValueError(f"No complete replay frame in {source}")
                first = Image.frombytes("RGB", (WIDTH, HEIGHT), raw)
                for second in range(PREROLL, 0, -1):
                    frame = overlay(first.copy(), replay, actions, sample_rate, 0., str(second))
                    encoder.stdin.write(frame * FPS)
                frames = 0
                while len(raw) == size:
                    last = Image.frombytes("RGB", (WIDTH, HEIGHT), raw)
                    frame = overlay(last.copy(), replay, actions, sample_rate, frames / FPS,
                                    "GO" if frames < .6 * FPS else None)
                    encoder.stdin.write(frame)
                    frames += 1
                    raw = decoder.stdout.read(size)
                if decoder.wait() or raw:
                    raise ValueError(f"Incomplete replay decode for {source}")
                frame = overlay(last, replay, actions, sample_rate, frames / FPS, results=True)
                for _ in range(hold):
                    encoder.stdin.write(frame * FPS)
                encoder.stdin.close()
                if encoder.wait():
                    raise RuntimeError(f"Could not encode {destination}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def attach_exports(directory, replay):
    if not replay.get("timing"):
        return {}
    controls = json.loads((directory / replay["controls_url"]).read_text())
    exports = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = []
        for view, source_key, export_key in (("fpv", "url", "download_url"),
                                             ("chase", "chase_url", "chase_download_url")):
            if source := replay.get(source_key):
                filename = f"export-{view}-{replay['version']}.mp4"
                jobs.append((pool.submit(render, directory / source, directory / filename, replay, controls),
                             export_key, filename))
        for job, key, filename in jobs:
            job.result()
            exports[key] = filename
    return exports
