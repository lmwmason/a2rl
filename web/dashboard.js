"use strict";

const byId = (id) => document.getElementById(id);
const setText = (id, value) => {
  const element = byId(id);
  if (element.textContent !== value) element.textContent = value;
};
const number = new Intl.NumberFormat("en-US");
const dateTime = new Intl.DateTimeFormat("en-US", { dateStyle: "medium", timeStyle: "short" });
const count = (value) => Number.isFinite(value) ? number.format(value) : "—";
const percent = (value) => Number.isFinite(value) ? count(Math.round(value * 100)) + "%" : "—";
const quantity = (value, unit) => Number.isFinite(value) ? value.toFixed(2) + " " + unit : "—";
const curriculumLabels = {
  finish: "Stage 1 · Clean two-lap finish",
  pace: "Stage 2 · Imitation ≤30 s",
  speed: "Stage 3 · Speed refinement",
};
const camera = byId("fpv");
const chart = byId("history");
const chartContext = chart.getContext("2d");
const trend = { values: [], key: "", limit: 22 };
const finishChart = byId("finish-history");
const finishTrend = { values: [], key: "", speed: false };
let playing = null;
let queuedReplay = null;
let raceEndTimer = null;
let countdownTimer = null;
let sound = null;
let soundEnabled = false;
let gateAudioTime = null;
const countdown = byId("race-countdown");
let displayedControl = -1;
let pilot = null;
let studentStatus = null;
let preferredCamera = new URLSearchParams(window.location.search).get("camera") === "chase" ? "chase" : "fpv";
let cameraSelection = 0;
let selection = 0;
const hosted = !["127.0.0.1", "localhost", "[::1]"].includes(window.location.hostname);
const feeds = hosted ? [
  "https://beam-snapshot-down-museums.trycloudflare.com/",
  "https://a2rl-fpv-feed.server.cbgpu.com/",
] : ["/"];

async function requestData(url, video = false) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), video ? 90_000 : 3000);
  try {
    const response = await fetch(url, { cache: "no-store", credentials: "omit", signal: controller.signal });
    if (!response.ok) throw new Error("Training feed unavailable");
    return { url: response.url, value: video ? await response.blob() : await response.json() };
  } finally {
    window.clearTimeout(timer);
  }
}

async function readFeed(name, view) {
  if (view === "teacher") return requestData("/snapshot/" + name);
  const requests = feeds.map(async (feed) => {
    const response = await requestData(feed + name);
    const { course, physics_engine: physics, phase, kind } = response.value;
    const matchingRun = name === "replay.json"
      ? physics === "elodin" && kind === "student-evaluation"
      : response.value.live_feed_version === 1;
    if (hosted && (course !== "2026-practice" || !matchingRun || !["IMITATION", "SPEED RL"].includes(phase))) {
      throw new Error("Student evaluation feed unavailable");
    }
    return response;
  });
  try {
    if (typeof Promise.any === "function") return await Promise.any(requests);
    const responses = await Promise.all(requests.map((request) => request.catch(() => null)));
    const first = responses.find(Boolean);
    if (first) return first;
    throw new Error("Training feed unavailable");
  } catch {
    return requestData("/student/" + name);
  }
}

function setConnection(label, state = "") {
  setText("connection", label);
  byId("connection").className = state ? "connection " + state : "connection";
}

function showSticks(time = camera.currentTime) {
  const controls = playing?.controls;
  if (!controls || !Number.isFinite(time)) return;
  const index = Math.max(0, Math.min(controls.actions.length - 1, Math.floor(time * controls.sample_rate + 0.0001)));
  if (index === displayedControl) return;
  displayedControl = index;
  const [yaw, throttle, roll, pitch] = controls.actions[index];
  for (const [id, x, y] of [["left-stick", yaw, throttle], ["right-stick", roll, pitch]]) {
    byId(id).style.left = (50 + 46 * x) + "%";
    byId(id).style.top = (50 - 46 * y) + "%";
  }
  setText("stick-throttle", Math.round(50 * (throttle + 1)) + "%");
}

function replayTiming(timing) {
  if (!timing || timing.laps !== 2 || !Number.isFinite(timing.video_start_seconds)
      || !(timing.start_seconds === null || Number.isFinite(timing.start_seconds))
      || !(timing.stop_seconds === null || Number.isFinite(timing.stop_seconds))
      || !Array.isArray(timing.lap_end_seconds) || timing.lap_end_seconds.length > timing.laps
      || !timing.lap_end_seconds.every((end, index, ends) => Number.isFinite(end)
        && end >= (ends[index - 1] ?? timing.start_seconds))) return null;
  return timing;
}

function showRaceTimer(time = camera.currentTime) {
  const timing = playing?.timing;
  if (!timing || !Number.isFinite(time)) return;
  const start = timing.start_seconds;
  const timestamp = Math.min(time + timing.video_start_seconds, timing.stop_seconds ?? Infinity);
  const lapOneEnd = timing.lap_end_seconds[0];
  const secondLap = start !== null && Number.isFinite(lapOneEnd) && timestamp >= lapOneEnd;
  const lapStart = secondLap ? lapOneEnd : start;
  const elapsed = (from) => from === null ? "0.000" : Math.max(0, timestamp - from).toFixed(3);
  setText("timer-lap-label", secondLap ? "Lap 2 / 2" : "Lap 1 / 2");
  setText("timer-lap", elapsed(lapStart));
  setText("timer-total", elapsed(start));
  if (Array.isArray(timing.gate_seconds)) {
    const passed = timing.gate_seconds.filter(crossing => crossing <= timestamp).length;
    setText("timer-gates", `${passed} / ${playing.gates_per_race}`);
    byId("race-progress").value = passed;
  }
}

function updateReplayOverlays(time = camera.currentTime) {
  showSticks(time);
  showRaceTimer(time);
  updateMotorSound();
  updateGateSound();
}

function updateGateSound() {
  const timing = playing?.timing;
  if (!Array.isArray(timing?.gate_seconds) || !Number.isFinite(camera.currentTime)) return;
  const timestamp = camera.currentTime + timing.video_start_seconds;
  const previous = gateAudioTime;
  gateAudioTime = timestamp;
  if (!soundEnabled || camera.seeking || (camera.paused && !camera.ended)
      || previous === null || timestamp <= previous || timestamp - previous > .8 * Math.max(1, camera.playbackRate)) return;
  let delay = 0;
  timing.gate_seconds.forEach((crossing, index) => {
    if (previous >= crossing || crossing > timestamp) return;
    if (playing.status === 4 && index + 1 === playing.gates_per_race) {
      playFinishSound(delay);
    } else {
      playTone(660, .085, .095, delay, "triangle");
      playTone(880, .12, .07, delay + .045);
      delay += .13;
    }
  });
}

function updateMotorSound() {
  if (!sound) return;
  const controls = playing?.controls;
  const time = sound.context.currentTime;
  const active = soundEnabled && controls && !camera.paused && !camera.ended && countdownTimer === null;
  const row = active && controls.actions[Math.min(controls.actions.length - 1,
    Math.floor(camera.currentTime * controls.sample_rate))];
  const throttle = row && (1 + row[1]) / 2;
  sound.volume.gain.setTargetAtTime(throttle > .005 ? .018 + .14 * throttle : 0, time, .035);
  if (throttle > .005) {
    const pitch = 74 + 380 * Math.sqrt(throttle);
    sound.motors.forEach((motor, index) => {
      const mix = index % 2 ? row[2] : row[3];
      motor.frequency.setTargetAtTime(pitch * (1 + (index - 1) * .055 + .14 * mix), time, .035);
    });
  }
}

byId("race-sound").addEventListener("click", async () => {
  const button = byId("race-sound");
  if (!sound) {
    const Audio = window.AudioContext ?? window.webkitAudioContext;
    if (!Audio) return;
    const context = new Audio();
    const volume = context.createGain();
    volume.gain.value = 0;
    const effects = context.createGain();
    effects.gain.value = 0;
    effects.connect(context.destination);
    const filter = context.createBiquadFilter();
    filter.frequency.value = 1250;
    filter.connect(volume).connect(context.destination);
    const motors = Array.from({ length: 3 }, (_, index) => {
      const motor = context.createOscillator();
      motor.type = index ? "triangle" : "sawtooth";
      motor.connect(filter);
      motor.start();
      return motor;
    });
    sound = { context, volume, effects, motors };
  }
  soundEnabled = !soundEnabled;
  sound.effects.gain.setTargetAtTime(soundEnabled ? 1 : 0, sound.context.currentTime, .015);
  if (soundEnabled) await sound.context.resume().catch(() => {});
  button.setAttribute("aria-pressed", String(soundEnabled));
  setText("race-sound", soundEnabled ? "Sound on" : "Sound off");
  updateMotorSound();
});

function playTone(frequency, duration, level, delay = 0, type = "sine") {
  if (!soundEnabled || sound?.context.state !== "running") return;
  const { context, effects } = sound;
  const oscillator = context.createOscillator();
  const volume = context.createGain();
  const now = context.currentTime + delay;
  oscillator.type = type;
  oscillator.frequency.value = frequency;
  volume.gain.setValueAtTime(.001, now);
  volume.gain.exponentialRampToValueAtTime(level, now + .008);
  volume.gain.exponentialRampToValueAtTime(.001, now + duration);
  oscillator.connect(volume).connect(effects);
  oscillator.start(now);
  oscillator.stop(now + duration + .01);
}

function playFinishSound(delay) {
  [523.25, 659.25, 783.99].forEach((frequency, index) => {
    playTone(frequency, .17, .11, delay + index * .09, "triangle");
  });
  [523.25, 783.99, 1046.5].forEach(frequency => playTone(frequency, .44, .065, delay + .27));
}

function countdownBeep(go) {
  playTone(go ? 920 : 490, .17, .16);
}

function cancelCountdown() {
  clearTimeout(countdownTimer);
  countdownTimer = null;
  countdown.hidden = true;
  delete countdown.dataset.go;
}

function startRacePlayback() {
  if (!playing?.timing) {
    camera.play().catch(() => {});
    return;
  }
  cancelCountdown();
  const start = performance.now();
  let previousNumber = null;
  function tick() {
    const number = Math.max(0, 3 - Math.floor((performance.now() - start) / 1000));
    if (number !== previousNumber) {
      previousNumber = number;
      setText("race-countdown", number || "GO");
      countdown.hidden = false;
      countdownBeep(number === 0);
    }
    if (!number) {
      countdown.dataset.go = "";
      countdownTimer = window.setTimeout(cancelCountdown, 600);
      camera.play().catch(cancelCountdown);
      return;
    }
    countdownTimer = window.setTimeout(tick, Math.max(0, start + (4 - number) * 1000 - performance.now()));
  }
  tick();
}

function drawHistory() {
  const scale = window.devicePixelRatio || 1;
  chart.width = Math.round(chart.clientWidth * scale);
  chart.height = Math.round(chart.clientHeight * scale);
  const width = chart.width / scale;
  const height = chart.height / scale;
  chartContext.scale(scale, scale);
  chartContext.strokeStyle = "#dedfd8";
  for (const y of [2, height / 2, height - 2]) {
    chartContext.beginPath();
    chartContext.moveTo(0, y);
    chartContext.lineTo(width, y);
    chartContext.stroke();
  }
  const values = trend.values;
  const averages = values.map((_, index) => {
    const window = values.slice(Math.max(0, index - 7), index + 1);
    return window.reduce((sum, value) => sum + value, 0) / window.length;
  });
  for (const [series, color, weight] of [[values, "#696b66", 1.5], [averages, "#262724", 2.5]]) {
    chartContext.strokeStyle = chartContext.fillStyle = color;
    chartContext.lineWidth = weight;
    chartContext.beginPath();
    series.forEach((value, index) => {
      const x = 3 + (width - 6) * index / Math.max(1, series.length - 1);
      const y = height - 3 - (height - 6) * Math.max(0, Math.min(value, trend.limit)) / trend.limit;
      if (index === 0) {
        chartContext.moveTo(x, y);
        if (series.length === 1) chartContext.fillRect(x - 2, y - 2, 4, 4);
      } else {
        chartContext.lineTo(x, y);
      }
    });
    chartContext.stroke();
  }
}

function drawFinishHistory() {
  const { values, speed } = finishTrend;
  const times = values.flatMap((point) => [point.standard_s, point.comparison_s]).filter(Number.isFinite);
  byId("finish-history-plot").hidden = !times.length;
  byId("finish-history-empty").hidden = Boolean(times.length);
  byId("finish-history-key").hidden = !times.length;
  if (!times.length) return;

  const width = finishChart.clientWidth;
  const height = finishChart.clientHeight;
  if (!width || !height) return;
  const scale = window.devicePixelRatio || 1;
  finishChart.width = Math.round(width * scale);
  finishChart.height = Math.round(height * scale);
  const context = finishChart.getContext("2d");
  context.scale(scale, scale);
  context.font = "12px ui-monospace, monospace";
  context.fillStyle = "#565952";
  const left = 68, right = width - 5, top = 12, bottom = height - 28;
  const min = Math.min(...times);
  const max = Math.max(...times);
  const padding = Math.max(.08, (max - min) * .12);
  const low = Math.max(0, min - padding);
  const high = max + padding;
  const y = (seconds) => bottom - (bottom - top) * (seconds - low) / (high - low);
  const first = values[0].at;
  const last = values[values.length - 1].at;
  const x = (index) => first === last ? (left + right) / 2
    : left + (right - left) * (index - first) / (last - first);

  context.strokeStyle = "#dedfd8";
  context.textAlign = "right";
  for (const seconds of [high, (high + low) / 2, low]) {
    context.beginPath();
    context.moveTo(left, y(seconds));
    context.lineTo(right, y(seconds));
    context.stroke();
    context.fillText(seconds.toFixed(2) + " s", left - 6, y(seconds) + 4);
  }
  const axisLabel = (index) => speed ? count(index) + " steps" : "Round " + count(index);
  context.textAlign = first === last ? "center" : "left";
  context.fillText(axisLabel(first), x(first), height - 5);
  if (first !== last) {
    context.textAlign = "right";
    context.fillText(axisLabel(last), right, height - 5);
  }

  for (const [field, color] of [["comparison_s", "#696b66"], ["standard_s", "#262724"]]) {
    let previous = false;
    context.strokeStyle = context.fillStyle = color;
    context.lineWidth = 2;
    context.setLineDash(field === "comparison_s" ? [4, 3] : []);
    context.beginPath();
    for (const point of values) {
      const seconds = point[field];
      if (Number.isFinite(seconds)) {
        if (previous) context.lineTo(x(point.at), y(seconds));
        else context.moveTo(x(point.at), y(seconds));
        context.fillRect(x(point.at) - 2, y(seconds) - 2, 4, 4);
      }
      previous = Number.isFinite(seconds);
    }
    context.stroke();
  }
}

function showsChase(replay) {
  return /^chase-[0-9]+\.mp4$/.test(replay?.chase_url);
}

function updateCameraButtons() {
  const chase = playing?.perspective === "chase";
  byId("show-fpv").setAttribute("aria-pressed", String(!chase));
  byId("show-chase").setAttribute("aria-pressed", String(chase));
  byId("show-chase").disabled = !showsChase(playing);
}

async function selectCamera(perspective) {
  const current = playing;
  preferredCamera = perspective;
  const request = ++cameraSelection;
  if (!current || current.perspective === perspective || (perspective === "chase" && !showsChase(current))) return;
  try {
    const filename = perspective === "chase" ? current.chase_url : current.url;
    const { value: video } = await requestData(new URL(filename, current.manifestUrl), true);
    if (request !== cameraSelection || current !== playing) return;
    discardQueuedReplay();
    const blob = URL.createObjectURL(video.slice(0, video.size, "video/mp4"));
    const page = new URL(window.location.href);
    page.searchParams.set("camera", perspective);
    window.history.replaceState(null, "", page.href);
    startReplay({ ...current, blob, perspective }, camera.currentTime, !camera.paused);
  } catch {
    if (request === cameraSelection && current === playing) preferredCamera = current.perspective;
  }
}

function startReplay(replay, resumeTime = 0, autoplay = true) {
  const previous = playing;
  if (previous?.version !== replay.version) cancelCountdown();
  playing = replay;
  gateAudioTime = null;
  camera.src = playing.blob;
  updateCameraButtons();
  if (resumeTime > 0) {
    camera.addEventListener("loadedmetadata", () => {
      if (playing === replay) camera.currentTime = Math.min(resumeTime, camera.duration);
    }, { once: true });
  }
  camera.muted = true;
  camera.loop = !playing.timing;
  camera.hidden = false;
  byId("mode2-sticks").hidden = !playing.controls;
  byId("race-timer").hidden = !playing.timing;
  const hasGates = Array.isArray(playing.timing?.gate_seconds);
  byId("race-gate-row").hidden = !hasGates;
  byId("race-progress").hidden = !hasGates;
  byId("race-progress").max = playing.gates_per_race || 24;
  displayedControl = -1;
  updateReplayOverlays(resumeTime);
  const download = byId("download-replay");
  const exportName = playing.perspective === "chase" ? playing.chase_download_url : playing.download_url;
  const hasExport = /^export-(?:fpv|chase)-[0-9]+\.mp4$/.test(exportName);
  if (hasExport) download.href = new URL(exportName, playing.manifestUrl);
  else download.removeAttribute("href");
  const firmwareCheck = playing.kind === "takeoff-check";
  const teacher = playing.kind === "teacher-evaluation";
  const season = playing.course === "2026-practice" ? "2026-practice" : "2025";
  const progress = playing.phase === "SPEED RL" ? playing.steps : playing.round;
  download.download = teacher ? `a2rl-${season}-flow-teacher.mp4`
    : firmwareCheck ? `a2rl-${season}-elodin-takeoff.mp4`
    : !Number.isFinite(progress) ? `a2rl-${season}-student-${playing.version}.mp4`
    : playing.phase === "SPEED RL" ? `a2rl-${season}-rl-${playing.steps}.mp4`
    : `a2rl-${season}-imitation-round-${playing.round}.mp4`;
  download.download = download.download.replace(/\.mp4$/, `-${playing.perspective}.mp4`);
  download.hidden = !hasExport;
  byId("waiting").hidden = true;
  const stage = teacher ? "Flow teacher evaluation"
    : firmwareCheck ? "Firmware takeoff check"
    : playing.phase === "SPEED RL" ? (Number.isFinite(playing.steps)
      ? "Reinforcement learning · " + count(playing.steps) + " steps" : "Reinforcement learning checkpoint")
    : playing.round ? "Imitation · Round " + count(playing.round) : "Imitation checkpoint";
  setText("replay-stage", stage);
  setText("flight-gates", count(playing.gates) + " / " + count(playing.gates_per_race ?? 22));
  setText("replay-result", firmwareCheck ? "Takeoff verified"
    : { 1: "Crash", 2: "Timed out", 3: "Stalled", 4: "Finished two laps" }[playing.status] || "—");
  setText("replay-physics", physicsLabel(playing));
  const recorded = byId("replay-recorded");
  const recordedDate = new Date(Number(playing.version) / 1_000_000);
  if (Number.isFinite(recordedDate.getTime())) {
    recorded.dateTime = recordedDate.toISOString();
    recorded.textContent = dateTime.format(recordedDate);
    recorded.closest("div").hidden = false;
  } else {
    recorded.closest("div").hidden = true;
  }
  if (pilot === "student") renderStudentEvaluation();
  if (previous) URL.revokeObjectURL(previous.blob);
  if (autoplay) {
    if (resumeTime > 0) camera.play().catch(() => {});
    else startRacePlayback();
  }
}

function cancelRaceEnd() {
  clearTimeout(raceEndTimer);
  raceEndTimer = null;
  byId("race-results").hidden = true;
}

function discardQueuedReplay() {
  if (queuedReplay) URL.revokeObjectURL(queuedReplay.blob);
  queuedReplay = null;
}

camera.addEventListener("ended", () => {
  if (!playing?.timing || raceEndTimer !== null) return;
  const finished = playing.status === 4;
  const { start_seconds: start, stop_seconds: stop, video_start_seconds: offset,
          lap_end_seconds: [first, second] } = playing.timing;
  const seconds = (value) => Number.isFinite(value) ? `${value.toFixed(3)} s` : "—";
  const split = (end, from) => Number.isFinite(end) && Number.isFinite(from) ? seconds(Math.max(0, end - from)) : "—";
  const status = {1: "Crash", 2: "Timed out", 3: "Stalled"}[playing.status] ?? "Race ended";
  setText("race-results-title", finished ? "Race complete" : `${status} · ${playing.gates} / ${playing.gates_per_race} gates`);
  setText("result-total-label", finished ? "Finish time" : "Elapsed time");
  setText("result-lap-one", split(first, start));
  setText("result-lap-two", split(second, first));
  setText("result-elapsed-one", split(first, start));
  setText("result-total", finished && Number.isFinite(playing.official_seconds)
    ? seconds(playing.official_seconds) : split(stop ?? (camera.currentTime + offset), start));
  setText("result-elapsed-two", finished ? byId("result-total").textContent : split(second, start));
  byId("race-results").hidden = false;
  raceEndTimer = window.setTimeout(() => {
    cancelRaceEnd();
    if (queuedReplay) {
      const next = queuedReplay;
      queuedReplay = null;
      startReplay(next);
      return;
    }
    camera.currentTime = 0;
    startRacePlayback();
  }, 5000);
});

camera.addEventListener("seeking", () => {
  gateAudioTime = null;
  if (camera.currentTime < camera.duration - .1) cancelRaceEnd();
});

let overlayAnimation = null;
function animateOverlays() {
  overlayAnimation = null;
  updateReplayOverlays();
  if (!camera.paused && !camera.ended) startOverlayAnimation();
}

function startOverlayAnimation() {
  if (overlayAnimation === null && (playing?.controls || playing?.timing)) {
    overlayAnimation = requestAnimationFrame(animateOverlays);
  }
}

camera.addEventListener("playing", () => {
  if (!Object.hasOwn(countdown.dataset, "go")) cancelCountdown();
  cancelRaceEnd();
  startOverlayAnimation();
});
for (const event of ["timeupdate", "seeking", "seeked", "loadeddata", "pause", "ended"]) {
  camera.addEventListener(event, () => updateReplayOverlays());
}

function physicsLabel(run) {
  return run.physics_engine === "elodin" ? "Elodin + Betaflight SITL" : "—";
}

function renderStatus(info) {
  const connected = info.connection === "connected" && Date.now() / 1000 - info.snapshot_at < 15;
  const speedTraining = info.phase === "SPEED RL";
  const season2 = info.course === "2026-practice";
  const teacher = info.phase === "TEACHER";
  for (const row of document.querySelectorAll(".evaluation")) {
    row.hidden = info.phase === "VALIDATION" || (teacher && row.classList.contains("training-counter"));
  }
  setText("course", season2 ? "2026 race reconstruction" : info.course === "2025" ? "2025 baseline" : "—");
  if (info.pending) {
    setConnection(teacher ? "Loading teacher evaluation" : "Awaiting student evaluation", "stale");
  } else if (info.archived) {
    setConnection("Recorded evaluation", "stale");
  } else if (connected) {
    const activity = info.starting_speed_trainer ? "RL rollouts starting"
      : speedTraining ? "RL training" : "Training";
    setConnection(info.active ? "GPU online · " + activity : "GPU online · Training idle", info.active ? "live" : "");
  } else {
    setConnection(info.connection === "connecting" ? "Connecting to GPU" : "GPU feed offline", "stale");
  }
  setText("phase", teacher ? "Teacher validation"
    : info.phase === "VALIDATION" ? "Firmware validation" : info.target_beaten ? "Target reached"
    : info.awaiting_speed_trainer ? "Stage 3 · Awaiting Elodin reinforcement learning"
    : curriculumLabels[info.curriculum_stage] ?? (speedTraining ? "Reinforcement learning"
      : info.full_course ? "Full-course imitation" : info.phase ? "Imitation learning" : "—"));
  setText("progress-label", speedTraining ? "RL iteration" : "Current training round");
  setText("progress", count(info.round));
  setText("steps-label", speedTraining ? "Environment steps" : "Training samples");
  setText("steps", count(info.steps));
  byId("rl-rollouts").closest("div").hidden = !speedTraining || !Number.isFinite(info.rollouts);
  setText("rl-rollouts", count(info.rollouts));
  byId("rl-return").closest("div").hidden = !speedTraining || !Number.isFinite(info.mean_return);
  setText("rl-return", quantity(info.mean_return, "").trim());
  if (pilot === "student") {
    studentStatus = info;
    renderStudentEvaluation();
  }
  else renderEvaluation(info, speedTraining);
}

function renderStudentEvaluation() {
  if (studentStatus?.phase === "SPEED RL") {
    renderEvaluation(studentStatus, true);
    return;
  }
  const replay = playing?.physics_engine === "elodin" ? playing : null;
  const evaluations = replay?.history ?? (replay ? [replay] : []);
  const finishes = evaluations.map((evaluation) => evaluation.official_seconds);
  const completed = finishes.filter(Number.isFinite);
  renderEvaluation({
    gates_per_race: replay?.gates_per_race ?? 24,
    recent_gates: evaluations.map((evaluation) => evaluation.gates),
    recent_finish_times: finishes.map((seconds, index) => ({ at: index + 1, comparison_s: seconds ?? null })),
    finish_rate: evaluations.length ? completed.length / evaluations.length : null,
    collision_rate: evaluations.length ? evaluations.filter((evaluation) => evaluation.status === 1).length / evaluations.length : null,
    best_two_lap_seconds: completed.length ? Math.min(...completed) : null,
  }, false);
}

function renderEvaluation(info, speedTraining) {
  setText("history-title", speedTraining ? "Gates per RL rollout batch" : "Gates per evaluation");
  setText("gates-series", speedTraining ? "Batch average" : "Individual evaluations");
  setText("finish-rate", percent(info.finish_rate));
  setText("collision-rate", percent(info.collision_rate));
  setText("best-time-label", speedTraining ? "Best rollout time" : "Best two-lap time");
  setText("best-time", quantity(info.best_two_lap_seconds, "s"));
  const meanSpeed = info.mean_flight_speed_m_s;
  byId("mean-speed").closest("div").hidden = !Number.isFinite(meanSpeed);
  setText("mean-speed", quantity(meanSpeed, "m/s"));
  const meanThrottle = info.mean_throttle_fraction;
  byId("mean-throttle").closest("div").hidden = !Number.isFinite(meanThrottle);
  if (Number.isFinite(meanThrottle)) {
    setText("mean-throttle", (100 * meanThrottle).toFixed(2) + "%");
  }
  const highThrottle = info.high_throttle_seconds;
  const highFraction = info.high_throttle_fraction;
  const throttleMeasured = Number.isFinite(highThrottle) && Number.isFinite(highFraction);
  byId("high-throttle").closest("div").hidden = !throttleMeasured;
  if (throttleMeasured) {
    setText("high-throttle", `${(100 * highFraction).toFixed(1)}% · ${quantity(highThrottle, "s")}`);
  }
  setText("crossing-speed", quantity(info.mean_gate_crossing_speed_m_s, "m/s"));
  setText("approach-delay", quantity(info.mean_gate_approach_delay_seconds, "s"));
  byId("crossing-speed").closest("div").hidden = !Number.isFinite(info.mean_gate_crossing_speed_m_s);
  byId("approach-delay").closest("div").hidden = !Number.isFinite(info.mean_gate_approach_delay_seconds);

  const values = Array.isArray(info.recent_gates) ? info.recent_gates : [];
  const limit = info.gates_per_race ?? 22;
  const key = JSON.stringify([limit, values]);
  if (key !== trend.key) {
    trend.key = key;
    trend.values = values;
    trend.limit = limit;
    setText("gates-axis-max", count(limit));
    setText("gates-axis-mid", count(limit / 2));
    drawHistory();
  }
  const latestGates = values.at(-1);
  setText("history-count", values.length ? (speedTraining
    ? `Latest average · ${latestGates.toFixed(1)} / ${count(limit)}` : pilot === "student"
    ? `Latest · ${count(latestGates)} / ${count(limit)}`
    : "Last " + count(values.length) + " evaluations") : "Awaiting evaluation");

  const finishTimes = Array.isArray(info.recent_finish_times) ? info.recent_finish_times : [];
  const finishKey = JSON.stringify([pilot, speedTraining, finishTimes]);
  if (finishKey !== finishTrend.key) {
    finishTrend.key = finishKey;
    finishTrend.values = finishTimes;
    finishTrend.speed = speedTraining;
    byId("finish-history-standard").hidden = !speedTraining;
    setText("finish-history-comparison", speedTraining ? "Mean of clean finishes"
      : pilot === "student" ? "Recorded run" : "Fastest completed run");
    const latest = finishTimes[finishTimes.length - 1];
    const seconds = latest?.standard_s ?? latest?.comparison_s;
    setText("finish-history-latest", Number.isFinite(seconds) ? "Latest · " + seconds.toFixed(2) + " s"
      : latest ? "Latest evaluation · No finish" : "Awaiting evaluation");
    drawFinishHistory();
  }
}

async function fetchStatus(poll = true) {
  const selected = selection;
  try {
    const { value } = await readFeed("status.json", pilot);
    if (selected === selection) renderStatus(value);
  } catch {
    if (selected === selection) {
      setConnection(pilot === "student" ? "Awaiting student evaluation" : "Teacher unavailable", "stale");
    }
  } finally {
    if (poll) window.setTimeout(fetchStatus, 2000);
  }
}

function renderExperiments(experiments) {
  const teacher = experiments.teacher;
  const targetMet = Number.isFinite(teacher?.verified_worst_seconds)
    && teacher.verified_worst_seconds <= teacher.target_seconds;
  setText("experiment-teacher-state", teacher
    ? targetMet ? "Target met" : teacher.active ? "Running" : "Stopped"
    : "Unavailable");
  const teacherDetails = teacher ? [`${count(teacher.evaluations)} evaluations`] : [];
  if (Number.isFinite(teacher?.verified_worst_seconds)) {
    teacherDetails.push(`Worst of 3 verified starts ${teacher.verified_worst_seconds.toFixed(2)} s`);
  }
  setText("experiment-teacher-detail", teacherDetails.join(" · "));

  for (const name of ["segment-5-6", "segment-11-12"]) {
    const run = experiments[name];
    setText(`experiment-${name}-state`, run ? (run.active ? "Running" : "Stopped") : "Unavailable");
    const details = run ? [`Round ${count(run.round)}`, `${count(run.rollouts)} rollouts`] : [];
    if (Number.isFinite(run?.latest_best_finish_seconds)) {
      details.push(`Latest round best ${run.latest_best_finish_seconds.toFixed(2)} s`);
    }
    if (run?.promoted_round != null) {
      details.push(run.promoted_round ? `Promoted in round ${count(run.promoted_round)}` : "No promotion");
    }
    setText(`experiment-${name}-detail`, details.join(" · "));
  }
}

async function fetchExperiments() {
  try {
    const { value } = await Promise.any(feeds.map(async (feed) => {
      const response = await requestData(feed + "experiments.json");
      if (!response.value?.teacher && !response.value?.["segment-5-6"] && !response.value?.["segment-11-12"]) {
        throw new Error("Experiment feed unavailable");
      }
      return response;
    }));
    renderExperiments(value);
    byId("experiment-runs").hidden = false;
    setText("experiments-connection", "Connected");
  } catch {
    if (!byId("experiment-runs").hidden) {
      for (const name of ["teacher", "segment-5-6", "segment-11-12"]) {
        setText(`experiment-${name}-state`, "Last status unavailable");
      }
    }
    setText("experiments-connection", "Feed unavailable");
  } finally {
    window.setTimeout(fetchExperiments, 2000);
  }
}

async function fetchReplay(poll = true) {
  const selected = selection;
  const view = pilot;
  try {
    const manifest = await readFeed("replay.json", view);
    if (selected !== selection) return;
    try {
      await loadReplay(manifest, selected);
    } catch (error) {
      if (!hosted || view !== "student" || manifest.value.archived) throw error;
      if (selected === selection) await loadReplay(await requestData("/student/replay.json"), selected);
    }
  } catch {
    if (selected === selection && !playing) {
      setText("waiting", view === "student" ? "Waiting for a student replay" : "Teacher replay unavailable");
    }
  } finally {
    if (poll) window.setTimeout(fetchReplay, 3000);
  }
}

function selectPilot(view, refresh = true) {
  if (pilot === view) return;
  pilot = view;
  selection++;
  cancelRaceEnd();
  cancelCountdown();
  discardQueuedReplay();
  const page = new URL(window.location.href);
  page.searchParams.set("pilot", view);
  window.history.replaceState(null, "", page.href);
  byId("show-teacher").setAttribute("aria-pressed", String(view === "teacher"));
  byId("show-student").setAttribute("aria-pressed", String(view === "student"));
  camera.pause();
  camera.hidden = true;
  camera.removeAttribute("src");
  camera.load();
  if (playing) URL.revokeObjectURL(playing.blob);
  playing = null;
  updateCameraButtons();
  byId("download-replay").hidden = true;
  byId("mode2-sticks").hidden = true;
  byId("race-timer").hidden = true;
  byId("replay-recorded").closest("div").hidden = true;
  byId("waiting").hidden = false;
  setText("waiting", "Loading replay");
  for (const id of ["replay-stage", "flight-gates", "replay-result", "replay-physics"]) setText(id, "—");
  renderStatus({ course: "2026-practice", pending: true, phase: view === "teacher" ? "TEACHER" : "IMITATION", gates_per_race: 24 });
  if (refresh) {
    fetchStatus(false);
    fetchReplay(false);
  }
}

byId("show-teacher").addEventListener("click", () => selectPilot("teacher"));
byId("show-student").addEventListener("click", () => selectPilot("student"));
byId("show-fpv").addEventListener("click", () => selectCamera("fpv"));
byId("show-chase").addEventListener("click", () => selectCamera("chase"));
byId("download-replay").addEventListener("click", async event => {
  const link = event.currentTarget;
  if (!link.href) return;
  event.preventDefault();
  if (link.dataset.busy) return;
  const filename = link.download;
  link.dataset.busy = "1";
  setText("download-replay", "Downloading…");
  try {
    const { value } = await requestData(link.href, true);
    const localUrl = URL.createObjectURL(value);
    const save = document.createElement("a");
    save.href = localUrl;
    save.download = filename;
    save.click();
    window.setTimeout(() => URL.revokeObjectURL(localUrl), 60_000);
    setText("download-replay", "Download MP4");
  } catch {
    setText("download-replay", "Retry download");
  } finally {
    delete link.dataset.busy;
  }
});

function isNewReplay(info, selected) {
  if (selected !== selection) return false;
  if (!/^[0-9]+$/.test(info.version)) throw new Error("Invalid replay version");
  const current = queuedReplay ?? playing;
  if (!current) return true;
  if (Boolean(info.archived) !== Boolean(current.archived)) return !info.archived;
  return BigInt(info.version) > BigInt(current.version);
}

async function loadReplay({ url, value: info }, selected) {
  if (!isNewReplay(info, selected)) return;
  if (!/^replay-[0-9]+\.mp4$/.test(info.url)) throw new Error("Invalid replay filename");
  const controlsRequest = /^controls-[0-9]+\.json$/.test(info.controls_url)
    ? requestData(new URL(info.controls_url, url)).then(({ value }) => (
      Number.isFinite(value.sample_rate) && value.sample_rate > 0
        && Array.isArray(value.actions) && value.actions.length
        && value.actions.every((axes) => Array.isArray(axes) && axes.length === 4
          && axes.every((axis) => Number.isFinite(axis) && Math.abs(axis) <= 1)) ? value : null
    )).catch(() => null) : null;
  const cameraRequest = cameraSelection;
  const perspective = preferredCamera === "chase" && showsChase(info) ? "chase" : "fpv";
  const filename = perspective === "chase" ? info.chase_url : info.url;
  const [video, controls] = await Promise.all([requestData(new URL(filename, url), true), controlsRequest]);
  if (cameraRequest !== cameraSelection || !isNewReplay(info, selected)) return;
  const blob = URL.createObjectURL(video.value.slice(0, video.value.size, "video/mp4"));
  const replay = { ...info, blob, controls, timing: replayTiming(info.timing), perspective, manifestUrl: url };
  if (raceEndTimer !== null || (playing?.timing && !playing.archived)) {
    discardQueuedReplay();
    queuedReplay = replay;
    return;
  }
  startReplay(replay);
}

new ResizeObserver(drawHistory).observe(chart);
new ResizeObserver(drawFinishHistory).observe(finishChart);
selectPilot(new URLSearchParams(window.location.search).get("pilot") === "teacher" ? "teacher" : "student", false);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) return;
  if (!camera.paused) startOverlayAnimation();
  if (pilot === "student") fetchReplay(false);
});
window.addEventListener("pageshow", (event) => {
  if (event.persisted && pilot === "student") fetchReplay(false);
});
fetchStatus();
fetchExperiments();
fetchReplay();
