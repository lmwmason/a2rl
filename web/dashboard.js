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
const camera = byId("fpv");
const chart = byId("history");
const chartContext = chart.getContext("2d");
const trend = { values: [], key: "", limit: 22 };
const finishChart = byId("finish-history");
const finishTrend = { values: [], key: "", speed: false };
let playing = null;
let pending = null;
let displayedControl = -1;
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

async function readFeed(name) {
  const requests = feeds.map(async (feed) => {
    const response = await requestData(feed + name);
    if (hosted && response.value.physics_engine !== "elodin") throw new Error("Elodin feed unavailable");
    return response;
  });
  try {
    if (typeof Promise.any === "function") return await Promise.any(requests);
    const responses = await Promise.all(requests.map((request) => request.catch(() => null)));
    const first = responses.find(Boolean);
    if (first) return first;
    throw new Error("Training feed unavailable");
  } catch (error) {
    if (!hosted) throw error;
    return requestData("/snapshot/" + name);
  }
}

function pill(id, label, state = "") {
  setText(id, label);
  byId(id).className = state ? "pill " + state : "pill";
}

function showSticks(time = camera.currentTime) {
  const controls = playing?.controls;
  if (!controls || !Number.isFinite(time)) return;
  const index = Math.min(controls.actions.length - 1, Math.floor(time * controls.sample_rate + 0.0001));
  if (index === displayedControl) return;
  displayedControl = index;
  const [yaw, throttle, roll, pitch] = controls.actions[index];
  for (const [id, x, y] of [["left-stick", yaw, throttle], ["right-stick", roll, pitch]]) {
    byId(id).style.left = (50 + 46 * x) + "%";
    byId(id).style.top = (50 - 46 * y) + "%";
  }
  setText("stick-throttle", Math.round(50 * (throttle + 1)) + "%");
}

function drawHistory() {
  const scale = window.devicePixelRatio || 1;
  chart.width = Math.round(chart.clientWidth * scale);
  chart.height = Math.round(chart.clientHeight * scale);
  const width = chart.width / scale;
  const height = chart.height / scale;
  chartContext.scale(scale, scale);
  chartContext.strokeStyle = "#37454e";
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
  for (const [series, color, weight] of [[values, "#4c8871", 1.5], [averages, "#74e8ac", 2.5]]) {
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
  context.fillStyle = "#a6b2b9";
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

  context.strokeStyle = "#37454e";
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

  for (const [field, color] of [["comparison_s", "#92a7ed"], ["standard_s", "#74e8ac"]]) {
    let previous = false;
    context.strokeStyle = context.fillStyle = color;
    context.lineWidth = 2;
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

function startReplay() {
  if (pending) {
    if (playing) URL.revokeObjectURL(playing.blob);
    playing = pending;
    pending = null;
    camera.src = playing.blob;
    camera.muted = true;
    camera.hidden = false;
    byId("mode2-sticks").hidden = !playing.controls;
    displayedControl = -1;
    showSticks(0);
    const download = byId("download-replay");
    download.href = playing.blob;
    const firmwareCheck = playing.kind === "takeoff-check";
    const teacher = playing.kind === "teacher-evaluation";
    const season = playing.course === "2026-practice" ? "2026-practice" : "2025";
    download.download = teacher ? `a2rl-${season}-flow-teacher.mp4`
      : firmwareCheck ? `a2rl-${season}-elodin-takeoff.mp4`
      : playing.phase === "SPEED RL" ? `a2rl-${season}-rl-${playing.steps}.mp4`
      : `a2rl-${season}-imitation-round-${playing.round}.mp4`;
    download.hidden = false;
    byId("waiting").hidden = true;
    const stage = teacher ? "Flow teacher evaluation"
      : firmwareCheck ? "Firmware takeoff check"
      : playing.phase === "SPEED RL" ? "Reinforcement learning · " + count(playing.steps) + " steps"
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
  } else if (playing && !playing.archived) {
    camera.currentTime = 0;
  } else {
    return;
  }
  if (playing) camera.play().catch(() => {});
}
camera.addEventListener("ended", startReplay);
camera.addEventListener("seeked", () => showSticks());
if ("requestVideoFrameCallback" in camera) {
  const updateSticks = (_, frame) => {
    showSticks(frame.mediaTime);
    camera.requestVideoFrameCallback(updateSticks);
  };
  camera.requestVideoFrameCallback(updateSticks);
} else {
  camera.addEventListener("timeupdate", () => showSticks());
}

function physicsLabel(run) {
  return { elodin: "Elodin + Betaflight SITL", rust: "Rust" }[run.physics_engine] ?? "—";
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
  if (info.archived) {
    pill("connection", "No live training", "stale");
  } else if (connected) {
    pill("connection", info.active ? "GPU online · Training" : "GPU online · Training idle", info.active ? "live" : "");
  } else {
    pill("connection", info.connection === "connecting" ? "Connecting to GPU" : "GPU feed offline", "stale");
  }
  setText("phase", teacher ? "Teacher validation"
    : info.phase === "VALIDATION" ? "Firmware validation" : info.target_beaten ? "Target reached" : speedTraining ? "Reinforcement learning"
    : info.full_course ? "Full-course imitation" : info.phase ? "Imitation learning" : "—");
  setText("progress-label", speedTraining ? "Reinforcement learning steps" : "Imitation round");
  setText("progress", count(speedTraining ? info.steps : info.round));
  setText("steps", count(info.steps));
  setText("finish-rate", percent(info.finish_rate));
  setText("collision-rate", percent(info.collision_rate));
  setText("best-time", quantity(info.best_two_lap_seconds, "s"));
  const meanSpeed = info.mean_flight_speed_m_s;
  byId("mean-speed").closest("div").hidden = !Number.isFinite(meanSpeed);
  setText("mean-speed", quantity(meanSpeed, "m/s"));
  const highThrottle = info.high_throttle_seconds;
  const highFraction = info.high_throttle_fraction;
  const throttleMeasured = Number.isFinite(highThrottle) && Number.isFinite(highFraction);
  byId("high-throttle").closest("div").hidden = !throttleMeasured;
  if (throttleMeasured) {
    setText("high-throttle", `${(100 * highFraction).toFixed(1)}% · ${quantity(highThrottle, "s")}`);
  }
  setText("crossing-speed", quantity(info.mean_gate_crossing_speed_m_s, "m/s"));
  setText("approach-delay", quantity(info.mean_gate_approach_delay_seconds, "s"));

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
  setText("history-count", values.length ? "Last " + count(values.length) + " evaluations" : "Awaiting evaluation");

  const finishTimes = Array.isArray(info.recent_finish_times) ? info.recent_finish_times : [];
  const finishKey = JSON.stringify([speedTraining, finishTimes]);
  if (finishKey !== finishTrend.key) {
    finishTrend.key = finishKey;
    finishTrend.values = finishTimes;
    finishTrend.speed = speedTraining;
    byId("finish-history-standard").hidden = !speedTraining;
    setText("finish-history-comparison", speedTraining ? "Mean of completed runs" : "Fastest completed run");
    const latest = finishTimes[finishTimes.length - 1];
    const seconds = latest?.standard_s ?? latest?.comparison_s;
    setText("finish-history-latest", Number.isFinite(seconds) ? "Latest · " + seconds.toFixed(2) + " s"
      : latest ? "Latest evaluation · No finish" : "Awaiting evaluation");
    drawFinishHistory();
  }
}

async function fetchStatus() {
  try {
    const { value } = await readFeed("status.json");
    renderStatus(value);
  } catch {
    pill("connection", "Waiting for training feed", "stale");
  } finally {
    window.setTimeout(fetchStatus, 2000);
  }
}

async function fetchReplay() {
  try {
    const manifest = await readFeed("replay.json");
    try {
      await queueReplay(manifest);
    } catch (error) {
      if (!hosted || manifest.value.archived) throw error;
      await queueReplay(await requestData("/snapshot/replay.json"));
    }
  } catch {
    if (!playing) setText("waiting", "Waiting for replay");
  } finally {
    window.setTimeout(fetchReplay, 3000);
  }
}

async function queueReplay({ url, value: info }) {
  if (info.version === playing?.version || info.version === pending?.version) return;
  if (!/^replay-[0-9]+\.mp4$/.test(info.url)) throw new Error("Invalid replay filename");
  const controlsRequest = /^controls-[0-9]+\.json$/.test(info.controls_url)
    ? requestData(new URL(info.controls_url, url)).then(({ value }) => (
      Number.isFinite(value.sample_rate) && value.sample_rate > 0
        && Array.isArray(value.actions) && value.actions.length
        && value.actions.every((axes) => Array.isArray(axes) && axes.length === 4
          && axes.every((axis) => Number.isFinite(axis) && Math.abs(axis) <= 1)) ? value : null
    )).catch(() => null) : null;
  const [video, controls] = await Promise.all([requestData(new URL(info.url, url), true), controlsRequest]);
  const blob = URL.createObjectURL(video.value.slice(0, video.value.size, "video/mp4"));
  if (pending) URL.revokeObjectURL(pending.blob);
  pending = { ...info, blob, controls };
  if (!playing || camera.ended) startReplay();
}

new ResizeObserver(drawHistory).observe(chart);
new ResizeObserver(drawFinishHistory).observe(finishChart);
fetchStatus();
fetchReplay();
