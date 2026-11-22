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
let pilot = "teacher";
let selection = 0;
let studentView = "evaluation";
let liveUrl = null;
let liveDriver = null;
let liveAge = 0;
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
  } catch (error) {
    if (!hosted) throw error;
    return requestData("/student/" + name);
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
  } else if (playing && !playing.archived) {
    camera.currentTime = 0;
  } else {
    return;
  }
  applyStudentView();
  if (playing && !camera.hidden) camera.play().catch(() => {});
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
  if (info.pending) {
    pill("connection", teacher ? "Loading teacher evaluation" : "Awaiting student evaluation", "stale");
  } else if (info.archived) {
    pill("connection", "Recorded evaluation", "stale");
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

async function fetchStatus(poll = true) {
  const selected = selection;
  try {
    const { url, value } = await readFeed("status.json", pilot);
    if (selected !== selection) return;
    updateStudentFeed(url, value);
    renderStatus(value);
  } catch {
    if (selected === selection) {
      updateStudentFeed(null, {});
      pill("connection", pilot === "student" ? "Awaiting student evaluation" : "Teacher unavailable", "stale");
    }
  } finally {
    if (poll) window.setTimeout(fetchStatus, 2000);
  }
}

async function fetchReplay(poll = true) {
  const selected = selection;
  const view = pilot;
  try {
    const manifest = await readFeed("replay.json", view);
    if (selected !== selection) return;
    try {
      await queueReplay(manifest, selected);
    } catch (error) {
      if (!hosted || view !== "student" || manifest.value.archived) throw error;
      if (selected === selection) await queueReplay(await requestData("/student/replay.json"), selected);
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
  const page = new URL(window.location.href);
  page.searchParams.set("pilot", view);
  window.history.replaceState(null, "", page.href);
  byId("show-teacher").setAttribute("aria-pressed", String(view === "teacher"));
  byId("show-student").setAttribute("aria-pressed", String(view === "student"));
  liveUrl = null;
  liveDriver = null;
  camera.pause();
  camera.hidden = true;
  camera.removeAttribute("src");
  camera.load();
  if (playing) URL.revokeObjectURL(playing.blob);
  if (pending) URL.revokeObjectURL(pending.blob);
  playing = pending = null;
  byId("download-replay").hidden = true;
  byId("mode2-sticks").hidden = true;
  byId("replay-recorded").closest("div").hidden = true;
  byId("waiting").hidden = false;
  setText("waiting", "Loading replay");
  for (const id of ["replay-stage", "flight-gates", "replay-result", "replay-physics"]) setText(id, "—");
  applyStudentView();
  renderStatus({ course: "2026-practice", pending: true, phase: view === "teacher" ? "TEACHER" : "IMITATION", gates_per_race: 24 });
  if (refresh) {
    fetchStatus(false);
    fetchReplay(false);
  }
}

function updateStudentFeed(origin, info) {
  const next = pilot === "student" && info.active && info.live_url === "/live.png"
    ? new URL(info.live_url, origin).href : null;
  const changed = liveUrl !== next;
  liveUrl = next;
  liveDriver = info.live_driver || null;
  liveAge = info.live_age_seconds ?? 0;
  setText("live-gates", count(info.live_gates) + " / " + count(info.gates_per_race));
  applyStudentView();
  if (changed && liveUrl) refreshLiveFrame();
}

function applyStudentView() {
  const showLive = pilot === "student" && studentView === "live" && Boolean(liveUrl);
  const frame = byId("live-fpv");
  byId("student-source").hidden = pilot !== "student";
  byId("show-live").disabled = !liveUrl;
  byId("show-live").setAttribute("aria-pressed", String(showLive));
  byId("show-evaluation").setAttribute("aria-pressed", String(!showLive));
  frame.hidden = !showLive;
  camera.hidden = showLive || !playing;
  byId("mode2-sticks").hidden = showLive || !playing?.controls;
  byId("download-replay").hidden = showLive || !playing;
  byId("waiting").hidden = showLive || Boolean(playing);
  byId("replay-stage").hidden = showLive;
  byId("live-stage").hidden = !showLive;
  byId("replay-gate-count").hidden = showLive;
  byId("live-gate-count").hidden = !showLive;
  byId("replay-result").hidden = showLive;
  if (showLive) {
    camera.pause();
    const source = liveAge > 3 ? "Last training frame" : "Live training";
    setText("live-stage", liveDriver ? `${source} · ${liveDriver} · Rust` : `${source} · Rust`);
  }
}

function refreshLiveFrame() {
  if (pilot !== "student" || studentView !== "live" || !liveUrl) return;
  const frame = new URL(liveUrl);
  frame.searchParams.set("t", Date.now());
  byId("live-fpv").src = frame.href;
}

byId("show-live").addEventListener("click", () => {
  studentView = "live";
  applyStudentView();
  refreshLiveFrame();
});
byId("show-evaluation").addEventListener("click", () => {
  studentView = "evaluation";
  applyStudentView();
  if (playing) camera.play().catch(() => {});
});
window.setInterval(refreshLiveFrame, 320);

byId("show-teacher").addEventListener("click", () => selectPilot("teacher"));
byId("show-student").addEventListener("click", () => selectPilot("student"));

async function queueReplay({ url, value: info }, selected) {
  if (selected !== selection) return;
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
  if (selected !== selection || info.version === playing?.version || info.version === pending?.version) return;
  const blob = URL.createObjectURL(video.value.slice(0, video.value.size, "video/mp4"));
  if (pending) URL.revokeObjectURL(pending.blob);
  pending = { ...info, blob, controls };
  if (!playing || camera.ended) startReplay();
}

new ResizeObserver(drawHistory).observe(chart);
new ResizeObserver(drawFinishHistory).observe(finishChart);
if (new URLSearchParams(window.location.search).get("pilot") === "student") {
  selectPilot("student", false);
}
fetchStatus();
fetchReplay();
