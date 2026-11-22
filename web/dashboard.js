"use strict";

const byId = (id) => document.getElementById(id);
const setText = (id, value) => {
  const element = byId(id);
  if (element.textContent !== value) element.textContent = value;
};
const number = new Intl.NumberFormat("en-US");
const count = (value) => Number.isFinite(value) ? number.format(value) : "—";
const percent = (value) => Number.isFinite(value) ? count(Math.round(value * 100)) + "%" : "—";
const camera = byId("fpv");
const chart = byId("history");
const chartContext = chart.getContext("2d");
const trend = { values: [], key: "" };
let playing = null;
let pending = null;
const hosted = !["127.0.0.1", "localhost", "[::1]"].includes(window.location.hostname);
const feeds = hosted ? [
  "https://a2rl-fpv-feed.server.cbgpu.com/",
  "https://beam-snapshot-down-museums.trycloudflare.com/",
] : ["/"];

async function dataFile(name, manifestUrl) {
  const urls = manifestUrl ? [new URL(name, manifestUrl)] : feeds.map((feed) => feed + name);
  for (const url of urls) {
    try {
      const response = await fetch(url, { cache: "no-store", credentials: "omit", signal: AbortSignal.timeout(5000) });
      if (response.ok) return response;
    } catch {
      // Try the GPU-hosted feed if the managed feed is unavailable.
    }
  }
  return null;
}

function pill(id, label, state = "") {
  setText(id, label);
  byId(id).className = state ? "pill " + state : "pill";
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
      const y = height - 3 - (height - 6) * Math.max(0, Math.min(value, 22)) / 22;
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

function startReplay() {
  if (pending) {
    if (playing) URL.revokeObjectURL(playing.blob);
    playing = pending;
    pending = null;
    camera.src = playing.blob;
    camera.hidden = false;
    const download = byId("download-replay");
    download.href = playing.blob;
    download.download = playing.phase === "SPEED RL" ? `a2rl-rl-${playing.steps}.mp4`
      : `a2rl-imitation-round-${playing.round}.mp4`;
    download.hidden = false;
    byId("waiting").hidden = true;
    const stage = playing.phase === "SPEED RL" ? "Reinforcement learning · " + count(playing.steps) + " steps"
      : playing.round ? "Imitation · Round " + count(playing.round) : "Imitation checkpoint";
    setText("replay-stage", stage);
    setText("flight-gates", count(playing.gates) + " / 22");
    setText("replay-result", { 1: "Crash", 2: "Timed out", 3: "Stalled", 4: "Finished two laps" }[playing.status] || "—");
    pill("video-state", "Student model replay", "live");
  } else if (playing) {
    camera.currentTime = 0;
  }
  if (playing) camera.play().catch(() => {});
}
camera.addEventListener("ended", startReplay);

function renderStatus(info) {
  const connected = info.connection === "connected" && Date.now() / 1000 - info.snapshot_at < 15;
  const speedTraining = info.phase === "SPEED RL";
  if (connected) {
    pill("connection", info.active ? "GPU online · Training" : "GPU online · Training stopped", info.active ? "live" : "");
  } else {
    pill("connection", info.connection === "connecting" ? "Connecting to GPU" : "GPU feed offline", "stale");
  }
  setText("phase", info.target_beaten ? "Target reached" : speedTraining ? "Reinforcement learning" : info.phase ? "Imitation learning" : "—");
  setText("progress-label", speedTraining ? "Reinforcement learning steps" : "Imitation round");
  setText("progress", count(speedTraining ? info.steps : info.round));
  setText("steps", count(info.steps));
  setText("finish-rate", percent(info.finish_rate));
  setText("collision-rate", percent(info.collision_rate));
  setText("best-time", Number.isFinite(info.best_two_lap_seconds) ? info.best_two_lap_seconds.toFixed(2) + " s" : "—");
  setText("target", "Under " + (info.target_two_lap_seconds ?? 16.56).toFixed(2) + " s");

  const values = Array.isArray(info.recent_gates) ? info.recent_gates : [];
  const key = values.join(",");
  if (key !== trend.key) {
    trend.key = key;
    trend.values = values;
    drawHistory();
  }
  setText("history-count", values.length ? "Last " + count(values.length) + " evaluations" : "Awaiting evaluation");
}

async function fetchStatus() {
  try {
    const response = await dataFile("status.json");
    if (!response) throw new Error("Status unavailable");
    renderStatus(await response.json());
  } catch {
    pill("connection", "Waiting for training feed", "stale");
  } finally {
    window.setTimeout(fetchStatus, 2000);
  }
}

async function fetchReplay() {
  try {
    const response = await dataFile("replay.json");
    if (!response) return;
    const info = await response.json();
    if (info.version === playing?.version || info.version === pending?.version) return;
    if (!/^replay-[0-9]+\.mp4$/.test(info.url)) return;
    const video = await dataFile(info.url, response.url);
    if (!video) return;
    const blob = URL.createObjectURL(await video.blob());
    if (pending) URL.revokeObjectURL(pending.blob);
    pending = { ...info, blob };
    if (!playing || camera.ended) startReplay();
    else pill("video-state", "Next replay ready", "live");
  } catch {
    if (!playing) pill("video-state", "Preparing replay");
  } finally {
    window.setTimeout(fetchReplay, 3000);
  }
}

new ResizeObserver(drawHistory).observe(chart);
fetchStatus();
fetchReplay();
