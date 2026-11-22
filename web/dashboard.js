"use strict";

const byId = (id) => document.getElementById(id);
const setText = (id, value) => {
  const element = byId(id);
  if (element.textContent !== value) element.textContent = value;
};
const number = new Intl.NumberFormat("ko-KR");
const count = (value) => Number.isFinite(value) ? number.format(value) : "—";
const percent = (value) => Number.isFinite(value) ? count(Math.round(value * 100)) + "%" : "—";
const camera = byId("fpv");
const chart = byId("history");
const chartContext = chart.getContext("2d");
const trend = { values: [], key: "" };
let playing = null;
let pending = null;

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
  chartContext.strokeStyle = chartContext.fillStyle = "#74e8ac";
  chartContext.lineWidth = 2;
  chartContext.beginPath();
  trend.values.forEach((value, index) => {
    const x = 3 + (width - 6) * index / Math.max(1, trend.values.length - 1);
    const y = height - 3 - (height - 6) * Math.max(0, Math.min(value, 22)) / 22;
    if (index === 0) {
      chartContext.moveTo(x, y);
      if (trend.values.length === 1) chartContext.fillRect(x - 2, y - 2, 4, 4);
    } else {
      chartContext.lineTo(x, y);
    }
  });
  chartContext.stroke();
}

function startReplay() {
  if (pending) {
    if (playing) URL.revokeObjectURL(playing.blob);
    playing = pending;
    pending = null;
    camera.src = playing.blob;
    camera.hidden = false;
    byId("waiting").hidden = true;
    const stage = playing.phase === "SPEED RL" ? "강화학습 · " + count(playing.steps) + " 스텝"
      : playing.round ? "모방학습 · " + count(playing.round) + " 라운드" : "모방학습 체크포인트";
    setText("replay-stage", stage);
    setText("flight-gates", count(playing.gates) + " / 22");
    setText("replay-result", { 1: "충돌", 2: "시간 초과", 3: "진행 정체", 4: "2랩 완주" }[playing.status] || "—");
    pill("video-state", "학생 모델 리플레이", "live");
  } else if (playing) {
    camera.currentTime = 0;
  }
  if (playing) camera.play().catch(() => {});
}
camera.addEventListener("ended", startReplay);

function renderStatus(info) {
  const connected = info.connection === "connected" && Date.now() / 1000 - info.snapshot_at < 6;
  const speedTraining = info.phase === "SPEED RL";
  if (connected) {
    pill("connection", info.active ? "GPU 연결 · 학습 중" : "GPU 연결 · 학습 종료", info.active ? "live" : "");
  } else {
    pill("connection", info.connection === "connecting" ? "GPU 연결 중" : "GPU 연결 끊김", "stale");
  }
  setText("phase", info.target_beaten ? "목표 달성" : speedTraining ? "강화학습" : info.phase ? "모방학습" : "—");
  setText("progress-label", speedTraining ? "강화학습 스텝" : "모방학습 라운드");
  setText("progress", count(speedTraining ? info.steps : info.round));
  setText("steps", count(info.steps));
  setText("finish-rate", percent(info.finish_rate));
  setText("collision-rate", percent(info.collision_rate));
  setText("best-time", Number.isFinite(info.best_two_lap_seconds) ? info.best_two_lap_seconds.toFixed(2) + " 초" : "—");
  setText("target", (info.target_two_lap_seconds ?? 16.56).toFixed(2) + " 초 미만");

  const values = Array.isArray(info.recent_gates) ? info.recent_gates : [];
  const key = values.join(",");
  if (key !== trend.key) {
    trend.key = key;
    trend.values = values;
    drawHistory();
  }
  setText("history-count", values.length ? "최근 " + count(values.length) + "회" : "평가 대기");
}

async function fetchStatus() {
  try {
    const response = await fetch("status.json", { cache: "no-store" });
    if (!response.ok) throw new Error("Status unavailable");
    renderStatus(await response.json());
  } catch {
    pill("connection", "로컬 데이터 연결 대기", "stale");
  } finally {
    window.setTimeout(fetchStatus, 1000);
  }
}

async function fetchReplay() {
  try {
    const response = await fetch("replay.json", { cache: "no-store" });
    if (!response.ok) return;
    const info = await response.json();
    if (info.version === playing?.version || info.version === pending?.version) return;
    const video = await fetch(info.url);
    if (!video.ok) return;
    const blob = URL.createObjectURL(await video.blob());
    if (pending) URL.revokeObjectURL(pending.blob);
    pending = { ...info, blob };
    if (!playing || camera.ended) startReplay();
    else pill("video-state", "다음 리플레이 준비됨", "live");
  } catch {
    if (!playing) pill("video-state", "리플레이 준비 중");
  } finally {
    window.setTimeout(fetchReplay, 1000);
  }
}

new ResizeObserver(drawHistory).observe(chart);
fetchStatus();
fetchReplay();
