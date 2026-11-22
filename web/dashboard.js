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
const videoContext = camera.getContext("2d");
const chart = byId("history");
const chartContext = chart.getContext("2d");
const trend = { values: [], key: "" };
let lastFrame = null;

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

function drawFrame(info, pixels) {
  if (!info.width || !info.height || pixels.length !== info.width * info.height * 3) return;
  if (info.frame_at === lastFrame) return;
  lastFrame = info.frame_at;
  if (camera.width !== info.width || camera.height !== info.height) {
    camera.width = info.width;
    camera.height = info.height;
  }
  const image = videoContext.createImageData(info.width, info.height);
  for (let i = 0, j = 0; i < pixels.length; i += 3, j += 4) {
    image.data[j] = pixels[i];
    image.data[j + 1] = pixels[i + 1];
    image.data[j + 2] = pixels[i + 2];
    image.data[j + 3] = 255;
  }
  videoContext.putImageData(image, 0, 0);
  camera.hidden = false;
  byId("waiting").hidden = true;
  setText("driver", { Teacher: "교사", Student: "학생", Policy: "정책" }[info.driver] || "—");
  setText("flight-gates", count(info.gates) + " 게이트");
  setText("resolution", info.width + " × " + info.height);
}

function render(info, pixels) {
  const connected = info.connection === "connected" && Date.now() / 1000 - info.snapshot_at < 6;
  const age = info.frame_at ? Date.now() / 1000 - info.frame_at : Infinity;
  const speedTraining = info.phase === "SPEED RL";
  if (connected) {
    pill("connection", info.active ? "GPU 연결 · 학습 중" : "GPU 연결 · 학습 종료", info.active ? "live" : "");
  } else {
    pill("connection", info.connection === "connecting" ? "GPU 연결 중" : "GPU 연결 끊김", "stale");
  }
  if (connected && info.active && age < 2) {
    pill("video-state", "실시간", "live");
  } else if (lastFrame !== null || pixels.length) {
    pill("video-state", "마지막 수신 프레임", "stale");
  } else {
    pill("video-state", "프레임 대기");
  }
  setText("phase", info.target_beaten ? "목표 달성" : speedTraining ? "강화학습" : info.phase ? "모방학습" : "—");
  setText("progress-label", speedTraining ? "강화학습 스텝" : "모방학습 라운드");
  setText("progress", count(speedTraining ? info.steps : info.round));
  setText("steps", count(info.steps));
  setText("finish-rate", percent(info.finish_rate));
  setText("collision-rate", percent(info.collision_rate));
  setText("best-time", Number.isFinite(info.best_two_lap_seconds) ? info.best_two_lap_seconds.toFixed(2) + " 초" : "—");
  setText("target", (info.target_two_lap_seconds ?? 16.56).toFixed(2) + " 초 미만");
  drawFrame(info, pixels);

  const values = Array.isArray(info.recent_gates) ? info.recent_gates : [];
  const key = values.join(",");
  if (key !== trend.key) {
    trend.key = key;
    trend.values = values;
    drawHistory();
  }
  setText("history-count", values.length ? "최근 " + count(values.length) + "회" : "평가 대기");
}

async function fetchSnapshot() {
  try {
    const response = await fetch("snapshot.bin", { cache: "no-store" });
    if (!response.ok) throw new Error("Snapshot unavailable");
    const buffer = await response.arrayBuffer();
    const header = new DataView(buffer);
    const metaLength = header.getUint32(0, true);
    const pixelLength = header.getUint32(4, true);
    if (8 + metaLength + pixelLength !== buffer.byteLength) throw new Error("Invalid snapshot");
    const info = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, metaLength)));
    render(info, new Uint8Array(buffer, 8 + metaLength, pixelLength));
  } catch {
    pill("connection", "로컬 데이터 연결 대기", "stale");
    if (lastFrame !== null) pill("video-state", "마지막 수신 프레임", "stale");
  } finally {
    window.setTimeout(fetchSnapshot, 120);
  }
}

new ResizeObserver(drawHistory).observe(chart);
fetchSnapshot();
