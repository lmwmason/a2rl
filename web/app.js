const el = id => document.getElementById(id);
const canvas = el('camera');
const context = canvas.getContext('2d', { alpha: false });
const map = el('course').getContext('2d');
const keys = new Set();
const controls = new Set(['KeyW', 'KeyS', 'KeyA', 'KeyD', 'KeyQ', 'KeyE', 'Space', 'ShiftLeft', 'ShiftRight']);
let config, pixels, running = false, busy = false, generation = 0, finished = false;

const point = ([x, y]) => [25 + (x + 8) * 5.35, 195 - (y + 12) * 5.75];
const clock = seconds => seconds == null ? '—' : seconds.toFixed(2) + ' s';
const held = code => Number(keys.has(code));

function drawMap(stats) {
  map.clearRect(0, 0, 320, 220);
  map.strokeStyle = '#223449';
  map.lineWidth = 1;
  for (let x = 20; x < 320; x += 20) {
    map.beginPath(); map.moveTo(x, 0); map.lineTo(x, 220); map.stroke();
  }
  for (let y = 20; y < 220; y += 20) {
    map.beginPath(); map.moveTo(0, y); map.lineTo(320, y); map.stroke();
  }
  map.beginPath();
  config.gates.forEach(gate => map.lineTo(...point(gate)));
  map.closePath();
  map.strokeStyle = '#676098';
  map.lineWidth = 2;
  map.stroke();
  config.gates.forEach((gate, index) => {
    const [x, y] = point(gate);
    map.beginPath();
    map.arc(x, y, index === stats.gate ? 6 : 3.5, 0, Math.PI * 2);
    map.fillStyle = index === stats.gate ? '#62ecbe' : '#a88af9';
    map.fill();
    if (index === stats.gate) {
      map.font = 'bold 11px monospace';
      map.fillText(String(index + 1).padStart(2, '0'), x + 9, y - 8);
    }
  });
  map.save();
  map.translate(...point(stats.position));
  map.rotate(-stats.yaw);
  map.beginPath();
  map.moveTo(8, 0); map.lineTo(-5, 5); map.lineTo(-5, -5); map.closePath();
  map.fillStyle = '#fff';
  map.fill();
  map.restore();
}

function paint(data, stats) {
  for (let src = 0, dst = 0; src < data.length; src += 3, dst += 4) {
    pixels.data[dst] = data[src];
    pixels.data[dst + 1] = data[src + 1];
    pixels.data[dst + 2] = data[src + 2];
    pixels.data[dst + 3] = 255;
  }
  context.putImageData(pixels, 0, 0);
  drawMap(stats);
  el('speed').textContent = stats.speed.toFixed(1);
  el('height').textContent = stats.height.toFixed(1);
  el('gate').textContent = String(stats.gate + 1).padStart(2, '0') + ' / ' + config.gates.length;
  el('laps').textContent = stats.laps;
  for (const field of ['time', 'last', 'best']) el(field).textContent = clock(stats[field]);

  finished = stats.status !== 0;
  if (finished) running = false;
  const messages = [running ? '' : '비행 시작 버튼을 누르세요.', '게이트와 충돌했습니다.', '코스 경계 또는 바닥과 충돌했습니다.', '비행 시간이 끝났습니다.', '14초 안에 다음 게이트를 통과하지 못했습니다.'];
  el('notice').textContent = messages[stats.status];
  el('notice').hidden = running;
  el('status').textContent = finished ? (stats.status >= 3 ? '시간 종료' : '충돌') : running ? '비행 중' : '대기';
  el('play').textContent = running ? '일시정지' : '비행 시작';
  el('play').disabled = finished;
}

async function request(path, body) {
  const current = generation;
  const start = performance.now();
  busy = true;
  try {
    const options = body === undefined ? {} : {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    };
    const response = await fetch(path, options);
    if (!response.ok) throw new Error('HTTP ' + response.status);
    const data = new Uint8Array(await response.arrayBuffer());
    if (current === generation) paint(data, JSON.parse(response.headers.get('x-race')));
  } catch (error) {
    running = false;
    el('status').textContent = '연결 오류';
    el('notice').hidden = false;
    el('notice').textContent = '로컬 시뮬레이터에 연결할 수 없습니다.';
    console.error(error);
  } finally {
    if (current === generation) {
      busy = false;
      if (running) setTimeout(loop, Math.max(0, 1000 / 30 - (performance.now() - start)));
    }
  }
}
function loop() {
  if (!running || busy) return;
  request('/fly', [
    held('Space') - Number(keys.has('ShiftLeft') || keys.has('ShiftRight')),
    held('KeyD') - held('KeyA'), held('KeyW') - held('KeyS'), held('KeyQ') - held('KeyE'),
  ]);
}

window.addEventListener('keydown', (event) => {
  if (!controls.has(event.code) || !running) return;
  event.preventDefault();
  keys.add(event.code);
});
window.addEventListener('keyup', event => keys.delete(event.code));
window.addEventListener('blur', () => keys.clear());
el('play').addEventListener('click', () => {
  if (finished) return;
  running = !running;
  keys.clear();
  if (running) {
    el('play').blur();
    loop();
  } else if (!busy) request('/frame');
});
el('reset').addEventListener('click', () => {
  running = false;
  keys.clear();
  // Wait for the in-flight command before resetting its race.
  const tryReset = () => {
    if (busy) return setTimeout(tryReset, 20);
    generation++;
    request('/reset', null);
  };
  tryReset();
});

try {
  const response = await fetch('/config');
  if (!response.ok) throw new Error('HTTP ' + response.status);
  config = await response.json();
  canvas.width = config.width;
  canvas.height = config.height;
  pixels = context.createImageData(config.width, config.height);
  await request('/frame');
  el('reset').disabled = false;
} catch (error) {
  el('status').textContent = '연결 오류';
  el('notice').textContent = '로컬 시뮬레이터를 시작해 주세요.';
  console.error(error);
}
