# A2RL FPV 드론 레이싱

Rust로 구현한 실내 순환 코스 시뮬레이터입니다. 데스크톱 앱에서 FPV 비행을 조종하거나 Python에서 RGB 영상만 보는 정책을 학습할 수 있습니다. 두 실행 경로는 동일한 기체, 카메라 모델과 게이트 판정을 사용합니다.

## GUI로 비행하기

Rust 1.88 이상에서 프로젝트 루트에서 실행합니다.

```bash
cargo run --release
```

데스크톱 창에서 **Start flight**를 누르거나 `Space`를 누릅니다. 3·2·1 카운트다운 중 W/S로 스로틀을 미리 맞추면 GO와 함께 물리 스텝이 시작됩니다. 우상단 HUD에 현재 랩과 전체 경과 시간이 표시됩니다. Linux에서는 Wayland 또는 X11 세션에서 실행합니다.

| 조종 | 키 |
| --- | --- |
| 스로틀 증가 / 감소 | W / S |
| 요 왼쪽 / 오른쪽 | A / D |
| 피치 앞으로 / 뒤로 | ↑ / ↓ |
| 롤 왼쪽 / 오른쪽 | ← / → |
| 시작 / 일시정지 / 재개 | Space |
| 비행 리셋 | R |
| 전체 화면 전환 | F11 |
| HDZero 영상 열화 전환 | N |

Mode 2 Acro에서 요·피치·롤 키를 놓으면 각속도 입력이 중앙으로 돌아옵니다. 기체는 기울어진 자세를 유지합니다. 스로틀은 키를 놓아도 마지막 값에 머물고, 리셋하면 스로틀 0으로 지상에 돌아옵니다. 공식 Grand Challenge 형식대로 11개 게이트를 두 번 통과합니다. 타이머는 첫 1번 게이트에서 시작하고 두 번째 11번 게이트에서 멈춥니다. 학습 에피소드에서는 반복 주행을 위해 2랩 이후에도 비행을 이어갑니다.

## 기체와 카메라

| 항목 | 적용 |
| --- | --- |
| 기준 기체 | 2025 A2RL × DCL 카본 프레임 기체. 966 g, 약 30 × 28 × 10 cm, 5인치급 프로펠러, XNOVA Black Thunder 2207 2100 KV, 6S 1400 mAh |
| 동역학 | 240 Hz 6자유도 강체 적분, 4개 모터 응답 지연 및 추력, 자이로스코프 피드백을 근사한 기체 내부 각속도 제어, 이차 공기저항, 게이트 프레임의 스윕 충돌 검사 |
| 카메라 | 전방 단안 RGB. 실제 IMX219 로그의 어안 렌즈 내부 파라미터와 장착 피치 51.33°를 적용합니다. 실기체는 820 × 616, 90 Hz이며 시뮬레이션 관측과 스틱 입력은 30 Hz입니다. |
| 코스 | 2025 Grand Challenge 결승 공개 비행 계획의 11개 게이트, 1.5 m 개구부, 두 개의 더블 프레임과 5→6 스플릿 S |

실제 게이트 사진을 펼쳐 전면과 후면의 재질로 사용했습니다. 홀의 바닥, 커튼, 천장과 조명은 사진을 참고해 모델링했습니다. 천장 높이와 조명 배치는 재구성값이며, 현장의 3D 스캔, 카메라 롤링셔터, 모션 블러와 노출 응답은 재현 범위 밖입니다. HDZero 효과는 약한 영상 링크에서 나타나는 국소 픽셀 열화를 근사합니다.

### 실제 비행 로그와 비교

MonoRace Log 8의 대회 비행에서 공개된 모터 회전수·IMU·EKF로 몸체 수직축 비력을 비교했습니다. 55–69.8초를 20ms 구간으로 나눠 보정에 쓰지 않은 후반 367개 구간에서 RMSE가 **29.81 → 3.41 m/s²**로 줄었습니다. 이 테스트는 측정된 모터 속도를 입력으로 사용하므로 모터 명령 응답, 관성 및 전체 6자유도 궤적의 검증은 별도로 필요합니다. 모터 회전 상한과 30ms 지연은 참가팀 M16 설정을 사용합니다.

```bash
cargo build --release --lib --no-default-features
python -m tools.check_flight_log path/to/imu.csv path/to/ekf.csv
```

- [Grand Challenge 비행 계획](https://github.com/tudelft/optimal_quad_control_RL/blob/a2rl2025/flight_plans/grand_challenge.json)
- [4TU MonoRace 비행 로그와 카메라 캘리브레이션](https://doi.org/10.4121/039abd82-094c-4c93-baa1-e7446f562e0e)
- [TU Delft MonoRace 논문](https://arxiv.org/html/2601.15222v1)
- [Azhari 외 참가팀의 기체 사양](https://onlinelibrary.wiley.com/doi/10.1002/rob.70371)
- [HDZero 픽셀 단위 열화 설명](https://www.hd-zero.com/motion-clarity)

## 보상

- 다음 게이트에 가까워지면 양의 보상, 같은 거리를 물러서면 음의 보상을 줍니다. 게이트 진행 보상은 구간당 제한합니다.
- 유효한 게이트 통과는 +2, 랩 완주는 기본 +20입니다. 기준 24초 대비 빠를수록 추가 보상이 최대 +8까지 올라갑니다. 같은 세션에서 최고 기록을 줄이면 단축 1초당 0.18, 최대 +3을 추가합니다. 연습 중간 지점에서 출발하면 결승선을 처음 지나고부터 랩을 잽니다.
- 경과 시간은 초당 −0.20, 충돌은 −18입니다. 다음 게이트 미통과가 14초 지속되면 −10으로 종료합니다. 충돌 없이 계속 완주하면 180초까지 랩을 반복합니다.
- 정책의 입력은 연속한 RGB 4장입니다. 위치, 속도, 자세, 게이트 좌표와 GUI 오버레이는 정책에 전달하지 않습니다. 시뮬레이터 내부 상태로 계산한 보상은 학습에 쓰고, 랩 타임과 충돌률은 별도로 평가합니다.

## 학습 및 평가

PyTorch에서 그 환경의 NVIDIA 드라이버와 호환되는 CUDA 빌드를 준비합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r training/requirements.txt
cargo build --release --lib --no-default-features
python -m training.train --device cuda --steps 1000000 --envs 8 --eval-episodes 5
```

CPU에서도 `--device cpu`로 실행할 수 있습니다. 기본 96 × 72 RGB 카메라를 병렬로 실행하고, 리플레이는 각 프레임을 `uint8`로 한 번만 저장합니다. 정책의 네 출력은 실제 Mode 2 스틱 순서로 왼쪽 X·Y와 오른쪽 X·Y입니다. 데스크톱과 동일한 변환으로 요, 스로틀, 롤, 피치를 조종합니다. 학습에는 공유 CNN, 프레임 스택 전체에 동일한 랜덤 시프트, 이중 Q와 다중 스텝 리턴을 쓰는 DrQ-v2 계열 기준선을 적용했습니다. 지상 론치와 스플릿 S·슬라럼·급턴 앞에서 출발하는 에피소드를 섞고, 진행에 따라 지상 출발 비중을 높입니다. 일부 학습 에피소드에 HDZero 픽셀 열화를 적용하며 `--hdzero-max`로 강도를 조절합니다. 출발선 평가에서는 첫 게이트, 스플릿 S, 슬라럼 및 완주 통과율을 기록합니다.

```bash
python -m training.train --device cuda --evaluate runs/drq/best.pt --eval-episodes 5
```

`runs/drq/`에 설정, 평가 결과, 최근 및 최상위 가중치를 저장합니다. 해상도나 프레임 수를 변경했다면 평가 때 동일한 값을 지정합니다. 평가의 첫 에피소드는 정규 출발이고, 이후에는 고정 시드로 생성한 작은 출발 위치·노출 변화를 사용합니다.

- [DrQ-v2, ICLR 2022](https://arxiv.org/abs/2107.09645)
- [DreamerV3, Nature 2025](https://www.nature.com/articles/s41586-025-08744-2)
- [Ego-Foresight, ICLR 2026](https://openreview.net/pdf/955430a2085b49a89db12b6bf0d46d60a701caa0.pdf)
- [DR.Q, ICML 2026](https://proceedings.mlr.press/v306/lyu26f.html)

DreamerV3, Ego-Foresight와 DR.Q는 후속 비교 후보입니다. 현재 수렴 성능이나 알고리즘 간 우위를 측정한 결과는 없습니다.

## 하드코딩 비행 녹화

아래 예제는 고정 웨이포인트를 따라 실제 물리 스텝으로 지상 론치부터 두 바퀴, 총 22개 게이트를 통과합니다. 웨이포인트 검증에는 시뮬레이터 상태를 사용합니다. FPV 학습 정책은 RGB만 받습니다.

```bash
cargo run --release --no-default-features --example grand_challenge -- runs/grand-challenge-cpu.mp4
cargo run --release --no-default-features --example grand_challenge -- --poses runs/grand-challenge.poses
```

CUDA 렌더러는 Rust에서 내보낸 자세, 렌즈 및 게이트 형상을 픽셀별로 렌더링합니다. GPU 4의 Quadro RTX 5000에서는 다음과 같이 빌드합니다. 마지막 인수는 HDZero 열화 강도입니다. 영상의 비행 구간은 2배속입니다.

```bash
nvcc -O3 -arch=sm_75 tools/render_fpv.cu -o render_fpv
./render_fpv runs/grand-challenge.poses assets/grand-challenge-gate.rgb 0.32 |
  ffmpeg -f rawvideo -pixel_format rgb24 -video_size 820x616 -framerate 30 \
    -i - -an -c:v libx264 -pix_fmt yuv420p runs/grand-challenge-gpu.mp4
```

## 실제 비행 모방학습 후 강화학습

4TU의 MAVLab Log 5는 2025 Grand Challenge 실제 비행이며, Log 7은 비행 단위 홀드아웃 평가에만 사용합니다. 원본은 직접 모터 명령을 쓰므로 Mode 2 스틱 라벨은 저장돼 있지 않습니다. 시간 정렬한 모터 회전수와 자이로 측정값에서 근사 스틱 목표를 만들고, 범위를 넘는 값은 클리핑하고 가중치를 낮춥니다. 시뮬레이터 데모에는 적용 직전의 정확한 스틱을 저장합니다. 정책 입력은 두 경우 모두 RGB 네 프레임입니다.

```bash
python -m pip install Pillow
python -m tools.prepare_monorace_demos runs/monorace
cargo run --release --no-default-features --example grand_challenge -- --dataset runs/monorace/sim-clean 0
cargo run --release --no-default-features --example grand_challenge -- --dataset runs/monorace/sim-noise 7
python -m training.imitate --real runs/monorace/log5.npz --held-out runs/monorace/log7.npz \
  --sim runs/monorace/sim-clean runs/monorace/sim-noise --out runs/bc
python -m training.train --device cuda --initialize runs/bc/best.pt \
  --demo-sim runs/monorace/sim-clean runs/monorace/sim-noise --run-dir runs/bc-rl
```

사전학습 체크포인트는 시뮬레이터 롤아웃을 기준으로 고릅니다. 강화학습에서는 초기 전이를 사전학습 정책으로 모으고, Q 함수가 준비되는 동안 정책 업데이트를 지연합니다. 이후 시뮬레이터 데모에 대한 보조 손실을 점진적으로 줄입니다. 홀드아웃 스틱 오차, 첫 게이트 통과율, 충돌률 및 2랩 완주율을 기록합니다.

- [공개 대회 비행 로그와 데이터 라이선스](https://doi.org/10.4121/039abd82-094c-4c93-baa1-e7446f562e0e)

### 1번 게이트에서 2번 게이트로 이어지는 구간 훈련

기존 최상위 정책에서 추가로 500만 전이를 학습합니다. 일반 출발과 게이트 연습에 1번 게이트 직후의 이동 중 출발을 섞습니다. 정규 출발의 `gate_two_rate`와 `crash_rate`를 기록하고, 구간 연습의 `gate_exit_to_two_rate`도 함께 기록합니다.

```bash
python -m training.train --device cuda --initialize runs/bc-rl/best.pt \
  --focus-gate-two --steps 5000000 --eval-every 25000 --eval-episodes 5 \
  --checkpoint-every 100000 --run-dir runs/gate-two-5m
```

`training-state.zip`에 정책, 옵티마이저, 리플레이와 난수 상태를 저장합니다. 재개할 때는 같은 환경·리플레이 설정으로 `--initialize` 인수 대신 `--resume runs/gate-two-5m/training-state.zip`을 사용합니다. 시뮬레이터는 새 에피소드에서 시작하고, 중단 시 마지막 카메라 프레임을 유지해 이전 에피소드의 리플레이가 새 프레임과 섞이지 않게 합니다.

### 완주를 익힌 뒤 속도 최적화

MonoRace Log 5의 EKF 위치에서 연속 게이트 간 비행선을 추출했습니다. 시뮬레이터 교사는 이 비행선과 게이트 통과 방향을 기준으로 론치, 스플릿 S의 상단→하단 진입, 슬라럼과 급턴을 수행합니다. 실제 팀의 속도나 순간 자세를 그대로 재생하기보다, 같은 구간을 느린 속도로 통과하며 실제 Mode 2 입력을 계산합니다. 학생이 비행 중 벗어난 상태에도 교사 입력을 붙여 다시 학습합니다. 학생의 입력은 최근 RGB 여덟 프레임입니다.

```bash
cargo build --release --lib --no-default-features
python -m training.racing_school --device cuda --rounds 0 --live \
  --consolidation-rounds 6 --out runs/racing-school
test -s runs/racing-school/qualified.pt &&
  python -m training.train --device cuda --initialize runs/racing-school/qualified.pt \
    --stack 8 --steps 5000000 --batch 96 --live --official-eval --eval-episodes 3 \
    --eval-every 25000 --checkpoint-every 100000 --run-dir runs/speed-rl
```

첫 2랩 평가 통과 이후 최소 6라운드를 더 학습합니다. 이후에도 교사 개입 없이 연속 두 차례 평가에서 출발 위치와 노출이 다른 3회 모두 2랩을 완주해야 `qualified.pt`가 저장됩니다. 커리큘럼은 정규 출발 평가에서 현재 목표 게이트를 안정적으로 통과하면 다음 게이트로 넘어갑니다. `imitation-state.zip`은 옵티마이저, 교사 데이터와 커리큘럼 상태를 보관하며 `--resume`으로 복원합니다.

속도 강화학습에는 교사 행동을 따르는 손실이나 정해진 비행선을 두지 않습니다. 초기 완주 모델을 보존하고, 정책은 완주율, 충돌률, 공식 2랩 평균 시간 순으로 비교합니다.

### 데스크톱 학습 관전

```bash
cargo run --release -- --watch cbgpu-gpu4
```

`--live`로 시작한 GPU 학습의 FPV 샘플과 최근 평가 기록을 SSH로 읽습니다. 화면의 **Teacher** 또는 **Student** 표시는 해당 수집 프레임에서 실제로 조종한 쪽입니다. 라이브 프레임이 없거나 작업이 종료된 경우에는 체크포인트 재생임을 화면에 표시합니다. 창을 닫아도 GPU 학습은 계속됩니다.

## CBGPU GPU 4에서 실행

승인된 예약 시간에 등록된 SSH 별칭으로 GPU와 드라이버를 확인합니다.

```bash
ssh cbgpu-gpu4 nvidia-smi
```

이 작업공간에서는 Git 메타데이터가 `.a2rl-git/`에 있습니다. 커밋한 소스만 컨테이너로 복사하려면 다음 명령을 실행합니다.

```bash
git --git-dir=.a2rl-git --work-tree=. archive HEAD | ssh cbgpu-gpu4 'mkdir -p ~/a2rl-racing && tar -x -C ~/a2rl-racing'
ssh cbgpu-gpu4
cd ~/a2rl-racing
```

컨테이너에서 위 학습 명령을 실행합니다. 기존 예약 환경의 CUDA PyTorch가 있으면 그 환경을 사용할 수 있습니다. 교내망 접속은 `cbgpu-gpu4-school` 별칭을 사용합니다.

## 확인 명령

```bash
cargo test
cargo build --release --lib --no-default-features
python -m unittest discover -s tests -v
python -m training.train --device cpu --steps 160 --envs 2 --warmup 16 --replay 256 --batch 8 --eval-every 80 --run-dir /tmp/a2rl-smoke
git --git-dir=.a2rl-git --work-tree=. status
```
