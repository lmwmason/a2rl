# A2RL FPV 드론 레이싱

Rust로 구현한 실내 순환 코스 시뮬레이터입니다. 브라우저에서 비행을 조종하거나 Python에서 RGB 영상만 보는 정책을 학습할 수 있습니다. 두 실행 경로는 동일한 기체, 카메라 모델과 게이트 판정을 사용합니다.

## GUI로 비행하기

Rust 1.88 이상에서 프로젝트 루트에서 실행합니다.

```bash
cargo run --release
```

브라우저에서 <http://127.0.0.1:8765>를 열고 **비행 시작**을 누릅니다.

| 조종 | 키 |
| --- | --- |
| 스로틀 증가 / 감소 | W / S |
| 요 왼쪽 / 오른쪽 | A / D |
| 피치 앞으로 / 뒤로 | ↑ / ↓ |
| 롤 왼쪽 / 오른쪽 | ← / → |

Mode 2 Acro에서 요·피치·롤 키를 놓으면 각속도 입력이 중앙으로 돌아옵니다. 기체는 기울어진 자세를 유지합니다. 스로틀은 키를 놓아도 마지막 값에 머물고, 리셋하면 공중 출발에 필요한 호버 추력으로 돌아옵니다. 코스 지도에 다음 게이트와 현재 기체 위치가 표시됩니다. 마지막 게이트가 출발·결승선이고, 게이트는 순서대로 정방향으로 통과합니다.

## 기체와 카메라

| 항목 | 적용 |
| --- | --- |
| 기준 기체 | 2025 A2RL × DCL 카본 프레임 기체. 966 g, 약 30 × 28 × 10 cm, 5인치급 프로펠러, XNOVA Black Thunder 2207 2100 KV, 6S 1400 mAh |
| 동역학 | 240 Hz 6자유도 강체 적분, 4개 모터 응답 지연 및 추력, 자이로스코프 피드백을 근사한 기체 내부 각속도 제어, 이차 공기저항, 게이트 프레임의 스윕 충돌 검사 |
| 카메라 | 전방 단안 RGB, 155° × 115° 등거리 투영 근사. 시뮬레이션 제어·관측은 30 Hz. 실기체 자료의 해상도·프레임레이트는 820 × 616, 90 Hz |
| 코스 | 자체 제작 12개 게이트 순환 코스 |

질량과 시야각은 참가팀 자료를 따릅니다. 모터 시정수, 추력·항력 계수, 관성, 카메라 장착각과 충돌 반경은 추정값입니다. 실기체 이전에는 모터 스텝 응답, 고속 항력, 렌즈 캘리브레이션과 롤링셔터를 실측해 검증해야 합니다. Rust 렌더러는 CPU용 기하 장면입니다. CBGPU 환경에서 RTX/Vulkan을 사용할 수 있는지 확인하면 Isaac Sim의 타일 카메라를 같은 과제의 렌더링 비교에 사용할 수 있습니다.

- [TU Delft MonoRace 논문](https://arxiv.org/html/2601.15222v1)
- [Azhari 외 참가팀의 기체 사양](https://onlinelibrary.wiley.com/doi/10.1002/rob.70371)
- [NVIDIA Isaac Sim 시스템 요구 사항](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html)

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
cargo build --release --lib
python -m training.train --device cuda --steps 1000000 --envs 8 --eval-episodes 5
```

CPU에서도 `--device cpu`로 실행할 수 있습니다. 기본 96 × 72 RGB 카메라를 병렬로 실행하고, 리플레이는 각 프레임을 `uint8`로 한 번만 저장합니다. 조종 출력은 추력과 롤·피치·요 각속도입니다. 학습에는 공유 CNN, 프레임 스택 전체에 동일한 랜덤 시프트, 이중 Q와 다중 스텝 리턴을 쓰는 DrQ-v2 계열 기준선을 적용했습니다. 여러 게이트 앞에서 시작하는 연습을 섞고, 진행에 따라 시작선 출발 비중을 높입니다. 완주 기록과 충돌률을 출발선에서 따로 평가합니다.

```bash
python -m training.train --device cuda --evaluate runs/drq/best.pt --eval-episodes 5
```

`runs/drq/`에 설정, 평가 결과, 최근 및 최상위 가중치를 저장합니다. 해상도나 프레임 수를 변경했다면 평가 때 동일한 값을 지정합니다. 평가의 첫 에피소드는 정규 출발이고, 이후에는 고정 시드로 생성한 작은 출발 위치·노출 변화를 사용합니다.

- [DrQ-v2, ICLR 2022](https://arxiv.org/abs/2107.09645)
- [DreamerV3, Nature 2025](https://www.nature.com/articles/s41586-025-08744-2)
- [Ego-Foresight, ICLR 2026](https://openreview.net/pdf/955430a2085b49a89db12b6bf0d46d60a701caa0.pdf)
- [DR.Q, ICML 2026](https://proceedings.mlr.press/v306/lyu26f.html)

DreamerV3, Ego-Foresight와 DR.Q는 후속 비교 후보입니다. 현재 수렴 성능이나 알고리즘 간 우위를 측정한 결과는 없습니다.

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
cargo build --release --lib
python -m unittest discover -s tests -v
python -m training.train --device cpu --steps 160 --envs 2 --warmup 16 --replay 256 --batch 8 --eval-every 80 --run-dir /tmp/a2rl-smoke
git --git-dir=.a2rl-git --work-tree=. status
```
