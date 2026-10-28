# PPO 전용 실험

`common.py`, `reward.py`, A2C 학습·평가·재생 파일을 수정하지 않고 PPO를 비교합니다.

## 파일별 역할

| 파일 | 역할 |
|---|---|
| `ppo_common.py` | PPO 설정, 환경 생성, 설정 검증, 실행 정보 |
| `ppo_reward.py` | PPO에서만 사용하는 보상 계수와 계산 |
| `train_ppo.py` | 학습, 체크포인트, CSV·TensorBoard 기록 |
| `evaluate_ppo.py` | 원래 시작점의 결정적·확률적 평가와 행동 기록 |
| `watch_ppo.py` | 저장한 PPO 모델을 기존 게임 그림으로 관찰 |

환경은 `RealtimeMansionEnv`, 관측은 636차원, 학습용 시작 상태는 기존
`Stage1RouteCurriculum`, 관측 보강은 `TaskGoalObservation`을 재사용합니다.
이 공용 두 파일을 바꾸면 A2C와 PPO 모두에 영향을 주므로, 비교 중에는 고정합니다.
PPO 실행에는 `rl.common`, `rl.reward`, A2C 학습·평가 모듈을 가져오지 않습니다.
기존 stable-baselines3 설치에 PPO도 포함되어 추가 설치는 필요하지 않습니다.

## 첫 비교 조건

환경·보상·관측·평가 seed·커리큘럼과 학습량은 직전 A2C와 같습니다.
`ppo_common.py`의 `CONFIG['ppo']`에서만 학습 조건을 바꿉니다.

- 환경 4개, `n_steps=100`: 업데이트마다 경험 400개 수집
- `batch_size=100`: 한 번의 학습에 경험 100개 사용
- `n_epochs=10`: 수집한 경험을 최대 10번 반복 학습
- 학습률 0.0002, gamma 0.9994, GAE lambda 1.0, 엔트로피 계수 0.01
- 정책/가치 신경망 각각 [64, 64], Tanh, CPU 스레드 1개
- PPO clip_range 0.2, advantage 정규화 사용, 기본 Adam 최적화
- seed당 600,000 환경 스텝, 200,000·400,000·600,000에 저장 및 평가

총 스텝은 네 환경의 합계이며 준비 경로의 프레임 수는 포함하지 않습니다.
PPO는 같은 경험을 반복 학습하므로 A2C보다 벽시계 시간이 길어질 수 있습니다.
이것은 알고리즘과 그 기본 최적화 방식의 비교이며, 동일한 경사 업데이트 횟수의 비교는 아닙니다.

`ppo_reward.py`의 초기 계수는 A2C 보상과 같습니다.
필수 물건 +1, 목표 온기까지 점당 +0.1, 초과 온기는 점당 +0.4,
성공 10 + 점수/20 + 별×2, 실패 -2, 실제 피해 -0.1입니다.
바꾸려면 PPO 보상 계수와 `REWARD_VERSION`, 실험 이름을 함께 갱신합니다.
재생과 재평가는 현재 기본 계수가 아니라 실행 폴더의 `config.json`에 저장된 계수를 사용합니다.

## 실행 순서

프로젝트 루트 PowerShell에서 먼저 짧게 점검합니다.

```powershell
.\.venv\Scripts\python.exe -m rl.train_ppo --seed 11 --smoke
```

2,000스텝 점검 결과는 `runs/ppo_smoke/seed_11_시각/`에 저장됩니다.
smoke의 평가 성공률 0%만으로 학습 성공 여부를 판단하지 않습니다.
`train/episodes.csv`의 success와 curriculum_start/frontier를 함께 확인합니다.

본 실험은 우선 seed 11 하나를 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m rl.train_ppo --seed 11
```

결과는 `runs/RTenv_s01_ppo_route_v3_goal_v1_lr2e-4_ns100_600k/seed_11_시각/`에 저장됩니다.
기본 평가가 0스텝에서 먼저 수행되고, 각 체크포인트마다 결정적 50회와 확률적 100회를 실행합니다.
평가는 커리큘럼을 적용하지 않고 원래 시작점에서 진행합니다.
평가 중에는 학습 진행 메시지가 잠시 멈춥니다.

학습 로그는 10,000스텝마다 학습 성공/원래 시작점 성공/환경별 단계를 표시합니다.
`train/progress.json`도 같은 간격으로 갱신하며 `metadata.json`의 단계 수는 체크포인트에서 갱신합니다.
공식 성능은 학습 중 성공 수가 아니라 `evaluation/step_.../deterministic/summary.json`으로 판단합니다.
보조 확률적 성능, 6단계의 성공 수, 원래 시작점 학습 성공 수도 함께 비교합니다.

학습 결과를 다시 평가하거나 화면에서 보려면 실제 결과 경로를 넣습니다.

```powershell
.\.venv\Scripts\python.exe -m rl.evaluate_ppo --run-dir "runs/실험명/seed_11_시각"
.\.venv\Scripts\python.exe -m rl.watch_ppo "runs/실험명/seed_11_시각"
.\.venv\Scripts\python.exe -m rl.watch_ppo "runs/실험명/seed_11_시각" --checkpoint "checkpoints/model_0200000.zip"
```

화면에서는 R 재시작, ESC 종료입니다. `--stochastic`으로 확률적 플레이를 볼 수 있습니다.
학습을 재생하는 프로그램이므로 화면 재생 중 추가 학습은 없습니다.
`--headless --frames 5`로 창 없이 렌더 연결도 확인할 수 있습니다.

중단 시 가능한 경우 `interrupted_model.zip`을 저장하지만, 현재 학습 명령은 항상 새 모델로 시작합니다.
학습 재개 기능은 제공하지 않으며, A2C 모델을 PPO로 이어 학습할 수도 없습니다.

seed 11에서 개선이 확인된 뒤 22·33을 같은 설정으로 진행합니다.

```powershell
foreach ($trainSeed in @(22, 33)) {
    & .\.venv\Scripts\python.exe -m rl.train_ppo --seed $trainSeed
    if ($LASTEXITCODE -ne 0) { break }
}
```

성공 여부는 아직 검증 전입니다. smoke와 코드 검사는 전체 탐사 성능을 보장하지 않습니다.
