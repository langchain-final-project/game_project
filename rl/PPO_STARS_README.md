# PPO 별 1개 추가 학습: 탐색 강도 비교

성공한 seed 11의 PPO 가중치와 Adam 상태를 불러오고, 원래 시작점에서 온기 7점을
모으도록 추가 학습합니다. 기존 게임·A2C·귀환 PPO 파일과 성공 모델은 수정하지 않습니다.
이번 준비 작업에서는 학습과 smoke 학습을 실행하지 않습니다. 실행은 사용자가 합니다.

## 조건

- 설정: `ppo_stars_common.py`의 `CONFIG`, 탐색 계수는 `EXPLORATION_PRESETS`
- 원본: `runs/RTenv_s01_ppo_route_v3_goal_v1_lr2e-4_ns100_600k/seed_11_20261029_001614_962501/final_model.zip`
- 실제 게임 성공 조건: 필수 물건 + 온기 4점 + 출구 상호작용 그대로
- 학습용 수집 목표: 7점. 물건 획득 후 7점 미만이면 남은 온기 위치를 안내
- 목표가 7점이어도 4점으로 일찍 귀환하는 행동을 차단하거나 실패로 바꾸지 않음
- 학습과 평가 모두 원래 시작점. 기존 귀환 커리큘럼은 적용하지 않음
- 관측은 636차원을 유지하지만 목표 안내 의미가 달라 새 관측 버전으로 기록
- 보상은 원본 실행의 config.json에서 복사. 별 보너스 등 계수 변경 없음
- 추가 학습률 0.00005, 추가 학습 200,000스텝, 50,000스텝마다 저장·평가
- 나머지 PPO 조건과 결정적 50회/확률적 100회 평가 seed는 이전과 같음

## 이번에 비교하는 변수

v1은 ent_coef=0.01로 20만 스텝을 추가 학습했지만 온기 4점 귀환만 반복했습니다.
v2에서는 다른 조건을 고정하고 엔트로피 계수만 바꿉니다.

| CLI 조건 | ent_coef | 목적 |
|---|---:|---|
| baseline | 0.01 | 이전 실패 조건 재현용. 기존 결과를 참고하면 재실행은 선택 사항 |
| entropy03 | 0.03 | 첫 비교, 기본 선택 |
| entropy05 | 0.05 | 더 강한 탐색, 두 번째 비교 |

엔트로피 계수는 학습 중 행동 확률이 한 행동으로 지나치게 쏠리는 것을 완화하는 항의
가중치입니다. 실행 즉시 무작위 행동을 강제로 삽입하거나 평가 행동을 바꾸지는 않습니다.
설명: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

두 조건은 모두 원래 60만 스텝 성공 모델에서 출발합니다. entropy03의 결과에서 entropy05로
이어서 학습하면 출발점까지 달라지므로 비교가 아닙니다. --init-model은 기본값을 유지하세요.
보상·목표 온기 7·학습률 0.00005·20만 추가 스텝·커리큘럼 해제를 고정합니다.
탐색을 늘려도 실제 추가 수집이나 별 획득이 생긴다는 보장은 없습니다.

첫 단계 목표는 귀환 성공률 90% 이상, 별 1개 이상 획득률 80% 이상입니다.
Stage 1의 용기 3·56틱 이내 귀환에서는 온기 7점이 58점/별 1개가 됩니다.
이를 통과하기 전에는 온기 목표를 10점이나 14점으로 올리지 않습니다.

## 사용자 실행 순서

학습하지 않고 적용할 설정만 확인하려면 다음을 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m rl.train_ppo_stars --exploration entropy03 --show-config
```

첫 비교를 실행합니다. 아래 명령은 실제로 추가 학습을 시작합니다.

```powershell
.\.venv\Scripts\python.exe -m rl.train_ppo_stars --seed 11 --exploration entropy03
```

첫 결과를 확인한 뒤 두 번째 조건을 독립 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m rl.train_ppo_stars --seed 11 --exploration entropy05
```

짧은 연결 학습이 필요하면 위 명령에 --smoke를 붙입니다. --smoke도 실제 학습입니다.

다른 성공 모델을 출발점으로 쓰려면 `--init-model ".../final_model.zip"`을 지정합니다.
기존 실행의 config.json도 모델과 같은 실행 폴더에 있어야 합니다.
원본 가중치와 게임 조건/관측 크기를 확인하고 불러오므로, 파일이 없거나 다르면 새 모델로
조용히 대체하지 않고 오류로 종료합니다.

## 스텝과 결과 위치

- 첫 비교: `runs/RTenv_s01_ppo_star1_goal7_entropy03_ft200k_v2/seed_11_시각/`
- 두 번째 비교: `runs/RTenv_s01_ppo_star1_goal7_entropy05_ft200k_v2/seed_11_시각/`
- 기존 v1 결과는 기존 폴더에 그대로 보존하며, 설정에 exploration 항목이 없어도 재평가 가능
- 점검: `runs/ppo_star_smoke/seed_11_시각/`
- 이번 파일 이름과 그래프의 step 0~200000은 추가 학습량
- step 0 체크포인트는 원본 가중치 그대로이며, 새로운 온기 목표 관측에서 평가
- metadata.json의 initial_model에 원본 경로·해시·60만 스텝 이력을 기록
- cumulative_completed_timesteps는 원본 60만 + 추가 학습량, 완료 시 80만
- 학습 중 1만 스텝마다 별 획득 수와 목표 온기·조기 귀환 수도 표시

공식 별 획득률은 `evaluation/step_.../deterministic/summary.json`을 봅니다.
`at_least_one_star_rate`, `success_rate`, `mean_warmth`, `early_return_rate`,
`warmth_goal_rate`, `star_training_pass`를 함께 확인합니다.
실패한 회차도 별 획득률의 분모에 포함합니다.

5만 스텝마다 우선 학습의 mean_warmth가 4보다 커지는지, warmth_goal_met_count나
at_least_one_star_count가 0보다 커지는지 확인합니다. 실제 목표는 평가의 귀환 성공률 90% 이상,
별 1개 이상 비율 80% 이상입니다. 성공률이 90% 아래로 내려가면 해당 조건의 결과를 검토하고,
별이 늘지 않는 상태에서 같은 조건의 학습량만 계속 늘리지 않습니다.

```powershell
.\.venv\Scripts\python.exe -m rl.evaluate_ppo_stars --run-dir "runs/실험명/seed_11_시각"
.\.venv\Scripts\python.exe -m rl.watch_ppo_stars "runs/실험명/seed_11_시각"
```

별 수집 모델은 `watch_ppo_stars`로 봐야 해당 실행에 저장한 수집 목표가 적용됩니다.
이전 귀환 모델은 기존 `watch_ppo`로 그대로 볼 수 있습니다.

CLI는 기존 별 수집 실행을 정확히 재개하는 기능이 아닙니다. 지정된 모델에서 가중치와
옵티마이저를 받아 새 추가 학습 실험을 시작하며, 환경 상태와 난수 진행 상태는 새로 시작합니다.
별 수집 성공과 귀환 유지 여부는 실제 평가로 확인해야 합니다.
