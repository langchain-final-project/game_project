# 반복 시드 비교와 귀환 보강을 분리한 실행

기존 모델·이전 결과·rl 폴더는 유지합니다. 새 학습은 사용자가 실행합니다.
두 프로필은 학습 조건이 다르므로 같은 평균에 섞지 않습니다.
둘 다 고정한 학습 전 baseline_model.zip에서 새 A2C·PPO 최적화기로 시작합니다.

## 1. 기존 조건으로 학습 시드 22·33 반복

프로젝트 최상위에서 실행합니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m A2cToPpo.experiments --profile benchmark --seeds 22 33 --frames 300000
```

- seed 22: PPO 30만 물리 프레임 → A2C 30만 물리 프레임.
- seed 33: PPO 30만 물리 프레임 → A2C 30만 물리 프레임.
- 모두 순서대로 실행합니다. 추가 학습 예산 합계는 120만 물리 프레임입니다.
- 이미 완료한 seed 11의 아래 비교 보고서는 다시 학습하지 않고 집계에 포함합니다.

```text
runs/comparisons/seed_11_20261101_172005_698191.json
```

기존 보고서 경로는 config.py의 BENCHMARK_REFERENCE에 있습니다.
새 결과는 알고리즘별 runs/ppo_seed_22_... 및 runs/a2c_seed_22_... 형태로 저장하고,
각 시드의 비교 파일도 runs/comparisons에 남깁니다.

마지막에는 runs/suites/benchmark_실행시각/에 아래를 만듭니다.

| 파일 | 내용 |
|---|---|
| summary.json | 시드 11·22·33의 평균·표본 표준편차, 시드별 원래 값, 스테이지별 집계 |
| summary.md | 사람이 읽기 쉬운 요약 표 |
| progress.json | 완료한 비교 보고서 목록과 실행 상태 |

같은 시드의 PPO−A2C 차이도 집계합니다. 평가 1,200판을 독립 학습 1,200회로 세지 않습니다.
초기 A2C 사전 학습 가중치는 하나로 고정되므로, 반복 시드는 추가 학습 과정의 변동을 측정합니다.
처음부터 학습한 A2C·PPO 비교나 여러 사전 학습 시드의 일반화 결과와 구분합니다.

기존 결과와 공통 설정·예산·평가가 다르면 집계하지 않습니다.
처음부터 새 seed 11을 다시 실행하려면 기존 보고서 자동 포함을 끕니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m A2cToPpo.experiments --profile benchmark --seeds 11 --frames 300000 --include
```

## 2. 미해결 네 스테이지의 귀환 연습

첫 방향 확인은 같은 seed 11로 두 알고리즘을 실행합니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m A2cToPpo.train --profile return_practice --algorithm both --seed 11 --frames 300000
```

각 알고리즘 30만, 합계 60만 물리 프레임입니다. 기존 benchmark 결과와 합치지 않습니다.
이후 같은 귀환 보강 조건을 여러 시드로 비교하려면 아래를 실행합니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m A2cToPpo.experiments --profile return_practice --seeds 11 22 33 --frames 300000
```

이 명령은 세 시드 모두 새로 실행합니다. 위의 seed 11 한 쌍을 이미 실행했다면
해당 비교 파일을 --include에 지정하고 --seeds 22 33만 요청하면 중복 학습을 피할 수 있습니다.

### 귀환 연습 상태를 만드는 방법

별도 practice_states/return_v1.json에 시작 상태를 재현할 물리 행동열을 보관합니다.
준비 과정에는 학습이나 역전파가 없습니다.

1. 고정 기준 정책으로 3·4·9·10스테이지의 물건과 온기 14점을 실제로 수집합니다.
2. 벽과 닫힌 문을 반영한 경로를 실제 게임 물리로 이동하며 귀환 상태를 기록합니다.
3. 원본 코어에서 실제 별 3개로 완료되는 경로인지 검사합니다.
4. 출구까지 약 1·3·6·10·16·24칸과 가장 먼 상태를 골라 둡니다.

원래 시간·유령·문 개방·무적·피해·수집 상태를 보존합니다.
온기나 물건을 임의로 지급하거나 유령을 제거하지 않습니다.
초기화에서는 기록된 행동을 재생하거나 그 코어의 복사본을 사용합니다.
준비 중 획득한 물건·온기의 보상은 다시 지급하지 않습니다.
준비에 사용한 물리 프레임은 별도 prepared_frames로 기록하며 추가 학습 예산에는 넣지 않습니다.

준비 상태에 도달하는 구간은 기준 정책과 경로 계산을 사용한 커리큘럼입니다.
에이전트가 전체 경로를 배웠다는 성과가 아니며, 학습 중 행동은 A2C/PPO 정책만 선택합니다.
A2C와 PPO가 같은 상태 파일과 같은 설정을 사용합니다.

### 연습 비중과 역순 확장

| 시작 방식 | 회차 선택 비중 |
|---|---:|
| 3·4·9·10 중 하나의 물건 획득 후 귀환 상태 | 70% |
| 1~12 전체 중 하나의 원래 시작점 | 30% |

귀환 상태는 출구 가까이에서 시작합니다. 현재 거리에서 최근 8판 중 별 3개 6판이면
더 먼 구간으로 확장합니다. 20%는 이미 익힌 더 짧은 구간도 복습합니다.
네 환경은 같은 알고리즘 안에서 진급 기록을 공유하고, 알고리즘이 바뀌면 새 기록으로 시작합니다.
위 비중은 프레임 비중이 아니라 회차 선택 확률입니다.

### 공통 보상 보강

대상 네 스테이지에서 물건을 보유하고 온기가 12점 이상이면 추가합니다.

- 출구까지 벽·문을 반영한 거리 감소량 × 0.10. 멀어지면 음수입니다.
- 물리 프레임당 시간 비용 −0.0002.
- 실제 별 3개 귀환 완료 보너스 +0.5.

같은 경로를 왕복하면 거리 보상 합계는 0입니다. 시간 비용은 별도로 남습니다.
원래 게임 점수·별 기준·기존 보상은 유지하고 A2C와 PPO에 같은 보강을 적용합니다.
현재 설정은 실험 후보이며 성공을 보장하지 않습니다.

## 평가와 결과 해석

두 프로필 모두 최종 평가는 **1~12스테이지의 원래 시작점**에서만 실행합니다.
평가·게임 관람에서는 준비 상태와 추가 귀환 보상을 사용하지 않습니다.
따라서 귀환 연습에서 성공해도 전체 플레이가 성공했다고 판단하지 않습니다.

기존의 스테이지별 확률적 100판·결정적 10판과 최종 모델 채택 조건은 그대로 유지합니다.
최종 목표를 못 넘으면 best_model.zip은 학습 전 기준 모델로 남고,
학습한 실제 후보는 candidate_model.zip, 마지막 상태는 last_model.zip에 있습니다.

귀환 프로필에서는 아래도 확인합니다.

- train_episodes.csv의 start_type: full_start와 return_practice를 분리합니다.
- start_distance, prepared_frames, quest_collected_at_start, damage_at_start: 준비 상태 정보입니다.
- training_summary.json 및 정기 평가의 return_curriculum: 스테이지별 진행 거리와 연습 성공 횟수입니다.
- 최종 원래 시작점 평가에서 3·4·9·10의 성공률이 0%에서 벗어나는지 확인합니다.
- 나머지 8개 스테이지의 별 3개 비율도 유지되는지 확인합니다.

두 프로필을 섞어 비교하면 학습 조건이 바뀐 효과와 알고리즘 효과를 구분하기 어렵습니다.
집계기는 프로필·상태 파일·보상·학습 계수·예산·평가 조건이 다르면 결합을 거부합니다.

## 연결 확인과 이미 끝난 결과 집계

```powershell
# 기존 조건의 반복 연결만 확인: 실제 학습 없음
.\.venv\Scripts\python.exe -X utf8 -B -m A2cToPpo.experiments --profile benchmark --seeds 22 33 --check

# 귀환 보강 연결만 확인: 필요하면 준비 상태 파일을 생성하나 실제 학습은 없음
.\.venv\Scripts\python.exe -X utf8 -B -m A2cToPpo.train --profile return_practice --algorithm both --check

# 이미 끝난 세 비교 파일만 집계: 실제 경로로 교체
.\.venv\Scripts\python.exe -X utf8 -B -m A2cToPpo.experiments --aggregate-only --include "seed11비교.json" "seed22비교.json" "seed33비교.json"
```

반복 학습을 중단하면 이후 시드나 알고리즘을 자동 실행하지 않습니다.
완료한 파일 목록은 progress.json에 보존합니다.
기존 코드의 --algorithm both 명령도 유지하며 --profile을 생략하면 benchmark입니다.
