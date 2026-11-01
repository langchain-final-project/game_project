# Stage 2 물건 획득 독립 과제

이 부분 과제는 새 시드 평가에서 6번 94%, 7번 92%로 통과했습니다.
현재는 [자석·물건 연결 PPO](PPO_STAGE2_MAGNET_CONNECTION.md)를 먼저 실행합니다.
아래는 물건 전용 과제의 재현 방법이며, 이번 전체 학습에는 `--quest-only`를 사용하지 않습니다.

6번과 7번에서 아래쪽으로 이동하는 행동을 먼저 익히는 PPO 연습입니다.
자석을 이미 얻고 온기 4점·물건 미획득인 합법적인 시작 상태를 사용합니다.
실제 RealtimeMansionEnv의 이동·충돌·유령·자석 시간은 그대로 진행합니다.
관측 640개와 행동 마스크를 유지하며 정책이 행동을 선택합니다.

## 시작 모델과 탐색

`stage2_quest_first/quick/seed_11_20261101_013936_887142/models/best_model.zip`의
가중치를 가져와 새 최적화기로 시작합니다. 원본 ZIP은 변경하지 않습니다.
부분 과제에서만 행동 출력 가중치·편향에 0.3을 곱하고 엔트로피 계수는 0.03을 사용합니다.
전체 게임의 보상 척도에 맞춰진 가치 출력층은 부분 과제에서 0으로 초기화합니다.
정책의 특징 추출기·정책망 가중치는 전이하고, 실제 학습은 PPO가 수행합니다.
이 초기화로 아래쪽 선택 확률은 약 0.2%에서 11%로 증가했지만 아직 학습 결과는 아닙니다.

## 부분 보상과 종료

- 물건 획득: +10, 해당 물리 프레임에서 부분 과제 종료.
- 물건까지의 벽을 고려한 거리 감소/증가: 거리 변화 × 1.0.
- 시간: 프레임당 -0.005, 실제 피해: 회당 -1.0.
- 180프레임 내 미획득: -2, 시간 제한 중단 후 새로 시작.
- 추가 온기·별·전체 귀환 보상은 이 연습에 사용하지 않음.

`quest_task_episode`로 별도 기록하며 `rl_episode`나 게임 COMPLETE로 위장하지 않습니다.
PPO의 시간 제한 중단 처리와 물건 획득 성공의 terminal을 구분합니다.
환경 4개 중 2개는 6번, 2개는 7번에 배정합니다. 쉬운 8번은 학습에서 제외합니다.

## 실행

```powershell
# 학습·저장 없는 연결 확인
.\.venv\Scripts\python.exe -X utf8 -B -m rl.train_ppo --seed 11 --quest-only --check --quick

# 물건 획득 부분 과제 PPO 학습
.\.venv\Scripts\python.exe -X utf8 -B -m rl.train_ppo --seed 11 --quest-only --quick
```

추가 예산은 최대 90,000 물리 프레임이며 에피소드 종료 시점에 따라 행동 선택 횟수가 달라집니다.
6·7번 확률적 평가가 각각 90% 이상이면 예산 전에도 멈춥니다.
평가 주기는 9,000 물리 프레임이고 구간별 확률적 20판·결정적 3판을 기록합니다.
최적 모델은 두 구간 중 낮은 성공률을 우선하므로 한 구간만 잘하는 모델을 피합니다.
결과는 `reports/local/experiments/stage2_quest_acquisition/quick/seed_...`입니다.
`training_summary.json`의 `task_passed`는 물건 과제만의 결과이며 별 3개 성능이 아닙니다.

학습 없는 초기 평가 결과는 `checks/baseline.json`에 저장했습니다.
출력 완화 직후 구간별 확률적 20판에서 6번 20%, 7번 30%였습니다.
새 PPO 학습 후 이 기준보다 개선되는지 확인합니다.

## 평가와 전체 과제 복귀

```powershell
# 학습 없이 6·7번을 각각 새로운 시드 50개로 평가하고 결과 저장
.\.venv\Scripts\python.exe -X utf8 -B -m rl.train_ppo_quest --evaluate "부분과제실행폴더/models/best_model.zip" --heldout

# 두 구간의 새 시드 성공률이 각각 90% 이상일 때만 전체 과제 연결 확인
.\.venv\Scripts\python.exe -X utf8 -B -m rl.train_ppo --seed 11 --quick --check --from-quest "부분과제실행폴더/models/best_model.zip"

# 통과 후 전체 게임의 자석·추가 수집·귀환 학습으로 복귀
.\.venv\Scripts\python.exe -X utf8 -B -m rl.train_ppo --seed 11 --quick --from-quest "부분과제실행폴더/models/best_model.zip"
```

`--from-quest`는 매번 새 시드 평가로 두 구간을 확인하고, 미달이면 전체 학습을 시작하지 않습니다.
복귀 시에는 행동 출력 배율 1.0으로 가져오므로 0.3 완화를 다시 적용하지 않습니다.
전체 과제 설정·새 최적화기를 사용하며 이 경로 변경은 해당 실행에만 적용합니다.
현재 전체 결과는 `stage2_joint_review` 아래 새 실행 폴더에 저장합니다.
이후 실제 시작점에서의 성공률·별 3개 획득률·자석 활용·유령 피해를 다시 검증해야 합니다.
