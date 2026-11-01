# 같은 A2C 기준 모델에서 A2C·PPO 추가 학습 비교

현재는 기존 설정 반복용 **benchmark**와 귀환 보강용 **return_practice**를 구분합니다.
시드 22·33 자동 반복, 시드 11 결과 재사용, 귀환 커리큘럼 실행 방법은
[실험 프로필과 반복 시드 안내](EXPERIMENT_PROFILES.md)를 먼저 확인하세요.
아래의 기본 명령은 benchmark 조건을 유지합니다.

기존 rl의 PPO 학습은 그대로 둡니다. 이 실험의 코드와 결과는 A2cToPpo 안에서만 관리합니다.
이번 비교는 **같은 사전 학습 정책을 어느 알고리즘으로 추가 학습하는 것이 좋은가**를 확인합니다.
처음부터 A2C와 PPO를 각각 학습한 비교와 구분합니다.

## 출발점 고정

직전 실행의 학습 전 모델만 사용합니다.

```text
A2cToPpo/runs/seed_11_20261101_161343_544404/models/baseline_model.zip
```

이 파일은 A2C 선정 모델의 정책·가치망을 PPO 보관 형식으로 옮긴 **추가 학습 0프레임 가중치**입니다.
직전 best_model.zip이나 last_model.zip을 이어 학습하지 않습니다.
[config.py](config.py)의 BASELINE_CHECKPOINT와 SHA-256으로 출발 파일을 고정하고,
파일이 달라지거나 이미 추가 학습한 모델이면 실행을 거부합니다. 원본은 덮어쓰지 않습니다.

두 대조군은 이 가중치를 각각 새 PPO와 새 A2C에 복사합니다.
신경망 가중치·초기 행동 확률·가치 예측의 일치와 **빈 Adam 최적화기·0 학습 카운터**를 검증합니다.
A2C 대조군도 기존 최적화기 상태를 복원하지 않습니다.

## 사용자 실행 명령

game_project 최상위에서 실행합니다.

```powershell
# 두 알고리즘 연결만 확인: 학습·새 모델 저장 없음
.\.venv\Scripts\python.exe -X utf8 -B -m A2cToPpo.train --algorithm both --check

# PPO 추가 30만 물리 프레임 → A2C 추가 30만 물리 프레임 순차 실행 후 비교
.\.venv\Scripts\python.exe -X utf8 -B -m A2cToPpo.train --algorithm both --seed 11 --frames 300000
```

두 학습을 따로 실행할 수도 있습니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m A2cToPpo.train --algorithm ppo --seed 11 --frames 300000
.\.venv\Scripts\python.exe -X utf8 -B -m A2cToPpo.train --algorithm a2c --seed 11 --frames 300000
```

알고리즘을 생략하면 PPO만 실행합니다. both는 동시에 실행하지 않고 순서대로 실행합니다.
첫 실행이나 최종 평가를 Ctrl+C로 중단하면 두 번째 학습을 자동으로 시작하지 않습니다.
기존 가상환경을 사용하며 추가 패키지 설치는 필요하지 않습니다.

30만은 **알고리즘별로 환경 4개를 합산한 물리 프레임**입니다.
행동 선택은 약 16,700회이고, 게임이 유지 행동 도중 끝나면 더 많아집니다.
PPO와 A2C 모두 수집 묶음을 끝까지 갱신하므로 실제 프레임 수는 요청 예산을 약간 넘을 수 있습니다.
원본 A2C의 620만 스텝은 행동 선택 수이며 물리 프레임과 단위가 다릅니다.

## 공통 설정과 알고리즘 차이

| 항목 | A2C·PPO 공통 |
|---|---|
| 초기 가중치 | 위 baseline_model.zip 하나 |
| 학습률 | 0.00005 |
| 엔트로피 계수 | 0.005, 고정 |
| gamma / GAE | 기준 모델 값 유지, 0.997002999 / 1.0 |
| 환경 | 원본 UnifiedMansionEnv → RealtimeGameCore |
| 관측 | 지도 14×9×16 + 상태 224 = 2,240개 |
| 행동 | 이동 4방향 + 대기, 5개, 자동 상호작용 |
| 행동 유지 | 최대 18물리 프레임 |
| 게임·보상·신경망 | ZIP 원본 유지 |
| 스테이지 | 처음부터 1~12 전체, 회차마다 균등 선택 |
| 시작 상태 | 실제 게임 시작점, 준비된 경로 없음 |
| 환경·스레드 | DummyVecEnv 4개, Torch CPU 1스레드 |
| 수집 길이 | 환경당 128행동, 합계 512행동 |
| 최적화기 | 둘 다 새 Adam |
| 학습·평가 시드와 예산 | 두 군에 같은 값 사용 |

직전의 학습률 0.0002·엔트로피 약 0.0444를 그대로 쓰지 않습니다.
새 값은 정책 변화를 줄이기 위한 **실험 후보**이며 최적값이나 개선 보장은 아닙니다.
같은 값을 두 알고리즘에 적용해 이 공통 조건에서 추가 학습 효과를 비교합니다.

PPO는 배치 128, 최대 4회 반복, clipping 0.1, target KL 0.02를 사용합니다.
A2C는 같은 수집 묶음으로 A2C 갱신을 수행합니다. RMSProp은 사용하지 않습니다.
원본 A2C의 환경당 n_steps=32도 이번 대조 실험에서는 128로 맞춥니다.
원래 A2C의 운영 설정 전체와 새 PPO를 비교하는 실험은 아닙니다.

공통 설정은 COMMON_UPDATES, PPO 전용 설정은 PPO_UPDATES에서 바꿉니다.
기존 rl의 관측 640개·9프레임 PPO 설정에는 영향을 주지 않습니다.
새 방문 보상의 학습 진행도는 원래 620만 행동에 추가 행동 수를 더해 계산합니다.

## 평가와 최적 모델 채택

정기 평가는 10만 물리 프레임 간격으로 실행하며, 스테이지별 **확률적 50판·결정적 1판**입니다.
직전의 10판 평가보다 판 수를 늘렸습니다.
확률적 별 3개 비율·성공률·평균 점수·적은 피해 순으로 검증 대기 후보를 선택합니다.

기준 모델의 성공률·별 3개 비율뿐 아니라 **물건 획득률**도 보호합니다.
해당 지표가 기준에서 80% 이상인 스테이지에서 10%p를 초과해 하락한 후보는 선택하지 않습니다.
예를 들어 귀환 성공률이 0%여도 물건 획득률이 99%였던 능력은 보호합니다.

파일의 역할을 구분합니다.

| 파일 | 역할 |
|---|---|
| baseline_model.zip | 새 학습 전 기준 가중치, 보존 |
| candidate_model.zip | 정기 평가로 고른 검증 대기 후보 |
| last_model.zip | 요청 예산까지 학습한 마지막 상태 |
| best_model.zip | 최종 검증을 통과해 실제 채택한 모델. 검증 전에는 기준 모델 |
| *_frames_....zip | 정기 평가 시점 모델 |

학습 종료 후 **기준·후보·마지막 모델 각각**을 새 시드에서 스테이지별 확률적 100판·결정적 10판으로 평가합니다.
이 평가 시간은 학습 프레임 예산에 포함하지 않습니다. 두 군을 모두 실행하면 최종 평가량도 두 배입니다.

기본 채택 조건은 다음을 모두 만족하는 것입니다.

1. 기존 능력 보존 조건 통과.
2. 같은 최종 평가에서 기준보다 나은 선택 점수.
3. 모든 스테이지에서 성공률·별 3개 비율 각각 90% 이상.

3번은 EVALUATION의 require_all_stages_passed_for_adoption=True로 명시했습니다.
일부 스테이지만 개선된 후보는 연구 결과로 보관하되 최종 채택하지 않습니다.
최종 검증에 실패하거나 평가가 중단되면 **best_model.zip은 기준 가중치를 유지**합니다.
후보·마지막 모델은 삭제하지 않으므로 성능이 나빠진 결과도 분석할 수 있습니다.
학습량을 더 늘리는 것만으로 모든 조건을 만족한다고 보장하지 않습니다.

## 저장과 비교 해석

```text
A2cToPpo/runs/
  ppo_seed_11_실행시각/
  a2c_seed_11_실행시각/
    config.json
    transfer_check.json
    train_episodes.csv
    training_summary.json
    candidate_evaluation.json
    best_evaluation.json
    models/
      baseline_model.zip
      candidate_model.zip
      last_model.zip
      best_model.zip
    evaluations/
      baseline.json
      frames_....json
    final_verification/
      baseline.json
      candidate.json
      last.json
      best.json
      comparison.json
  comparisons/
    seed_11_실행시각.json
```

both 명령은 두 실행을 끝내고 자동 비교 파일을 만듭니다.
따로 실행한 결과는 다음 명령으로 비교합니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m A2cToPpo.compare --ppo-run "A2cToPpo/runs/ppo_실행폴더" --a2c-run "A2cToPpo/runs/a2c_실행폴더"
```

출발 가중치 해시·공통 계수·게임 설정·학습 시드·예산·최종 평가 조건이 다른 두 실행은 비교를 거부합니다.
최종 평가가 완료되어야 하며, 스테이지별 평가 판 수도 확인합니다.

비교 보고서에는 세 가지 결과를 함께 남깁니다.

- last: 같은 추가 예산을 쓴 마지막 모델의 결과.
- candidate: 같은 정기 평가 절차로 고른 후보의 결과.
- best: 최종 채택 모델의 결과.

둘 다 기준 모델로 복원됐다고 best끼리만 비교하면 학습의 성능 하락을 숨길 수 있으므로,
**알고리즘 비교에는 last와 candidate의 원래 결과도 함께 사용**합니다.
training_summary.json의 candidate_accepted, selected_model_origin,
selected_model_physical_frames를 보면 실제 채택 여부를 확인할 수 있습니다.
후보가 탈락했다면 selected_model_physical_frames는 0입니다.

처음에는 seed 11 한 쌍으로 동작과 방향을 확인할 수 있습니다.
정량적인 비교 결론을 내려면 같은 설정에서 여러 학습 시드를 반복하고 변동도 보고해야 합니다.
한 시드의 전이 학습 결과만으로 PPO나 A2C 자체가 우월하다고 결론 내리지 않습니다.

## 관람과 평가만 실행

```powershell
# 알고리즘은 실행 폴더의 config.json에서 자동 확인
.\.venv\Scripts\python.exe -X utf8 -B -m A2cToPpo.watch --run "A2cToPpo/runs/실행폴더" --stage 3

# 채택되지 않은 후보를 확인
.\.venv\Scripts\python.exe -X utf8 -B -m A2cToPpo.watch --run "A2cToPpo/runs/실행폴더" --stage 3 --checkpoint candidate_model.zip

# 같은 예산의 마지막 상태를 확인
.\.venv\Scripts\python.exe -X utf8 -B -m A2cToPpo.watch --run "A2cToPpo/runs/실행폴더" --stage 3 --checkpoint last_model.zip

# 평가만 재실행
.\.venv\Scripts\python.exe -X utf8 -B -m A2cToPpo.evaluate --run "A2cToPpo/runs/실행폴더" --episodes 100
```

관람 기본값은 확률적 플레이이고 --deterministic으로 바꿀 수 있습니다.
18개 물리 상태를 건너뛰지 않고 화면에 한 장씩 표시하며 60FPS·1배속을 목표로 합니다.
R 재시작, Space 일시정지, Esc 종료입니다.
기존 A2C_to_PPO 실행 폴더도 계속 관람·평가할 수 있습니다.
직전 실행의 ZIP이나 보고서를 소급해서 수정하지 않습니다.

## 소스와 검증

_vendor/a2c에는 ZIP의 필요한 원본 75개 파일을 그대로 보관합니다.
실행할 때 transfer_manifest.json의 해시를 검사하며 복사한 원본은 수정하지 않습니다.
기준 ZIP은 위 이전 실행 폴더에 있어야 합니다. runs와 _vendor는 Git에서 제외되어 있으므로
다른 PC로 옮길 때는 원본 ZIP과 기준 모델도 함께 준비해야 합니다.

```powershell
# 실제 학습·역전파를 실행하지 않는 검증
.\.venv\Scripts\python.exe -X utf8 -B -m pytest A2cToPpo/tests -q -p no:cacheprovider --basetemp=A2cToPpo/.pytest_tmp
```

가중치·Adam 초기화·기준 파일 고정·최종 채택/복원·비교 조건 검증·관람을 테스트합니다.
코드 수정 과정에서는 실제 추가 학습을 실행하지 않습니다.
