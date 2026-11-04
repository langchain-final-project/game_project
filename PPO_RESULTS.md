# PPO 학습 코드·저장 모델·발표 자료

`ppo` 브랜치에는 원본 A2C 공유본과 같은 게임 환경에서 처음부터 학습한 PPO 실험을 보관합니다.
코드와 게임 환경은 `ppo_2388/`, 실제 학습 결과 공유본은 `ppo_2388/trained_snapshot/`에 있습니다.

## 저장된 결과

두 알고리즘은 모두 seed 11의 단일 학습 실행이며, 최대 학습 예산은 환경 4개의 정책 결정 수 합계 1,000만 스텝입니다.
PPO는 새 경험 128개를 10 epochs 학습하며 clipping은 0.2입니다.
아래 발표 비교는 고정 초기 조건에서 맵당 1판 수행한 최대확률 행동 결과입니다.

| 지표 | A2C 공유 선정 모델 | PPO 최대확률 평가 기준 선정 모델 |
|---|---:|---:|
| 체크포인트 학습 스텝 | 9,709,436 | 9,609,344 |
| 85점 이상 귀환 맵 | 12/12 | 9/12 |
| 귀환 성공 맵 | 12/12 | 11/12 |
| 필수 기억 아이템 획득 맵 | 12/12 | 11/12 |
| 전체 맵 평균 게임 점수 | 97.42 | 86.75 |

PPO의 전체 학습과 최종 검증은 완료됐으며, 전12맵 목표는 통과하지 못했습니다.
최대확률 평가로 고른 발표 모델과 확률 행동 평가까지 고려한 자동 선정 모델은 서로 다릅니다.
각 모델의 구분과 실행 명령은 [학습 결과 공유본](ppo_2388/trained_snapshot/README.md)을 참고하세요.

## 설치 및 모델 검사

```powershell
cd ppo_2388
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py check --run trained_snapshot
```

발표용 PPO 모델의 Stage 9 플레이:

```powershell
.\.venv\Scripts\python.exe run.py watch --run trained_snapshot --model checkpoints/round_096_step_0009609344.zip --stage 9
```

대시보드:

```powershell
.\.venv\Scripts\python.exe run.py dashboard --run trained_snapshot
```

## 발표 자료

- [학습 곡선·원인 후보 분석](reports/presentation/ppo_maxprob_20261103/)
- [Stage 9 A2C·PPO GIF](reports/presentation/stage09_gifs_20261103/)
- [A2C Stage 9 GIF](reports/presentation/stage09_gifs_20261103/stage09_a2c_greedy.gif)
- [PPO Stage 9 GIF](reports/presentation/stage09_gifs_20261103/stage09_ppo_greedy.gif)

발표 자료 재생성 스크립트는 로컬 전체 실행 폴더가 없으면 `trained_snapshot/`을 읽습니다.

## 보관 범위

최종 재개 모델, 자동 선정 모델, 최대확률 기준 발표 모델, 초기 모델, 전체 정기 평가 집계·학습 진행 로그,
최종 검증과 주요 체크포인트의 재생 자료, 실행 당시 소스 스냅샷을 보관합니다.
중복된 중간 모델 200여 개와 대용량 판별 원시 로그 등은 Git 공유본에서 제외했습니다.
원본 학습 폴더는 로컬 `ppo_2388/runs/`에 그대로 보존되어 있습니다.
공유본의 파일 해시와 보관 범위는 `trained_snapshot/PACKAGE_MANIFEST.json`에 기록합니다.

추가 비교 실험을 수행하지 않았으므로 성능 차이의 원인은 확정하지 않았습니다.
