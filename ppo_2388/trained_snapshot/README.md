# 완료된 PPO 학습 결과 공유본

학습 seed 11, 입력 2,388개, 전12맵 공통 정책을 최대 1,000만 스텝 학습한 결과입니다.
전체 학습과 최종 검증을 마쳤으며 상태는 `complete_not_passed`입니다.
원본 실행은 `seed_11_20261102_230005_281926_io_retry_v1`입니다.

## 모델 구분

| 용도 | 파일 | 학습 스텝 | 최대확률 평가의 85점 이상 귀환 맵 |
|---|---|---:|---:|
| 발표용 최대확률 기준 선정 모델 | `checkpoints/round_096_step_0009609344.zip` | 9,609,344 | 9/12 |
| 확률 행동 성적까지 고려한 자동 선정 모델 | `selected_model.zip` | 8,708,480 | 8/12 |
| 마지막 학습·재개 체크포인트 | `resume_B.zip` | 10,000,000 | 5/12 |
| 학습 전 초기 모델 | `initial_model.zip` | 0 | 초기 기준 |

발표용 모델은 최대확률 평가의 85점 이상 귀환 맵 수를 우선하고, 동률이면 전체 맵 평균 점수가 높은 모델입니다.
최대확률 평가는 유령 초기 위상을 고정하고 맵마다 1판 수행했습니다.
확률 행동 최종 검증은 자동 선정 모델로 수행했으므로 발표용 모델의 최종 검증 성적으로 해석하지 않습니다.

## 저장 모델 검사와 관람

명령은 `ppo_2388` 폴더에서 실행합니다. 필요한 패키지는 상위 `requirements.txt`를 사용합니다.

```powershell
python run.py check --run trained_snapshot
python run.py watch --run trained_snapshot --model checkpoints/round_096_step_0009609344.zip --stage 9
python run.py dashboard --run trained_snapshot
```

`check`는 마지막 재개 체크포인트를 읽어 소스·설정 해시와 모델·Adam 상태 및 전12맵 입력을 검사합니다.
기본 `watch`의 모델은 자동 선정 모델이므로, 발표 모델 관람에는 위의 `--model`을 명시합니다.
완료된 실행에 `train --run trained_snapshot`을 실행하면 종료 상태를 확인하고 추가 학습 없이 반환합니다.
새 실험은 `python run.py train --new --hours 8`로 별도 `runs/` 폴더에 생성합니다.

## 평가 자료

- `greedy_selected_evaluation.json`: 발표 모델 평가, 평균 86.75점, 85점 이상 귀환 9/12맵.
- `selected_evaluation.json`: 자동 선정 모델의 정기 평가.
- `last_evaluation.json`: 마지막 모델의 정기 평가.
- `checks/`: 학습 전 기준 평가와 100개 정기 평가 집계.
- `evaluation/history.jsonl`: 전체 학습 곡선을 그릴 수 있는 맵별 평가 기록.
- `evaluation/round_087`, `round_096`, `round_100`: 주요 모델의 최대확률·확률 행동 평가와 재생 자료.
- `evaluation/final`: 자동 선정 모델의 두 시드 집합 최종 검증 결과.
- `train/`: 학습 진행·맵별 성적과 각 구간의 최적화 지표 CSV.
- `source_snapshot/`: 모델 재개 검증에 사용한 실행 당시 소스.
- `PACKAGE_MANIFEST.json`: 공유 파일별 SHA-256과 보관 범위.

수백 개 중간 모델, 다른 중간 시점의 재생 파일, 250MB 안팎의 `train/episodes.jsonl`은 공유본에서 제외했습니다.
집계 파일에 남아 있는 일부 중간 체크포인트·재생 경로는 이 축약 공유본에 없을 수 있습니다.
원본 전체 실행과 모든 원시 로그는 로컬 `runs/` 폴더에 보존했습니다.

패키지 구성 과정에서는 학습 업데이트를 수행하지 않았으며 모델 파일을 다시 저장하지 않았습니다.
모델·설정·소스 스냅샷은 원본 파일을 그대로 복사했습니다.
