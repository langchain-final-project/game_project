# 선택한 PPO 후보 모델 보관본

원본 실행:

```text
A2cToPpo/runs/ppo_seed_11_20261101_164722_768756
```

이 보관본에서 선택한 모델은 **models/candidate_model.zip**입니다.
모델, config.json, 학습 기록, 정기 평가, 최종 평가를 원본과 같은 바이트로 복사했습니다.
archive_manifest.json에 복사 파일별 SHA-256, 크기, 원본 위치를 기록했습니다.
원본 실행 폴더와 모델은 그대로 보존합니다.

training_summary.json과 best_evaluation.json을 포함한 과거 보고서는 당시 기록 그대로입니다.
이 폴더의 모델 선택 정보는 archive_manifest.json의 selected_checkpoint를 확인하세요.
평가 조건과 평가 범위는 수정하지 않았습니다.

## 게임 화면 재생

game_project 최상위에서 실행합니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m A2cToPpo.watch --run "A2cToPpo/final/ppo_benchmark_seed11" --checkpoint candidate_model.zip --stage 1 --stochastic --seed 11
```

--stage 뒤의 숫자를 바꾸면 해당 스테이지를 볼 수 있습니다.
R 재시작, Space 일시정지, Esc 종료입니다.
재생에는 프로젝트의 A2cToPpo/watch.py와 _vendor/a2c 복사본을 사용합니다.
source_reference/transfer_manifest.json은 해당 원본 자료의 해시 기록입니다.

이 보관 작업에서 학습이나 추가 평가는 실행하지 않았습니다.
