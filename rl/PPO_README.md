# 실시간 PPO 학습·평가·재생

현재 환경은 `RealtimeMansionEnv`입니다. 실행 코드는 `rl/`, 확정한 Stage 1은
`reports/local/stage1/`에 보관합니다. 격자 학습 코드와 중간 모델은 정리했습니다.

- [Stage 1 설정·결과](PPO_STAGE1_THREE_STARS.md)
- [현재 실행: Stage 3 수집 이후 귀환 연결 PPO](PPO_STAGE3_RETURN_CONNECTION.md)
- [Stage 2 무피해 추가 수집·귀환 실험](PPO_STAGE2_SAFE_RETURN_FINAL.md)
- [완료한 연결 모델의 0·1번 추가 복습](PPO_STAGE2_CONNECTION_EARLY_REVIEW.md)
- [첫 Stage 2 자석·물건 연결 PPO](PPO_STAGE2_MAGNET_CONNECTION.md)
- [이전 Stage 2 초반 수집·물건 획득 공동 복습](PPO_STAGE2_JOINT_REVIEW.md)
- [이전 Stage 2 목표 순서·혼합 연습·부분 진단](PPO_STAGE2_QUEST_FIRST.md)
- [완료한 6·7번 물건 획득 독립 PPO 과제](PPO_STAGE2_QUEST_ACQUISITION.md)
- [게임 화면 재생](PPO_WATCH_README.md)

```powershell
# 최종 모델 재생: 실제 시작점, 확률적 행동
.\.venv\Scripts\python.exe -X utf8 -B -m rl.watch_ppo "reports/local/stage1" --stochastic --seed 11

# 새 시드 평가: 학습 없음
.\.venv\Scripts\python.exe -X utf8 -B -m rl.evaluate_ppo --model "reports/local/stage1/models/best_model.zip" --seed-base 30000 --output-dir "reports/local/stage1/evaluation/heldout_seeds"

# 추가 학습 연결만 확인: 학습·저장 없음
.\.venv\Scripts\python.exe -X utf8 -B -m rl.train_ppo --seed 11 --check --quick
```

현재는 `--quick`으로 기존 Stage 3 최적 모델에서 긴 귀환 구간을 집중 학습합니다.
18번부터 역순으로 확장하며 13~18번 귀환을 40%, 현재 구간을 40% 비중으로 연습합니다.
짧은 실험은 추가 300,000 물리 프레임이며 `reports/local/experiments/stage3_return_connection/`에 저장합니다.
행동 유지 간격은 9프레임이고, 온기 목표는 실제 점수 계산에 따라 기본 12점부터 조절합니다.
학습 후 새 시드 200판으로 선택 모델을 자동 검증합니다. 자세한 기준은 현재 실행 안내를 참고하세요.
물건 획득 연습은 `--quest-only --quick`으로 실행하며 `stage2_quest_acquisition/`에 따로 저장합니다.
`stage1/config.json`은 당시 실험 기록입니다. 현재 설정은 `rl/ppo_config.py`입니다.
