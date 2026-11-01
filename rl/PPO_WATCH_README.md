# PPO 모델을 게임 화면으로 보기

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m rl.watch_ppo "reports/local/stage1" --stochastic --seed 11
```

보관한 `config.json`과 `models/best_model.zip`을 읽어 실제 Stage 1 시작점에서
온기 0점·물건 미획득으로 시작합니다. 커리큘럼 시작 상태는 적용하지 않습니다.
PPO가 행동을 선택하며 목표 경유지 안내와 이동 가능 마스크는 학습 조건을 유지합니다.
화면은 실제 실시간 물리 이동을 재생합니다. 학습은 실행하지 않습니다.

- `--stochastic`: 확률적 플레이. 생략하면 결정적 플레이.
- `--seed 11`: 환경 및 행동 추첨 시드.
- `--speed 0.5`: 절반 속도로 관찰.
- `--checkpoint best_model.zip`: `models/`에서 선택할 체크포인트.
- `R`: 같은 시드로 재시작, `Space`: 일시정지, `Esc`: 종료.

창 없이 로딩·렌더 연결을 확인할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m rl.watch_ppo "reports/local/stage1" --stochastic --headless --frames 3
```
