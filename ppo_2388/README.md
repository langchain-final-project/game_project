# PPO 2,388 · A2C 공유본과 같은 환경

`a2c_2388.zip`을 별도 폴더에 복사한 뒤 학습 알고리즘을 Stable Baselines3 PPO로 변경했습니다.
기존 게임 프로젝트와 ZIP 원본은 수정하지 않았습니다.

**GitHub 공유 결과는 [trained_snapshot/README.md](trained_snapshot/README.md)에 있습니다.**
다운로드한 저장 모델은 `python run.py check --run trained_snapshot`으로 검사하고,
발표용 모델은 `python run.py watch --run trained_snapshot --model checkpoints/round_096_step_0009609344.zip --stage 9`로 관람합니다.
아래 `prepare`·기본 `train` 명령은 새로운 로컬 학습을 시작할 때 사용합니다.

설치·새 모델 준비·검사·학습 명령은 아래와 같습니다.
Windows 파일 잠금 오류 이후 기존 학습을 재개하는 방법은 [RESUME_AFTER_IO_FIX.md](RESUME_AFTER_IO_FIX.md)에 있습니다.
수정 이유와 코드 읽는 순서는 [PPO_CHANGES.md](PPO_CHANGES.md)에 있습니다.

## 유지한 비교 조건

- Stage 1~12, basic 모드, 원래 출발점·물리·게임 점수·종료 규칙
- 2,388개 입력, `exit_distance_field_v1`, 기존 CNN·MLP·로짓 제한
- 5개 행동, `action_repeat=18`, 자동 상호작용, 유령 초기 위상 고정
- `RefinementEnv`, 기존 보상과 `return_progress_guard`
- 환경 4개 × `n_steps=32` = rollout당 새 경험 128개
- 학습률 0.0002, gamma 0.997002999, GAE lambda 1.0, 초기 entropy 계수 0.03
- 엔트로피 자동 조정, 취약 맵 집중 학습, 평가·선정·최종 검증 절차
- seed 11, 총 10,000,000스텝 예산, CPU·Torch 스레드 1개
- 원본 `requirements.txt`에 명시된 패키지 버전

PPO 전용 설정은 `batch_size=128`, `n_epochs=10`, `clip_range=0.2`입니다.
가치 함수 clipping과 KL 조기 종료는 사용하지 않습니다. 이 값은 첫 비교 실험용이며 최적값으로 검증되지는 않았습니다.
같은 학습 스텝 수에도 반복 최적화가 늘어나므로 실행 시간은 A2C와 달라질 수 있습니다.
취약 맵 선정은 각 모델의 성적을 따르므로, 같은 배분 규칙이라도 실제 맵별 경험량은 달라질 수 있습니다.

## Windows에서 다음에 실행할 명령

아래 명령은 이 README와 `run.py`가 있는 `ppo_2388` 폴더에서 실행합니다.
기존 프로젝트의 가상환경과 구분하려고 이 폴더 안에 `.venv`를 만듭니다.

### 1. 설치

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. 새 PPO 모델 준비

```powershell
.\.venv\Scripts\python.exe run.py prepare
```

원본 `shared_snapshot/config.json`을 읽어 PPO 설정을 덧붙인 뒤 새 실행 폴더에 저장합니다.
원본 A2C 모델 가중치는 불러오지 않습니다. 임의 가중치·빈 Adam·0스텝에서 시작합니다.
다시 실행하면 별도의 새 실험을 만들고 `runs/latest.json`이 그 실험을 가리킵니다.

### 3. 입력·환경·저장 모델 검사

```powershell
.\.venv\Scripts\python.exe run.py check
```

`status: passed`, `input_dimensions: 2388`, `empty_Adam: true`, `training_updates_added: 0`을 확인합니다.
위 값은 학습 전 새 모델 기준입니다. 이 검사는 전12맵의 입력과 행동 연결 및 저장 모델을 확인하며,
학습 업데이트나 전체 성능 평가는 수행하지 않습니다.

### 4. 학습 및 재개

```powershell
.\.venv\Scripts\python.exe run.py train --hours 8
```

첫 실행에서 학습 전 성능을 평가한 뒤 학습합니다. 같은 명령을 다시 실행하면 최신 실행을 이어갑니다.
시간 예산에는 평가도 포함되며, 진행 중인 평가 작업 때문에 종료가 지연될 수 있습니다.
`Ctrl+C`로 중단할 수 있습니다. 재개 시에는 저장된 가중치·Adam·카운터를 복원하지만 진행 중 에피소드와 난수 상태는 복원하지 않습니다.

별도의 실험을 새로 시작하려면 다음 명령을 사용합니다.

```powershell
.\.venv\Scripts\python.exe run.py train --new --hours 8
```

새 모델을 준비한 뒤에는 소스 파일과 해당 실행의 설정 파일을 수정하지 마세요.
재개 시 저장 당시의 파일 해시와 비교합니다. 코드를 변경해 실험하려면 새 실행을 만드세요.

### 5. 학습 현황 및 선정 모델 관람

별도 터미널에서 실행합니다.

```powershell
.\.venv\Scripts\python.exe run.py dashboard
```

브라우저 주소: http://127.0.0.1:8769

```powershell
.\.venv\Scripts\python.exe run.py watch
```

`Tab`: 다음 맵, `Shift+Tab`: 이전 맵, `Space`: 일시정지, `Esc`: 종료.
초기에는 선정 모델도 0스텝 모델입니다. 학습 중 최신 모델은 `run.py watch --model latest`로 관람할 수 있습니다.
특정 실행은 `--run runs/실행폴더이름`으로 지정합니다.

## 파일과 기록 구분

- `game_source/`: 원본 게임 파일. 변경하지 않았습니다.
- `shared_snapshot/`: 원본 A2C 설정·모델·결과. 변경하지 않았고 PPO 성적으로 사용하지 않습니다.
- `runs/`: 새 PPO 실험 결과와 기존 체크포인트에서 이어가는 재개 폴더입니다.
- `a2c_game/`: 기존 패키지 경로를 유지한 PPO 실행 코드입니다.
- `a2c` 설정 키와 `UnifiedA2CPolicy` 클래스 이름은 기존 환경·신경망과의 호환을 위해 유지합니다.
- `PACKAGE_MANIFEST.json`: ZIP 원본의 파일 해시 기록입니다. 수정한 PPO 소스의 검증 자료로 사용하지 않습니다.
  PPO 학습 재개에는 각 실행의 `metadata.json`과 `source_snapshot/` 해시를 사용합니다.

## 학습 횟수 읽는 법

| 기록 | 단위 | 새 경험 128개를 학습했을 때 |
|---|---|---:|
| `steps` | 네 환경의 행동 수 합계 | 128 |
| `rollout_updates` | 완료한 rollout 수 | 1 |
| `updates` | PPO epochs / Adam 갱신 수 | 10 |

128개 전체 배치를 매 epoch 한 번 학습하므로 이 설정에서는 epoch 수와 Adam 갱신 수가 같습니다.
로그·평가·엔트로피 조정 간격은 원본의 rollout·환경 스텝 단위를 유지합니다.
고정 배치·epochs·KL 설정은 저장 검증의 전제이므로 `ppo_settings()`에서 다른 값을 거부합니다.

macOS에서는 같은 순서로 `python run.py prepare`, `check`, `train`을 실행합니다.
설치는 `setup.command`, 관람과 현황은 각각 `watch.command`, `dashboard.command`를 사용할 수 있습니다.
