# 까망이와 잠 못 드는 저택 · v0.2.0

## 시작하기

Python 3.12를 사용합니다. Windows는 `RUN_WINDOWS.bat`, macOS/Linux는 `sh run.sh`로 실행합니다. 첫 실행에는 패키지 다운로드 연결이 필요합니다.

직접 실행하는 공통 명령:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

게임 실행에는 Pygame만 필요합니다. Gymnasium·NumPy는 강화학습 환경/검증용입니다. Pillow는 화면 증거 재생성 도구에만 필요합니다.

## 이번 버전의 목표

각 방에서 **필수 물건 1개 + 온기 4점 이상**을 챙겨, 출발했던 문에서 E를 누르세요. 악보·책·담요 등 물건을 빠뜨리면 온기가 충분해도 완료되지 않습니다. 조건 미충족 귀환은 안내를 띄우며 탐사를 계속합니다.

- 방향키/WASD: 이동 및 온기 자동 수집, E: 같은 칸의 필수 물건/샛문 장치/귀환, Space: 대기, Esc: 일시정지.
- 이동하지 않아도 0.3초마다 시간이 흐릅니다. 메뉴·일시정지·창 포커스 상실 중에는 멈춥니다.
- 고양이는 칸 중간에서도 방향을 바꾸고 키를 떼면 멈춥니다. 입력은 다음 1/60초 이동 프레임에 반영되며, 한 칸 거리를 이동하는 속도는 0.3초로 유지합니다. E는 한 이동 프레임을 사용하고, 누르고 있는 방향키로 이어서 이동합니다.
- 온기는 +1 8개, +3 2개로 합계 14점입니다. 큰 온기는 순찰 주변에 있습니다.
- 2번·7번 방의 자석은 밟으면 즉시 소모됩니다. 주변 3×3의 +1 온기만 벽·닫힌 문과 관계없이 끌어오며, +3 온기는 직접 밟아야 획득합니다. 남은 13틱은 세계 틱마다 감소합니다. 효과는 방을 다시 시작하면 초기화됩니다.
- 3번 `[7,5]`·5번 `[12,2]`·11번 `[14,6]`의 무적 아이템은 밟으면 즉시 피해를 막습니다. 남은 10틱은 세계 틱마다 감소하며, 보호 표시와 추격 유령의 접촉 정지가 유지됩니다. 다음 방으로 효과를 가져갈 수 없습니다.
- 순찰 유령 접촉 시 용기 1 감소, 다음 3틱 보호. 온기는 차감하지 않습니다.
- 제한시간 또는 용기 소진 시 실패하며 그 판의 확정 점수는 0입니다. 이전 사진·꾸미기는 유지됩니다.
- 필수 물건은 점수를 대신하지 않습니다. 기본 완료로 방 복원·기억 사진·다음 방을 얻습니다.
- 온기 10점 이상 귀환 / 무피해 완료는 각각 선택 배지 1개. 중복 획득 없이 배지 4·8·12개에 목도리·러그·액자 선택지가 열립니다.

## 작업 구분

| 파일 | 수정 영역 |
|---|---|
| `kkamang/core.py`, `realtime.py`, `levels.py` | 공통 게임 판정·연속 이동·레벨 검증 |
| `data/levels/stage_01.json` … `stage_12.json` | 맵·필수 물건·순찰·샛문·시간 |
| `kkamang/ui/illustration.py` | 모든 화면이 공유하는 고양이·유령·가구·쉼터 |
| `kkamang/ui/cat_sprites.py` | 고양이 PNG의 프레임 좌표·발 기준 정렬·이미지 캐시 |
| `kkamang/ui/ghost_sprites.py` | 유령 A/B/C의 PNG·방향별 자세·몸체 정렬·이미지 캐시 |
| `kkamang/ui/field.py`, `menus.py`, `theme.py` | 탐사 화면·메뉴·공통 버튼·텍스트 |
| `kkamang/app.py` | 입력·1/60초 이동 프레임·0.3초 세계 틱·장면·저장 연결 |
| `kkamang/campaign.py`, `story.py` | 지속 보상·세이브·대사 |
| `kkamang/env.py` | 자유 이동 `RealtimeMansionEnv`: 6행동·623차원 (`vector_v6_realtime`), 기존 칸 단위 `MansionEnv`: 620차원 |

새 그림에는 글자를 굽지 않습니다. 고양이는 `assets/art/kkamang.png`의 4방향 대기와 방향별 걷기 6프레임을 탐사·쉼터·대화에서 공유합니다. 원본 PNG를 수정하지 않고 불규칙한 프레임 영역을 잘라 같은 몸통·발 기준으로 정렬합니다. 현재 표정 전환과 목도리 색상 변경은 적용하지 않으며 이미지의 빨간 목도리를 사용합니다. 무적 보호막과 피격 중 걷기는 유지합니다. 배경·가구는 기존 표시를 사용하며 독립적인 새 게임 판정은 UI에 추가하지 않습니다.

탐사 유령 A/B/C는 `assets/art/a.png`, `b.png`, `c.png`를 각각 사용합니다. 4열×2행의 8방향 자세 중 이동 방향에 해당하는 그림을 선택하고 기존 부유·직선 보간을 유지합니다. 정지·충돌 후 대기·일시정지 재개 시 마지막 이동 방향을 유지합니다. 쉼터·대화·호감도 화면은 보송=a, 이파리=b, 도레=c로 같은 PNG를 사용합니다. 엔딩 ID `ghost_b`(도레)와 탐사 유령 ID B(이파리 그림)는 별도 의미이므로 혼동하지 않습니다.

## 개발용 바로 실행

```sh
python main.py --dev-stage 7
python main.py --dev-screen home
python main.py --dev-stage 7 --dev-screen story
python main.py --headless --no-audio --frames 5 --dev-stage 12
```

개발 옵션은 임시 저장을 사용하고 모든 방을 미리 열어 줍니다. 일반 플레이의 완료 기록을 만들지 않습니다. 지정한 `--save-path`도 개발 옵션과 함께 쓰면 무시됩니다. 일반 실행의 기본 저장은 사용자 데이터 폴더의 `KkamangManor/save_v02.json`이며 v0.1 저장을 덮어쓰지 않습니다.

## 같은 기준으로 검증

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
python tools/replay_validation.py
python tools/evaluate.py --episodes 10 --output reports/local
```

현재 검증 결과와 미검증 범위는 `docs/VALIDATION.md`, 변경 전체 목록은 `docs/CHANGELOG.md` 및 `validation/change_manifest.json`에 있습니다. 실제 화면은 `evidence/review_board.png`, 걷기/표정은 `evidence/cat_animation.gif`로 확인할 수 있습니다.

현재 게임의 자유 이동을 학습하려면 `from kkamang.env import RealtimeMansionEnv`를 사용합니다. 한 행동은 1/60초이며 소수점 좌표와 세계 틱 진행률을 관측합니다. 기존 620차원 모델은 호환되지 않습니다. `tools/plan_routes.py`, `tools/evaluate.py` 및 기존 36개 경로 검증은 칸 단위 환경의 검사입니다. 그 결과를 자유 이동의 무피해 경로나 난이도 평가로 해석할 수 없습니다. 새 플레이 기록은 `python tools/replay_validation.py --record <기록.json>`으로 재생 검증합니다.

강화학습 모델은 아직 학습하지 않았습니다. Windows 실행 파일은 Windows에서 `BUILD_WINDOWS.bat`로 만들고 실제 PC에서 확인해야 합니다. 이 묶음은 팀 개발용 소스 배포본입니다.
