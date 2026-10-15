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

각 방에서 **필수 물건 1개 + 온기 4점 이상**을 챙겨, 출발했던 문에서 E를 누르세요. 악보·책·담요 등 물건을 빠뜨리면 온기가 충분해도 완료되지 않습니다. 조건 미충족 귀환은 안내를 띄우며 1틱을 소비하고 탐사를 계속합니다.

- 방향키/WASD: 이동, E: 같은 칸의 물건·온기 수집/샛문 장치/귀환, Space: 대기, Esc: 일시정지.
- 이동하지 않아도 0.3초마다 시간이 흐릅니다. 메뉴·일시정지·창 포커스 상실 중에는 멈춥니다.
- 온기는 +1 8개, +3 2개로 합계 14점입니다. 큰 온기는 순찰 주변에 있습니다.
- 순찰 유령 접촉 시 용기 1 감소, 다음 3틱 보호. 온기는 차감하지 않습니다.
- 제한시간 또는 용기 소진 시 실패하며 그 판의 확정 점수는 0입니다. 이전 사진·꾸미기는 유지됩니다.
- 필수 물건은 점수를 대신하지 않습니다. 기본 완료로 방 복원·기억 사진·다음 방을 얻습니다.
- 온기 10점 이상 귀환 / 무피해 완료는 각각 선택 배지 1개를 남깁니다. 꾸미기는 최신 `main`의 별점 규칙을 따라 누적 별 6·12·18개에 목도리·러그·액자 선택지가 열립니다.

## feature/story · 스토리 제안서 반영

비를 피해 들어온 저택에서 보송·이파리·도레의 기억과 체온을 되찾는 이야기입니다. 01~04방은 고양이들의 정체 발견, 05~08방은 우정과 네 번째 자리, 09~12방은 보호 마법의 비밀과 열린 정원문으로 이어집니다. 각 방의 필수 물건 이름·그림·설명과 기억 장면이 연결됩니다.

- 최신 `main`의 호감도와 대표 기억에 따른 **6종 엔딩 판정**을 유지합니다. 재도전 후 대표 기억 선택이 남아 있으면 먼저 유지/교체를 결정한 뒤 결말을 봅니다.
- 기본 후일담 **문은 열어 둘게**: 12개 방을 완료하면 선택된 엔딩 뒤에 이어집니다. 친구들은 까망이를 붙잡지 않고, 까망이는 스스로 돌아옵니다.
- 온기 회복 후일담 **함께 맞는 아침**: 각 방에서 한 번 이상 **온기 10점 이상 + 필수 물건**으로 귀환한 기록을 12개 모두 모으면 선택된 엔딩 뒤에 이어집니다. 세 친구가 고양이의 모습을 되찾고 쉼터에 새벽이 찾아옵니다. 이 추가 수집은 호감도 엔딩의 판정과 별도로 기록합니다.
- 완료한 방은 다시 도전할 수 있습니다. 마지막 부족한 온기 기록을 어느 방에서 채워도 결과 화면에서 후일담을 볼 수 있고, 이후 쉼터에서도 결말과 함께 다시 열 수 있습니다.
- 기록은 기존 `rich` 배지를 활용합니다. 무피해 배지는 온기 기록을 대신하지 않으며 기존 저장 파일도 그대로 이어집니다. 대사 건너뛰기·기억 재생은 보상과 엔딩 조건을 바꾸지 않습니다.
- 유령과의 접촉은 차가운 기억에 놀라는 소리·테두리·문구로 표현합니다. 최신 `main`의 추격 유령, 프레임 이동, 별점, 호감도, 레벨 배치와 610차원 관측 규칙을 그대로 사용합니다.

제안서의 진행별 BGM 레이어와 방별 고유 잔향은 후속 연출 작업으로 남겨 두었습니다. 현재 배경음악은 기존 피아노 곡을 사용합니다.

## 작업 구분

| 파일 | 수정 영역 |
|---|---|
| `kkamang/core.py`, `levels.py` | 유일한 게임 판정·레벨 검증 |
| `data/levels/stage_01.json` … `stage_12.json` | 맵·필수 물건·순찰·샛문·시간 |
| `kkamang/ui/illustration.py` | 모든 화면이 공유하는 고양이·유령·가구·쉼터 |
| `kkamang/ui/field.py`, `menus.py`, `theme.py` | 탐사 화면·메뉴·공통 버튼·텍스트 |
| `kkamang/app.py` | 입력·0.3초 틱·장면·저장 연결 |
| `kkamang/campaign.py`, `story.py` | 지속 보상·세이브·대사 |
| `kkamang/env.py` | 같은 코어를 쓰는 6행동·610차원 학습 환경 |

새 그림에는 글자를 굽지 않습니다. PNG 원본은 보존했지만 주요 배경·캐릭터는 동일한 코드 기반 일러스트로 표시합니다. 독립적인 새 게임 판정은 UI에 추가하지 않습니다.

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

강화학습 모델은 아직 학습하지 않았습니다. Windows 실행 파일은 Windows에서 `BUILD_WINDOWS.bat`로 만들고 실제 PC에서 확인해야 합니다. 이 묶음은 팀 개발용 소스 배포본입니다.
