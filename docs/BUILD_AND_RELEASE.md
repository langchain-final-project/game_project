# 실행·Git 공유·배포

## 팀 환경

Python 3.12, `requirements.txt`(게임), `requirements-rl.txt`(환경), `requirements-dev.txt`(테스트), `requirements-build.txt`(빌드)를 구분합니다. 각자의 `.venv`를 만들고 시스템 전역 환경을 공유하지 않습니다.

소스 ZIP을 풀면 최상위 `kkamang_game/` 아래 README와 main.py가 있습니다. 이 디렉터리를 Git 저장소 루트로 쓰세요. 기존 저장소에 넣는 경우에는 파일 비교 후 반영하고 기존 커밋을 강제로 덮어쓰지 마세요. 이 작업에서는 원격 저장소 생성/푸시를 하지 않았습니다.

```sh
git init
git add .
git commit -m "Integrate v0.2 mandatory missions and unified UI"
```

`.github/workflows/test.yml`은 Ubuntu/Windows Python 3.12에서 테스트와 경로 재생을 실행하도록 준비했습니다. 아직 원격 실행 결과는 없습니다.

## Windows 배포

Windows에서 `BUILD_WINDOWS.bat`를 실행합니다. `dist/KkamangManor/`의 exe와 `_internal` 등 **폴더 전체**를 ZIP으로 배포합니다. 소스 배포본에 Windows 실행 파일을 미리 포함했다고 표시하지 마세요. Windows 빌드·실제 PC 동작은 이번 환경에서 실행하지 않았습니다.

확인할 항목: 한글 폴더 경로, Python 없는 PC 실행, 사운드, 창 크기 변경, Alt-Tab 후 자동 일시정지, 새 게임/저장/재실행, 방 1 귀환, 엔딩, 오류 없는 종료.

## 데이터 보존

기본 저장은 `KkamangManor/save_v02.json`이며 v0.1 `save.json`을 덮어쓰지 않습니다. Windows는 LOCALAPPDATA, macOS는 Application Support, Linux는 XDG_DATA_HOME 또는 ~/.local/share를 사용합니다. 원자적 저장과 이전 정상 백업을 유지합니다. 중간 탐사 자체는 이어하기 저장 대상이 아닙니다.

개발용 `--dev-stage`, `--dev-screen`은 임시 저장과 임시 해금을 사용합니다. 일반 플레이 데이터에 검증용 완료 기록을 섞지 않습니다. `--save-path`만 사용하면 지정 파일이 실제 저장 대상입니다.

## 레벨 수정 후

```sh
python tools/plan_routes.py --include-safe
python -m pytest -q
python tools/replay_validation.py
python tools/evaluate.py --episodes 10 --output reports/local
```

경로 생성은 한정 폭 탐색입니다. 찾았다는 결과는 실행 가능한 경로의 증거이고, 못 찾았다고 반드시 불가능한 맵이라는 뜻은 아닙니다. 최단 경로 보장도 없습니다. 생성된 경로만으로 코어가 올바르다고 주장하지 않고 독립 규칙 테스트와 사람/AI 일치 검사도 함께 사용합니다.
