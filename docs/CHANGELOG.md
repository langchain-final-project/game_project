# 복구 점검과 v0.2 변경 기록

## 연결 복구 시 확인한 상태

기존 수정 파일 18개(12개 레벨 JSON, app/core/env/levels/story/field)와 새 illustration.py 1개를 확인했습니다. 그 시점에는 문법/레벨 로딩만 확인됐고 현재 규칙의 게임 실행·테스트·UI 통합·팀 문서는 미완료였습니다. 원본 ZIP을 보존하고 기존 작업본을 체크포인트로 남긴 뒤 이어서 수정했습니다.

## 이번에 완료한 통합

- 12개 방에 필수 물건을 연결하고 온기만 챙긴 조기 완료를 막음. 조건 미달 안내, 수집 이벤트, 정산 검증, 관측값을 동기화.
- +3 온기를 순찰 주변에 두고, 모든 방의 기본 안전 경로와 추가 수집 경로를 실제 재생 검증.
- 효과가 부족했던 장치 위치를 다시 수정. 특정 동선에서 장치 조작을 포함한 3~9틱의 정적 이득 확인.
- 공통 캐릭터 rig/4방향 발 움직임/대기·수집·놀람·기쁨, 종이 버튼과 라이브 텍스트, 가구·발 위치 정렬·조명 통합.
- 홈·대화·앨범의 서로 다른 캐릭터 그림을 공통 렌더링으로 교체. 대화 화면의 중복 등장과 함수명 충돌 수정.
- 방 복원 소품과 주민의 작은 활동, 단계별 등불, 목도리·러그·액자 반영.
- 개발용 방/화면 바로 실행을 임시 세이브로 격리. 기본 save_v02.json으로 기존 세이브 보호.
- JSON 스키마/602 관측 계약/버전/문서를 실제 구현과 일치시킴. 오래된 검증 결과를 현재 결과에서 분리.
- 실행/빌드/개발 의존성, Windows 실행 스크립트, .gitignore, CI, 팀 인터페이스 문서 추가/정리.

## 충돌과 중복 수정 방지

v0.1 원본을 덮어쓰지 않았습니다. 기존 부분 작업을 이어서 수정했으며 별도의 두 번째 게임 구현을 만들지 않았습니다. 공통 core·illustration을 한 곳에서 사용합니다. 파일별 해시 비교는 `validation/change_manifest.json`에 있습니다. 원격 Git 작업은 수행하지 않았으므로 업로드 전 이 소스 폴더를 팀 기준 커밋으로 삼으면 됩니다.

## 남은 사항

자동 검사 범위를 넘는 Windows 배포 실기기 확인, 사람의 재미/아트 피드백, 강화학습 훈련은 후속 작업입니다. 현재 검증으로 버그가 전혀 없거나 모든 사람이 재미있게 느낀다고 보장하지 않습니다.

## 파일별 변경 목록

| 상태 | 파일 |
|---|---|
| added | `.github/workflows/test.yml` |
| added | `.gitignore` |
| modified | `BUILD_WINDOWS.bat` |
| modified | `CONTRACT.md` |
| modified | `README.md` |
| modified | `RUN_WINDOWS.bat` |
| modified | `data/levels/level.schema.json` |
| added | `data/levels/observation_contract.json` |
| removed_from_current_package | `data/levels/observation_contract_proposal.json` |
| modified | `data/levels/stage_01.json` |
| modified | `data/levels/stage_02.json` |
| modified | `data/levels/stage_03.json` |
| modified | `data/levels/stage_04.json` |
| modified | `data/levels/stage_05.json` |
| modified | `data/levels/stage_06.json` |
| modified | `data/levels/stage_07.json` |
| modified | `data/levels/stage_08.json` |
| modified | `data/levels/stage_09.json` |
| modified | `data/levels/stage_10.json` |
| modified | `data/levels/stage_11.json` |
| modified | `data/levels/stage_12.json` |
| modified | `docs/ARCHITECTURE.md` |
| modified | `docs/BUILD_AND_RELEASE.md` |
| added | `docs/CHANGELOG.md` |
| modified | `docs/CREDITS.md` |
| added | `docs/DESIGN_SYSTEM.md` |
| added | `docs/LEVELS.md` |
| modified | `docs/PLAYTEST_GUIDE.md` |
| added | `docs/RL_READY.md` |
| modified | `docs/VALIDATION.md` |
| removed_from_current_package | `field_qa.png` |
| added | `historical_v01/evidence/cosmetic_cat_variants.png` |
| added | `historical_v01/evidence/exploration_live.png` |
| added | `historical_v01/evidence/field_live_collected4.png` |
| added | `historical_v01/evidence/field_stage01.png` |
| added | `historical_v01/evidence/field_stage05.png` |
| added | `historical_v01/evidence/field_stage12.png` |
| added | `historical_v01/evidence/frozen_title.png` |
| added | `historical_v01/evidence/home_live.png` |
| added | `historical_v01/evidence/result_live.png` |
| added | `historical_v01/evidence/title_live.png` |
| added | `historical_v01/field_qa.png` |
| added | `historical_v01/observation_contract_proposal.json` |
| added | `historical_v01/validation/all_tests.txt` |
| added | `historical_v01/validation/all_tests.xml` |
| added | `historical_v01/validation/app_integration_results.json` |
| added | `historical_v01/validation/app_integration_tests.txt` |
| added | `historical_v01/validation/core_unit_results.json` |
| added | `historical_v01/validation/core_unit_tests.txt` |
| added | `historical_v01/validation/final_test_results.json` |
| added | `historical_v01/validation/linux_build_log.txt` |
| modified | `kkamang/__init__.py` |
| modified | `kkamang/app.py` |
| modified | `kkamang/campaign.py` |
| modified | `kkamang/core.py` |
| modified | `kkamang/env.py` |
| modified | `kkamang/levels.py` |
| modified | `kkamang/story.py` |
| modified | `kkamang/ui/field.py` |
| added | `kkamang/ui/illustration.py` |
| modified | `kkamang/ui/menus.py` |
| modified | `kkamang/ui/theme.py` |
| modified | `main.py` |
| added | `requirements-build.txt` |
| modified | `requirements-dev.txt` |
| added | `requirements-review.txt` |
| modified | `tests/test_core.py` |
| modified | `tests/test_env.py` |
| added | `tests/test_upgrade.py` |
| added | `tools/capture_review.py` |
| added | `tools/check_level_schema.py` |
| modified | `tools/evaluate.py` |
| added | `tools/plan_routes.py` |
| modified | `tools/replay_validation.py` |
