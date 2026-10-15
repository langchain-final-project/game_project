# 복구 점검과 v0.2 변경 기록

## 2026-10-16 · 최신 main 위로 story 리베이스

- `5e59762`의 최신 팀 변경을 기반으로 스토리를 다시 적용. 앱·캠페인·탐사 화면·메뉴·참조 경로 파일의 충돌 해결.
- `main`의 추격 유령/프레임 이동/610차원 관측/별점/호감도/대표 기억 선택과 저장 버전 4를 보존. 호감도에 따른 6종 엔딩 판정과 감상 기록도 유지.
- 6종 결말 뒤에 기존 스토리의 기본/온기 회복 후일담을 연결. 방별 rich 배지 12개 조건은 별도로 유지하여 스토리와 팀 기능이 서로의 판정을 덮어쓰지 않도록 통합.
- 최신 레벨의 배치·유령·시간·별점 목표는 보존하고 스토리 아이템명/그림/기억 설명만 결합. 최신 참조 행동열을 유지하고 통합된 레벨의 해시를 갱신.

## 2026-10-16 · feature/story 스토리 제안서 적용

- 프롤로그와 열두 기억을 발견 → 우정 → 보호 마법의 진실로 재구성. 물건 이름·기억 설명·그림을 제안서와 일치시킴.
- 기본 엔딩 `문은 열어 둘게`와 12개 방 온기 10점 달성에 따른 확장 엔딩 `함께 맞는 아침` 추가. 마지막 기록을 앞방 재도전에서 얻어도 엔딩을 열 수 있고 쉼터에서 재생 가능.
- 기존 완료·rich 배지에서 진행도를 계산해 저장 형식과 기존 기록 유지. quiet 배지나 중복 배지는 온기를 대신하지 않음.
- 확장 엔딩과 이후 쉼터에 새벽과 세 친구의 고양이 모습을 표시. 처음 쉼터 비교 화면은 홈에만 적용하여 엔딩으로 상태가 새지 않도록 수정.
- 새 긴 아이템명에 맞게 탐사·준비 화면의 글자 크기 조정. 접촉을 짧은 찬바람·냉기 테두리로 표현.
- 게임 판정·배치·시간·기본 귀환·관측 계약은 변경하지 않음. 레벨의 서사 필드만 바뀌었음을 비교한 뒤 참조 파일 해시만 갱신하고 기존 행동 경로를 재검증.
- 진행별 BGM 레이어와 공간별 고유 잔향은 후속 작업.

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
