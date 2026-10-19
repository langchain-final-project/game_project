# 복구 점검과 v0.2 변경 기록

## 2026-10-19 · 보송 귀·꼬리 원화 v3와 main c89d8d7 통합

- 내장 이미지 생성 도구의 정밀 편집으로 기억 원화 세 장의 보송을 교정. 사용 도구·대상·그림별 프롬프트는 `assets/art/memories/generation_prompts-anatomy-v3.json`에 보존하고 최종 `memory_01.png`~`memory_03.png`는 도구가 반환한 PNG 그대로 적용.
- 보송의 양쪽 귀를 끝만 굽은 삼각형이 아니라 머리 앞쪽으로 낮게 접힌 전체 귀 형태로 통일. 첫 그림의 중복 꼬리를 제거하고, 두 번째 그림은 한 가닥 꼬리와 뒷발이 구분되도록 정리. 세 번째 그림은 세 고양이의 기존 꼬리를 각각 한 가닥씩 유지하며 보송 귀만 맞춤.
- 인물·소품·원화 구도와 기존 공개 마스크를 유지. 과거에 까망이를 추가하지 않고 폭풍의 밤·같은 몸의 유령화·상부 온기 이탈을 보존. 현재 장면의 `resident_cat`도 흰색 보송인 `variant=0`만 접힌 귀로 맞추고 다른 캐릭터와 기본 PNG API는 유지.
- 최신 `main`의 `c89d8d7`에 통합. 자석이 +1 온기만 끌어오는 최신 판정, 로비의 앉은 고양이 PNG, ESC 저장 후 타이틀 복귀 동작과 최신 참조 경로를 보존. 레벨의 스토리 표시 필드 이외 맵·좌표·규칙도 해당 `main`과 동일하게 유지.
- 로컬 전체 자동 검사 **453개와 하위 사례 137개** 통과, 소요 **26.25초**. 최신 경로 재생 **36/36**, 레벨 해시 **12/12**, JSON 스키마 **12/12** 통과. 생성 PNG와 배포 자산의 해시 일치, 누적 공개 화면 15개와 귀·꼬리의 독립 시각 검토도 확인. 아래 450개·`ac043a6` 기록은 이전 통합 이력으로 보존.
- 실제 앱의 800×450 앨범·기억 세 장·해방·진엔딩·7번 방·로비를 확인하고, 홈 ESC의 저장 후 타이틀 복귀를 앱 명령으로 검사. 5프레임 CLI 실행도 통과. 첫째·셋째 앨범과 진엔딩 이미지를 직접 확인하고 검토 산출물을 대화 작업공간의 `outputs/memory-anatomy-v3/`에 보존.
- 원격 CI는 마지막 커밋의 실행 결과에서 별도로 확인. 실제 사람 플레이·실기기 확인·실행 파일 빌드·오디오 청취는 미실시.

## 2026-10-19 · 원화 스타일 v2와 최신 main 통합

- 최신 `main`의 `ac043a6`를 기준으로 스토리와 원화를 다시 통합. PNG 고양이·유령·맵·아이템 API, 자유 이동, 자동 수집, 시간제 효과와 게임 규칙을 유지. 모든 레벨은 물건 이름·그림 종류·기능 표시를 제외한 맵·좌표·시간·판정을 해당 `main`과 동일하게 보존.
- 내장 이미지 생성 도구로 기억 원화 세 장을 게임 화풍에 맞춰 재제작하여 `assets/art/memories/memory_01.png`~`memory_03.png`에 적용. 스타일 참조와 사용 프롬프트는 `assets/art/memories/generation_prompts-style-v2.json`에 보존.
- 작업 중 추가된 안내 문구도 통합. 유령 명칭·배지 아이콘·이번 탐험의 별 개수 안내를 유지하고, 수집 배지 10점 귀환과 필수 물건+온기 4점 귀환 조건을 실제 판정에 맞춰 표시.
- 큰 호박색 눈·둥근 체형·선명한 윤곽의 화풍으로 통일. 보송은 흰 털과 빨간 방울 장식, 이파리는 주황·흰 털과 꽃 달린 밀짚모자, 도레는 회색·흰 털과 빨간 리본으로 구분. 과거의 원화에 까망이를 추가하지 않고 세 번째 그림은 폭풍의 밤을 유지.
- 새 배치에 맞춰 공개 마스크를 조정하고, 원화·누적 공개·축소 화면을 자동 픽셀 검사와 별도로 시각 검토. 앞서 공개된 그림을 유지하면서 마지막 얼굴·유령화·온기 이탈이 조기 노출되지 않도록 확인.
- 최신 `main`의 여유 모드를 연습으로 유지. 점수는 보여 주되 완료·기억 영역·온기 회수·별점·꾸미기·호감도·엔딩 진행과 캠페인 저장 기록을 늘리지 않음.
- 탐험의 남은 시간과 결과의 정밀 시간 표시는 이동 프레임을 반영하되 공식 결과·리플레이·저장·점수 계산과 분리. 기억 세 장과 기본·진엔딩 후일담의 진행 방식은 유지.
- 로컬 전체 자동 검사 **450개와 하위 사례 137개** 통과, 소요 **26.49초**. 참조 경로 재생 **36/36**, 레벨 해시 **12/12**, JSON 스키마 **12/12** 통과. 실제 앱의 800×450 앨범·기억 공개·1/2/7/12번 방·연습 결과를 렌더하고 주요 이미지를 직접 확인. 아래 `d1a91f2`·434개 검사 기록은 이전 통합 시점의 이력으로 보존.
- 실제 사람 플레이, 실기기 확인, 실행 파일 빌드, 오디오 청취는 미실시. 원격 CI 결과는 해당 커밋의 GitHub Actions 실행에서 별도로 확인.

## 2026-10-19 · 기억 래스터 원화 적용

- 상세 장면 배치에 맞춘 완성 원화 세 장을 내장 이미지 생성 도구로 제작하고 `assets/art/memories/memory_01.png`부터 `memory_03.png`까지의 자산으로 적용. 각 원화는 네 번의 누적 공개와 앨범에서 공유하고 기존 배포 자산 설정에 포함. 사용 프롬프트도 함께 보존.
- 원화의 인물·소품 배치에 맞춰 공개 마스크를 재조정. 과거 장면에 까망이를 넣지 않고 세 번째 폭풍 원본은 진엔딩 이후에도 유지. 하부 몸·꼬리의 온기 점을 원화에서 정리해 담요·꼬리 공개와 마지막 얼굴·유령화 공개를 구분.
- 공개 화면·앨범·썸네일에서 원화를 자르거나 늘이지 않고 원래 비율로 표시. 파일 누락·손상·잘못된 비율·투명 원본을 명확하게 처리하고 원본 로드와 크기 맞춤은 캐시.
- 저장 형식, 영역 해금, 아이템·귀환 조건, 기본·진엔딩 흐름은 유지. 게임 실행에는 Pygame만 사용하는 구성을 유지.
- 실제 원화·누적 공개·축소 화면·게임 화면을 확인하고 저장·진행·엔딩 회귀 검사도 실행. 검증 결과는 `docs/VALIDATION.md`에 기록.
- 작업 중 갱신된 `main`의 `d1a91f2`에 리베이스. 새 PNG 캐릭터·유령·맵·가구·아이템·유령 방향 추적과 로비 단순화를 보존하고, 난로·문·시계는 시네마틱에만 연결. 진엔딩의 네 고양이 잠든 모습과 해방 후 시계 재개를 보존.

## 2026-10-19 · 스토리 제안서 개정 3.0 적용

- 최신 `main`의 자유 이동, 자동 온기 수집, 추격 유령, 시간제 자석·무적, 별점·호감도·대표 기억 선택과 저장 버전 4를 유지. 레벨의 필수 물건 이름·그림 종류·기능 표시만 변경하고 맵·좌표·시간·게임 규칙은 보존.
- 본편 프롤로그·쉼터·기억 공개의 대사와 설명 자막을 제거. 조작·아이템명·귀환 조건·수집 진행도는 기능 안내로 유지.
- 12개 별도 사진 화면을 세 장의 고정 기억 그림으로 교체. 스테이지 1~4, 5~8, 9~12를 각 그림의 네 영역에 연결하고 첫 성공 완료에서만 영구 공개.
- 이미 열린 픽셀을 유지하는 불규칙 마스크와 불투명 가림막을 추가. 세 번째 그림의 얼굴·온기 이탈·유령 윤곽은 마지막 영역에 함께 배치하며 진엔딩 뒤에도 과거 그림을 변경하지 않음.
- 앨범을 세 슬롯과 `0/4~4/4` 진행도로 변경. 부분 공개는 약 2초 후 복귀하고, 완성 그림과 다시 보기는 직접 넘기도록 구성.
- 필수 물건을 수건·안내 지도·발자국 사진·정원 안내서·스카프 고리·씨앗 주머니·잃어버린 악보·찻잔 받침·커튼 끈·담요·마지막 악보·정원 손잡이로 적용.
- 화면 엔딩을 기본 `네 자리는 여기`와 진엔딩 `조금만 비켜 줘`로 구성. 정원문·해방은 무대사로 연출하고 각 후일담에서만 대사 세 줄을 표시. 기존 호감도 6종 분류 함수와 기록은 저장 호환을 위해 보존하되 결말 대사를 덧붙이지 않음.
- 진엔딩은 각 방의 성공 귀환 온기 10점 기록을 모두 요구. 무피해·별점·전량 수집과 독립이며 기본 엔딩 후 마지막 기록을 채우면 쉼터에서 해방을 한 번 연출.
- `completed`에서 그림 영역을 계산하고 기존 `photos`·대표 사진·별점·호감도·6종 감상 기록은 보존. 증거 없는 온기 기록을 생성하지 않음. `narrative`에 기본·진엔딩·해방 감상 상태를 별도 저장하고 잘못된 형식은 거절.
- 쉼터의 자리 변화와 행동, 기억 완성에 따른 음악 구절, 해방 후 새벽 소리·현재 친구의 몸 상태를 연결. 재탐험으로 해방 상태가 되돌아가지 않도록 구분.
- 그림 불변성·가림막·기억 해금·구저장·진엔딩·기억 선택·기존 자유 이동을 함께 검사하는 회귀 테스트를 추가하고 기존 스토리 테스트를 새 흐름에 맞게 갱신.

## 2026-10-16 · 이전 스토리 적용과 리베이스 기록

이 시점의 대사 중심 본편·방별 사진·엔딩 연결 화면은 위의 개정 3.0 구현으로 교체했습니다. 당시 작업에서는 `5e59762`의 팀 변경을 기반으로 앱·캠페인·메뉴·레벨·참조 경로 충돌을 해결하고, 기본 귀환 및 온기 회수 기록을 스토리와 연결했습니다. 게임 규칙과 저장 보존 원칙은 이번 개정에서도 이어집니다.

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
