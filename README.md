# 까망이와 유령 저택

**KKAMANG GHOST MANOR**

고양이 까망이가 열두 방에서 온기와 물건을 모아 친구들의 기억을 되찾는 Pygame 게임입니다. 이 README가 현재 개발 문서의 단일 기준입니다. 게임 버전은 `0.2.0`, 규칙은 `cat_mansion_0_2`, 저장 형식은 버전 5입니다. 정확한 맵·시간·배치는 `data/levels/stage_*.json`, 판정은 `kkamang/core.py`와 `realtime.py`를 따릅니다.

## 실행과 개발 환경

Python 3.12를 기준으로 합니다. 일반 실행에는 `requirements.txt`의 pygame-ce만 필요합니다.

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

Windows에서는 `RUN_WINDOWS.bat`, Linux/macOS에서는 `sh run.sh`도 사용할 수 있습니다. 개발·테스트는 `python -m pip install -r requirements-dev.txt`, 강화학습 연결만 필요하면 `requirements-rl.txt`를 설치합니다.

| 조작 | 동작 |
|---|---|
| 방향키 / WASD | 이동·메뉴 선택 |
| E | 같은 칸의 필수 물건 회수·샛문 장치·귀환 상호작용 |
| Space | 탐사 대기, 장면에서 다음 |
| Enter | 메뉴 확인·장면에서 다음 |
| Esc | 일시정지·뒤로·장면 넘기기 |

개발 옵션은 임시 저장을 만들며 실제 플레이 저장과 분리됩니다. `--save-path`를 함께 줘도 개발 미리보기에서는 임시 저장을 사용합니다.

```sh
python main.py --dev-stage 7
python main.py --dev-screen home
python main.py --dev-stage 12 --dev-screen story
python main.py --dev-likes happy_all
python main.py --headless --no-audio --frames 5 --dev-stage 7
```

`--dev-likes`는 `overview`, `choice`, `happy_all`, `sad`, `ghost_a`, `ghost_b`, `ghost_c`, `normal`을 지원합니다. 일반 저장은 운영체제의 사용자 데이터 폴더 아래 `KkamangManor/save_v02.json`입니다. `--save-path <경로>`로 검증용 저장을 별도 지정할 수 있습니다. 버전 5는 옛 기록을 이관하며, 손상 파일의 백업과 더 높은 저장 버전의 덮어쓰기 방지 동작을 유지합니다.

## 폴더와 파일 역할

```text
GAME_PROJECT_TEAM/
├─ main.py                       실행 옵션과 GameApp 시작
├─ kkamang/                      게임·저장·화면·환경 연결
│  ├─ app.py                     입력, 장면 전환, 탐사 진행, 저장 연결
│  ├─ core.py                    격자 규칙·충돌·수집·종료 판정
│  ├─ realtime.py                실시간 이동과 세계 틱 연결
│  ├─ levels.py                  레벨 로드와 규칙 검증
│  ├─ scoring.py                 점수·별 계산
│  ├─ campaign.py                진행·보상·저장·이관
│  ├─ story.py                   기억 공개와 현재 이야기 상태
│  ├─ likes.py                   대표 기억·호감도·엔딩 선택
│  ├─ likes_preview.py           개발용 호감도·엔딩 미리보기
│  ├─ ending_content.py          엔딩·낮밤 진입 PNG·내레이션·해방 대상
│  ├─ collection.py              수집품과 해금 판정
│  ├─ audio.py                   장면별 BGM·효과음·음량
│  ├─ input.py                   키 상태와 입력 처리
│  ├─ paths.py                   배포 자산·사용자 저장 경로
│  ├─ env.py                     Gymnasium 관측·행동·보상
│  └─ ui/                        Pygame 화면과 PNG 표시
│     ├─ theme.py                폰트·텍스트·버튼·공통 자산
│     ├─ menus.py                제목·결과·설정·엔딩·후일담
│     ├─ field.py                탐사 화면과 상태 안내
│     ├─ lobby.py                쉼터의 물건·주민 PNG·깊이 정렬
│     ├─ memories.py             기억 원화·가림막·온기 복원 효과
│     ├─ illustration.py         PNG 캐릭터와 공용 코드 연출
│     ├─ cat_sprites.py           까망이 PNG 프레임·발 기준 정렬
│     ├─ ghost_sprites.py         유령 PNG 방향·정렬
│     ├─ map_sprites.py           맵·소품 PNG 표시
│     └─ collection.py / likes.py 수집장·호감도 화면
├─ data/levels/                  12개 레벨, 목록, JSON 스키마, 관측 계약
├─ assets/
│  ├─ art/                       배경·캐릭터·맵·수집품·기억·엔딩 PNG
│  ├─ sound/                     게임에서 재생하는 WAV
│  ├─ fonts/                     폰트와 원문 OFL 허가문
│  ├─ icons/                     아이템 아이콘
│  ├─ provenance/                현재 그림·음원 출처와 재제작 명세
│  └─ licenses/                  외부 샘플의 원문 허가·출처 고지
├─ tests/                        코어·실시간·저장·환경·화면 회귀 검사
├─ tools/
│  ├─ check_level_schema.py      12개 레벨 JSON 구조 검사
│  ├─ replay_validation.py       기준 경로·내보낸 기록 재생 검사
│  ├─ plan_routes.py             격자 환경의 기준 경로 탐색
│  ├─ balance_test.py            격리된 사람 플레이 세션·기록 집계
│  ├─ evaluate.py                격자 환경의 기준 정책 비교
│  └─ audio/                     음원 변환·효과음 제작·피아노 복원
├─ validation/
│  ├─ reference_routes.json      테스트가 읽는 기준 경로·추적 상태
│  ├─ safe_routes.json           테스트가 읽는 경로 증거
│  └─ upgrade_report.json        보존한 레벨 메타데이터의 최신 결과 안내
├─ reports/latest/               최신 통합 검증 결과와 화면 확인 자료
├─ .github/workflows/test.yml    Windows·Linux Python 3.12 검사
├─ requirements*.txt             실행·RL·개발·빌드 의존성
├─ RUN_WINDOWS.bat / run.sh      실행 스크립트
├─ BUILD_WINDOWS.bat             Windows 폴더 배포 빌드
├─ KkamangManor.spec             실행 파일과 assets·data 묶음
└─ README.md                    현재 개발 문서
```

`__init__.py`는 패키지 식별·버전 정보를 제공합니다. `.gitignore`, `.gitattributes`, `pytest.ini`는 생성물 제외, 레벨 줄바꿈, 테스트 설정을 담당합니다. 가상환경·캐시·빌드·임시 평가 결과는 배포 원본이나 최신 검증 결과에 포함하지 않습니다. `reports/evaluation*`와 `reports/local`은 도구를 다시 실행하면 만들어지는 작업용 출력입니다.

출처 JSON의 당시 작업 경로·명령은 제작 기록입니다. 게임이 해당 경로를 따라 파일을 읽지는 않습니다. 폰트·샘플의 원문 허가문은 문서 통합 대상에서 제외해 보존합니다.

## 게임 규칙과 점수

- 모든 방은 16×9 격자입니다. 좌표는 왼쪽 위가 `[0,0]`이고 오른쪽이 x, 아래가 y입니다. 시작점과 귀환점은 같습니다.
- 용기 3으로 시작합니다. 1점 온기 8개와 3점 온기 2개가 있어 총 14점입니다. 온기는 밟으면 자동 수집합니다.
- 필수 물건을 같은 칸에서 E로 회수하고 온기 4점 이상을 가진 채 귀환점에서 E를 누르면 성공합니다. 실패·중도 포기의 반납 온기는 0입니다.
- 데스크톱은 초당 60회 이동을 계산합니다. 한 칸 이동은 18프레임, 세계 틱은 0.3초입니다. 칸 중간에서도 방향을 바꿀 수 있고 대각 이동은 없습니다. 남은 시간·유령 이동은 세계 틱을, 수집·접촉·상호작용은 이동 프레임을 따릅니다.
- 유령 접촉은 한 세계 틱에 최대 용기 1을 줄이고, 이후 보호 시간 3틱을 둡니다. 보유 온기는 잃지 않습니다. 종료가 겹치면 용기 소진, 성공 귀환, 시간 초과 순서로 처리합니다.
- 자석은 획득 즉시 13틱 동안 3×3 범위의 **1점 온기만** 수집합니다. 벽·닫힌 샛문 너머도 범위에 포함하며 3점 온기는 직접 밟아야 합니다. 무적 아이템은 10틱 동안 접촉 피해를 막습니다. 피격 후 보호 시간과는 별도입니다.
- 샛문은 장치와 상호작용하면 해당 탐사에서 열립니다. 추격 유령은 필수 물건 회수 이후 활성화됩니다. 상세 경로·대기 시간·장치 좌표는 레벨 JSON에 있습니다.
- 여유 모드는 제한 시간만 기본의 1.5배로 올림한 연습입니다. 성공한 방 번호는 수집장 연습 기록에 남지만 캠페인 완료·기억 공개·반납 온기·별·배지·호감도·엔딩을 갱신하지 않습니다.

성공 점수는 온기 60 + 속도 25 + 용기 15 = 100점입니다. 온기는 필수 4점에서 추가로 반납한 양을 14점까지, 속도는 제한 틱에서 `star_fast_ticks`까지의 여유를, 용기는 남은 용기/3을 각 배점으로 환산하고 정수 반올림·범위 제한합니다. 45·65·85점 이상이면 별 1·2·3개입니다. 실패·중단 점수는 0입니다.

`rich` 배지는 온기 10점 이상, `quiet` 배지는 무피해 성공 기록입니다. 수집장은 외형 24종과 방별 필수 물건 그림 12종을 사용하며 해금 조건은 `assets/art/collection/catalog.json`을 따릅니다. 수집 배지와 엔딩 판정은 별개입니다.

## 열두 방과 난이도 기준

제한 시간은 기본 모드 기준입니다. `빠른 목표`는 속도 점수 만점 기준 틱이며 클리어 제한이 아닙니다. 순찰은 수/이동 간격(틱), 추격은 존재 여부를 표시합니다.

| 방 | 이름 | 필수 물건 / 좌표 | 제한 틱(초) | 빠른 목표 | 순찰 | 추격 | 샛문 | 특수 아이템 |
|---|---|---|---:|---:|---|---|---:|---|
| 1 | 젖은 현관 | 수건 [11,1] | 80 (24.0) | 56 | 없음 | 없음 | 0 | 없음 |
| 2 | 긴 복도 | 안내 지도 [7,6] | 80 (24.0) | 56 | 1 / 2 | 없음 | 0 | 자석 [6,4] |
| 3 | 액자 회랑 | 발자국 사진 [11,7] | 100 (30.0) | 70 | 1 / 1 | 없음 | 0 | 무적 [7,5] |
| 4 | 작은 서재 | 정원 안내서 [11,7] | 100 (30.0) | 70 | 1 / 2 | 있음 | 0 | 없음 |
| 5 | 서재 뒤 샛길 | 스카프 고리 [11,1] | 95 (28.5) | 67 | 1 / 2 | 있음 | 1 | 무적 [12,2] |
| 6 | 온실 외곽 | 씨앗 주머니 [12,2] | 90 (27.0) | 63 | 1 / 1 | 있음 | 1 | 없음 |
| 7 | 음악실 | 잃어버린 악보 [12,2] | 95 (28.5) | 67 | 2 / 1 | 없음 | 1 | 자석 [8,4] |
| 8 | 식탁 옆 회랑 | 찻잔 받침 [12,2] | 78 (23.4) | 55 | 2 / 2 | 없음 | 1 | 없음 |
| 9 | 두 갈래 응접실 | 커튼 끈 [13,4] | 100 (30.0) | 70 | 2 / 2 | 없음 | 2 | 없음 |
| 10 | 유리 회랑 | 담요 [11,6] | 100 (30.0) | 70 | 2 / 2 | 있음 | 2 | 없음 |
| 11 | 시계 아래 복도 | 마지막 악보 [14,7] | 90 (27.0) | 63 | 2 / 2 | 있음 | 2 | 무적 [14,6] |
| 12 | 정원 앞 홀 | 정원 손잡이 [14,7] | 72 (21.6) | 50 | 2 / 2 | 있음 | 2 | 없음 |

1번은 기본 동선, 2~4번은 자석·빠른 순찰·추격, 5~8번은 샛문과 복합 동선, 9~12번은 두 샛문과 추격·시간 압박을 조합합니다. 설계 요소가 늘어나는 구성이며 실제 사람의 체감 난이도가 매번 상승한다고 검증한 결과는 아닙니다.

격자 기준 경로의 목표 온기 4/10/14점별 소요 틱은 아래와 같습니다. 기록된 경로가 현재 코어에서 성공하는지 확인하는 증거이며 최단 경로·실시간 입력·사람의 성공률을 의미하지 않습니다. 3·5·11번은 아이템 무적 중 접촉을 허용하는 경로입니다.

| 방 | 4점 / 10점 / 14점 경로 틱 |
|---|---|
| 1 | 16 / 24 / 30 |
| 2 | 18 / 22 / 22 |
| 3 | 50 / 50 / 56 |
| 4 | 51 / 53 / 61 |
| 5 | 37 / 39 / 48 |
| 6 | 38 / 39 / 48 |
| 7 | 37 / 37 / 48 |
| 8 | 38 / 41 / 53 |
| 9 | 58 / 59 / 67 |
| 10 | 40 / 46 / 61 |
| 11 | 37 / 45 / 58 |
| 12 | 37 / 45 / 55 |

## 기억·호감도·엔딩

기억 원화 세 장은 과거의 고정 그림입니다. 1~4번, 5~8번, 9~12번 방의 첫 기본 모드 성공마다 해당 그림의 영역 하나를 공개합니다. 실패·연습·같은 방 반복은 새 영역을 열지 않습니다. 부분 공개는 약 2초 뒤 쉼터로 돌아가며, 그림 완성과 앨범 다시 보기는 입력을 기다립니다. 세 번째 그림은 어떤 엔딩 이후에도 폭풍의 밤을 유지합니다.

대표 기억의 온기·속도·용기 점수를 각 배점으로 정규화하고 열두 방에 걸쳐 합산해 친구별 0~100 호감도를 계산합니다. 보송=온기, 도레=속도, 이파리=용기입니다. 70 이상이 높은 호감도입니다. 대표 기억의 유지/교체 선택은 먼저 마쳐야 합니다. 열두 방 완료, 점수 있는 대표 기억 열두 개와 방별 최고 별 기록 열두 개가 있어야 다음 우선순위로 결말을 정합니다.

| 우선순위 | 조건 | 엔딩 ID / 친구 |
|---|---|---|
| 1 | 세 친구 모두 호감도 70 이상 | `happy_all` / 모두 해방 |
| 2 | 호감도 70 이상인 친구가 없음 | `sad` / 친구 없음 |
| 3 | 높은 호감도 정확히 1명 | `ghost_a` 보송 / `ghost_b` 도레 / `ghost_c` 이파리 |
| 4 | 호감도 70 이상인 친구가 정확히 2명 | `normal` / 세 유령 |

엔딩 종류는 친구별 호감도로 결정하며 최고 별의 합계는 새드 판정에 사용하지 않습니다. 호감도가 정확히 70이면 높은 호감도에 포함합니다. 이미 감상한 엔딩·현재 쉼터 상태는 저장된 값을 유지하고, 다음 엔딩 판정부터 이 조건을 적용합니다.

탐사 유령 A=보송·B=이파리·C=도레, 주민 값 0=보송·1=이파리·2=도레입니다. 엔딩의 `ghost_b`=도레·`ghost_c`=이파리와 의미가 다르므로 연결을 바꾸지 않습니다.

첫 감상은 낮·밤 진입 PNG → 엔딩 PNG 감상 → 같은 그림에 한 문장 내레이션 → 쉼터입니다. 베스트·노멀은 `ending_intro_day.png`, 새드·보송·도레·이파리는 `ending_intro_night.png`를 사용합니다. 진입은 캐릭터 없이 원화 전체를 비율대로 표시하며 0.45초 페이드인 후에도 입력을 기다립니다. ‘계속’·Enter·Space 또는 ‘넘기기’·Esc로 엔딩 그림으로 넘어갑니다. 기존 정원·변신·시계 재개·새벽 변화 코드 장면은 사용하지 않습니다. 명시적 다시 보기는 엔딩 PNG부터 열리며 현재 세계를 바꾸지 않습니다. `likes.seen_endings`는 감상 이력, `narrative.active_ending`은 현재 세계 상태입니다. 저장 후 불러오기와 재탐험·앨범은 현재 해방 상태를 보존합니다. 옛 `basic_seen`, `true_seen`, `liberation_seen`는 호환 필드이며 새 엔딩을 자동 생성하는 근거로 쓰지 않습니다.

쉼터의 까망이와 친구들은 PNG입니다. 베스트는 세 고양이, 개별 엔딩은 해당 고양이와 나머지 두 유령 PNG, 노멀은 세 유령 PNG, 새드는 친구를 숨깁니다. 진행 중에는 기존 완료 기록에 따라 주민·회수 물건이 추가됩니다.

| 엔딩 | 확정 내레이션 |
|---|---|
| 베스트 | 저주에서 풀려나 다시 고양이가 된 세 친구와 함께, 까망이는 저택에서 오래오래 행복하게 살았답니다. |
| 보송 | 까망이는 고양이로 돌아온 보송이와 벽난로의 온기를 나누며, 저택에서 오래오래 행복하게 살았답니다. |
| 도레 | 까망이는 고양이로 돌아온 도레와 저택을 떠나, 나란히 달리며 세상 곳곳으로 즐거운 여행을 이어 갔답니다. |
| 이파리 | 까망이는 고양이로 돌아온 이파리와 마을에서 가장 아름다운 정원을 가꾸며, 꽃향기 가득한 여유로운 나날을 보냈답니다. |
| 노멀 | 저주는 아직 풀리지 않았지만, 까망이는 세 유령 친구와 함께 저택에서 따뜻한 나날을 보냈답니다. |
| 새드 | 까망이가 돌아온 저택에는, 친구들의 작은 흔적만 조용히 남아 있었답니다. |

캔버스는 1280×720입니다. 엔딩은 `(32,64,1216,506)` 안에서 원화 비율을 유지하고, 아래 `(32,586,1216,102)` 패널에 내레이션 최대 두 줄과 쉼터 버튼을 배치합니다. 제공 원본의 보라색 가장자리 정리는 보송 좌우 4px, 노멀 위·왼쪽 2px, 베스트 아래 9px를 표시할 때만 제외합니다. 원본 PNG는 바꾸지 않습니다. 고양이는 알파 여백을 제외해 몸 크기·발 위치·가구 깊이를 맞춥니다.

## 강화학습 연결과 검증 범위

| 항목 | 현재 데스크톱 연결 | 격자 호환 연결 |
|---|---|---|
| 환경 | `RealtimeMansionEnv` | `MansionEnv` |
| 코어 | `RealtimeGameCore` | `GameCore` |
| 한 행동 | 1/60초 이동 프레임 | 세계 틱 1회 |
| 관측 | 623차원 `vector_v6_realtime` | 620차원 `vector_v5_invincibility` |
| 공통 | `Discrete(6)`, `Box(0,1)`, `float32` | 동일 |

행동은 위 0·아래 1·왼쪽 2·오른쪽 3·상호작용 4·대기 5입니다. 관측의 정확한 슬롯·정규화는 `data/levels/observation_contract.json`, `realtime_observation_contract.json`에 있습니다. 기본 620차원은 벽·닫힌 문·귀환점·플레이어·시간·온기·순찰 경로와 위상·샛문·필수 물건·추격·자석·무적 상태를 포함합니다. 실시간 환경은 소수점 위치 2개와 세계 틱 진행률 1개를 추가합니다. 620차원 모델을 623차원 환경에 그대로 연결할 수 없습니다.

```python
from kkamang.env import RealtimeMansionEnv

env = RealtimeMansionEnv(stage=1, mode='basic')
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(5)
env.close()
```

기본 보상은 종료 시 `2 × 성공 여부 + 반납 온기 / 14`이며 실패·중단은 0입니다. 선택적인 `safety_coefficient`는 실제 피해에 벌점을 주고 기본값은 0입니다. 게임의 시간 초과는 `terminated`, 외부 `max_episode_steps` 제한은 `truncated`입니다. 제한 단위는 실시간 환경에서는 이동 프레임, 격자 환경에서는 세계 틱입니다. 환경은 캠페인 저장·화면·엔딩 진행을 포함하지 않습니다.

현재 검증은 Gymnasium API/결정성/관측 범위, 행동과 판정, 저장·화면 연결 회귀 검사입니다. `plan_routes.py`, `evaluate.py`와 36개 기준 경로는 **격자 환경**을 사용합니다. `evaluate.py`는 무작위·시간을 고려한 기준 정책 비교이며 학습된 정책 평가가 아닙니다. 공개 맵·위상과 탐색 예산을 쓰므로 성공 결과를 정책 일반화로 해석하지 않습니다. `balance_test.py`는 별도 저장으로 실제 사람의 실시간 플레이를 실행하고 기록을 집계하는 도구입니다. 이 정리 작업에서 사람 플레이 세션은 수행하지 않았습니다.

PPO·DQN 등 학습은 수행하지 않았습니다. 실시간 정책의 성능·일반화·사람 기준 난이도도 아직 검증하지 않았습니다. 학습을 연결할 때는 환경·관측 버전과 행동 단위, 보상·종료·제한 정책을 고정하고 학습/검증/시험 seed를 분리해야 합니다.

## 재현 가능한 검증

```sh
python -m pytest -q
python tools/check_level_schema.py
python tools/replay_validation.py
python main.py --headless --no-audio --frames 5 --dev-stage 7
```

스키마 검사는 `validation/schema_results.json`, 기본 경로 재생은 `validation/runtime_replay_results.json`을 생성합니다. 재생 결과는 `--output <경로>`로 분리할 수 있습니다. 사람의 입력을 내보낸 기록은 `python tools/replay_validation.py --record <기록.json>`으로 재생합니다. 원본 기준 데이터인 `reference_routes.json`, `safe_routes.json`은 보고서로 오인해 삭제하지 않습니다.

격자 기준 정책을 새로 측정하려면 다음을 사용합니다. 출력은 작업용이며 현재 검증을 대체하는 학습 결과가 아닙니다.

```sh
python tools/evaluate.py --episodes 10 --output reports/local/evaluation
```

최신 실제 검증 결과는 아래 한 묶음만 보존합니다. 세부 증거와 환경·명령·범위는 [통합 검증 JSON](reports/latest/verification.json), 화면은 [낮·밤 진입](reports/latest/intro_board.png), [여섯 엔딩](reports/latest/ending_board.png), [여섯 쉼터](reports/latest/lobby_board.png), [프롤로그](reports/latest/prologue.png)에 있습니다.

2026-10-20 Windows·Python 3.12 로컬 검증 결과입니다.

| 검사 | 실제 결과 |
|---|---|
| 전체 pytest | 674개 통과 + 하위 검사 137개 통과, 실패·건너뜀 0, 36.19초 |
| 레벨 스키마 | 12/12 통과 |
| 격자 기준 경로 재생 | 36/36 성공, 12개 위상 설정, 레벨 해시 12/12 일치 |
| 헤드리스 실행 | 베스트 엔딩 미리보기 5프레임 정상 종료 |
| 진입·엔딩·쉼터 | 여섯 분기의 화면 18장 캡처, 배경 선택·버튼 배치·주민 PNG 확인 |
| 기억 보존 | 기억 원화 3장과 가림막 3개의 이전 RGB 해시 일치 |
| 제공 PNG·레벨·음원 | 제공 PNG 12개 원본 일치, 기존 보존 자산·레벨·기준 데이터 변경 0 |
| 제목·배포명 | 한글·영문 제목 배치 확인, 창 제목·CLI 설명 일치, 배포명 `KkamangGhostManor`, 기존 저장 폴더 보존 |

여섯 엔딩의 첫 감상·다시 보기·저장 후 불러오기, 내레이션 배치·쉼터 PNG 분기는 관련 회귀 테스트에 포함됩니다. 낮·밤 진입 선택, 입력 대기·계속·넘기기·페이드인, PNG 누락·손상 시 명시적 오류와 대체 그림 미사용을 확인합니다. 새드 조건의 정확히 70인 경계, 과거 최고 별 기록과 무관한 판정, 대표 기억 교체에 따른 엔딩 전환도 검증합니다. 레벨의 옛 보고서 안내 필드는 설정 해시 보존을 위해 그대로 두고 `validation/upgrade_report.json`에서 최신 결과로 연결합니다.

Windows/Linux CI 설정은 보존했으며 이 로컬 검사로 원격 CI 실행까지 완료했다고 표시하지 않습니다. 실제 Windows 배포 실행 파일 생성·실기기 조작·사람의 전체 플레이·오디오 청음은 별도 확인 범위입니다.

## Windows 배포

Windows에서 `BUILD_WINDOWS.bat`를 실행합니다. `requirements-build.txt`와 `KkamangManor.spec`를 사용해 `dist/KkamangGhostManor/KkamangGhostManor.exe`를 만듭니다. **`dist/KkamangGhostManor/` 폴더 전체**를 전달해야 합니다. spec은 `assets/`와 `data/`를 포함하고 개발·학습 의존성을 게임에 포함하지 않습니다. 빌드 설정 파일명과 기존 사용자 저장 폴더 `KkamangManor`는 호환성을 위해 유지합니다. 배포 전 실제 PC에서 실행·한글·저장·입력·오디오를 확인해야 합니다.

## 자산 출처와 재제작 자료

2026-10-20 제공받은 PNG 12개의 원본 SHA-256과 배포 경로는 `assets/provenance/art/user_art_manifest.json`에 있습니다. 작가·제작 도구·별도 라이선스를 임의로 부여하지 않습니다. `best`→베스트, `ghost1`→보송, `ghost3`→도레, `ghost2`→이파리, `normal`→노멀, `sad`→새드, `coming in`→프롤로그이며 고양이 세 장은 쉼터 주민입니다. `daybackground`·`nightbackground`는 낮·밤 엔딩 진입입니다. 파일은 원본 바이트를 보존합니다.

기억 세 장은 내장 `image_gen`으로 제작한 고정 원화이며 까망이가 등장하지 않는 과거입니다. 현재 귀·꼬리 수정 기록은 `assets/provenance/art/memory_generation.json`에 보존합니다. 원화는 1672×941이고 게임은 Pygame으로 표시합니다. 얼굴·발·소품 배치가 바뀌면 공개 가림막도 함께 검토해야 합니다.

한국어 UI는 Noto Sans KR, 제목은 Nanum Myeongjo를 사용합니다. SIL Open Font License 원문은 `assets/fonts/`에 보존합니다. 사용 라이브러리의 라이선스는 각 패키지를 따르며 프로젝트 전체에 임의의 새 라이선스를 부여하지 않습니다.

| 음원 | 출처·권리 고지 |
|---|---|
| 탐험·엔딩 BGM 5곡 | 프로젝트 악보를 VCSL Kawai 그랜드 피아노 녹음으로 렌더링. Versilian Studios LLC / VCSL contributors, CC0-1.0. 소스 커밋 `c1ea7bcc3c7309650ab0da9d15c9cd1fbc4a4c7e`. [소스](https://github.com/sgossner/VCSL), [공식 프로젝트](https://versilian-studios.com/vcsl/). |
| `bgm_lobby.wav` | 제공 MP3 `fireside_rain_complete_90s_discord.mp3`에서 변환, 90초. 내장 태그는 Alexander Holm의 Salamander Grand Piano 샘플, CC BY 3.0을 명시. [샘플](https://github.com/Tonejs/audio/tree/master/salamander), [라이선스](https://creativecommons.org/licenses/by/3.0/). 제공 편집본에 편곡·조옮김·EQ·잔향·환경음 정리가 기록되어 있음. 개별 환경음 출처는 미확인. |
| `sfx_ghost_contact.wav` | 제공 MP3 `collision_cat_long_600ms_project.mp3`에서 변환, 0.6초. 내장 태그는 tuberatanka의 고양이 울음, Freesound 110011, CC0-1.0과 절차 합성 유령 소리를 명시. [원본 표기](https://freesound.org/people/tuberatanka/sounds/110011/), [CC0](https://creativecommons.org/publicdomain/zero/1.0/). 제공본의 단축·크로스페이드·게인 이력을 보존. |

VCSL 원문 허가·출처는 `assets/licenses/`, 악보 다섯 개·샘플 해시·렌더 측정은 `assets/provenance/audio_scores/`에 있습니다. 사용자 음원의 원문 태그·변환 명령·해시는 `assets/provenance/audio/`에 있습니다. 사용자 음원 고지는 제공 태그의 보존이며 외부 권리를 독립 재검증한 결과는 아닙니다. 게임용 변환은 22,050Hz·스테레오·PCM16 WAV 형식 변환이며 새 편곡·EQ·게인 조정은 하지 않았습니다.

오디오 도구는 일반 실행 의존성이 아닙니다. 재제작에 NumPy·SciPy, MP3 변환에는 FFmpeg가 필요합니다. 효과음 제작 순서는 `generate_sfx.py` → `generate_pickups_v3.py` → `generate_timed_v4.py` → `generate_shortcut_v5.py`입니다. 버전 숫자는 제작 단계이므로 단순한 구버전 중복으로 삭제하지 않습니다. 초기 WAV 별칭도 현재 오디오 대체 경로에서 사용하므로 보존합니다.

피아노 재제작은 작업 폴더에 `v7_sample_manifest.json`을 복사한 뒤 샘플을 복원하고 명시적으로 경로를 지정합니다. 기본 샘플 뱅크는 저장소에 포함하지 않습니다.

```sh
python tools/audio/restore_piano_samples.py --manifest .audio_work/v7_sample_manifest.json
python tools/audio/generate_map_bgm_v7.py --sample-manifest .audio_work/v7_sample_manifest.json --output .audio_work/rendered
```

재제작 출력은 작업 폴더에서 확인한 후 검토합니다. 단순 정리 작업에서 배포 음원을 재생성하지 않습니다.
