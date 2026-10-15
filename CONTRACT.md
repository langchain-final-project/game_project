# 팀 공통 계약 · v0.2

이 파일은 팀원이 같은 규칙으로 수정하기 위한 기준입니다. 특정 작업자에게 파일을 독점 배정하지 않습니다. 규칙/관측 변경은 PR에서 명시하고 관련 레벨·테스트·문서를 함께 바꿉니다.

## 변경 전에 고정할 것

- 규칙 버전 `cat_mansion_0_2`, 관측 버전 `vector_v2_quest`, 프로그램 버전 `0.2.0`.
- 맵 16×9, 전체 공개. 좌표 `[x,y]`, 좌상단 0,0.
- 행동 `UP=0, DOWN=1, LEFT=2, RIGHT=3, INTERACT=4, WAIT=5`.
- 한 행동=한 틱. 사람은 0.3초마다, AI는 `step` 호출마다 진행합니다.
- 같은 칸에서 E. 자동 수집이나 대각 이동은 없습니다.
- 필수 물건 회수 + 온기 4 이상 + 귀환점 E → COMPLETE.
- 미충족 E는 `return_blocked` 이벤트와 시간 소비만 발생합니다. v0.2 코어는 RETURN_LOW를 생성하지 않습니다.
- 입력→순찰 이동→최종 겹침/자리 교차 접촉→피해/보호→시간 차감→종료 판정.
- 종료 우선순위는 용기 0 > 조건 충족 귀환 > 시간 0. 마지막 틱 귀환 가능.
- 접촉은 한 틱 최대 피해 1, 다음 3틱 보호. 점수 손실 없음. 실패·중단 확정 점수 0.
- 필수 물건/장치/온기/문은 겹치지 않습니다. 필수 물건은 모든 샛문이 닫혀도 도달 가능해야 합니다.

## 경계

`GameCore(config, mode='basic', seed=0, randomize_phase=False)`는 파일·Pygame·Gym·캠페인에 의존하지 않습니다.
`reset(seed=None, phase_overrides=None)`와 `snapshot()`은 독립 복사본을 반환합니다.
`step(int)` → `StepResult(snapshot, events, terminated, outcome)`.
`abort()`는 앱 명령으로, 일곱 번째 AI 행동이 아닙니다.
종료 후 `step`은 거절됩니다.

snapshot의 새 필드는 `quest={id,name,kind,position,restoration,collected}`입니다.
새 이벤트는 `quest_collect`, `return_blocked`입니다. outcome의 `quest_collected`를 정산에서도 검사합니다.

UI는 snapshot을 읽기만 합니다. `illustration.cat`을 탐사·대화·홈에서 공유합니다. 장식과 가구는 실제 충돌 셀을 임의로 바꾸지 않습니다. 캐릭터/가구의 그리기 순서는 발 위치의 y값입니다.

`MansionEnv`는 같은 코어에 연결합니다. 관측의 마지막 3개 값은 필수 물건 x/15,y/8,회수 여부입니다. 관측 상세는 `data/levels/observation_contract.json`을 따릅니다. 599차원 모델은 이 버전과 호환되지 않습니다.

## 팀 수정 절차

1. 현재 배포본을 기준 브랜치로 커밋하고 기능별 브랜치에서 작업합니다.
2. `core.py`, `env.py`, 레벨 구조는 여러 사람이 동시에 다른 방식으로 수정하지 않도록 PR 범위를 미리 표시합니다.
3. 레벨 변경 후 `validate_level`만 통과했다고 완료로 보지 않습니다. 경로를 재생성하고 시간 내 귀환·무피해/고보상 경로를 재검증합니다.
4. UI 변경 후 실제 화면·버튼 영역·설명·확대/축소를 확인합니다. 프레임 지연이 행동 타이밍을 바꾸지 않아야 합니다.
5. `pytest`, 경로 재생 검사 통과 후 병합합니다. CI 설정은 포함되어 있으나 실제 원격 CI 실행은 아직 하지 않았습니다.
6. 저장·리플레이·가상환경·빌드 산출물은 커밋하지 않습니다. `.gitignore`가 포함되어 있습니다.
