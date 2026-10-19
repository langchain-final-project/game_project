# 제작 자산과 출처

- 게임 규칙·스토리·맵: 「까망이와 잠 못 드는 저택」 통합 기획서 기준.
- v0.1의 AI 지원 PNG 원본은 보존했습니다. v0.2 주요 화면의 캐릭터·저택·소품·애니메이션은 illustration.py의 공통 코드 기반 렌더링을 사용합니다.
- 타이틀·메뉴·게임 상태 텍스트는 Pygame에서 실제 데이터로 표시합니다.
- 비교 연구에서 확인한 다른 게임의 캐릭터·스크린샷·상표 로고는 게임 자산에 포함하지 않았습니다.
- 한국어 UI: Noto Sans KR, SIL Open Font License. 실행 환경에서 굵기가 일정하도록 원본 가변 폰트의400·700 인스턴스를 생성했습니다. 원본과 허가문은 `assets/fonts/`에 있습니다.
- 일부 제목: Nanum Myeongjo, SIL Open Font License. 허가문 동봉.
- 초기 효과음·배경 선율(`manor.wav` 등): `tools/make_audio.py`로 생성한 자체 합성 WAV입니다. 이 초기 제작 기록을 현재의 모든 오디오에 적용하지 않습니다. 현재 배포 음원에는 아래 외부 녹음 샘플과 사용자 제공 편집본도 포함됩니다.
- 라이브러리: pygame-ce, 선택 학습 환경 Gymnasium·NumPy, 개발 검사 pytest, 배포 PyInstaller. 각각의 라이선스는 해당 배포 패키지를 따릅니다.

## 현재 오디오 출처 / Current audio credits

- **탐험·엔딩 BGM 5곡** — `bgm_act1.wav`, `bgm_act2.wav`, `bgm_act3.wav`, `bgm_stage12.wav`, `bgm_ending.wav`. 이 프로젝트용 새 악보·편곡을 **VCSL의 Kawai 그랜드 피아노 녹음**으로 렌더링했습니다. 샘플 제공: **Versilian Studios LLC / VCSL contributors**. 라이선스: **CC0 1.0 Universal**. [VCSL 공식 사이트](https://versilian-studios.com/vcsl/), [소스 저장소](https://github.com/sgossner/VCSL), [고정 커밋의 라이선스](https://github.com/sgossner/VCSL/blob/c1ea7bcc3c7309650ab0da9d15c9cd1fbc4a4c7e/LICENSE). 소스 커밋: `c1ea7bcc3c7309650ab0da9d15c9cd1fbc4a4c7e`. 가까운 건반 재생·강도 층 혼합·키 해제 감쇠·작은 잔향·음량 조정을 거쳐 WAV로 렌더링했습니다. [샘플 매니페스트](audio_scores/v7_sample_manifest.json), [CC0 원문](audio_scores/VCSL_CC0_LICENSE.txt), [제작 기록](audio_scores/COMPOSITION_NOTES_V7.md)에 세부 내용을 보존했습니다.
- **사용자 제공 로비 BGM** — `fireside_rain_complete_90s_discord.mp3` → `bgm_lobby.wav`. 제공 파일의 내장 표기: **Salamander Grand Piano samples by Alexander Holm**, **CC BY 3.0**. [Tone.js 샘플 소스](https://github.com/Tonejs/audio/tree/master/salamander), [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). 제공 편집본에는 편곡·조옮김·EQ·잔향 및 환경음 밸런스 수정이 기록돼 있습니다. 이는 피아노 샘플에 대한 표기이며, 로비에 포함된 모든 환경음의 출처·라이선스까지 확인됐다는 뜻은 아닙니다.
- **사용자 제공 접촉·피격음** — `collision_cat_long_600ms_project.mp3` → `sfx_ghost_contact.wav`. 제공 파일의 내장 표기: 절차 합성 유령 소리와 **tuberatanka의 고양이 울음**, Freesound **110011**, **CC0 1.0**. [원본 소리](https://freesound.org/people/tuberatanka/sounds/110011/), [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). 제공 편집본에는 울음 중간 부분의 단축·크로스페이드와 원래 꼬리 보존이 기록돼 있습니다.
- 사용자 제공 두 파일의 출처·편집·권리 표기는 **파일에 포함된 메타데이터를 근거로 보존**했습니다. 이 배포용 변환에서 외부 원본의 라이선스를 독립적으로 재검증한 것으로 표시하지 않습니다. 파일별 출처·해시·변환 범위와 확인되지 않은 환경음 정보는 [사용자 오디오 출처 기록](audio/USER_AUDIO_PROVENANCE.md)을 참고하세요. 변환은 재생 형식만 바꾸며, 이 프로젝트가 원본에 새로운 라이선스를 부여하지 않습니다.
- 그 밖의 이번 작업용 효과음·환경음은 `tools/audio/generate_sfx.py`, `generate_pickups_v3.py`, `generate_timed_v4.py`, `generate_shortcut_v5.py`의 합성 제작 기록을 따릅니다. 사용자 로비 파일 안의 환경음과 별도 합성 앰비언스 파일은 구분합니다.

이 문서는 제작 출처를 기록한 것이며, 프로젝트 전체에 새로운 오픈소스 라이선스를 임의로 부여하지 않습니다.
