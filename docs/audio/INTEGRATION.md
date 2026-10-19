# 스토리 브랜치 오디오 적용

`feature/story`의 기존 스토리·기억 연출 위에 오디오 재생 경로를 연결한다. 게임 규칙, 레벨, 저장 형식, 강화 효과의 지속 틱은 변경하지 않는다.

| 상황 | 적용 음원 |
|---|---|
| 타이틀·로비 | 사용자 제공 `fireside_rain_complete_90s_discord.mp3`를 변환한 `bgm_lobby.wav`, 90초 반복 |
| 1–4 스테이지 | `bgm_act1.wav` |
| 5–8 스테이지 | `bgm_act2.wav` |
| 9–11 스테이지 | `bgm_act3.wav` |
| 12 스테이지 | `bgm_stage12.wav` |
| 엔딩 화면 | `bgm_ending.wav`, 한 번 재생 |
| 실제 피해 | 사용자 제공 `collision_cat_long_600ms_project.mp3`를 변환한 `sfx_ghost_contact.wav`, 0.6초 |
| 일반·큰 온기 | 서로 다른 길이·음형의 `sfx_warmth_small/large.wav` |
| 중요 아이템 | 재질별 `sfx_memory_*.wav`, 공통 획득음이 포함된 파일을 한 번 재생 |
| 보호·자석 획득 | `sfx_badge_shield.wav` 3초, `sfx_magnet_activate.wav` 4초 |
| 자석으로 온기 흡수 | `sfx_magnet_absorb_01/02/03.wav`, 일반 온기음을 중복 재생하지 않음 |
| 지름길 개방 | `sfx_door_open.wav`, 걸쇠·짧은 움직임·멈춤 |

사용자 음원은 MP3의 실제 오디오 프레임을 디코딩해 PCM16·22,050Hz·스테레오로 변환했다. EQ, 게인, 페이드, 추가 음표, 임의 자르기를 적용하지 않았다. 로비 파일에 이미 들어 있는 비·불소리에 환경음을 중복해서 더하지 않는다. 정확한 파일 해시는 `deployed_asset_manifest.json`, 사용자 파일의 원본 해시·내장 출처와 변환 검증은 `user_audio_conversion_*.json`에 기록한다.

일시정지는 게임 효과음과 오디오 내부 시간을 함께 멈추고 BGM을 낮춘다. UI 선택음은 계속 사용할 수 있다. 실제 코어의 `invincibility_end`·`magnet_end` 이벤트가 먼저 도착하면 긴 효과음을 중단하고 짧은 종료음을 한 번 재생한다. 게임의 기존 10틱·13틱 지속 시간은 그대로이며, 3초·4초 음원 길이와 코어의 실제 종료 시각은 구분한다. 실시간 이동 이벤트에는 발걸음 간격 제한을 두고, 자석 동시 흡수도 짧은 간격으로 묶는다.

기존 `configure(completed=..., released=...)` 및 기억 공개·차가운 기억·기억 완성 API를 유지한다. 진행도별 선율과 해방 이후 새벽 레이어는 별도 예약 채널을 사용한다. 기존 일곱 WAV는 누락 파일을 위한 대체 음원으로 보존한다. 출처는 `../CREDITS.md`와 `USER_AUDIO_PROVENANCE.md`를 참고한다.

## 음량 변경과 예약 레이어

진행도별 레이어는 SDL 채널의 비동기 fade-in/fade-out을 사용하지 않습니다. 믹서 스레드의 페이드가 이후 음소거·음량 복구 설정을 덮는 Windows 경합을 피하며, WAV에 들어 있는 부드러운 시작과 BGM의 기존 소프트웨어 전환은 유지합니다. 설정을 적용할 때 활성 레이어가 자기 음원을 재생 중인지 확인하고, 정지하거나 다른 소리에 점유된 채널은 복구합니다. 비활성 레이어는 정지하며 모든 레이어 음량은 설정값에서 적용합니다.

## 재생성

완성 WAV만으로 게임을 실행할 수 있다. 아래 도구는 제작용이며 NumPy·SciPy, 사용자 음원 변환에는 FFmpeg가 필요하다. 게임의 실행 의존성에는 추가하지 않는다.

맵·엔딩 피아노를 다시 만들려면 저장소 루트에서 고정된 샘플 매니페스트를 복사해 작업 폴더에 복원한다.

```bash
mkdir -p .audio_work/piano
cp docs/audio_scores/v7_sample_manifest.json .audio_work/piano/sample_manifest.json
python tools/audio/restore_piano_samples.py --manifest .audio_work/piano/sample_manifest.json
python tools/audio/generate_map_bgm_v7.py --sample-manifest .audio_work/piano/sample_manifest.json --output .audio_work/maps
```

사용자 제공 원본 MP3 두 개를 별도 폴더에 보관하고 다음과 같이 변환할 수 있다.

```bash
python tools/audio/convert_user_media.py --source-dir /path/to/user_audio_sources --output .audio_work/user
```

효과음 생성기는 `generate_sfx.py` → `generate_pickups_v3.py` → `generate_timed_v4.py` → `generate_shortcut_v5.py` 순서로 같은 임시 출력 폴더에 실행한다. 이 초기 생성기의 피격음은 사용하지 않고, 최종 사용자 변환 파일로 교체한다. 맵 생성기·효과음 생성기를 곧바로 런타임 폴더에 실행해 검증된 배포 파일을 덮어쓰지 않는다. 완성된 파일을 비교할 때는 `deployed_asset_manifest.json`을 기준으로 한다.
