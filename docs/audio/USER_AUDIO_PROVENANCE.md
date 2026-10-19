# 사용자 제공 오디오 출처 / User-supplied audio provenance

사용자가 제공한 두 MP3를 게임용 WAV로 변환한 기록입니다. 아래 원제작자·라이선스·편집 내용은 **제공 파일의 내장 태그에 적힌 정보**를 보존한 것입니다. 변환 작업에서 외부 원본의 권리·라이선스를 독립적으로 재검증한 것으로 표시하지 않습니다. 탐험·엔딩 BGM의 VCSL CC0 피아노 출처는 [CREDITS.md](../CREDITS.md)와 별도 샘플 매니페스트에 기록합니다.

| 사용자 제공 파일 | 게임 배포 파일 | 실제 재생 길이 |
|---|---|---:|
| `fireside_rain_complete_90s_discord.mp3` | `assets/sound/bgm_lobby.wav` | 90초 |
| `collision_cat_long_600ms_project.mp3` | `assets/sound/sfx_ghost_contact.wav` | 0.6초 |

## 로비 BGM / Lobby music

- 제공 파일 제목: **Fireside Rain - complete - compact 90 seconds**.
- 태그에 적힌 피아노 샘플: **Salamander Grand Piano by Alexander Holm**, **Creative Commons Attribution 3.0**.
- 표기된 소스: https://github.com/Tonejs/audio/tree/master/salamander
- 표기된 라이선스: https://creativecommons.org/licenses/by/3.0/
- 제공 편집본에 기록된 변경: 피아노 편곡·조옮김·EQ·잔향. 지속적인 공기·비·불 노이즈를 줄이고 기존 음악과 장작 타닥임을 유지한 환경음 정리.

위 크레딧은 **피아노 샘플**에 대한 표기입니다. 파일에 섞인 비·공기·장작 환경음 각각의 원본 URL·제작자·라이선스는 제공 태그만으로 확인되지 않았습니다. 그 부분을 CC0 또는 CC BY로 임의 분류하지 않습니다. Alexander Holm을 이 사용자 편집곡 전체의 작곡자로 표시하지도 않습니다.

## 접촉·피격음 / Contact and damage cue

- 제공 파일 제목: **Ghost brush and cat - 0.6 second collision**.
- 태그에 적힌 구성: 절차 합성 유령 소리(제공본 제작 시 게인 -7dB) + **tuberatanka**의 고양이 울음.
- 표기된 원본: https://freesound.org/people/tuberatanka/sounds/110011/
- 표기된 라이선스: **CC0 1.0**, https://creativecommons.org/publicdomain/zero/1.0/
- 제공 편집본에 기록된 변경: 울음의 피치를 유지하고, 지속되는 가운데 모음을 위상을 맞춘 크로스페이드로 단축했으며, 원래 울음 꼬리는 보존. 전체 0.60초, 고양이 울음 시작 약 0.11초.

이 단축·크로스페이드·게인 표기는 이미 제공된 MP3의 제작 이력입니다. 이번 WAV 변환에서 해당 편집을 다시 적용하지 않았습니다.

## 배포용 변환 / Deployment conversion

두 파일 모두 FFmpeg로 **22,050Hz · 스테레오 · WAV PCM16**으로 변환했습니다. 로비는 44,100Hz에서, 접촉음은 48,000Hz에서 리샘플링했습니다. 파일 태그를 복사했고, 새 음표·EQ·게인 변경·정규화·페이드·크로스페이드·명시적인 길이 자르기는 적용하지 않았습니다. MP3의 인코더 패딩은 FFmpeg의 기본 gapless 처리로 해석됩니다. 따라서 MP3 컨테이너의 90.044082초·0.624초와 실제 디코딩된 90초·0.6초를 구분합니다.

| 파일 | 제공 MP3 SHA-256 | 변환 WAV SHA-256 |
|---|---|---|
| 로비 | `7c2fa0f5f4887e65cf2938112d1ef1c59ddf05f9aab20bb4355c083e30ec7aac` | `544f15eecfdd04bd808d21198c278a79596611435617d2ac6efd56489ccb15a5` |
| 접촉음 | `baaa82df7074592baf72d14cd05a27ee3a552ed64f4884cbebc7a6bd5ee6fdb5` | `1eb978f54ee804a4d7c72654f05cb232280e58ef9b5a10ee5d25137575280daa` |

정확한 내장 태그 원문·실행 명령·프레임 수·측정값은 [user_audio_conversion_metadata.json](user_audio_conversion_metadata.json)에, 수치 검증은 [user_audio_conversion_qc.json](user_audio_conversion_qc.json)에 보존했습니다. 원본과 WAV는 형식이 달라 파일 바이트가 같지는 않습니다. 재변환한 WAV의 해시 일치를 확인했으며 사람의 청음 완료를 주장하지 않습니다.

**English attribution note:** The lobby file's embedded metadata credits Salamander Grand Piano samples by Alexander Holm under CC BY 3.0, sourced through the Tone.js Salamander directory; it describes arrangement, transposition, EQ and reverb. The contact file credits a cat meow by tuberatanka, Freesound sound 110011, under CC0 1.0, combined with procedural ghost audio. These are preserved user-supplied attributions, not an independent license audit. Individual ambient sources within the lobby mix are unspecified. Deployment changes only the audio format; it grants no new license to the supplied media.
