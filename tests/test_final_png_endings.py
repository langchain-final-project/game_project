"""엔딩 PNG·한 문장 내레이션·쉼터 표시와 저장 복원을 검증한다."""
from copy import deepcopy
import os
from unittest.mock import patch

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import pygame
import pytest

from kkamang.app import GameApp, CINEMATIC_DURATIONS
from kkamang.campaign import Campaign, SaveStore
from kkamang.ending_content import ENDING_CONTENT, released_variants
from kkamang.likes_preview import configure_preview
from kkamang.ui import menus, theme


NARRATIONS = {
    'happy_all': '저주에서 풀려나 다시 고양이가 된 세 친구와 함께, 까망이는 저택에서 오래오래 행복하게 살았답니다.',
    'ghost_a': '까망이는 고양이로 돌아온 보송이와 벽난로의 온기를 나누며, 저택에서 오래오래 행복하게 살았답니다.',
    'ghost_b': '까망이는 고양이로 돌아온 도레와 저택을 떠나, 나란히 달리며 세상 곳곳으로 즐거운 여행을 이어 갔답니다.',
    'ghost_c': '까망이는 고양이로 돌아온 이파리와 마을에서 가장 아름다운 정원을 가꾸며, 꽃향기 가득한 여유로운 나날을 보냈답니다.',
    'normal': '저주는 아직 풀리지 않았지만, 까망이는 세 유령 친구와 함께 저택에서 따뜻한 나날을 보냈답니다.',
    'sad': '까망이가 돌아온 저택에는, 친구들의 작은 흔적만 조용히 남아 있었답니다.',
}


@pytest.fixture
def app(tmp_path):
    application = GameApp(save_path=tmp_path / 'save.json', headless=True, no_audio=True)
    application.campaign.data['intro_seen'] = True
    application.campaign.data['tutorial_seen'] = True
    application.transition('home')
    yield application
    application.close()


def finish_resolution(app):
    while app.scene == 'story':
        assert app.story_kind in ('garden', 'liberation')
        app.advance(CINEMATIC_DURATIONS[app.story_kind])
    assert app.scene == 'ending' and app.ending_art_only


@pytest.mark.parametrize('ending_id', NARRATIONS)
def test_each_ending_has_the_approved_narrator_and_keeps_its_png_after_continue(app, ending_id):
    assert ENDING_CONTENT[ending_id]['lines'] == [('', NARRATIONS[ending_id])]
    configure_preview(app, ending_id)
    finish_resolution(app)
    before = deepcopy(app.campaign.data)
    with patch.object(menus, 'draw_cinematic') as procedural, \
         patch.object(menus, '_ending_picture', wraps=menus._ending_picture) as picture:
        app.draw()
        assert picture.call_args.args[2] == ENDING_CONTENT[ending_id]['art']
        app.next_dialogue()
        assert app.scene == 'ending' and not app.ending_art_only

        app.draw()
        assert picture.call_args.args[2] == ENDING_CONTENT[ending_id]['art']
    procedural.assert_not_called()
    assert app.campaign.data == before
    assert len(app.controls) == 1
    assert app.controls[0]['id'] == 'ending_continue'
    assert app.controls[0]['label'] == '쉼터로'
    app.dispatch('ending_continue')
    assert app.scene == 'home'
    assert app.campaign.data == before


@pytest.mark.parametrize('ending_id', NARRATIONS)
def test_ending_png_fits_entire_composition_above_the_caption(app, ending_id):
    art = ENDING_CONTENT[ending_id]['art']

    assert (app.assets.root / 'art' / art).is_file()
    source = menus._ending_source(app.assets, art)
    canvas = pygame.Surface((1280, 720))
    picture = menus._ending_picture(canvas, app.assets, art)
    frame = pygame.Rect(menus.ENDING_ART_RECT)
    panel = pygame.Rect(menus.ENDING_PANEL_RECT)
    assert picture == menus._ending_art_rect(source.get_size())
    assert frame.contains(picture)
    assert not picture.colliderect(panel)
    assert picture.center == frame.center
    factor = min(frame.w / source.get_width(), frame.h / source.get_height())
    assert abs(picture.w - source.get_width() * factor) <= 1
    assert abs(picture.h - source.get_height() * factor) <= 1
    scaled = pygame.transform.smoothscale(source, picture.size)
    expected = pygame.Surface(picture.size)
    expected.blit(scaled, (0, 0))

    assert pygame.image.tobytes(canvas.subsurface(picture), 'RGB') == pygame.image.tobytes(expected, 'RGB')


@pytest.mark.parametrize('ending_id', NARRATIONS)
def test_narrator_fits_two_lines_without_speaker_page_count_or_button_overlap(app, ending_id):
    configure_preview(app, ending_id)
    finish_resolution(app)
    app.next_dialogue()
    app.dialogue_time = 99.
    caption = pygame.Rect(menus.ENDING_TEXT_RECT)
    caption_draws = []
    original_wrapped = menus.wrapped
    original_text = theme.text

    def draw_caption(*args, **kwargs):
        rows = []

        def draw_row(*row_args, **row_kwargs):
            result = original_text(*row_args, **row_kwargs)
            rows.append((str(row_args[2]), result))
            return result

        with patch.object(theme, 'text', side_effect=draw_row):
            result = original_wrapped(*args, **kwargs)
        caption_draws.append((args[2], rows))
        return result

    with patch.object(menus, 'wrapped', side_effect=draw_caption), \
         patch.object(menus, 'text', wraps=menus.text) as labels:
        app.draw()
    assert caption_draws
    for content, rows in caption_draws:
        assert content.split() == NARRATIONS[ending_id].split()
        assert 1 <= len(rows) <= 2
        assert ''.join(''.join(text.split()) for text, _ in rows) == ''.join(content.split())
        assert all(caption.contains(rect) for _, rect in rows)
    label_values = [str(call.args[2]) for call in labels.call_args_list]
    assert not {'보송', '도레', '이파리', '1 / 1', '1 / 3'} & set(label_values)
    assert all(not caption.colliderect(control['rect']) for control in app.controls)


@pytest.mark.parametrize('ending_id', NARRATIONS)
def test_first_ending_save_load_and_replay_keep_png_and_active_lobby(app, ending_id):
    configure_preview(app, ending_id)
    finish_resolution(app)
    app.next_dialogue()
    app.dialogue_time = 99.
    app.next_dialogue()
    assert app.scene == 'home'
    expected = Campaign(app.campaign.data).data
    assert SaveStore(app.store.path).load().data == Campaign(expected).data
    restored = GameApp(save_path=app.store.path, headless=True, no_audio=True)
    try:
        assert restored.campaign.data == expected
        restored.dispatch('continue')
        assert restored.scene == 'home'
        assert restored.context()['released_variants'] == released_variants(ending_id)
        restored.draw()
        restored.start_ending(replay=True)
        assert restored.scene == 'ending' and restored.ending_art_only
        assert restored.ending_id == ending_id
        with patch.object(menus, 'draw_cinematic') as procedural, \
             patch.object(menus, '_ending_picture', wraps=menus._ending_picture) as picture:
            restored.draw()
            restored.next_dialogue()
            restored.dialogue_time = 99.
            restored.draw()
        procedural.assert_not_called()
        assert all(call.args[2] == ENDING_CONTENT[ending_id]['art'] for call in picture.call_args_list)
        restored.next_dialogue()
        assert restored.scene == 'home'
        assert restored.campaign.data == expected
        assert SaveStore(app.store.path).load().data == expected
    finally:
        restored.close()


@pytest.mark.parametrize('ending_id', NARRATIONS)
def test_replay_of_a_different_result_never_replaces_the_activated_world(app, ending_id):
    configure_preview(app, ending_id)
    finish_resolution(app)
    active = 'normal' if ending_id == 'happy_all' else 'happy_all'
    app.campaign.record_ending(active, activate=True)
    app.persist('home')
    expected = deepcopy(app.campaign.data)
    app.start_ending(replay=True)
    assert app.ending_id == ending_id
    assert app.campaign.active_ending() == active
    app.next_dialogue()
    app.dialogue_time = 99.
    app.next_dialogue()
    assert app.scene == 'home'
    assert app.context()['released_variants'] == released_variants(active)
    assert app.campaign.data == expected
    assert SaveStore(app.store.path).load().data == Campaign(expected).data


@pytest.mark.parametrize('kind', ('ending', 'prologue'))
@pytest.mark.parametrize('failure', ('missing', 'corrupt'))
def test_required_story_pngs_fail_explicitly_without_decorative_fallback(tmp_path, kind, failure):
    art_dir = tmp_path / 'art'
    art_dir.mkdir()
    name = 'ending_normal.png' if kind == 'ending' else 'prologue.png'
    if failure == 'corrupt':
        (art_dir / name).write_bytes(b'not a PNG')
    assets = theme.Assets(tmp_path)
    expected = FileNotFoundError if failure == 'missing' else pygame.error
    with patch.dict(menus._COVER_CACHE, clear=True), \
         patch.object(assets, 'image', side_effect=AssertionError('decorative fallback')), \
         patch.object(menus, 'draw_cinematic') as procedural, \
         pytest.raises(expected):
        if kind == 'ending':
            menus._ending_picture(pygame.Surface((1280, 720)), assets, name)
        else:
            menus._prologue_picture(pygame.Surface((1280, 720)), assets)
    procedural.assert_not_called()
