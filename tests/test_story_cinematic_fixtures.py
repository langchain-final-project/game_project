"""엔딩 진입의 낮·밤 PNG, 입력 전환, 원본 보존과 오류 처리를 검증한다."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import pygame
import pytest

from kkamang.app import GameApp
from kkamang.ending_content import ENDING_CONTENT
from kkamang.likes_preview import configure_preview
from kkamang.ui import illustration, memories, menus, theme

BACKGROUNDS = {
    'happy_all': 'ending_intro_day.png', 'normal': 'ending_intro_day.png',
    'sad': 'ending_intro_night.png', 'ghost_a': 'ending_intro_night.png',
    'ghost_b': 'ending_intro_night.png', 'ghost_c': 'ending_intro_night.png',
}


@pytest.fixture
def app(tmp_path):
    instance = GameApp(save_path=tmp_path / 'save.json', headless=True, no_audio=True)
    yield instance
    instance.close()


@pytest.mark.parametrize('ending,background', BACKGROUNDS.items())
@pytest.mark.parametrize('action', ['story_next', 'story_skip', 'enter', 'escape'])
def test_intro_waits_for_input_then_opens_the_selected_ending(app, ending, background, action):
    configure_preview(app, ending)
    before = deepcopy(app.campaign.data)
    assert app.scene == 'story' and app.story_kind == 'ending_intro'
    assert app.context()['silent'] and app.context()['dialogue'] is None
    assert ENDING_CONTENT[ending]['intro_art'] == background
    app.advance(10.)
    assert app.scene == 'story' and app.campaign.data == before
    with patch.object(menus, '_ending_intro_picture', wraps=menus._ending_intro_picture) as picture, \
         patch.object(illustration, 'shelter', side_effect=AssertionError('procedural room')), \
         patch.object(menus, 'draw_story_character', side_effect=AssertionError('portrait')):
        app.draw()
    assert picture.call_args.args[2] == background
    assert {c['id'] for c in app.controls} == {'story_next', 'story_skip'}
    if action in ('enter', 'escape'):
        key = pygame.K_RETURN if action == 'enter' else pygame.K_ESCAPE
        app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0))
    else:
        control = next(c for c in app.controls if c['id'] == action)
        app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=control['rect'].center, button=1))
    assert app.scene == 'ending' and app.ending_id == ending and app.ending_art_only
    assert app.campaign.active_ending() == ending
    assert not hasattr(memories, 'draw_cinematic')


@pytest.mark.parametrize('name', ['ending_intro_day.png', 'ending_intro_night.png'])
def test_intro_preserves_full_composition_and_finishes_its_fade(name):
    assets = theme.Assets()
    source = pygame.image.load(str(assets.root / 'art' / name))
    scale = min(1280 / source.get_width(), 720 / source.get_height())
    size = (round(source.get_width() * scale), round(source.get_height() * scale))
    expected = pygame.Surface((1280, 720)); expected.fill(theme.NIGHT)
    image = pygame.transform.smoothscale(source, size)
    expected.blit(image, image.get_rect(center=expected.get_rect().center))
    frames = []
    for elapsed in (0., .2, .45, 10.):
        frame = pygame.Surface((1280, 720))
        menus._ending_intro_picture(frame, assets, name, elapsed)
        frames.append(pygame.image.tobytes(frame, 'RGB'))
    assert frames[0] != frames[1] != frames[2]
    assert frames[2] == frames[3] == pygame.image.tobytes(expected, 'RGB')


@pytest.mark.parametrize('name', ['ending_intro_day.png', 'ending_intro_night.png'])
@pytest.mark.parametrize('failure', ['missing', 'corrupt'])
def test_required_intro_png_never_uses_a_fallback(tmp_path, name, failure):
    art = tmp_path / 'art'; art.mkdir()
    if failure == 'corrupt':
        (art / name).write_bytes(b'not a PNG')
    assets = theme.Assets(tmp_path)
    with patch.dict(menus._COVER_CACHE, clear=True), \
         patch.object(assets, 'image', side_effect=AssertionError('decorative fallback')), \
         pytest.raises(FileNotFoundError if failure == 'missing' else pygame.error):
        menus._ending_intro_picture(pygame.Surface((1280, 720)), assets, name, 1.)


def test_intro_assets_match_provenance_and_leave_past_memories_unchanged():
    assets = theme.Assets()
    original = [pygame.image.tobytes(memories.memory_original(n), 'RGB') for n in (1, 2, 3)]
    masks = [pygame.image.tobytes(memories.memory_region_mask(n), 'RGB') for n in (1, 2, 3)]
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'assets/provenance/art/user_art_manifest.json').read_text(encoding='utf-8'))
    entries = {entry['path']: entry for entry in manifest['files']}
    for name in set(BACKGROUNDS.values()):
        path = root / 'assets/art' / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entries[path.relative_to(root).as_posix()]['sha256']
        menus._ending_intro_picture(pygame.Surface((1280, 720)), assets, name, 1.)
    assert original == [pygame.image.tobytes(memories.memory_original(n), 'RGB') for n in (1, 2, 3)]
    assert masks == [pygame.image.tobytes(memories.memory_region_mask(n), 'RGB') for n in (1, 2, 3)]
