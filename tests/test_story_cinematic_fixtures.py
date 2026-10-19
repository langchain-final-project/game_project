"""Present-day fixtures must preserve the frozen past and the playable lobby."""
from copy import deepcopy
import os
from unittest.mock import patch

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import pygame
import pytest

from kkamang.ui import illustration, memories


def render(kind, elapsed, **kwargs):
    surface=pygame.Surface(memories.SIZE)
    memories.draw_cinematic(surface,kind,elapsed,**kwargs)
    return surface


def clock_face(kind, elapsed, **kwargs):
    # The inner dial excludes the time-dependent room light and other actors.
    return pygame.image.tobytes(render(kind,elapsed,**kwargs).subsurface((350,125,35,35)),'RGB')


@pytest.mark.parametrize('kind',('prologue','garden','basic_end'))
def test_ordinary_scenes_keep_the_clock_stopped(kind):
    assert clock_face(kind,0.)==clock_face(kind,8.7)


def test_liberation_clock_starts_only_after_the_dawn_transition():
    stopped=clock_face('liberation',0.)
    assert clock_face('liberation',2.19)==stopped
    assert clock_face('liberation',2.2)==stopped
    assert clock_face('liberation',3.3)!=stopped
    assert clock_face('liberation',3.3)==clock_face('true_end',1.1)


def test_true_ending_clock_ticks_without_changing_the_hour_and_minute():
    assert clock_face('true_end',0.)!=clock_face('true_end',1.1)
    assert clock_face('true_end',1.1)==clock_face('true_end',1.9)
    assert clock_face('true_end',1.1)==clock_face('true_end',61.1)


@pytest.mark.parametrize('variants',((),(0,),(1,),(2,)))
def test_partial_liberation_keeps_night_and_the_clock_stopped(variants):
    # Completing the picture alone must not turn an individual ending into
    # the best ending's dawn or restart its frozen clock.
    assert clock_face('liberation',0.,released_variants=variants)==clock_face(
        'liberation',8.7,released_variants=variants)


def test_explicit_best_liberation_keeps_the_existing_dawn_timing():
    options={'released_variants':(0,1,2)}
    assert clock_face('liberation',2.19,**options)==clock_face('liberation',0.,**options)
    assert clock_face('liberation',3.3,**options)==clock_face('true_end',1.1,**options)


def test_true_ending_has_four_sleepers_without_the_awake_only_png_atlas():
    # The main PNG atlas has no sleeping frame. Only this cinematic must use
    # the established native closed-eye rig, leaving the playable atlas intact.
    with patch.object(illustration,'sprite_frame',side_effect=AssertionError('awake PNG')):
        with patch.object(memories,'_sleeping',wraps=memories._sleeping) as asleep:
            render('true_end',4.)
    assert asleep.call_count==4
    assert [call.args[2] for call in asleep.call_args_list]==[0,2,None,1]


def test_cinematic_fixtures_leave_all_past_originals_masks_and_lobby_unchanged():
    originals=[pygame.image.tobytes(memories.memory_original(n),'RGB') for n in (1,2,3)]
    masks=[pygame.image.tobytes(memories.memory_region_mask(n),'RGB') for n in (1,2,3)]
    completed=list(range(1,13));cosmetics={'scarf':'lavender'}
    inputs=deepcopy((completed,cosmetics))
    lobby=pygame.Surface(memories.SIZE)
    illustration.shelter(lobby,completed,cosmetics,time=0.,show_actors=False)
    before=pygame.image.tobytes(lobby,'RGB')
    for kind in ('prologue','garden','liberation','basic_end','true_end'):
        memories.draw_cinematic(pygame.Surface(memories.SIZE),kind,4.,
                                completed=completed,cosmetics=cosmetics,released=True)
    illustration.shelter(lobby,completed,cosmetics,time=0.,show_actors=False)
    assert pygame.image.tobytes(lobby,'RGB')==before
    assert (completed,cosmetics)==inputs
    for index,n in enumerate((1,2,3)):
        assert pygame.image.tobytes(memories.memory_original(n),'RGB')==originals[index]
        assert pygame.image.tobytes(memories.memory_region_mask(n),'RGB')==masks[index]
