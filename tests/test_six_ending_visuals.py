"""Rendered actor identities must agree with the six existing ending results."""
import os
from unittest.mock import patch

os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')

import pygame
import pytest

from kkamang.ending_content import released_variants
from kkamang.ui import illustration,memories


ENDINGS=(('happy_all',(0,1,2)),('ghost_a',(0,)),('ghost_b',(2,)),
         ('ghost_c',(1,)),('normal',()),('sad',()))


def canvas():
    return pygame.Surface(memories.SIZE)


def called_variants(mock,argument):
    return tuple(call.args[argument] for call in mock.call_args_list)


@pytest.mark.parametrize('ending,expected',ENDINGS)
def test_lobby_restores_only_the_selected_friend_and_keeps_other_ghosts(ending,expected):
    assert released_variants(ending)==expected
    surface=canvas()
    with patch.object(illustration,'resident_cat',wraps=illustration.resident_cat) as alive, \
         patch.object(illustration,'ghost',wraps=illustration.ghost) as spectral:
        illustration.shelter(surface,range(1,13),time=2.,story_time=3.,
                             released=True,released_variants=expected)
    assert called_variants(alive,4)==expected
    assert called_variants(spectral,5)==tuple(n for n in range(3) if n not in expected)

    # The explicit set wins over the legacy true flag. Only restoring all
    # three changes the room's light and weather to the existing dawn scene.
    illustration.shelter(surface,range(1,13),show_actors=False,
                         released=True,released_variants=expected)
    legacy=canvas()
    illustration.shelter(legacy,range(1,13),show_actors=False,released=len(expected)==3)
    assert pygame.image.tobytes(surface,'RGB')==pygame.image.tobytes(legacy,'RGB')


@pytest.mark.parametrize('ending,expected',ENDINGS)
def test_completed_liberation_restores_the_target_and_leaves_other_two_as_ghosts(ending,expected):
    with patch.object(memories,'resident_cat',wraps=memories.resident_cat) as alive, \
         patch.object(memories,'ghost',wraps=memories.ghost) as spectral, \
         patch.object(memories,'_cinematic_fixtures',wraps=memories._cinematic_fixtures) as room:
        memories.draw_cinematic(canvas(),'liberation',4.,released=True,released_variants=expected)
    assert called_variants(alive,4)==expected
    assert called_variants(spectral,5)==tuple(n for n in range(3) if n not in expected)
    assert room.call_args.kwargs['dawn'] is (len(expected)==3)


@pytest.mark.parametrize('variants',((0,1,2),(0,),(1,),(2,),()))
def test_garden_draws_current_residents_and_liberation_starts_with_three_ghosts(variants):
    with patch.object(memories,'resident_cat',wraps=memories.resident_cat) as alive, \
         patch.object(memories,'ghost',wraps=memories.ghost) as spectral:
        memories.draw_cinematic(canvas(),'garden',1.,released_variants=variants)
    assert called_variants(alive,4)==variants
    assert called_variants(spectral,5)==tuple(n for n in range(3) if n not in variants)
    with patch.object(memories,'resident_cat',wraps=memories.resident_cat) as alive, \
         patch.object(memories,'ghost',wraps=memories.ghost) as spectral:
        memories.draw_cinematic(canvas(),'liberation',0.,released_variants=variants)
    assert not alive.called
    assert called_variants(spectral,5)==(0,1,2)
    with patch.object(memories,'resident_cat',wraps=memories.resident_cat) as alive, \
         patch.object(memories,'ghost',wraps=memories.ghost) as spectral:
        memories.draw_cinematic(canvas(),'liberation',1.6,released_variants=variants)
    assert called_variants(alive,4)==variants
    assert called_variants(spectral,5)==(0,1,2)


def test_sad_lobby_hides_all_friends_but_keeps_the_player():
    with patch.object(illustration,'resident_cat',wraps=illustration.resident_cat) as alive, \
         patch.object(illustration,'ghost',wraps=illustration.ghost) as spectral, \
         patch.object(illustration,'_png_image',wraps=illustration._png_image) as image, \
         patch.object(illustration,'prop',wraps=illustration.prop) as objects:
        illustration.shelter(canvas(),range(1,13),released_variants=(),show_residents=False)
    assert not alive.called and not spectral.called
    assert sum(call.args[0]=='kkamang_wait.png' for call in image.call_args_list)==1
    assert [call.args[1] for call in objects.call_args_list]==['towel']


def test_comparison_before_stays_ghostly_and_legacy_release_still_restores_all():
    assert released_variants(None)==()
    with patch.object(illustration,'resident_cat',wraps=illustration.resident_cat) as alive, \
         patch.object(illustration,'ghost',wraps=illustration.ghost) as spectral:
        illustration.shelter(canvas(),range(1,13),before=True,
                             released=True,released_variants=(0,1,2))
    assert not alive.called
    assert called_variants(spectral,5)==(0,)
    with patch.object(illustration,'resident_cat',wraps=illustration.resident_cat) as alive:
        illustration.shelter(canvas(),released=True)
    assert called_variants(alive,4)==(0,1,2)


def test_explicit_best_ending_has_four_sleepers_and_no_ghosts():
    with patch.object(memories,'_sleeping',wraps=memories._sleeping) as asleep, \
         patch.object(memories,'ghost',wraps=memories.ghost) as spectral:
        memories.draw_cinematic(canvas(),'true_end',4.,released_variants=(0,1,2))
    assert called_variants(asleep,2)==(0,2,None,1)
    assert not spectral.called
