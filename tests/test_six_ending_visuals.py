"""Rendered actor identities must agree with the six existing ending results."""
import os
from unittest.mock import patch

os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')

import pygame
import pytest

from kkamang.ending_content import released_variants
from kkamang.ui import illustration,lobby,memories


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
    assert sum(call.args[0]=='character/kkamang_wait.png' for call in image.call_args_list)==1
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


@pytest.mark.parametrize('ending,expected',ENDINGS)
def test_png_lobby_six_active_states_replace_only_their_own_resident(ending,expected):
    lobby._scene.cache_clear()
    with patch.object(lobby,'resident_cat',wraps=lobby.resident_cat) as alive, \
         patch.object(lobby,'_sprite',wraps=lobby._sprite) as sprite, \
         patch.object(lobby,'_image',wraps=lobby._image) as image:
        lobby.draw_lobby(canvas(),range(1,13),released_variants=expected,
                         show_residents=ending!='sad')
    assert called_variants(alive,4)==expected
    ghosts={call.args[1].name for call in sprite.call_args_list
            if call.args[0]=='character.png' and call.args[1].name!='까망이'}
    if ending=='normal':
        # Main's supplied final art already has all three ghosts and Kkamang.
        assert [call.args[0] for call in image.call_args_list]==['after.png']
        assert not sprite.called
    else:
        assert ghosts==({name for name,variant in lobby.RESIDENT_VARIANTS.items()
                         if variant not in expected} if ending!='sad' else set())
        assert sum(call.args[1].name=='까망이' for call in sprite.call_args_list)==1
        assert sum(call.args[0]=='item.png' for call in sprite.call_args_list)==11
    for call in alive.call_args_list:
        spec=lobby.CHARACTERS[('A','B','C')[call.args[4]]]
        assert call.args[1]==spec.foot
        assert call.args[2]==spec.size[0]


def test_png_lobby_default_final_art_and_variation_caches_stay_separate():
    lobby._scene.cache_clear()
    completed=list(range(1,13))
    surface=canvas();lobby.draw_lobby(surface,completed)
    expected=pygame.transform.smoothscale(lobby._image('after.png'),surface.get_size())
    original=pygame.image.tobytes(surface,'RGB')
    assert original==pygame.image.tobytes(expected,'RGB')
    frames=[]
    for living,residents in (((0,),True),((1,),True),((2,),True),((0,1,2),True),((),False)):
        lobby.draw_lobby(surface,completed,released_variants=living,show_residents=residents)
        frames.append(pygame.image.tobytes(surface,'RGB'))
    assert len(set(frames+[original]))==6
    lobby.draw_lobby(surface,completed)
    assert pygame.image.tobytes(surface,'RGB')==original
    lobby.draw_lobby(surface,completed,before=True,released_variants=(0,1,2))
    initial=canvas();lobby.draw_lobby(initial,())
    assert pygame.image.tobytes(surface,'RGB')==pygame.image.tobytes(initial,'RGB')
    assert completed==list(range(1,13))


@pytest.mark.parametrize('completed,expected',(((),()),((1,),(0,)),((4,),(2,)),((7,),(1,))))
def test_png_lobby_released_bodies_keep_main_arrival_stages(completed,expected):
    lobby._scene.cache_clear()
    with patch.object(lobby,'resident_cat',wraps=lobby.resident_cat) as alive:
        lobby.draw_lobby(canvas(),completed,released_variants=(0,1,2))
    assert called_variants(alive,4)==expected


def test_png_lobby_released_bodies_keep_depth_and_bosongs_folded_ear_rig():
    lobby._scene.cache_clear();depths=[]
    actual_sprite=lobby._sprite;actual_resident=lobby.resident_cat
    def sprite(filename,spec):
        depths.append(spec.foot[1])
        return actual_sprite(filename,spec)
    def resident(surface,foot,size,time,variant):
        depths.append(foot[1])
        return actual_resident(surface,foot,size,time,variant)
    with patch.object(lobby,'_sprite',side_effect=sprite), \
         patch.object(lobby,'resident_cat',side_effect=resident), \
         patch.object(illustration,'_resident_cat_frame',wraps=illustration._resident_cat_frame) as rig:
        lobby.draw_lobby(canvas(),range(1,13),released_variants=(0,1,2))
    assert depths==sorted(depths)
    assert [call.kwargs['folded_ears'] for call in rig.call_args_list]==[True,False,False]
