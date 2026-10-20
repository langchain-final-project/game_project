"""여섯 엔딩에 맞는 캐릭터·쉼터 조명·PNG 배치를 검증한다."""
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


    illustration.shelter(surface,range(1,13),show_actors=False,
                         released=True,released_variants=expected)
    legacy=canvas()
    illustration.shelter(legacy,range(1,13),show_actors=False,released=len(expected)==3)
    assert pygame.image.tobytes(surface,'RGB')==pygame.image.tobytes(legacy,'RGB')






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




@pytest.mark.parametrize('ending,expected',ENDINGS)
def test_png_lobby_six_active_states_replace_only_their_own_resident(ending,expected):
    lobby._scene.cache_clear()
    with patch.object(lobby,'_resident_sprite',wraps=lobby._resident_sprite) as alive, \
         patch.object(lobby,'_sprite',wraps=lobby._sprite) as sprite, \
         patch.object(lobby,'_image',wraps=lobby._image) as image, \
         patch.object(illustration,'resident_cat') as code_cat:
        lobby.draw_lobby(canvas(),range(1,13),released_variants=expected,
                         show_residents=ending!='sad')
    assert called_variants(alive,0)==expected
    code_cat.assert_not_called()
    ghosts={call.args[1].name for call in sprite.call_args_list
            if call.args[0]=='character.png' and call.args[1].name!='까망이'}
    if ending=='normal':

        assert [call.args[0] for call in image.call_args_list]==['after.png']
        assert not sprite.called
    else:
        assert ghosts==({name for name,variant in lobby.RESIDENT_VARIANTS.items()
                         if variant not in expected} if ending!='sad' else set())
        assert sum(call.args[1].name=='까망이' for call in sprite.call_args_list)==1
        assert sum(call.args[0]=='item.png' for call in sprite.call_args_list)==11
    for call in alive.call_args_list:
        spec=lobby.CHARACTERS[('A','B','C')[call.args[0]]]
        assert call.args[1]==spec.size


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
    with patch.object(lobby,'_resident_sprite',wraps=lobby._resident_sprite) as alive:
        lobby.draw_lobby(canvas(),completed,released_variants=(0,1,2))
    assert called_variants(alive,0)==expected


def test_png_lobby_released_bodies_keep_furniture_depth():
    lobby._scene.cache_clear();depths=[]
    actual_sprite=lobby._sprite;actual_resident=lobby._resident_sprite
    def sprite(filename,spec):
        depths.append(spec.foot[1])
        return actual_sprite(filename,spec)
    def resident(variant,size):
        depths.append(lobby.CHARACTERS[('A','B','C')[variant]].foot[1])
        return actual_resident(variant,size)
    with patch.object(lobby,'_sprite',side_effect=sprite), \
         patch.object(lobby,'_resident_sprite',side_effect=resident):
        lobby.draw_lobby(canvas(),range(1,13),released_variants=(0,1,2))
    assert depths==sorted(depths)


@pytest.mark.parametrize('variant,filename',[(0,'resident_bosong.png'),
                                         (1,'resident_ipari.png'),
                                         (2,'resident_dore.png')])
def test_resident_png_is_trimmed_scaled_and_cached_without_changing_source(variant,filename):
    lobby._resident_image.cache_clear();lobby._resident_sprite.cache_clear()
    path=lobby.ASSET_DIR/'art'/'character'/filename
    original=pygame.image.load(str(path))
    bounds=original.get_bounding_rect(min_alpha=10)
    with patch.object(pygame.image,'load',wraps=pygame.image.load) as load:
        trimmed=lobby._resident_image(variant)
        spec=lobby.CHARACTERS[('A','B','C')[variant]]
        fitted=lobby._resident_sprite(variant,spec.size)
        assert lobby._resident_sprite(variant,spec.size) is fitted
    assert [call.args[0] for call in load.call_args_list]==[str(path)]
    assert trimmed.get_size()==bounds.size
    assert pygame.image.tobytes(trimmed,'RGBA')==pygame.image.tobytes(
        original.subsurface(bounds),'RGBA')

    destination=fitted.get_rect(midbottom=spec.foot)
    assert destination.midbottom==spec.foot
    assert destination.width<=spec.size[0] and destination.height<=spec.size[1]
    assert destination.height>=spec.size[1]*.9
    assert abs(fitted.get_width()/fitted.get_height()-bounds.width/bounds.height)<.01


def test_png_lobby_residents_are_blitted_at_their_original_feet():
    lobby._scene.cache_clear()
    calls=[]
    class RecordingScene(pygame.Surface):
        def copy(self):
            return self
        def blit(self,image,destination,*args,**kwargs):
            calls.append((image,pygame.Rect(destination)))
            return super().blit(image,destination,*args,**kwargs)
    scene=RecordingScene((1672,941))
    resident_images={variant:lobby._resident_sprite(
        variant,lobby.CHARACTERS[('A','B','C')[variant]].size) for variant in range(3)}
    with patch.object(lobby,'_image',return_value=scene):
        lobby._scene(tuple(range(1,13)),memories.SIZE,True,(0,1,2),True)
    for variant,image in resident_images.items():
        positions=[rect.midbottom for drawn,rect in calls if drawn is image]
        assert positions==[lobby.CHARACTERS[('A','B','C')[variant]].foot]
    lobby._scene.cache_clear()


def test_missing_resident_png_raises_instead_of_drawing_a_code_cat():
    lobby._resident_image.cache_clear();lobby._resident_sprite.cache_clear()
    with patch.object(pygame.image,'load',side_effect=FileNotFoundError('missing PNG')), \
         patch.object(illustration,'resident_cat') as code_cat:
        with pytest.raises(FileNotFoundError,match='missing PNG'):
            lobby._resident_sprite(0,(208,225))
    code_cat.assert_not_called()
