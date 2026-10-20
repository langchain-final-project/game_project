"""기억 원화·잠금 마스크·네 조각 공개 상태의 화면 계약을 검증한다."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from unittest.mock import patch

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import numpy as np
import pygame
import pytest

from kkamang.ui import memories
from kkamang.ui.illustration import cat, cat_frame, shelter


def pixels(surface):
    return pygame.surfarray.array3d(surface)


def draw(memory_id, regions, size=(1280,720), **kwargs):
    surface=pygame.Surface(size,pygame.SRCALPHA)
    surface.fill((251,2,199,0))
    memories.draw_memory(surface,memory_id,regions,surface.get_rect(),**kwargs)
    return surface


@pytest.mark.parametrize('memory_id',(1,2,3))
def test_original_copy_and_complete_boolean_reveal_are_identical(memory_id):
    original=memories.memory_original(memory_id)
    expected=pixels(original)
    original.fill((255,0,255))
    np.testing.assert_array_equal(pixels(memories.memory_original(memory_id)),expected)
    np.testing.assert_array_equal(pixels(draw(memory_id,[True]*4)),expected)


@pytest.mark.parametrize('memory_id',(1,2,3))
def test_irregular_labels_cover_every_pixel_once(memory_id):
    mask=memories.memory_region_mask(memory_id)
    labels=pixels(mask)[:,:,0]
    assert mask.get_size()==(1280,720)
    assert set(np.unique(labels))=={1,2,3,4}
    counts=[np.count_nonzero(labels==region) for region in (1,2,3,4)]
    assert min(counts)>10000 and len(set(counts))>1

    for region in (1,2,3,4):
        xx,yy=np.nonzero(labels==region)
        assert len(xx)<(xx.max()-xx.min()+1)*(yy.max()-yy.min()+1)


@pytest.mark.parametrize('memory_id',(1,2,3))
@pytest.mark.parametrize('size',((1280,720),(637,359)))
def test_locked_pixels_are_fully_opaque_and_contain_no_original(memory_id,size):
    fog=draw(memory_id,[False]*4,size)
    labels=pixels(pygame.transform.scale(memories.memory_region_mask(memory_id),size))[:,:,0]
    for count in (1,2,3):
        flags=[n<count for n in range(4)]
        image=draw(memory_id,flags,size)
        locked=labels>count
        np.testing.assert_array_equal(pixels(image)[locked],pixels(fog)[locked])
        assert np.all(pygame.surfarray.array_alpha(image)==255)


@pytest.mark.parametrize('memory_id',(1,2,3))
def test_animation_cannot_change_previous_reveals_or_expose_other_regions(memory_id):
    labels=pixels(memories.memory_region_mask(memory_id))[:,:,0]
    previous=[True,False,False,False];regions=[True,True,False,False]
    still=draw(memory_id,previous)
    for progress in (0.,.25,.8,1.):
        animated=draw(memory_id,regions,previous_regions=previous,reveal_region=1,
                      progress=progress,item_kind='map',elapsed=4.)
        np.testing.assert_array_equal(pixels(animated)[labels==1],pixels(still)[labels==1])
        np.testing.assert_array_equal(pixels(animated)[labels>2],pixels(still)[labels>2])


def test_final_storm_reveal_keeps_faces_warmth_and_threshold_together():
    labels=pixels(memories.memory_region_mask(3))[:,:,0]
    # 얼굴·흘러나오는 온기·정원 손잡이는 12조각을 모은 뒤에만 공개한다.
    for point in ((545,430),(678,445),(837,450),(516,337),(825,338),
                  (349,458),(955,487),(1032,515),(75,227),
                  (456,351),(504,307),(568,346),(614,337),(653,317),
                  (739,324),(777,347),(883,357),(945,402),(992,448),
                  (402,405),(287,491),(323,462),(957,435),(1035,494)):
        assert labels[point]==4,point
    hidden=draw(3,[True,True,True,False]);fog=draw(3,[False]*4)
    np.testing.assert_array_equal(pixels(hidden)[labels==4],pixels(fog)[labels==4])
    assert labels[1100,250]==1
    assert labels[740,575]==2

    for point in ((360,595),(448,615),(565,610),(650,625),(875,628),(1010,617),
                  (475,499),(565,496),(613,502),(698,515),(774,522)):
        assert labels[point]==2,point
    assert labels[850,295]==3
    assert labels[630,266]==3
    assert labels[789,31]==3


@pytest.mark.parametrize('memory_id,landmarks',(
    (1,((800,600,1),(400,590,1),(100,200,2),(400,674,2),
        (840,455,3),(1180,400,3),(620,565,3),(665,560,3),
        (649,449,4),(668,505,4),(535,510,4),(550,377,4),(619,345,4),
        (950,220,4))),
    (2,((270,475,1),(180,530,1),(25,210,1),(454,279,2),(330,220,2),
        (1140,280,3),(1025,250,3),(670,431,4),(550,451,4))),
))
def test_welcome_and_afternoon_landmarks_follow_their_raster_reveal_order(memory_id,landmarks):
    labels=pixels(memories.memory_region_mask(memory_id))[:,:,0]
    for x,y,region in landmarks:
        assert labels[x,y]==region,(memory_id,x,y,region)


@pytest.mark.parametrize('memory_id',(1,2,3))
def test_memory_uses_its_raster_source_and_loads_it_only_once(memory_id):
    memories._original.cache_clear()
    path=memories.MEMORY_ART_DIR/f'memory_{memory_id:02d}.png'
    source=pygame.image.load(str(path))
    expected=pygame.transform.smoothscale(source,memories.SIZE)
    with patch.object(pygame.image,'load',wraps=pygame.image.load) as load:
        first=memories.memory_original(memory_id)
        second=memories.memory_original(memory_id)
    load.assert_called_once_with(str(path))
    np.testing.assert_array_equal(pixels(first),pixels(expected))
    np.testing.assert_array_equal(pixels(second),pixels(expected))


@pytest.mark.parametrize('problem',('missing','corrupt','wrong_aspect','transparent'))
def test_missing_or_invalid_memory_art_reports_the_asset_path(tmp_path,monkeypatch,problem):
    path=tmp_path/'memory_01.png'
    monkeypatch.setattr(memories,'MEMORY_ART_DIR',tmp_path)
    memories._original.cache_clear()
    expected=FileNotFoundError
    if problem=='corrupt':
        path.write_bytes(b'not a PNG')
        expected=RuntimeError
    elif problem in ('wrong_aspect','transparent'):
        image=pygame.Surface((160,90) if problem=='transparent' else (100,100),pygame.SRCALPHA)
        image.fill((120,130,140,0 if problem=='transparent' else 255))
        pygame.image.save(image,str(path))
        expected=ValueError
    try:
        with pytest.raises(expected,match='memory_01.png'):
            memories.memory_original(1)
    finally:
        memories._original.cache_clear()


def test_present_day_endings_do_not_repaint_the_past_storm():
    original=pixels(memories.memory_original(3))
    completed=list(range(1,13));cosmetics={'scarf':'peach','rug':'leaves'}
    before=deepcopy((completed,cosmetics))
    surface=pygame.Surface((1280,720))
    for kind in ('prologue','garden','liberation','basic_end','true_end'):
        memories.draw_cinematic(surface,kind,4.,completed=completed,
                                 cosmetics=cosmetics,released=True)
    assert (completed,cosmetics)==before
    np.testing.assert_array_equal(pixels(memories.memory_original(3)),original)


def test_shelter_relationship_action_and_latest_main_walk_api_are_preserved():
    a=pygame.Surface((1280,720));b=a.copy()
    shelter(a,range(1,9),story_time=0.)
    shelter(b,range(1,9),story_time=3.)
    assert not np.array_equal(pixels(a),pixels(b))
    idle=cat_frame(160,mood='happy',walking=False)
    walking=cat_frame(160,phase=2.25,mood='happy',walking=True)
    assert not np.array_equal(pixels(idle),pixels(walking))
    cat(a,(200,200),walking=True,walk_progress=.5)


def test_real_app_boolean_context_reveals_the_complete_fourth_region(tmp_path):
    from kkamang.app import GameApp
    app=GameApp(save_path=tmp_path/'save.json',headless=True,no_audio=True)
    try:
        app.campaign.data['completed']=list(range(1,13))
        app.start_memory(12)
        ctx=app.context()
        assert ctx['regions']==[True,True,True,True]
        assert ctx['previous_regions']==[True,True,True,False]
        assert ctx['reveal_region']==3
        image=draw(ctx['memory_id'],ctx['regions'],previous_regions=ctx['previous_regions'],
                   reveal_region=ctx['reveal_region'],progress=1.,item_kind='handle')
        np.testing.assert_array_equal(pixels(image),pixels(memories.memory_original(3)))
    finally:
        app.close()


def test_game_art_renders_in_a_fresh_process_without_numpy(tmp_path):
    # 픽셀 검사용 NumPy가 게임의 실행 의존성을 숨기지 않도록 새 프로세스에서 검사한다.
    script=textwrap.dedent('''
        import importlib.abc
        import importlib.util
        import os
        import sys
        os.environ['SDL_VIDEODRIVER']='dummy'
        os.environ['SDL_AUDIODRIVER']='dummy'
        os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
        class WithoutArrays(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname=='numpy' or fullname.startswith('numpy.') or fullname=='pygame.surfarray':
                    raise ImportError('array dependency deliberately unavailable')
        sys.meta_path.insert(0,WithoutArrays())
        actual_find_spec=importlib.util.find_spec
        def without_numpy_spec(name,package=None):
            if name=='numpy' or name.startswith('numpy.'):
                return None
            return actual_find_spec(name,package)
        # Pygame probes availability before importing optional array modules.
        # Missing NumPy normally returns None at this probe rather than raising.
        importlib.util.find_spec=without_numpy_spec
        import pygame
        from pathlib import Path
        from kkamang.ui.memories import draw_memory, memory_original, memory_region_mask, draw_cinematic
        from kkamang.app import GameApp
        app=GameApp(save_path=Path(sys.argv[1]),headless=True,no_audio=True)
        try:
            for memory_id in (1,2,3):
                full=pygame.Surface((1280,720))
                draw_memory(full,memory_id,[True]*4,full.get_rect())
                assert pygame.image.tobytes(full,'RGB')==pygame.image.tobytes(memory_original(memory_id),'RGB')
                for size in ((1280,720),(637,359)):
                    empty=pygame.Surface(size)
                    draw_memory(empty,memory_id,[False]*4,empty.get_rect())
                    labels=pygame.transform.scale(memory_region_mask(memory_id),size)
                    for region in range(4):
                        flags=[index<=region for index in range(4)]
                        old=[index<region for index in range(4)]
                        frame=pygame.Surface(size,pygame.SRCALPHA)
                        draw_memory(frame,memory_id,flags,frame.get_rect(),previous_regions=old,
                                    reveal_region=region,progress=.45,item_kind='music_final')
                        for x in range(0,size[0],17):
                            for y in range(0,size[1],17):
                                assert frame.get_at((x,y)).a==255
                                if labels.get_at((x,y)).r>region+1:
                                    assert frame.get_at((x,y))[:3]==empty.get_at((x,y))[:3]
            for kind in ('prologue','garden','liberation','basic_end','true_end'):
                draw_cinematic(app.canvas,kind,3.2)
            app.campaign.data['completed']=list(range(1,13))
            app.start_memory(12)
            app.scene_elapsed=1.
            app.draw()
            app.transition('album')
            app.draw()
            assert 'numpy' not in sys.modules
            assert 'pygame.surfarray' not in sys.modules
            print('no-numpy-render-ok')
        finally:
            app.close()
    ''')
    result=subprocess.run([sys.executable,'-c',script,str(tmp_path/'no_numpy_save.json')],
                          cwd=Path(__file__).resolve().parents[1],capture_output=True,
                          text=True,timeout=30)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    assert 'no-numpy-render-ok' in result.stdout
