"""게임 사건에 따른 음향 전환·중복 재생 방지·음량 처리를 검증한다."""
from copy import deepcopy
from pathlib import Path
import os
import random
from unittest.mock import Mock, patch

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import pygame
import pytest

from kkamang import audio
from kkamang.audio import AudioEngine


class Sound:
    def __init__(self, name):
        self.name, self.volume = name, 1.

    def set_volume(self, value):
        self.volume = value


class Channel:
    def __init__(self):
        self.busy, self.paused = False, False
        self.sound, self.volume = None, 1.
        self.plays, self.stops = [], 0

    def play(self, sound, loops=0, fade_ms=0):
        self.sound, self.busy = sound, True
        self.plays.append((sound.name, loops))

    def get_busy(self):
        return self.busy

    def get_sound(self):
        return self.sound

    def stop(self):
        self.stops += 1
        self.sound, self.busy, self.paused = None, False, False

    def pause(self):
        self.paused = True

    def unpause(self):
        self.paused = False

    def set_volume(self, value):
        self.volume = value

    def fadeout(self, milliseconds):
        self.stop()


class Music:
    def __init__(self):
        self.loads, self.plays = [], []
        self.volume, self.stops = 1., 0

    def load(self, path):
        self.loads.append(Path(path).stem)

    def play(self, loops):
        self.plays.append(loops)

    def stop(self):
        self.stops += 1

    def set_volume(self, value):
        self.volume = value


class Mixer:
    def __init__(self):
        self.initialized, self.reserved = False, 0
        self.channels = [Channel() for _ in range(8)]
        self.music = Music()

    def get_init(self):
        return (22050, -16, 2) if self.initialized else None

    def init(self, **kwargs):
        self.initialized = True

    def get_num_channels(self):
        return len(self.channels)

    def set_num_channels(self, number):
        self.channels.extend(Channel() for _ in range(number-len(self.channels)))

    def set_reserved(self, number):
        self.reserved = number

    def Channel(self, index):
        return self.channels[index]

    def Sound(self, path):
        if Path(path).stem == 'broken':
            raise pygame.error('invalid cue')
        return Sound(Path(path).stem)


@pytest.fixture
def make_engine(tmp_path, monkeypatch):
    instances = []
    monkeypatch.setattr(audio, '_tone', lambda pg, notes, duration, gain=.14: Sound(f'tone-{notes}-{duration}'))
    monkeypatch.setattr(audio, '_ambience', lambda pg, dawn=False: Sound('dawn' if dawn else 'rain'))
    monkeypatch.setattr(audio, '_cold_wind', lambda pg: Sound('cold-wind'))

    def create(names=(), enabled=True):
        folder = tmp_path / f'sound-{len(instances)}'
        folder.mkdir()
        for name in names:
            (folder / (name+'.wav')).touch()
        mixer = Mixer()
        monkeypatch.setattr(pygame, 'mixer', mixer)
        engine = AudioEngine(enabled, sound_dir=folder, rng=random.Random(42))
        instances.append(engine)
        return engine, mixer

    yield create
    for engine in instances:
        engine.close()


def names(mixer):
    return [name for channel in mixer.channels[AudioEngine.RESERVED_CHANNELS:] for name, _ in channel.plays]


def test_streamed_stage_groups_and_story_overlays_do_not_restart_tracks(make_engine):
    tracks = ('bgm_lobby', 'bgm_act1', 'bgm_act2', 'bgm_act3', 'bgm_stage12', 'bgm_ending')
    engine, mixer = make_engine(tracks)
    assert mixer.music.volume == pytest.approx(.35)
    engine.set_scene('home', 1)
    engine.set_scene('briefing', 1)
    assert mixer.music.loads == ['bgm_lobby']
    for stage, track in [(1, 'bgm_act1'), (4, 'bgm_act1'), (5, 'bgm_act2'),
                         (8, 'bgm_act2'), (9, 'bgm_act3'), (11, 'bgm_act3'), (12, 'bgm_stage12')]:
        engine.set_scene('game', stage)
        engine.update(1.2)
        assert engine.current_track == track
    assert mixer.music.loads == list(tracks[:-1])
    previous = mixer.music.loads[:]
    for scene in ('pause', 'settings', 'pause', 'game', 'result', 'story', 'briefing'):
        engine.set_scene(scene, 12, story_kind='memory')
        engine.update(.2)
    assert mixer.music.loads == previous
    engine.set_scene('ending', 12, story_kind='ending')
    engine.update(1.2)
    assert mixer.music.loads[-1] == 'bgm_ending' and mixer.music.plays[-1] == 0
    engine.update(90)
    engine.set_scene('ending', 12)
    assert mixer.music.loads.count('bgm_ending') == 1


def test_rapid_fade_destination_changes_and_cancel_remain_nonblocking(make_engine):
    engine, mixer = make_engine(('bgm_lobby', 'bgm_act1', 'bgm_act2'))
    engine.set_scene('game', 1)
    engine.update(.3)
    assert mixer.music.loads == ['bgm_lobby']
    assert mixer.music.volume == pytest.approx(.175)
    engine.set_scene('game', 5)
    engine.update(.3)
    assert mixer.music.loads == ['bgm_lobby', 'bgm_act2']
    engine.update(.6)
    engine.set_scene('game', 1)
    engine.update(.2)
    engine.set_scene('game', 5)
    engine.update(1)
    assert mixer.music.loads == ['bgm_lobby', 'bgm_act2']
    assert mixer.music.volume == pytest.approx(.35)


def test_supplied_lobby_does_not_duplicate_fire_or_rain_and_keeps_story_progression(make_engine):
    engine, mixer = make_engine(('bgm_lobby', 'amb_fireplace', 'amb_rain'))
    engine.configure({'music': .6, 'sfx': .4}, completed=range(1, 9), released=False)
    assert not mixer.channels[0].plays and not mixer.channels[1].plays
    assert engine.story_active == {'rain': True, 'dawn': False, 'phrase1': True, 'phrase2': True}
    assert engine.story_channels['rain'].busy and engine.story_channels['rain'].volume == 0.
    assert engine.story_channels['phrase1'].volume == pytest.approx(.6*.65)
    counts = {name: len(channel.plays) for name, channel in engine.story_channels.items()}
    engine.configure({'music': .6, 'sfx': .4}, completed=range(1, 9), released=False)
    assert counts == {name: len(channel.plays) for name, channel in engine.story_channels.items()}
    engine.configure({'music': .6, 'sfx': .4}, completed=range(1, 13), released=True)
    assert engine.story_active['dawn'] and not engine.story_active['rain']
    assert engine.story_channels['dawn'].volume == pytest.approx(.6*.45)


def test_reserved_story_channels_are_not_stolen_by_sfx_and_slider_is_not_squared(make_engine):
    engine, mixer = make_engine(('sfx_warmth_small',))
    engine.configure({'music': .6, 'sfx': .4}, completed=range(1, 9), released=False)
    engine.set_scene('game', 1)
    stories = {name: channel.sound for name, channel in engine.story_channels.items()}
    for _ in range(50):
        engine.play_event({'type': 'collect', 'value': 1})
    assert mixer.reserved == 7 and len(mixer.channels) == 20
    assert len(names(mixer)) == 13
    assert stories == {name: channel.sound for name, channel in engine.story_channels.items()}
    assert mixer.channels[7].volume == pytest.approx(.92)
    assert engine.sounds['sfx_warmth_small'].volume == pytest.approx(.4)
    engine.configure({'music': 0., 'sfx': 0.})
    assert all(channel.volume == 0. for channel in engine.story_channels.values())
    assert engine.sounds['sfx_warmth_small'].volume == 0.


def test_successful_realtime_moves_have_independent_quiet_footstep_cadence(make_engine):
    engine, mixer = make_engine(tuple(f'sfx_cat_step_{index:02}' for index in range(1, 4)))
    engine.set_scene('game', 1)
    state = random.getstate()
    for _ in range(60):
        engine.update(1/60)
        engine.play_event({'type': 'move', 'from': [1., 1.], 'to': [1.055, 1.]})
    footsteps = names(mixer)
    assert 5 <= len(footsteps) <= 6
    assert all(a != b for a, b in zip(footsteps, footsteps[1:]))
    engine.play_event({'type': 'blocked'})
    engine.play_event({'type': 'move', 'from': [1., 1.], 'to': [1., 1.]})
    engine.set_scene('pause', 1)
    engine.update(10)
    engine.play_event({'type': 'move', 'from': [1., 1.], 'to': [1.055, 1.]})
    assert names(mixer) == footsteps and random.getstate() == state


def test_actual_power_events_play_once_and_core_end_cancels_without_double_release(make_engine):
    engine, mixer = make_engine(('sfx_badge_shield', 'sfx_magnet_activate', 'sfx_shield_end', 'sfx_magnet_end'))
    engine.set_scene('game', 1)
    engine.play_event({'type': 'invincibility_collect', 'duration_ticks': 10})
    engine.play_event({'type': 'magnet_collect', 'duration_ticks': 13})
    assert set(engine._timed_effects) == {'shield', 'magnet'}
    engine.update(2.7)
    engine.play_event({'type': 'invincibility_end'})
    engine.play_event({'type': 'invincibility_end'})
    engine.update(.9)
    engine.play_event({'type': 'magnet_end'})
    engine.play_event({'type': 'magnet_end'})
    assert names(mixer).count('sfx_badge_shield') == 1
    assert names(mixer).count('sfx_magnet_activate') == 1
    assert names(mixer).count('sfx_shield_end') == 1
    assert names(mixer).count('sfx_magnet_end') == 1
    assert not engine._timed_effects
    engine.play_event({'type': 'invincibility_collect', 'duration_ticks': 10})
    engine.update(3.)
    engine.play_event({'type': 'invincibility_end'})
    assert names(mixer).count('sfx_shield_end') == 1


def test_magnetic_bool_coalesces_cluster_without_ordinary_warmth_or_damage_changes(make_engine):
    engine, mixer = make_engine(('sfx_magnet_absorb_01', 'sfx_magnet_absorb_02', 'sfx_warmth_small',
                                 'sfx_warmth_large', 'sfx_ghost_contact'))
    engine.set_scene('game', 1)
    engine.play_event({'type': 'damage'})
    clock = engine._last_damage
    for _ in range(8):
        engine.play_event({'type': 'collect', 'value': 1, 'magnetic': True})
    assert len([name for name in names(mixer) if name.startswith('sfx_magnet_absorb')]) == 1
    assert 'sfx_warmth_small' not in names(mixer)
    engine.play_event({'type': 'collect', 'value': 3, 'magnetic': False})
    assert names(mixer).count('sfx_warmth_large') == 1
    engine.update(.15)
    engine.play_event({'type': 'collect', 'value': 1, 'magnetic': True})
    absorbed = [name for name in names(mixer) if name.startswith('sfx_magnet_absorb')]
    assert len(absorbed) == 2 and absorbed[0] != absorbed[1]
    assert engine._last_damage == clock


def test_pause_freezes_effects_ambient_signatures_and_direct_retrigger_keeps_handle(make_engine):
    engine, mixer = make_engine(('sfx_badge_shield', 'sfx_ui_select', 'amb_manor_air', 'amb_stage01'))
    engine.set_scene('game', 1)
    engine.play_event({'type': 'invincibility_collect'})
    handle = deepcopy(engine._timed_effects['shield'])
    engine._signature_remaining = 1.
    engine.update(.5)
    engine.set_scene('pause', 1)
    engine.set_scene('settings', 1)
    engine.update(20)
    engine.play_badge('shield')
    assert engine._timed_effects['shield'] == handle and engine._world_time == .5
    assert mixer.channels[0].paused and mixer.channels[7].paused
    engine.play('select')
    assert not mixer.channels[8].paused and mixer.channels[8].sound.name == 'sfx_ui_select'
    engine.set_scene('game', 1)
    assert not mixer.channels[7].paused
    engine.update(.5)
    assert mixer.channels[2].sound.name == 'amb_stage01'
    engine.set_scene('home', 1)
    assert not engine._timed_effects and not mixer.channels[0].busy


def test_actual_quest_kind_mapping_and_supplied_damage_alias(make_engine):
    from kkamang.levels import load_levels
    expected = ['fabric', 'paper', 'paper', 'paper', 'metal', 'leaf', 'music', 'ceramic',
                'fabric', 'fabric', 'music', 'metal']
    assert [AudioEngine.memory_material(level['quest']) for level in load_levels()] == expected
    engine, mixer = make_engine(('sfx_ghost_contact', 'sfx_memory_metal', 'sfx_memory_common'))
    assert engine.sounds['damage'] is engine.sounds['sfx_ghost_contact']
    engine.set_scene('game', 1)
    engine.play_event({'type': 'quest_collect', 'name': '스카프 고리'}, quest={'kind': 'scarf_hook'})
    assert names(mixer) == ['sfx_memory_metal']


def test_partial_files_disabled_audio_and_device_loss_are_nonfatal(make_engine):
    disabled, mixer = make_engine(enabled=False)
    disabled.configure({'music': .6, 'sfx': .4}, completed=range(1, 13), released=True)
    disabled.set_scene('game', 1)
    disabled.play_event({'type': 'invincibility_collect'})
    disabled.update(10)
    assert not mixer.initialized
    engine, mixer = make_engine(('manor', 'sfx_ui_select'))
    engine.set_scene('game', 12)
    engine.update(2)
    assert mixer.music.loads == ['manor']
    engine.play('select')
    assert names(mixer) == ['sfx_ui_select']
    with patch.object(mixer.channels[2], 'stop', side_effect=pygame.error('device lost')):
        engine.set_scene('home', 12)
        engine.update(.1)
    engine.close()
    assert not engine.available and engine.story_active == {}


def test_app_forwards_actual_magnet_events_once_and_preserves_visual_flights(make_engine, tmp_path):
    from kkamang.app import GameApp
    engine, mixer = make_engine(('sfx_magnet_activate', 'sfx_magnet_absorb_01', 'sfx_warmth_small'))
    app = GameApp(save_path=tmp_path/'save.json', headless=True, no_audio=True, movement_mode='grid')
    app.audio = engine
    try:
        app.campaign.data['completed'] = [1]
        app.stage = 2
        app.start_run()
        app.core._state['position'] = list(app.core.snapshot()['magnet']['position'])
        with patch.object(engine, 'play_event', wraps=engine.play_event) as events:
            result = app._perform_action(5)
        assert any(event['type'] == 'magnet_collect' for event in result.events)
        assert events.call_count == len(result.events)
        assert names(mixer).count('sfx_magnet_activate') == 1
        assert 'sfx_warmth_small' not in names(mixer)
        assert len(app.magnet_flights) == len([event for event in result.events
                                              if event['type'] == 'collect' and event.get('magnetic')])
        assert app.core.snapshot() == result.snapshot
    finally:
        app.close()
