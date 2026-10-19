"""Real SDL dummy-mixer checks for layered music, cues, muting and shutdown."""
from array import array
import math
import os
from unittest.mock import Mock, patch

os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import pygame
import pytest

from kkamang.audio import AudioEngine


SETTINGS = {'music': .6, 'sfx': .4}


@pytest.fixture
def engine():
    pygame.mixer.quit()
    instance = AudioEngine(enabled=True)
    assert instance.available, 'SDL dummy mixer should support the generated sound bank'
    yield instance
    instance.close()
    pygame.mixer.quit()


def pcm(sound):
    values = array('h')
    values.frombytes(sound.get_raw())
    return values


def test_disabled_audio_never_opens_device_and_all_operations_are_safe():
    pygame.mixer.quit()
    instance = AudioEngine(enabled=False)
    instance.configure(SETTINGS, completed=range(1, 13), released=True)
    instance.play('memory_complete')
    instance.close()
    instance.close()
    assert not instance.available and pygame.mixer.get_init() is None


def test_device_failure_is_nonfatal_and_a_later_instance_can_recover():
    pygame.mixer.quit()
    with patch('pygame.mixer.init', side_effect=pygame.error('audio device unavailable')):
        unavailable = AudioEngine()
    assert not unavailable.available
    unavailable.configure(SETTINGS, completed=range(1, 13), released=True)
    unavailable.play('damage')
    unavailable.close()
    restored = AudioEngine()
    try:
        assert restored.available
        restored.configure(SETTINGS, completed=[], released=False)
        assert restored.story_channels['rain'].get_busy()
    finally:
        restored.close()
        pygame.mixer.quit()


def test_distinct_cues_are_nonempty_unclipped_and_use_sfx_volume(engine):
    required = {'damage', 'warmth_small', 'warmth_large', 'memory_reveal', 'memory_cold',
                'memory_complete', 'quest_cloth', 'quest_paper', 'quest_ceramic'}
    assert required <= engine.sounds.keys()
    engine.configure(SETTINGS, completed=[], released=False)
    raw = []
    for name in sorted(required):
        samples = pcm(engine.sounds[name])
        assert samples and max(abs(value) for value in samples) < 32767
        assert any(samples)
        engine.play(name)
        assert engine.sounds[name].get_volume() == pytest.approx(.4, abs=1/128)
        raw.append(engine.sounds[name].get_raw())
    assert len(set(raw)) == len(raw)
    assert engine.sounds['warmth_large'].get_length() > engine.sounds['warmth_small'].get_length()
    assert any(pygame.mixer.Channel(index).get_busy()
               for index in range(4, pygame.mixer.get_num_channels()))
    engine.play('unknown optional cue')


def test_music_layers_require_their_complete_memories_and_reconfigure_does_not_restart(engine):
    engine.configure(SETTINGS, completed=[1, 2, 3], released=False)
    assert engine.story_active == {'rain': True, 'dawn': False, 'phrase1': False, 'phrase2': False}
    engine.configure(SETTINGS, completed=range(1, 5), released=False)
    assert engine.story_channels['phrase1'].get_busy()
    assert not engine.story_channels['phrase2'].get_busy()
    engine.configure(SETTINGS, completed=range(1, 9), released=False)
    assert engine.story_channels['phrase2'].get_busy()
    engine.story_channels = {name: Mock(wraps=channel)
                             for name, channel in engine.story_channels.items()}
    engine.configure(SETTINGS, completed=range(1, 9), released=False)
    for channel in engine.story_channels.values():
        channel.play.assert_not_called()


def test_recovery_sounds_are_gated_by_release_even_when_all_rooms_are_complete(engine):
    engine.configure(SETTINGS, completed=range(1, 13), released=False)
    assert not engine.story_active['dawn']
    assert not engine.story_channels['dawn'].get_busy()
    engine.configure(SETTINGS, completed=range(1, 13), released=True)
    assert engine.story_active['dawn'] and not engine.story_active['rain']
    assert engine.story_channels['dawn'].get_sound() is engine.story_sounds['dawn']
    # Volume-only changes preserve the already released world state.
    engine.configure({'music': .25, 'sfx': .7})
    assert engine.story_state == (True, True, True)
    assert engine.story_active['dawn']


def test_volume_zero_mutes_both_music_and_cues_without_losing_progress(engine):
    engine.configure({'music': 0., 'sfx': 0.}, completed=range(1, 13), released=True)
    assert pygame.mixer.music.get_volume() == 0
    assert all(channel.get_volume() == 0 for channel in engine.story_channels.values())
    engine.play('memory_complete')
    assert engine.sounds['memory_complete'].get_volume() == 0
    # Let SDL mix several buffers: an asynchronous channel fade used to undo
    # the mute after configure() had returned, even without another cue.
    pygame.time.wait(100)
    assert all(channel.get_volume() == 0 for channel in engine.story_channels.values())
    engine.configure(SETTINGS)
    assert pygame.mixer.music.get_volume() == pytest.approx(.6, abs=1/128)
    assert engine.story_channels['dawn'].get_volume() == pytest.approx(.6*.45, abs=1/128)
    assert engine.story_channels['phrase1'].get_volume() == pytest.approx(.6*.65, abs=1/128)
    engine.play('memory_complete')
    assert engine.sounds['memory_complete'].get_volume() == pytest.approx(.4, abs=1/128)
    pygame.time.wait(100)
    assert engine.story_channels['dawn'].get_volume() == pytest.approx(.6*.45, abs=1/128)
    assert engine.story_channels['phrase1'].get_volume() == pytest.approx(.6*.65, abs=1/128)
    assert engine.story_state == (True, True, True)


def test_saturated_sfx_and_automatic_play_cannot_replace_reserved_story_layers(engine):
    engine.configure(SETTINGS, completed=range(1, 13), released=True)
    expected = {name: channel.get_sound() for name, channel in engine.story_channels.items()}
    cue = engine.sounds['memory_complete']
    # Persistent blockers make channel pressure independent of CI wall time.
    for index in range(engine.RESERVED_CHANNELS, pygame.mixer.get_num_channels()):
        pygame.mixer.Channel(index).play(cue, loops=-1)
    engine.play('memory_complete')
    assert cue.play() is None
    assert expected == {name: channel.get_sound() for name, channel in engine.story_channels.items()}
    free = pygame.mixer.Channel(engine.RESERVED_CHANNELS)
    free.stop()
    automatic = cue.play(loops=-1)
    assert automatic is not None and free.get_sound() is cue
    assert expected == {name: channel.get_sound() for name, channel in engine.story_channels.items()}


def test_reconfigure_recovers_stopped_or_replaced_story_channels_without_losing_state(engine):
    engine.configure(SETTINGS, completed=range(1, 13), released=True)
    cue = engine.sounds['memory_complete']
    engine.story_channels['dawn'].play(cue, loops=-1)
    engine.story_channels['phrase1'].stop()
    engine.story_channels['rain'].play(cue, loops=-1)
    engine.configure(SETTINGS)
    assert engine.story_state == (True, True, True)
    for name in ('dawn', 'phrase1', 'phrase2'):
        assert engine.story_channels[name].get_sound() is engine.story_sounds[name]
    assert not engine.story_channels['rain'].get_busy()
    pygame.time.wait(100)
    assert engine.story_channels['dawn'].get_volume() == pytest.approx(.6*.45, abs=1/128)


def test_generated_dawn_loop_is_faded_and_contains_a_quiet_low_body_sound(engine):
    samples = pcm(engine.story_sounds['dawn'])
    rate, sample_format, channels = pygame.mixer.get_init()
    assert sample_format == -16 and len(samples) == rate*channels*4
    assert samples[:channels] == array('h', [0]*channels)
    assert max(abs(value) for value in samples[-channels:]) < 32
    assert 0 < max(abs(value) for value in samples) < .15*32767
    # The bird phrase is silent here. This span now carries the restored cats'
    # low purr; it would be all zero in the original birds-only dawn layer.
    body = samples[int(.65*rate)*channels:int(.95*rate)*channels:channels]
    rms = math.sqrt(sum(value*value for value in body)/len(body))
    assert 50 < rms < 700


def test_close_stops_reserved_loops_and_is_safe_after_device_shutdown(engine):
    engine.configure(SETTINGS, completed=range(1, 13), released=True)
    channels = list(engine.story_channels.values())
    assert any(channel.get_busy() for channel in channels)
    engine.close()
    assert not any(channel.get_busy() for channel in channels)
    assert not pygame.mixer.music.get_busy()
    assert engine.story_active == {} and engine.story_state is None
    assert not engine.available
    engine.configure(SETTINGS, completed=range(1, 13), released=True)
    engine.play('memory_complete')
    assert not any(channel.get_busy() for channel in channels)
    pygame.mixer.quit()
    engine.close()


def test_close_is_nonfatal_when_device_disappears_while_engine_is_active(engine):
    engine.configure(SETTINGS, completed=range(1, 13), released=True)
    assert engine.available
    pygame.mixer.quit()
    engine.close()
    assert not engine.available and engine.story_active == {}
