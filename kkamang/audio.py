"""Contextual streamed music, authored cues and preserved story-layer fallbacks.

Mixer/device and individual asset failures remain optional presentation errors.
Sound randomness and timing never change the game core or its replay state.
"""
from array import array
from pathlib import Path
import math
import random

from .paths import ASSET_DIR


def _cold_wind(pygame):
    """A soft rush and fading tremor for contact with an anxious memory."""
    rate, sample_format, channels = pygame.mixer.get_init()
    if sample_format != -16:
        raise ValueError('Cold wind requires the configured signed 16-bit mixer')
    noise = random.Random(17)
    samples = array('h')
    previous = 0.
    duration = .45
    for index in range(int(rate * duration)):
        t = index / rate
        value = noise.uniform(-1., 1.)
        breath = value - previous * .8
        previous = value
        envelope = math.sin(math.pi * t / duration) ** 2
        tremor = math.sin(2 * math.pi * 610 * t) * .04 * math.exp(-9 * t)
        sample = int((breath * .11 * envelope + tremor) * 32767)
        samples.extend([sample] * channels)
    return pygame.mixer.Sound(buffer=samples.tobytes())


def _tone(pygame, notes, duration, gain=.14):
    rate, _, channels = pygame.mixer.get_init()
    values = [0.] * int(rate*duration)
    for start, frequency, length in notes:
        for index in range(int(rate*length)):
            at = int(rate*start)+index
            if at >= len(values):
                break
            t = index/rate
            envelope = min(1.,t/.018)*math.exp(-4*t/max(.01,length))
            values[at] += gain*envelope*(math.sin(math.tau*frequency*t)+.12*math.sin(math.tau*2*frequency*t))
    samples = array('h')
    for value in values:
        samples.extend([int(max(-.85,min(.85,value))*32767)]*channels)
    return pygame.mixer.Sound(buffer=samples.tobytes())


def _ambience(pygame, dawn=False):
    rate, _, channels = pygame.mixer.get_init()
    rng = random.Random(103 if dawn else 37)
    samples = array('h')
    duration=4.
    for index in range(int(rate*duration)):
        t=index/rate
        fade=min(1.,t/.1,(duration-t)/.1)
        if dawn:
            phase=t%1.1
            value=(math.sin(math.tau*(1700+650*math.sin(t*19))*t)*.018*
                   max(0.,1-phase/.16) if phase<.16 else 0.)
            # Restored bodies and the current clock are heard only in this
            # released-state layer. Stagger the soft breaths instead of using
            # a steady low drone, and leave quiet gaps around each clock tick.
            for start,end in ((.54,1.34),(2.03,3.42)):
                if start<t<end:
                    breath=math.sin(math.pi*(t-start)/(end-start))**2
                    pulse=math.sin(math.tau*26*t)+.28*math.sin(math.tau*52*t)
                    value += .012*breath*pulse
            for start in (.42,1.49,2.55,3.61):
                age=t-start
                if 0.<=age<.025:
                    value += .021*math.sin(math.tau*1180*age)*math.exp(-age*195)
        else:
            # Soft distant rain and sparse fireplace crackle, without speech.
            value=rng.uniform(-.014,.014)
            if rng.random()<.0012:
                value += rng.uniform(-.06,.06)
        samples.extend([int(value*fade*32767)]*channels)
    return pygame.mixer.Sound(buffer=samples.tobytes())


class AudioEngine:
    FADE_SECONDS = .6
    DAMAGE_INTERVAL = .7
    MAGNET_INTERVAL = .14
    STEP_INTERVAL = .18
    TIMED_EFFECTS = {'shield': (3., 'sfx_shield_end'), 'magnet': (4., 'sfx_magnet_end')}
    BADGE_SOUNDS = {
        'collect': 'sfx_badge_collect', 'heal': 'sfx_badge_heal',
        'shield': 'sfx_badge_shield', 'speed': 'sfx_badge_speed',
        'magnet': 'sfx_magnet_activate',
    }
    RESERVED_CHANNELS = 7
    CHANNEL_COUNT = 20
    LEGACY = {
        'select': ('sfx_ui_select', 'select'),
        'collect': ('sfx_warmth_small', 'collect'),
        'open': ('sfx_door_open', 'open'),
        'damage': ('sfx_ghost_contact', 'damage'),
        'return': ('sfx_return', 'return'),
        'fail': ('sfx_fail', 'fail'),
        'warmth_small': ('sfx_warmth_small', 'warmth_small', 'collect'),
        'warmth_large': ('sfx_warmth_large', 'warmth_large', 'collect'),
        'quest_cloth': ('sfx_memory_fabric', 'quest_cloth', 'collect'),
        'quest_paper': ('sfx_memory_paper', 'quest_paper', 'collect'),
        'quest_ceramic': ('sfx_memory_ceramic', 'quest_ceramic', 'collect'),
    }
    MATERIALS = {
        'towel': 'fabric', 'map': 'paper', 'frame': 'paper', 'book': 'paper',
        'scarf': 'fabric', 'plant': 'leaf', 'music': 'music', 'tea': 'ceramic',
        'window': 'fabric', 'garden': 'metal', 'cushion': 'fabric',
        'blanket': 'fabric', 'seeds': 'leaf', 'cup': 'ceramic', 'bowl': 'ceramic',
        'footprint_photo': 'paper', 'scarf_hook': 'metal', 'saucer': 'ceramic',
        'cord': 'fabric', 'music_final': 'music', 'handle': 'metal',
    }
    LOOP_GAINS = {'amb_fireplace': .50, 'amb_rain': .14, 'amb_manor_air': .27}
    OVERLAYS = {'pause', 'settings', 'confirm', 'tutorial', 'save_error'}

    def __init__(self, enabled=True, *, sound_dir=None, rng=None):
        self.available = False
        self.sounds = {}
        self.music_volume = .35
        self.music_level = .35
        self.story_channels = {}
        self.story_sounds = {}
        self.story_state = None
        self.story_active = {}
        self.sfx_volume = .65
        self.sound_dir = Path(sound_dir) if sound_dir is not None else ASSET_DIR / 'sound'
        self.rng = rng if rng is not None else random.Random()
        self.scene = None
        self.stage = 1
        self.current_track = None
        self._current_music_path = None
        self._desired_track = None
        self._fade_phase = None
        self._fade_level = 1.
        self._scene_gain = 1.
        self._ambient_context = None
        self._ambient_names = [None, None]
        self._world_paused = False
        self._world_time = 0.
        self._last_damage = -math.inf
        self._last_step = None
        self._last_step_time = -math.inf
        self._last_magnet_absorb = -math.inf
        self._last_magnet_variant = None
        self._signature_remaining = 0.
        self._effect_world = {}
        self._channel_tokens = {}
        self._next_sound_token = 0
        self._timed_effects = {}
        self._channels = []
        if not enabled:
            return
        try:
            import pygame
            self.pg = pygame
            self._mixer = pygame.mixer
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            pygame.mixer.set_num_channels(max(self.CHANNEL_COUNT, pygame.mixer.get_num_channels()))
            pygame.mixer.set_reserved(self.RESERVED_CHANNELS)
            self._channels = [pygame.mixer.Channel(index) for index in range(self.CHANNEL_COUNT)]
        except Exception:
            return
        # A malformed cue must not disable every other sound or streamed BGM.
        names = {name for chain in self.LEGACY.values() for name in chain}
        names.update({'sfx_warmth_large', 'sfx_memory_common'})
        names.update(self.BADGE_SOUNDS.values())
        names.update(f'sfx_{effect}_{part}' for effect in self.TIMED_EFFECTS
                     for part in ('start', 'loop', 'end'))
        names.update(f'sfx_magnet_absorb_{index:02d}' for index in range(1, 4))
        names.update(f'sfx_cat_step_{index:02d}' for index in range(1, 4))
        names.update(f'sfx_memory_{material}' for material in set(self.MATERIALS.values()) | {'wood'})
        names.update(self.LOOP_GAINS)
        names.update(f'amb_stage{stage:02d}' for stage in range(1, 13))
        for name in sorted(names):
            path = self.sound_dir / f'{name}.wav'
            if path.is_file():
                try:
                    self.sounds[name] = pygame.mixer.Sound(str(path))
                except Exception:
                    pass
        self.available = True
        self._init_story_bank()
        self.set_scene('title')
        # Legacy callers configure volume before the first update() or draw.
        # The first title track starts normally; later scene changes still fade.
        self._fade_phase = None
        self._fade_level = 1.
        self._apply_volumes()

    @staticmethod
    def _volume(value, default):
        try:
            value = float(value)
            return max(0., min(1., value)) if math.isfinite(value) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def stage_track(stage):
        if stage <= 4:
            return 'bgm_act1'
        if stage <= 8:
            return 'bgm_act2'
        return 'bgm_act3' if stage <= 11 else 'bgm_stage12'

    @classmethod
    def memory_material(cls, quest=None, name=''):
        quest = quest or {}
        name = name or quest.get('name', '')
        # Actual item names take precedence: stage 5's scarf *ring* is metal.
        if any(word in name for word in ('고리', '손잡이', '방울')):
            return 'metal'
        return cls.MATERIALS.get(quest.get('kind'))

    def _init_story_bank(self):
        """Keep the story branch's cue names and four progression layers."""
        definitions = {
            'warmth_small': ('sfx_warmth_small', [(0,659.25,.22)], .3, .14),
            'warmth_large': ('sfx_warmth_large', [(0,659.25,.24),(.08,987.77,.3)], .45, .14),
            'memory_reveal': (None, [(0,329.63,.8),(.2,493.88,.75)], 1.2, .10),
            'memory_cold': (None, [(0,329.63,.7),(.18,369.99,.7),(.38,466.16,.8)], 1.4, .08),
            'memory_complete': (None, [(0,329.63,.6),(.16,392,.6),(.32,493.88,.8)], 1.3, .12),
            'quest_cloth': ('sfx_memory_fabric', [(0,440,.3),(.09,659.25,.45)], .65, .10),
            'quest_paper': ('sfx_memory_paper', [(0,587.33,.3),(.09,659.25,.45)], .65, .10),
            'quest_ceramic': ('sfx_memory_ceramic', [(0,1046.5,.3),(.09,659.25,.45)], .65, .10),
        }
        for alias, (canonical, notes, duration, gain) in definitions.items():
            try:
                self.sounds[alias] = self.sounds[canonical] if canonical in self.sounds else _tone(self.pg, notes, duration, gain)
            except Exception:
                pass
        if 'sfx_ghost_contact' in self.sounds:
            self.sounds['damage'] = self.sounds['sfx_ghost_contact']
        elif 'damage' not in self.sounds:
            try:
                self.sounds['damage'] = _cold_wind(self.pg)
            except Exception:
                pass
        builders = {
            'rain': lambda: _ambience(self.pg),
            'dawn': lambda: _ambience(self.pg, dawn=True),
            'phrase1': lambda: _tone(self.pg, [(0,329.63,1),(.9,392,1),(1.8,493.88,1.2)], 4., .06),
            'phrase2': lambda: _tone(self.pg, [(.45,440,1),(1.35,392,1),(2.25,329.63,1.3)], 4., .055),
        }
        for index, (name, build) in enumerate(builders.items(), 3):
            try:
                self.story_sounds[name] = build()
                self.story_channels[name] = self._channels[index]
            except Exception:
                pass

    def configure(self, settings, *, completed=None, released=None):
        self.music_volume = self._volume(settings.get('music', .35), .35)
        self.music_level = self.music_volume
        self.sfx_volume = self._volume(settings.get('sfx', .65), .65)
        if self.available:
            if completed is not None:
                done = set(completed)
                self.story_state = (all(n in done for n in range(1, 5)),
                                    all(n in done for n in range(5, 9)), bool(released))
            self._sync_story_layers()
        self._apply_volumes()

    def _sync_story_layers(self):
        if self.story_state is None:
            return
        first, second, dawn = self.story_state
        active = {'rain': not dawn, 'dawn': dawn, 'phrase1': first, 'phrase2': second}
        for name, channel in self.story_channels.items():
            try:
                if active[name]:
                    if not channel.get_busy():
                        channel.play(self.story_sounds[name], loops=-1, fade_ms=700)
                        if self._world_paused:
                            channel.pause()
                elif self.story_active.get(name, False):
                    channel.fadeout(500)
            except Exception:
                pass
        self.story_active = active

    def _music_candidates(self, track):
        return [path for path in (self.sound_dir / f'{track}.wav', self.sound_dir / 'manor.wav')
                if path.is_file()]

    def _request_music(self, track):
        if track == self.current_track and self._fade_phase != 'out':
            return
        candidates = self._music_candidates(track)
        if not candidates:
            return  # Keep an already playing track if the requested asset is absent.
        self._desired_track = track
        if candidates[0] == self._current_music_path:
            self.current_track = track
            if self._fade_phase == 'out':
                self._fade_phase = 'in'
            return
        if self._current_music_path is None:
            self._start_music(track)
        else:
            self._fade_phase = 'out'  # Rapid requests replace the pending destination.

    def _start_music(self, track):
        for path in self._music_candidates(track):
            try:
                self._mixer.music.load(str(path))
                self._mixer.music.set_volume(0.)
                self._mixer.music.play(0 if track == 'bgm_ending' else -1)
                self.current_track = track
                self._current_music_path = path
                self._fade_phase = 'in'
                self._fade_level = 0.
                return
            except Exception:
                continue
        self.current_track = None
        self._current_music_path = None
        self._fade_phase = None

    def set_scene(self, scene, stage=None, *, story_kind=None):
        """Update context once per scene change; overlays preserve their music.

        Pausing ducks the streamed music, freezes world sounds and the signature
        countdown, and leaves UI cues available. Resuming never reloads a track.
        """
        if not self.available:
            return
        try:
            self._set_scene(scene, stage)
        except Exception:
            pass

    def _set_scene(self, scene, stage):
        previous_scene = self.scene
        previous_context = self._ambient_context
        stage = self.stage if stage is None else int(stage)
        self.scene, self.stage = scene, stage
        was_paused = self._world_paused
        if scene == 'game':
            self._world_paused = False
            context = ('stage', stage)
            self._request_music(self.stage_track(stage))
            self._scene_gain = 1.
        elif scene in ('title', 'home'):
            self._world_paused = False
            context = ('lobby', None)
            self._request_music('bgm_lobby')
            self._scene_gain = 1.
        elif scene == 'ending':
            self._world_paused = False
            context = None
            self._request_music('bgm_ending')
            self._scene_gain = 1.
        elif scene in self.OVERLAYS and (previous_scene == 'game' or was_paused):
            self._world_paused = True
            context = self._ambient_context
            self._scene_gain = .42
        else:
            self._world_paused = False
            # Lobby atmosphere follows its menus; world atmosphere belongs to play.
            context = self._ambient_context if self._ambient_context == ('lobby', None) else None
            self._scene_gain = .72 if scene in ('result', 'story') else 1.
        if context != previous_context or (scene != 'game' and not self._world_paused):
            self._clear_timed_effects()
        if context != self._ambient_context:
            self._channels[2].stop()
            self._ambient_context = context
            self._signature_remaining = self.rng.uniform(8., 25.) if context and context[0] == 'stage' else 0.
            self._last_damage = -math.inf
            self._last_magnet_absorb = -math.inf
            self._last_step_time = -math.inf
        # The supplied lobby already contains its fireplace and rain bed.
        loops = ('amb_manor_air', 'amb_rain') if context and context[0] == 'stage' else (None, None)
        for index, name in enumerate(loops):
            if self._ambient_names[index] != name:
                self._channels[index].stop()
                self._ambient_names[index] = name
                if name in self.sounds:
                    try:
                        self._channels[index].play(self.sounds[name], loops=-1)
                    except Exception:
                        pass
        if self._world_paused != was_paused:
            method = 'pause' if self._world_paused else 'unpause'
            for index, channel in enumerate(self._channels):
                if index < self.RESERVED_CHANNELS or self._effect_world.get(index, (False, 0.))[0]:
                    getattr(channel, method)()
        self._apply_volumes()

    def _apply_volumes(self):
        if not self.available:
            return
        try:
            self._mixer.music.set_volume(self.music_volume * self._scene_gain * self._fade_level)
            for index, name in enumerate(self._ambient_names):
                self._channels[index].set_volume(self.music_volume * self._scene_gain * self.LOOP_GAINS.get(name, 0.))
            self._channels[2].set_volume(self.music_volume * .80)
            for sound in self.sounds.values():
                sound.set_volume(self.sfx_volume)
            for name, channel in self.story_channels.items():
                gain = .45 if name in ('rain', 'dawn') else .65
                if name == 'rain' and (
                        self._ambient_context == ('lobby', None) and (self.sound_dir / 'bgm_lobby.wav').is_file()
                        or self._ambient_context and self._ambient_context[0] == 'stage' and 'amb_rain' in self.sounds):
                    gain = 0.  # Keep logical progression without duplicating the supplied rain.
                channel.set_volume(self.music_level * self._scene_gain * gain)
            for index, (_, gain) in self._effect_world.items():
                self._channels[index].set_volume(gain)
        except Exception:
            pass

    def update(self, dt):
        if not self.available:
            return
        dt = max(0., float(dt))
        remaining = dt
        try:
            # At most two phases consume this frame; no wall-clock waits occur.
            for _ in range(2):
                if self._fade_phase == 'out':
                    take = min(remaining, self._fade_level * self.FADE_SECONDS)
                    self._fade_level = max(0., self._fade_level - take / self.FADE_SECONDS)
                    remaining = max(0., remaining - take)
                    if self._fade_level <= 1e-9:
                        self._mixer.music.stop()
                        self._current_music_path = None
                        self._start_music(self._desired_track)
                    else:
                        break
                elif self._fade_phase == 'in':
                    self._fade_level = min(1., self._fade_level + remaining / self.FADE_SECONDS)
                    if self._fade_level >= 1.:
                        self._fade_phase = None
                    break
                else:
                    break
            if self.scene == 'game' and not self._world_paused:
                self._world_time += dt
                for effect, active in list(self._timed_effects.items()):
                    if self._world_time >= active['deadline']:
                        # Full composites already contain their natural release.
                        self.stop_timed_effect(effect, play_end=False)
                if self.music_volume > 0.:
                    self._signature_remaining -= dt
                    channel = self._channels[2]
                    if self._signature_remaining <= 0.:
                        name = f'amb_stage{self.stage:02d}'
                        if name in self.sounds and not channel.get_busy():
                            channel.play(self.sounds[name])
                        self._signature_remaining = self.rng.uniform(8., 25.)
            self._apply_volumes()
        except Exception:
            # A device disconnect mid-session also must not interrupt gameplay.
            pass

    def _play(self, candidates, gain=1., *, world=True, return_handle=False):
        """Return True on play, or an ownership handle when explicitly requested."""
        if not self.available or self.sfx_volume <= 0. or (world and self._world_paused):
            return
        sound = next((self.sounds[name] for name in candidates if name in self.sounds), None)
        if sound is None:
            return
        try:
            for index in range(self.RESERVED_CHANNELS, len(self._channels)):
                channel = self._channels[index]
                if not channel.get_busy():
                    self._effect_world[index] = (world, gain)
                    sound.set_volume(self.sfx_volume)
                    channel.set_volume(gain)
                    channel.play(sound)
                    self._next_sound_token += 1
                    self._channel_tokens[index] = self._next_sound_token
                    return (index, self._next_sound_token) if return_handle else True
        except Exception:
            pass

    def play(self, name):
        """Legacy cue names and canonical sound stems are both supported."""
        self._play(self.LEGACY.get(name, (name,)), world=name not in ('select', 'sfx_ui_select'))

    def play_badge(self, effect=None):
        """Play one composite pickup/achievement cue; this grants no ability.

        Effect cues already contain their pickup sound. Missing/unknown variants
        fall back to the generic badge cue, then the existing safe-return cue.
        """
        stem = self.BADGE_SOUNDS.get(effect, 'sfx_badge_collect') if isinstance(effect, str) else 'sfx_badge_collect'
        candidates = (stem, 'sfx_badge_collect', 'sfx_return', 'return')
        if isinstance(effect, str) and effect in self.TIMED_EFFECTS:
            self._start_timed_effect(effect, candidates)
        else:
            self._play(candidates, .95)

    def _start_timed_effect(self, effect, candidates):
        if not self.available or self._world_paused or self.sfx_volume <= 0.:
            return  # Suppressed retriggers preserve an already running effect.
        self.stop_timed_effect(effect, play_end=False)
        handle = self._play(candidates, .95, return_handle=True)
        if handle is not None:
            self._timed_effects[effect] = {
                'handle': handle, 'deadline': self._world_time + self.TIMED_EFFECTS[effect][0],
            }

    def stop_timed_effect(self, effect, *, play_end=True):
        """Cancel presentation early, optionally with one short release cue.

        This does not change a gameplay timer. Channel generations prevent an
        expired/short fallback from stopping an unrelated sound that reused it.
        """
        if not isinstance(effect, str):
            return False
        active = self._timed_effects.pop(effect, None)
        if active is None:
            return False
        index, token = active['handle']
        if self.available and self._channel_tokens.get(index) == token:
            try:
                self._channels[index].stop()
            except Exception:
                pass
            self._channel_tokens.pop(index, None)
            self._effect_world.pop(index, None)
        if play_end:
            self._play((self.TIMED_EFFECTS[effect][1],), .82)
        return True

    def _clear_timed_effects(self):
        for effect in list(self._timed_effects):
            self.stop_timed_effect(effect, play_end=False)

    def play_magnet_absorb(self, value=1):
        """Coalesce magnetic warmth bursts using the presentation-only clock."""
        if (not self.available or self.scene != 'game' or self._world_paused
                or self.sfx_volume <= 0.
                or self._world_time - self._last_magnet_absorb < self.MAGNET_INTERVAL):
            return
        choices = [f'sfx_magnet_absorb_{index:02d}' for index in range(1, 4)
                   if f'sfx_magnet_absorb_{index:02d}' in self.sounds]
        warmth = 'sfx_warmth_large' if value == 3 else 'sfx_warmth_small'
        if choices:
            variants = [name for name in choices if name != self._last_magnet_variant] or choices
            selected = self.rng.choice(variants)
            candidates = (selected, warmth, 'collect')
        else:
            selected = None
            candidates = (warmth, 'collect')
        if self._play(candidates, .92):
            self._last_magnet_absorb = self._world_time
            self._last_magnet_variant = selected

    def play_event(self, event, *, quest=None):
        if not self.available or self.scene != 'game' or self._world_paused:
            return
        kind = event.get('type')
        if kind == 'effect_end':
            self.stop_timed_effect(event.get('effect'), play_end=event.get('play_end', True))
        elif kind in ('shield_end', 'magnet_end', 'invincibility_end'):
            effect = 'shield' if kind == 'invincibility_end' else kind[:-4]
            self.stop_timed_effect(effect, play_end=event.get('play_end', True))
        elif kind == 'invincibility_collect':
            self.play_badge('shield')
        elif kind == 'badge_collect':
            self.play_badge(event.get('effect'))
        elif kind in ('magnet_activate', 'magnet_collect'):
            self.play_badge('magnet')
        elif kind == 'magnet_absorb' or (kind == 'collect' and (event.get('magnetic') or event.get('source') == 'magnet')):
            self.play_magnet_absorb(event.get('value', 1))
        elif kind == 'collect':
            stem = 'sfx_warmth_large' if event.get('value') == 3 else 'sfx_warmth_small'
            alias = 'warmth_large' if event.get('value') == 3 else 'warmth_small'
            self._play((stem, alias, 'collect'), .92)
        elif kind == 'quest_collect':
            material = self.memory_material(quest, event.get('name', ''))
            # Material files already include the shared memory chime: play once.
            alias = 'quest_cloth' if material in ('fabric', 'leaf') else 'quest_ceramic' if material == 'ceramic' else 'quest_paper'
            self._play((f'sfx_memory_{material}', 'sfx_memory_common', alias, 'collect'))
        elif kind == 'open':
            self.play('open')
        elif kind == 'damage' and self._world_time - self._last_damage >= self.DAMAGE_INTERVAL:
            self._last_damage = self._world_time
            self.play('damage')
        elif (kind == 'move' and event.get('from') != event.get('to')
              and self._world_time - self._last_step_time >= self.STEP_INTERVAL):
            choices = [name for name in (f'sfx_cat_step_{index:02d}' for index in range(1, 4))
                       if name in self.sounds]
            if choices:
                variants = [name for name in choices if name != self._last_step] or choices
                self._last_step = self.rng.choice(variants)
                if self._play((self._last_step,), .80):
                    self._last_step_time = self._world_time

    def close(self):
        self._clear_timed_effects()
        self.story_active = {}
        self.story_state = None
        if not self.available:
            return
        try:
            self._mixer.music.stop()
            for channel in self._channels:
                channel.stop()
        except Exception:
            pass
        self.available = False
