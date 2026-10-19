"""Original small synthesized cues. Mixer failure is non-fatal."""
from array import array
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
    def __init__(self, enabled=True):
        self.available = False
        self.sounds = {}
        self.music_volume = -1
        self.story_channels = {}
        self.story_sounds = {}
        self.story_state = None
        self.story_active = {}
        self.music_level = .55
        self.sfx_volume = .65
        if not enabled:
            return
        try:
            import pygame
            self.pg = pygame
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            for name in ('select','collect','open','damage','return','fail'):
                path = ASSET_DIR / 'sound' / (name + '.wav')
                if path.exists():
                    self.sounds[name] = pygame.mixer.Sound(str(path))
            self.sounds['damage'] = _cold_wind(pygame)
            self.sounds['warmth_small'] = _tone(pygame,[(0,659.25,.22)],.3)
            self.sounds['warmth_large'] = _tone(pygame,[(0,659.25,.24),(.08,987.77,.3)],.45)
            self.sounds['memory_reveal'] = _tone(pygame,[(0,329.63,.8),(.2,493.88,.75)],1.2,.10)
            self.sounds['memory_cold'] = _tone(pygame,[(0,329.63,.7),(.18,369.99,.7),(.38,466.16,.8)],1.4,.08)
            self.sounds['memory_complete'] = _tone(pygame,[(0,329.63,.6),(.16,392,.6),(.32,493.88,.8)],1.3,.12)
            for name,frequency in [('quest_cloth',440),('quest_paper',587.33),('quest_ceramic',1046.5)]:
                self.sounds[name] = _tone(pygame,[(0,frequency,.3),(.09,659.25,.45)],.65,.10)
            self.story_sounds = {
                'rain':_ambience(pygame), 'dawn':_ambience(pygame,True),
                'phrase1':_tone(pygame,[(0,329.63,1),(.9,392,1),(1.8,493.88,1.2)],4.,.06),
                'phrase2':_tone(pygame,[(.45,440,1),(1.35,392,1),(2.25,329.63,1.3)],4.,.055),
            }
            pygame.mixer.set_reserved(4)
            self.story_channels = {name:pygame.mixer.Channel(index)
                                   for index,name in enumerate(self.story_sounds)}
            pygame.mixer.music.load(str(ASSET_DIR / 'sound' / 'manor.wav'))
            pygame.mixer.music.play(-1)
            self.available = True
        except Exception:
            self.available = False

    def configure(self, settings, *, completed=None, released=None):
        self.sfx_volume = float(settings.get('sfx', .65))
        self.music_level = float(settings.get('music', .35))
        if self.available:
            volume = float(settings.get('music', .35))
            if volume != self.music_volume:
                self.pg.mixer.music.set_volume(volume)
                self.music_volume = volume
            if completed is not None:
                done=set(completed)
                self.story_state=(all(n in done for n in range(1,5)),
                                  all(n in done for n in range(5,9)),bool(released))
            if self.story_state is not None:
                first,second,dawn=self.story_state
                active={'rain':not dawn,'dawn':dawn,'phrase1':first,'phrase2':second}
                for name,channel in self.story_channels.items():
                    channel.set_volume(self.music_level*(.45 if name in ('rain','dawn') else .65))
                    if active[name]:
                        if not channel.get_busy():
                            channel.play(self.story_sounds[name],loops=-1,fade_ms=700)
                    elif self.story_active.get(name, False):
                        channel.fadeout(500)
                self.story_active = active

    def play(self, name):
        sound = self.sounds.get(name)
        if self.available and sound:
            sound.set_volume(self.sfx_volume)
            sound.play()

    def close(self):
        if self.available and self.pg.mixer.get_init():
            for channel in self.story_channels.values():
                channel.stop()
            self.pg.mixer.music.stop()
        self.story_active = {}
        self.story_state = None
        self.available = False
