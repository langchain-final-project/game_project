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

class AudioEngine:
    def __init__(self, enabled=True):
        self.available = False
        self.sounds = {}
        self.music_volume = -1
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
            pygame.mixer.music.load(str(ASSET_DIR / 'sound' / 'manor.wav'))
            pygame.mixer.music.play(-1)
            self.available = True
        except Exception:
            self.available = False

    def configure(self, settings):
        self.sfx_volume = float(settings.get('sfx', .65))
        if self.available:
            volume = float(settings.get('music', .35))
            if volume != self.music_volume:
                self.pg.mixer.music.set_volume(volume)
                self.music_volume = volume

    def play(self, name):
        sound = self.sounds.get(name)
        if self.available and sound:
            sound.set_volume(self.sfx_volume)
            sound.play()

    def close(self):
        if self.available:
            self.pg.mixer.music.stop()
