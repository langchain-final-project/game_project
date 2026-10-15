"""Original small synthesized cues. Mixer failure is non-fatal."""
from .paths import ASSET_DIR

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
