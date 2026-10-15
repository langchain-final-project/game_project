"""Convert human input into the same six actions used by the headless core."""
import pygame

KEY_ACTION = {pygame.K_UP: 0, pygame.K_w: 0, pygame.K_DOWN: 1, pygame.K_s: 1,
              pygame.K_LEFT: 2, pygame.K_a: 2, pygame.K_RIGHT: 3, pygame.K_d: 3,
              pygame.K_e: 4, pygame.K_SPACE: 5}
PRIORITY = {4: 6, 0: 5, 3: 4, 1: 3, 2: 2, 5: 1}

class InputAdapter:
    def __init__(self):
        self.clear()

    def clear(self):
        self.held = {}
        self.pending = []
        self.serial = 0

    def handle(self, event):
        if event.type not in (pygame.KEYDOWN, pygame.KEYUP) or event.key not in KEY_ACTION:
            return
        if event.type == pygame.KEYUP:
            self.held.pop(event.key, None)
            return
        if event.key in self.held:  # OS key repeat must not repeat interactions.
            return
        self.serial += 1
        stamp = getattr(event, 'timestamp', self.serial)
        value = (stamp, PRIORITY[KEY_ACTION[event.key]], KEY_ACTION[event.key])
        self.held[event.key] = value
        # Retain short taps until the next logic tick; held directions continue.
        self.pending.append(value)

    def consume(self) -> int:
        candidates = [v for v in self.held.values() if v[2] != 4]
        candidates.extend(self.pending)
        self.pending = []
        return max(candidates)[2] if candidates else 5
