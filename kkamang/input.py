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

    def consume_interaction(self) -> bool:
        """새로 누른 E만 가져오고, 이동 입력은 남겨둡니다."""
        pressed = any(
            value[2] == 4
            for value in self.pending
        )

        self.pending = [
            value
            for value in self.pending
            if value[2] != 4
        ]

        return pressed

    def consume_movement(self) -> int:
        """E와 별도로 이동·대기 행동을 가져옵니다."""
        candidates = [
            value
            for value in self.held.values()
            if value[2] != 4
        ]

        candidates.extend(
            value
            for value in self.pending
            if value[2] != 4
        )

        # 아직 처리하지 않은 E는 이동 입력을 꺼내도 유지합니다.
        self.pending = [
            value
            for value in self.pending
            if value[2] == 4
        ]

        return max(candidates)[2] if candidates else 5

    def consume(self) -> int:
        pending = self.pending
        self.pending = []

        # 새로 누른 E는 나중에 들어온 방향키보다 먼저 처리합니다.
        if any(value[2] == 4 for value in pending):
            return 4

        # E를 누르지 않았다면 기존처럼 이동·대기를 선택합니다.
        candidates = [
            value
            for value in self.held.values()
            if value[2] != 4
        ]
        candidates.extend(pending)

        return max(candidates)[2] if candidates else 5
