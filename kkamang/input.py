"""키 입력을 여섯 행동으로 변환한다. 짧은 방향 입력과 상호작용을 보존해 논리 진행 주기에 맞춰 소비한다."""
import pygame

KEY_ACTION = {pygame.K_UP: 0, pygame.K_w: 0, pygame.K_DOWN: 1, pygame.K_s: 1,
              pygame.K_LEFT: 2, pygame.K_a: 2, pygame.K_RIGHT: 3, pygame.K_d: 3,
              pygame.K_e: 4, pygame.K_SPACE: 5}

class InputAdapter:
    def __init__(self):
        self.clear()

    def clear(self):
        self.held = {}
        self.pending = []
        self.serial = 0
        self.current_direction = None

    def handle(self, event):
        if event.type not in (pygame.KEYDOWN, pygame.KEYUP) or event.key not in KEY_ACTION:
            return
        if event.type == pygame.KEYUP:
            self.held.pop(event.key, None)
            return
        if event.key in self.held:  # 운영체제 키 반복으로 상호작용을 재실행하지 않는다.
            return
        self.serial += 1
        # 시간 정보가 같거나 없어도 이벤트 도착 순서를 유지한다.
        value = (self.serial, 0, KEY_ACTION[event.key])
        self.held[event.key] = value
        # 짧은 입력은 다음 논리 틱까지 보관하고 누른 방향은 계속 유지한다.
        self.pending.append(value)

    def consume_interaction(self) -> bool:
        """새로 누른 E만 가져오고 이동 입력은 남겨둔다."""
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

    def consume_movement(self, can_move=None) -> int:
        """상호작용과 별도로 이동·대기 행동을 가져온다."""
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

        # 이동 입력을 꺼내도 아직 처리하지 않은 상호작용은 남겨둔다.
        self.pending = [
            value
            for value in self.pending
            if value[2] == 4
        ]

        requested = max(candidates)[2] if candidates else 5
        if requested == 5:
            self.current_direction = None
            return requested

        if can_move is None or can_move(requested):
            self.current_direction = requested
            return requested

        # 누른 회전은 길이 열릴 때까지 요청하되 손을 뗀 짧은 입력은 한 번만 소비해 늦은 회전을 막는다.
        current = self.current_direction
        if (current is not None
                and any(value[2] == current for value in self.held.values())
                and can_move(current)):
            return current
        return requested

    def consume(self) -> int:
        pending = self.pending
        self.pending = []

        # 새 상호작용은 뒤에 들어온 방향 입력보다 먼저 처리한다.
        if any(value[2] == 4 for value in pending):
            return 4

        candidates = [
            value
            for value in self.held.values()
            if value[2] != 4
        ]
        candidates.extend(pending)

        return max(candidates)[2] if candidates else 5
