"""행동 종류에 관계없이 연속 정지와 좁은 범위 왕복을 추적합니다."""

from collections import deque
from copy import deepcopy

from rl.ppo_config import STAGNATION_CONFIG


class StagnationTracker:
    def __init__(self, position, settings=None):
        self.settings = deepcopy(STAGNATION_CONFIG if settings is None else settings)
        self.previous = tuple(position)
        self.positions = deque(maxlen=self.settings["local_window_frames"] + 1)
        self.positions.append(self.previous)
        self.stationary_streak = self.max_stationary_frames = 0
        self.stationary_frames = self.stagnant_frames = self.local_loop_frames = 0

    def update(self, position, *, made_progress=False):
        position = tuple(position)
        unchanged = max(abs(position[i] - self.previous[i]) for i in (0, 1)) < 1e-8
        self.stationary_frames += int(unchanged)
        self.stationary_streak = self.stationary_streak + 1 if unchanged else 0
        self.max_stationary_frames = max(self.max_stationary_frames, self.stationary_streak)
        self.previous = position
        if made_progress:
            # 수집으로 진행된 구간은 왕복 실패로 판정하지 않습니다.
            self.positions.clear()
            self.stationary_streak = 0
        self.positions.append(position)
        local_loop = (
            len(self.positions) == self.positions.maxlen
            and all(max(p[i] for p in self.positions) - min(p[i] for p in self.positions)
                    <= self.settings["local_max_span"] for i in (0, 1))
        )
        stagnant = not made_progress and (
            self.stationary_streak >= self.settings["stationary_grace_frames"] or local_loop
        )
        self.local_loop_frames += int(local_loop)
        self.stagnant_frames += int(stagnant)
        return bool(stagnant), bool(local_loop)

    def summary(self):
        return {name: getattr(self, name) for name in (
            "stationary_frames", "max_stationary_frames", "stagnant_frames", "local_loop_frames",
        )}
