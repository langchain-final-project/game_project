"""PPO 별 수집용 목표 안내. 실제 게임의 귀환 조건은 변경하지 않는다."""
import numpy as np

from rl.observation import interaction_ready
from rl.task_goal_observation import TaskGoalObservation, TASK_GOAL_VERSION


def observation_version(warmth_goal):
    return f"{TASK_GOAL_VERSION}_star_goal{warmth_goal}_v1"


class PPOStarObservation(TaskGoalObservation):
    def __init__(self, env, *, warmth_goal=7):
        if type(warmth_goal) is not int or not 4 <= warmth_goal <= 14:
            raise ValueError("수집 목표는 4~14의 정수여야 합니다.")
        super().__init__(env)
        self.warmth_goal = warmth_goal
        self.observation_version = observation_version(warmth_goal)

    def _augment(self, obs, info):
        snapshot = info["snapshot"]
        x, y = snapshot["motion"]["position"]
        if not snapshot["quest"]["collected"]:
            phase, target = 0, snapshot["quest"]["position"]
        elif snapshot["carried"] < max(snapshot["target"], self.warmth_goal):
            phase = 1
            active = [item for item in snapshot["items"] if item["active"]]
            if active:
                nearest = min(active, key=lambda item: (
                    abs(item["position"][0] - x) + abs(item["position"][1] - y)
                ))
                target = nearest["position"]
            else:
                target = snapshot["exit"]
        else:
            phase, target = 2, snapshot["exit"]
        extra = np.array([
            *[float(phase == i) for i in range(3)],
            0.5 + (target[0] - x) / 30,
            0.5 + (target[1] - y) / 16,
            # 실제 게임에서 가능한 상호작용을 표시한다. 온기 4점 조기 귀환도 허용된다.
            float(interaction_ready(snapshot)),
        ], dtype=np.float32)
        return np.concatenate((obs, extra)), {
            **info, "observation_version": self.observation_version,
            "star_warmth_goal": self.warmth_goal,
        }
