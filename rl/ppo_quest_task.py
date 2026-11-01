"""Stage 2 물건 획득 전용 환경. 부분 성공은 게임 클리어와 구분합니다."""

from copy import deepcopy

import gymnasium as gym

from kkamang.env import RealtimeMansionEnv
from rl import ppo_config as cfg
from rl.ppo_action_repeat import ActionRepeat
from rl.ppo_curriculum import Stage2ThreeStarCurriculum
from rl.ppo_observation import PPOGoalObservation
from rl.ppo_path_distance import path_distance


class QuestAcquisitionTask(gym.Wrapper):
    """물건 획득 즉시 종료하고 제한 시간에는 부분 과제를 중단합니다."""

    def __init__(self, env, *, start, settings):
        super().__init__(env)
        self.start = start
        self.settings = deepcopy(settings)
        if start not in (6, 7):
            raise ValueError("물건 획득 연습은 6·7번만 사용합니다.")
        limit = settings["frame_limit"]
        if type(limit) is not int or limit <= 0:
            raise ValueError("부분 과제 프레임 제한은 양의 정수여야 합니다.")
        self._finished = True

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        snapshot = info["snapshot"]
        if snapshot["quest"]["collected"] or not snapshot["magnet"]["collected"]:
            raise ValueError("자석 보유·물건 미획득 상태에서 시작해야 합니다.")
        self._distance = path_distance(snapshot, snapshot["quest"]["position"])
        self._damage = snapshot["damage_count"]
        self._frames = 0
        self._return = 0.0
        self._finished = False
        return obs, {**info, "task_kind": "quest_acquisition", "quest_task_start": self.start}

    def step(self, action):
        if self._finished:
            raise RuntimeError("부분 과제가 끝났습니다. reset()이 필요합니다.")
        obs, _, game_done, game_truncated, info = self.env.step(action)
        snapshot = info["snapshot"]
        self._frames += 1
        distance = path_distance(snapshot, snapshot["quest"]["position"])
        progress = (self._distance - distance
                    if self._distance is not None and distance is not None else 0.0)
        damage = snapshot["damage_count"] - self._damage
        success = bool(snapshot["quest"]["collected"])
        timeout = self._frames >= self.settings["frame_limit"] and not success
        terms = {
            "quest_task_progress": self.settings["progress_per_cell"] * progress,
            "quest_task_time": self.settings["time_per_frame"],
            "quest_task_damage": self.settings["damage"] * damage,
            "quest_task_success": self.settings["success_bonus"] if success else 0.0,
            "quest_task_timeout": self.settings["timeout"] if timeout else 0.0,
        }
        reward = float(sum(terms.values()))
        self._return += reward
        self._distance, self._damage = distance, snapshot["damage_count"]
        # 물건 획득은 부분 MDP의 종료이고, 시간 제한은 외부 중단입니다.
        # 원래 GameCore의 outcome은 COMPLETE로 바꾸지 않습니다.
        terminated = bool(success or game_done)
        truncated = bool(not terminated and (timeout or game_truncated))
        self._finished = terminated or truncated
        info = {**info, "task_kind": "quest_acquisition", "reward_terms": terms}
        if self._finished:
            info["quest_task_episode"] = {
                "start": self.start, "success": success,
                "reason": "quest_collected" if success else (
                    "local_timeout" if timeout else "game_ended"
                ),
                "frames": self._frames, "return": self._return,
                "damage": snapshot["damage_count"],
                "quest_collected": snapshot["quest"]["collected"],
                "game_status": (info.get("outcome") or {}).get("status", "RUNNING"),
            }
        return obs, reward, terminated, truncated, info


def quest_goals():
    goals = deepcopy(cfg.STAR_GOAL_CONFIG)
    goals.update(enabled=True, stage2_three_star=True, quest_after_magnet=True,
                 direction_encoding="local_clipped_v1", path_waypoints=True)
    return goals


def make_quest_env(start, settings=None):
    settings = deepcopy(cfg.QUEST_TASK_CONFIG if settings is None else settings)
    curriculum = deepcopy(cfg.CURRICULUM_CONFIG)
    curriculum.update(initial_frontier=start, full_start_probability=0, review_probability=0,
                      magnet_focus_probability=0, quest_focus_probability=0, ghost_focus_probability=0)
    environment = deepcopy(cfg.ENV_CONFIG)
    environment.update(stage=2, randomize_phase=False, max_episode_steps=None, render_mode=None)
    env = Stage2ThreeStarCurriculum(RealtimeMansionEnv(**environment), settings=curriculum)
    env = PPOGoalObservation(env, star_goal=settings.get("goals", quest_goals()), **cfg.ACTION_MASK_CONFIG)
    env = QuestAcquisitionTask(env, start=start, settings=settings)
    return ActionRepeat(env, repeat=cfg.ACTION_REPEAT)
