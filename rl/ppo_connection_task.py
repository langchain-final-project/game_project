"""자석 원격 수집과 물건 획득을 연결하는 실시간 부분 과제입니다."""

from copy import deepcopy

import gymnasium as gym
import numpy as np

from kkamang.env import RealtimeMansionEnv
from rl import ppo_config as cfg
from rl.ppo_action_repeat import ActionRepeat
from rl.ppo_curriculum import Stage2ThreeStarCurriculum
from rl.ppo_observation import PPOGoalObservation
from rl.ppo_reward import PPOReward


class ConnectionStarts(Stage2ThreeStarCurriculum):
    def __init__(self, env, starts, probabilities=None):
        self.starts = tuple(starts)
        if not self.starts or any(type(n) is not int or not 0 <= n <= 7 for n in self.starts):
            raise ValueError("연결 과제 시작 번호는 0~7이어야 합니다.")
        self.probabilities = probabilities
        if probabilities is not None:
            values = np.asarray(probabilities, dtype=float)
            if (values.shape != (len(self.starts),) or not np.isfinite(values).all()
                    or (values < 0).any() or not np.isclose(values.sum(), 1.0)):
                raise ValueError("시작 확률은 시작 번호별 유한한 음이 아닌 값이며 합이 1이어야 합니다.")
            self.probabilities = values / values.sum()
        settings = deepcopy(cfg.CURRICULUM_CONFIG)
        settings.update(initial_frontier=self.starts[0], full_start_probability=0,
                        review_probability=0, magnet_focus_probability=0,
                        quest_focus_probability=0, ghost_focus_probability=0)
        super().__init__(env, settings=settings)

    def _choose_start(self):
        self._first = False
        self._phase = "connection_focus"
        return int(self.env.np_random.choice(self.starts, p=self.probabilities))


class MagnetQuestConnection(gym.Wrapper):
    """전체 보상으로 연습하되 연결 성공 시에만 부분 종료합니다."""

    def __init__(self, env, settings):
        super().__init__(env)
        self.settings = deepcopy(settings)
        self._finished = True
        if type(settings["frame_limit"]) is not int or settings["frame_limit"] <= 0:
            raise ValueError("연결 과제 제한 시간은 양의 프레임 수여야 합니다.")

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.start = info["curriculum_start"]
        self.review = self.start in (6, 7)
        snapshot = info["snapshot"]
        if snapshot["quest"]["collected"] or bool(snapshot["magnet"]["collected"]) != self.review:
            raise ValueError("연결 과제의 시작 상태가 자석·물건 조건과 다릅니다.")
        self.limit = cfg.QUEST_TASK_CONFIG["frame_limit"] if self.review else self.settings["frame_limit"]
        self._frames, self._remote, self._return = 0, 0, 0.0
        self._finished = False
        return obs, info

    def step(self, action):
        if self._finished:
            raise RuntimeError("연결 과제가 끝났습니다. reset()이 필요합니다.")
        obs, reward, game_done, game_truncated, info = self.env.step(action)
        self._frames += 1
        self._remote += sum(e["value"] for e in info["events"] if e["type"] == "collect"
                            and e.get("magnetic", False) and e["source_position"] != e["position"])
        snapshot = info["snapshot"]
        success = bool(snapshot["quest"]["collected"] and
                       (self.review or snapshot["magnet"]["collected"] and self._remote > 0))
        timeout = self._frames >= self.limit and not success
        terms = dict(info["reward_terms"])
        terms["connection_success"] = self.settings["success_bonus"] if success and not self.review else 0.0
        terms["connection_timeout"] = self.settings["timeout"] if timeout else 0.0
        reward += terms["connection_success"] + terms["connection_timeout"]
        self._return += reward
        terminated = bool(success or game_done)
        truncated = bool(not terminated and (timeout or game_truncated))
        self._finished = terminated or truncated
        info = {**info, "reward_terms": terms}
        info.pop("rl_episode", None)
        if self._finished:
            info["connection_episode"] = {
                "start": self.start, "task_kind": "quest_review" if self.review else "magnet_quest_connection",
                "success": success, "frames": self._frames, "return": self._return,
                "reason": "task_success" if success else "local_timeout" if timeout else "game_ended",
                "magnet_collected": snapshot["magnet"]["collected"], "remote_warmth": self._remote,
                "quest_collected": snapshot["quest"]["collected"], "damage": snapshot["damage_count"],
                "game_status": (info.get("outcome") or {}).get("status", "RUNNING"),
            }
        return obs, reward, terminated, truncated, info


def make_connection_env(starts, settings=None, goals=None):
    settings = deepcopy(cfg.CONNECTION_TASK_CONFIG if settings is None else settings)
    environment = deepcopy(cfg.ENV_CONFIG)
    environment.update(stage=2, randomize_phase=False, max_episode_steps=None, render_mode=None)
    probabilities = connection_probabilities(starts, settings)
    env = ConnectionStarts(RealtimeMansionEnv(**environment), starts, probabilities)
    env = PPOGoalObservation(env, star_goal=goals, **cfg.ACTION_MASK_CONFIG)
    env = PPOReward(env)
    env = MagnetQuestConnection(env, settings)
    return ActionRepeat(env, repeat=cfg.ACTION_REPEAT)


def connection_probabilities(starts, settings):
    """학습은 가중 혼합, 고정 구간 평가와 물건 복습은 해당 구간만 사용합니다."""
    weights = settings.get("start_weights")
    if weights is None:
        return None
    values = np.asarray(weights, dtype=float)
    if (values.shape != (len(settings["starts"]),) or not np.isfinite(values).all()
            or (values < 0).any() or not np.isclose(values.sum(), 1.0)):
        raise ValueError("start_weights는 starts와 길이가 같고 합이 1인 확률이어야 합니다.")
    by_start = dict(zip(settings["starts"], values))
    if not all(start in by_start for start in starts):
        return None
    selected = np.asarray([by_start[start] for start in starts])
    if selected.sum() <= 0:
        raise ValueError("선택한 시작 구간의 확률 합은 양수여야 합니다.")
    return selected / selected.sum()
