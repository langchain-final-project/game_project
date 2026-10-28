from collections import deque
from copy import deepcopy

import gymnasium as gym


class Stage1RouteCurriculum(gym.Wrapper):
    # 실제 성공 경로의 앞부분들입니다.
    # 0: 원래 시작점
    # 1~7: 물건을 얻으러 가는 도중
    # 8: 물건 획득 직후
    # 9~15: 출구로 돌아오는 도중
    prefixes = (
        [],
        [3] * 18,
        [3] * 36,
        [3] * 54,
        [3] * 72,
        [3] * 90,
        [3] * 108,
        [3] * 126,
        [3] * 126 + [4],
        [3] * 126 + [4] + [2] * 18,
        [3] * 126 + [4] + [2] * 36,
        [3] * 126 + [4] + [2] * 54,
        [3] * 126 + [4] + [2] * 72,
        [3] * 126 + [4] + [2] * 90,
        [3] * 126 + [4] + [2] * 108,
        [3] * 126 + [4] + [2] * 126,
    )

    def __init__(self, env):
        super().__init__(env)

        base = env.unwrapped
        if base.core.config["stage"] != 1 or base.core.randomize_phase:
            raise ValueError("고정 위상의 Stage 1 전용 커리큘럼입니다.")

        self.frontier = len(self.prefixes) - 1
        self.history = deque(maxlen=20)
        self._prepared = {}
        self._first = True

    def _tag(self, info):
        info["curriculum_start"] = self.start
        info["curriculum_frontier"] = self.episode_frontier
        return info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        rng = self.env.np_random

        # 첫 에피소드는 출구에서 귀환 행동을 경험하게 합니다.
        if self._first:
            self.start = self.frontier
            self._first = False
        else:
            draw = rng.random()

            if self.frontier == 0:
                # 마지막 단계: 원래 시작점 80%, 이전 상태 복습 20%
                self.start = (
                    0 if draw < 0.8
                    else int(rng.integers(1, len(self.prefixes)))
                )
            elif draw < 0.1:
                # 처음부터 전체 과제도 계속 연습합니다.
                self.start = 0
            elif draw < 0.3 and self.frontier < len(self.prefixes) - 1:
                # 이미 배운 더 쉬운 상태를 복습합니다.
                self.start = int(
                    rng.integers(self.frontier + 1, len(self.prefixes))
                )
            else:
                self.start = self.frontier

        self.episode_frontier = self.frontier

        if self.start == 0:
            return obs, self._tag(info)

        base = self.env.unwrapped

        # 준비 경로는 상태별로 한 번만 실행합니다.
        if self.start in self._prepared:
            saved_core, saved_steps = self._prepared[self.start]
            base.core = deepcopy(saved_core)
            base._steps = saved_steps
            base._finished = False

            snapshot = base.core.snapshot()
            obs = base.encode(snapshot)
            info = base._info(snapshot)
        else:
            for action in self.prefixes[self.start]:
                obs, _, terminated, truncated, info = self.env.step(action)

                if terminated or truncated:
                    raise RuntimeError("준비 경로가 예상보다 일찍 끝났습니다.")

            self._prepared[self.start] = (
                deepcopy(base.core),
                base._steps,
            )

        return obs, self._tag(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            # 현재 학습 단계에서 시작한 결과만 진급 판단에 사용합니다.
            if self.start == self.episode_frontier:
                success = (
                    (info.get("outcome") or {}).get("status") == "COMPLETE"
                )
                self.history.append(success)

                if (
                    self.frontier > 0
                    and len(self.history) == 20
                    and sum(self.history) >= 16
                ):
                    self.frontier -= 1
                    self.history.clear()

        return (
            obs,
            reward,
            terminated,
            truncated,
            self._tag(info),
        )