### reset()에서 시작 상태 선택, 종료 시 난이도 진급 판단

from collections import deque
from copy import deepcopy

import gymnasium as gym

from kkamang.env import RealtimeMansionEnv
from rl.ppo_config import CURRICULUM_CONFIG


CURRICULUM_VERSION = "stage1_auto_return_v1"


class Stage1RouteCurriculum(gym.Wrapper):
    """Stage 1의 실제 이동 경로를 이용해 시작 상태를 조절합니다."""

    stage = 1
    version = CURRICULUM_VERSION

    # 3: 오른쪽, 2: 왼쪽. 숫자는 물리 프레임 수입니다.
    prefixes = (
        (),                              # 0: 원래 시작점
        (3,) * 18,                       # 1: 물건을 향해 이동 중
        (3,) * 36,                       # 2
        (3,) * 54,                       # 3
        (3,) * 72,                       # 4
        (3,) * 90,                       # 5
        (3,) * 108,                      # 6
        (3,) * 116,                      # 7: 물건 획득 직전
        (3,) * 126,                      # 8: 물건 획득 후
        (3,) * 126 + (2,) * 18,           # 9: 출구로 돌아오는 중
        (3,) * 126 + (2,) * 36,           # 10
        (3,) * 126 + (2,) * 54,           # 11
        (3,) * 126 + (2,) * 72,           # 12
        (3,) * 126 + (2,) * 90,           # 13
        (3,) * 126 + (2,) * 108,          # 14
        (3,) * 126 + (2,) * 117,          # 15: 출구 진입 직전
    )

    def __init__(self, env):
        super().__init__(env)

        # 상태 준비와 복원을 위해 기본 환경 바로 바깥에 연결합니다.
        if not isinstance(env, RealtimeMansionEnv):
            raise ValueError(
                "RealtimeMansionEnv 바로 바깥에 연결해야 합니다."
            )

        if (
            env.core.config["stage"] != self.stage
            or env.core.randomize_phase
        ):
            raise ValueError(
                f"고정 위상의 Stage {self.stage} 환경이 필요합니다."
            )

        self.settings = deepcopy(CURRICULUM_CONFIG)
        cfg = self.settings

        if not (
            1 <= cfg["required_successes"] <= cfg["history_size"]
        ):
            raise ValueError("진급 성공 횟수와 기록 개수를 확인하세요.")

        full = cfg["full_start_probability"]
        review = cfg["review_probability"]
        final_full = cfg["final_full_start_probability"]

        if not (
            0 <= full <= 1
            and 0 <= review <= 1
            and full + review <= 1
            and 0 <= final_full <= 1
        ):
            raise ValueError("커리큘럼 확률 설정을 확인하세요.")

        # 가장 쉬운 15번 상태부터 시작합니다.
        self.frontier = len(self.prefixes) - 1

        # 현재 난이도의 최근 성공·실패 기록입니다.
        self.history = deque(maxlen=cfg["history_size"])

        # 한 번 준비한 상태를 복원할 수 있도록 보관합니다.
        self._prepared = {}
        self._first = True

    def _choose_start(self):
        """이번 에피소드의 시작 상태 번호를 고릅니다."""
        rng = self.env.np_random

        if self._first:
            self._first = False
            return self.frontier

        draw = rng.random()

        # 마지막 단계: 원래 시작점 80%, 이전 상태 복습 20%.
        if self.frontier == 0:
            if draw < self.settings["final_full_start_probability"]:
                return 0

            return int(rng.integers(1, len(self.prefixes)))

        # 학습 도중에도 원래 시작점에서 전체 과제를 연습합니다.
        if draw < self.settings["full_start_probability"]:
            return 0

        review_limit = (
            self.settings["full_start_probability"]
            + self.settings["review_probability"]
        )

        # 이미 학습한 더 쉬운 상태를 복습합니다.
        if (
            draw < review_limit
            and self.frontier < len(self.prefixes) - 1
        ):
            return int(
                rng.integers(self.frontier + 1, len(self.prefixes))
            )

        return self.frontier

    def _tag(self, info):
        """시작 상태와 진급 정보를 기록합니다."""
        return {
            **info,
            "curriculum_version": self.version,
            "curriculum_start": self.start,
            "curriculum_frontier": self.episode_frontier,
            "curriculum_next_frontier": self.frontier,
        }

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError(
                "고정 시작 상태를 사용하므로 reset options는 지원하지 않습니다."
            )

        obs, info = self.env.reset(seed=seed)

        self.start = self._choose_start()
        self.episode_frontier = self.frontier

        # 0번은 기본 환경의 원래 시작 상태입니다.
        if self.start == 0:
            return obs, self._tag(info)

        base = self.env

        if self.start in self._prepared:
            # 저장된 상태의 복사본을 사용합니다.
            base.core = deepcopy(self._prepared[self.start])

        else:
            # 실제 게임 규칙으로 이동해 시작 상태를 준비합니다.
            for action in self.prefixes[self.start]:
                obs, _, terminated, truncated, info = base.step(action)

                if terminated or truncated:
                    raise RuntimeError(
                        f"시작 상태 {self.start} 준비 중 게임이 끝났습니다."
                    )

            self._prepared[self.start] = deepcopy(base.core)

        # 준비 이동은 에이전트의 에피소드 스텝에 포함하지 않습니다.
        # 게임 내부의 경과 시간과 획득 상태는 그대로 유지합니다.
        base._steps = 0
        base._finished = False

        snapshot = base.core.snapshot()
        obs = base.encode(snapshot)
        info = base._info(snapshot)

        return obs, self._tag(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        # 현재 난이도에서 시작한 게임의 최종 결과만 사용합니다.
        # 외부 제한으로 중단된 에피소드는 진급 판단에서 제외합니다.
        if terminated and self.start == self.episode_frontier:
            success = (
                (info.get("outcome") or {}).get("status") == "COMPLETE"
            )
            self.history.append(success)

            if (
                self.frontier > 0
                and len(self.history) == self.settings["history_size"]
                and sum(self.history) >= self.settings["required_successes"]
            ):
                self.frontier -= 1
                self.history.clear()

        return obs, reward, terminated, truncated, self._tag(info)

# Stage 2에서 검증한 안전 경로입니다.
# 행동 번호: 0=위, 1=아래, 2=왼쪽, 3=오른쪽
_STAGE2_SAFE_ROUTE = (
    (3,) * 36       # (4,1) → (6,1): 온기 수집
    + (1,) * 90     # (6,1) → (6,6): 자석과 추가 온기 획득
    + (3,) * 18     # (6,6) → (7,6): 필수 물건 획득
    + (2,) * 18     # (7,6) → (6,6)
    + (0,) * 90     # (6,6) → (6,1)
    + (2,) * 36     # 출구 방향으로 이동
)


# 경로의 몇 프레임까지 진행한 상태에서 시작할지 정합니다.
# 279프레임은 자동 귀환 직전입니다.
_STAGE2_PREFIX_FRAMES = (
    0, 18, 36, 54, 72, 90, 108, 126,
    135, 144, 162, 180, 198, 234, 270, 279,
)


class Stage2RouteCurriculum(Stage1RouteCurriculum):
    """Stage 2의 안전 경로를 사용하는 커리큘럼입니다."""

    stage = 2
    version = "stage2_safe_route_v1"

    prefixes = tuple(
        _STAGE2_SAFE_ROUTE[:frames]
        for frames in _STAGE2_PREFIX_FRAMES
    )