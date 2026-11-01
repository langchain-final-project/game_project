### reset()에서 시작 상태 선택, 종료 시 난이도 진급 판단

from collections import deque
from copy import deepcopy

import gymnasium as gym

from kkamang.env import RealtimeMansionEnv
from rl.ppo_config import CURRICULUM_CONFIG


CURRICULUM_VERSION = "stage1_balanced_return_v3"


class Stage1RouteCurriculum(gym.Wrapper):
    """Stage 1의 실제 이동 경로를 이용해 시작 상태를 조절합니다."""

    stage = 1
    version = CURRICULUM_VERSION
    required_stars = 0

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

    def __init__(self, env, *, settings=None):
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

        self.settings = deepcopy(CURRICULUM_CONFIG if settings is None else settings)
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

        # 설정한 시작 구간부터 연습합니다.
        self.frontier = cfg.get(
            "initial_frontier",
            len(self.prefixes) - 1,
        )

        if (
            type(self.frontier) is not int
            or not 0 <= self.frontier < len(self.prefixes)
        ):
            raise ValueError("initial_frontier는 유효한 시작 상태 번호여야 합니다.")

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
            "curriculum_required_stars": self.required_stars,
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
        if terminated:
            outcome = info.get("outcome") or {}
            success = (outcome.get("status") == "COMPLETE"
                       and outcome.get("score", {}).get("stars", 0) >= self.required_stars
                       and (not getattr(self, "required_damage_free", False)
                            or info["snapshot"]["damage_count"] == 0))
            info = {**info, "curriculum_task_success": success}
        if terminated and self.start == self.episode_frontier:
            self.history.append(success)

            if (
                self.frontier > 0
                and len(self.history) == self.settings["history_size"]
                and sum(self.history) >= self.settings["required_successes"]
            ):
                self.frontier -= 1
                self.history.clear()

        return obs, reward, terminated, truncated, self._tag(info)

# Stage 1: 온기 14점과 필수 물건을 수집한 뒤 귀환하는 실제 물리 이동 경로.
# 한 칸당 18프레임이며 자동 귀환은 532번째 프레임에 발생합니다.
_STAGE1_THREE_STAR_CELLS = (
    3, 3, 2, 1, 1, 3, 1, 1, 3, 3, 3, 1, 3, 0, 0,
    0, 0, 0, 2, 2, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2,
)
_STAGE1_THREE_STAR_ROUTE = tuple(
    action for action in _STAGE1_THREE_STAR_CELLS for _ in range(18)
)


class Stage1ThreeStarCurriculum(Stage1RouteCurriculum):
    """14점 수집·물건 획득·귀환을 역순으로 연습하고 별 3개로 진급합니다."""

    version = "stage1_three_star_v1"
    required_stars = 3
    route = _STAGE1_THREE_STAR_ROUTE
    prefix_frames = tuple(range(0, 540, 18))
    prefixes = tuple(_STAGE1_THREE_STAR_ROUTE[:n] for n in range(0, 540, 18))


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


# Stage 3: 왼쪽 벽은 아래로, 오른쪽 벽은 위로 우회합니다.
# 처음 네 개의 1점 온기를 수집하고 (7,5)의 무적 아이템을 통과합니다.
# 실제 물리 이동으로 검증한 경로이며 귀환 성공은 856프레임에 발생합니다.
_STAGE3_RETURN_ROUTE = (
    (3,) * 36 + (1,) * 36 + (3,) * 18 + (1,) * 54
    + (3,) * 54 + (0,) * 72 + (3,) * 72 + (1,) * 90
    + (0,) * 90 + (2,) * 72 + (1,) * 72 + (2,) * 108 + (0,) * 90
)
_STAGE3_PREFIX_FRAMES = (
    0, 36, 90, 144, 198, 270, 342, 422,
    432, 522, 594, 666, 684, 702, 720, 774, 846, 855,
)


class Stage3RouteCurriculum(Stage1RouteCurriculum):
    """출구 직전 귀환부터 물건 획득·전체 경로로 역순 확장합니다."""

    stage = 3
    version = "stage3_return_route_v2"
    route = _STAGE3_RETURN_ROUTE
    prefix_frames = _STAGE3_PREFIX_FRAMES
    prefixes = tuple(_STAGE3_RETURN_ROUTE[:frames] for frames in _STAGE3_PREFIX_FRAMES)


_STAGE3_THREE_STAR_ROUTE = (
    (3,) * 36 + (1,) * 36 + (3,) * 18 + (1,) * 54 + (3,) * 54
    + (0,) * 18 + (3,) * 18 + (0,) * 72 + (3,) * 90
    + (1,) * 90 + (2,) * 36 + (1,) * 18
    + (0,) * 90 + (2,) * 72 + (1,) * 72 + (2,) * 108 + (0,) * 90
)
_STAGE3_THREE_STAR_FRAMES = (
    0, 36, 90, 144, 198, 216, 252, 270, 306, 360, 450, 486,
    530, 540, 630, 702, 774, 792, 810, 828, 882, 954, 963,
)


class Stage3ThreeStarCurriculum(Stage1RouteCurriculum):
    """실제 온기 14점·무적·물건 수집 경로에서 별 3개 귀환부터 확장합니다."""
    stage = 3
    version = "stage3_three_star_v1"
    required_stars = 3
    route = _STAGE3_THREE_STAR_ROUTE
    prefix_frames = _STAGE3_THREE_STAR_FRAMES
    prefixes = tuple(_STAGE3_THREE_STAR_ROUTE[:n] for n in _STAGE3_THREE_STAR_FRAMES)


class Stage3EfficientCurriculum(Stage3ThreeStarCurriculum):
    """출구 직전부터 역순 확장하면서 수집·무적·물건 획득을 함께 복습합니다."""

    version = 'stage3_efficient_three_star_v2'

    def __init__(self, env, *, settings=None):
        super().__init__(env, settings=settings)
        probabilities = [self.settings['full_start_probability'], self.settings['review_probability'],
                         self.settings.get('stage3_collect_probability',0),
                          self.settings.get('stage3_quest_probability',0),
                          self.settings.get('stage3_return_probability',0)]
        if any(not 0 <= p <= 1 for p in probabilities) or sum(probabilities) > 1:
            raise ValueError('Stage 3 시작 구간 확률의 합은 1 이하여야 합니다.')
        for kind in ('collect','quest','return'):
            starts = self.settings.get(f'stage3_{kind}_starts',[])
            if (self.settings.get(f'stage3_{kind}_probability',0) > 0
                    and (not starts or any(type(i) is not int or not 0 < i < len(self.prefixes) for i in starts))):
                raise ValueError('Stage 3 집중 복습 시작 번호를 확인하세요.')

    def _choose_start(self):
        if self._first or self.frontier == 0:
            start = super()._choose_start()
            self._phase = 'full_start' if start == 0 else 'frontier' if start == self.frontier else 'review'
            return start
        rng = self.env.np_random
        draw = rng.random()
        limit = self.settings['full_start_probability']
        if draw < limit:
            self._phase = 'full_start'
            return 0
        for kind in ('collect','quest','return'):
            limit += self.settings.get(f'stage3_{kind}_probability',0)
            if draw < limit:
                self._phase = f'stage3_{kind}_focus'
                return int(rng.choice(self.settings[f'stage3_{kind}_starts']))
        limit += self.settings['review_probability']
        if draw < limit and self.frontier < len(self.prefixes)-1:
            self._phase = 'review'
            return int(rng.integers(self.frontier+1,len(self.prefixes)))
        self._phase = 'frontier'
        return self.frontier

    def _tag(self, info):
        return {**super()._tag(info), 'curriculum_phase':self._phase}


class Stage2MagnetCollectionCurriculum(gym.Wrapper):
    """실제 이동으로 준비한 자석 사용 후 수집 구간을 고정 비율로 연습합니다."""

    stage = 2
    version = "stage2_magnet_ten_v1"
    _collect_route = (3,) * 36 + (1,) * 90 + (3,) * 54
    # 유령 위상을 기다린 뒤 3점 온기를 수집하는 합법적인 준비 경로입니다.
    _nine_route = _collect_route + (5,) * 72 + (0,) * 54 + (2,) * 18
    prefixes = (
        (), _collect_route[:108], _collect_route[:144], _collect_route,
        _nine_route, _nine_route + (0,) * 36,
    )
    phases = ("full_start", "magnet_before_quest", "quest_warmth4",
              "quest_warmth6", "quest_warmth9", "return_warmth10")

    def __init__(self, env, *, settings=None):
        super().__init__(env)
        if not isinstance(env, RealtimeMansionEnv) or env.core.config["stage"] != 2 or env.core.randomize_phase:
            raise ValueError("고정 위상의 RealtimeMansionEnv Stage 2가 필요합니다.")
        self.settings = deepcopy(CURRICULUM_CONFIG if settings is None else settings)
        weights = self.settings["focus_start_weights"]
        if (len(weights) != len(self.prefixes) or any(w < 0 for w in weights)
                or abs(sum(weights) - 1.0) > 1e-9):
            raise ValueError("focus_start_weights는 합이 1인 6개 확률이어야 합니다.")
        self.first_start = self.settings["initial_frontier"]
        if type(self.first_start) is not int or not 0 <= self.first_start < len(self.prefixes):
            raise ValueError("initial_frontier는 0~5여야 합니다.")
        self._prepared = {}
        self._first = True

    def _tag(self, info):
        return {**info, "curriculum_version": self.version,
                "curriculum_start": self.start,
                "curriculum_frontier": self.start, "curriculum_next_frontier": self.start,
                "curriculum_phase": self.phases[self.start],
                "curriculum_magnet_remote_warmth_at_start": self.remote_at_start}

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("구간 준비에는 reset options를 지원하지 않습니다.")
        obs, info = self.env.reset(seed=seed)
        if self._first:
            self.start, self._first = self.first_start, False
        else:
            self.start = int(self.env.np_random.choice(len(self.prefixes), p=self.settings["focus_start_weights"]))
        self.remote_at_start = 0
        if self.start == 0:
            return obs, self._tag(info)
        if self.start not in self._prepared:
            for action in self.prefixes[self.start]:
                obs, _, terminated, truncated, info = self.env.step(action)
                self.remote_at_start += sum(
                    e["value"] for e in info["events"] if e["type"] == "collect"
                    and e.get("magnetic", False) and e["source_position"] != e["position"]
                )
                if terminated or truncated:
                    raise RuntimeError(f"구간 {self.start} 준비 도중 게임이 끝났습니다.")
            self._prepared[self.start] = (deepcopy(self.env.core), self.remote_at_start)
        else:
            core, self.remote_at_start = self._prepared[self.start]
            self.env.core = deepcopy(core)
        # 준비 시간·온기·피해·자석 잔여 시간은 실제 상태대로 유지합니다.
        self.env._steps, self.env._finished = 0, False
        snapshot = self.env.core.snapshot()
        return self.env.encode(snapshot), self._tag(self.env._info(snapshot))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self._tag(info)

# 실제 실시간 이동으로 검증한 자석 활용·별 3개 경로입니다.
_STAGE2_THREE_STAR_CELLS = (
    (3,) * 2
    + (1,) * 5
    + (3,) * 3
    + (5,)
    + (0,) * 5
    + (3,)
    + (1,) * 4
    + (2,) * 3
    + (1,)
    + (2,)
    + (0,) * 5
    + (2,) * 2
)

# 기본 유령 위상에서 무피해 경로가 되도록 한 틱 기다립니다.
_STAGE2_THREE_STAR_ROUTE = (5,) * 18 + tuple(
    action
    for action in _STAGE2_THREE_STAR_CELLS
    for _ in range(18)
)


class Stage2ThreeStarCurriculum(Stage1RouteCurriculum):
    """물건 획득 전·유령 근처·진행 구간을 섞어 별 3개 귀환을 연습합니다."""

    stage = 2
    version = "stage2_joint_review_v3"
    required_stars = 3
    route = _STAGE2_THREE_STAR_ROUTE

    prefix_frames = tuple(
        range(0, len(_STAGE2_THREE_STAR_ROUTE), 18)
    )
    prefixes = tuple(
        _STAGE2_THREE_STAR_ROUTE[:n]
        for n in range(0, len(_STAGE2_THREE_STAR_ROUTE), 18)
    )

    def __init__(self, env, *, settings=None):
        super().__init__(env, settings=settings)
        probabilities = [
            self.settings.get("magnet_focus_probability", 0.0),
            self.settings.get("quest_focus_probability", 0.0),
            self.settings.get("ghost_focus_probability", 0.0),
            self.settings["full_start_probability"],
            self.settings["review_probability"],
        ]
        if (any(not 0 <= p <= 1 for p in probabilities)
                or sum(probabilities) > 1 + 1e-9):
            raise ValueError("전체·자석·물건·유령·복습 확률의 합은 1 이하여야 합니다.")
        for kind in ("magnet", "quest", "ghost"):
            starts = self.settings.get(f"{kind}_focus_starts", [])
            if self.settings.get(f"{kind}_focus_probability", 0.0) > 0 and (
                not starts or any(type(n) is not int or not 0 < n < len(self.prefixes) for n in starts)
            ):
                raise ValueError(f"{kind}_focus_starts에 유효한 시작 번호가 필요합니다.")
        waits = self.settings.get("ghost_wait_frames", [0])
        maximum_wait = self.settings.get("ghost_max_wait_frames", 18)
        if (type(maximum_wait) is not int or not 0 <= maximum_wait <= 144
                or not waits or any(type(n) is not int or not 0 <= n <= maximum_wait for n in waits)):
            raise ValueError("ghost_wait_frames와 허용 대기 범위(최대 144프레임)를 확인하세요.")

    def _choose_start(self):
        if self._first or (self.frontier == 0 and not self.settings.get("keep_focus_at_frontier_zero", False)):
            start = super()._choose_start()
            self._phase = "full_start" if start == 0 else (
                "frontier" if start == self.frontier else "review"
            )
            return start
        rng = self.env.np_random
        draw = rng.random()
        limit = self.settings["full_start_probability"]
        if draw < limit:
            self._phase = "full_start"
            return 0
        for kind in ("magnet", "quest", "ghost"):
            limit += self.settings.get(f"{kind}_focus_probability", 0.0)
            if draw < limit:
                self._phase = f"{kind}_focus"
                return int(rng.choice(self.settings[f"{kind}_focus_starts"]))
        limit += self.settings["review_probability"]
        if draw < limit and self.frontier < len(self.prefixes) - 1:
            self._phase = "review"
            return int(rng.integers(self.frontier + 1, len(self.prefixes)))
        self._phase = "frontier"
        return self.frontier

    def _tag(self, info):
        return {
            **super()._tag(info),
            "curriculum_phase": self._phase,
            "curriculum_ghost_wait_frames": self._ghost_wait_frames,
        }

    def reset(self, *, seed=None, options=None):
        self._ghost_wait_frames = 0
        obs, info = super().reset(seed=seed, options=options)
        if self._phase != "ghost_focus":
            return obs, info
        if self.settings.get("ghost_safe_preparation", False):
            return self._safe_ghost_reset()
        self._ghost_wait_frames = int(self.env.np_random.choice(
            self.settings.get("ghost_wait_frames", [0])
        ))
        # 실제 대기로 유령 위상을 바꿉니다. 시간·자석 잔여 시간도 함께 경과합니다.
        for _ in range(self._ghost_wait_frames):
            obs, _, done, truncated, info = self.env.step(5)
            if done or truncated:
                raise RuntimeError("유령 연습의 준비 대기 중 게임이 끝났습니다.")
        self.env._steps = 0
        self.env._finished = False
        snapshot = self.env.core.snapshot()
        return self.env.encode(snapshot), self._tag(self.env._info(snapshot))

    def _safe_ghost_reset(self):
        """실제 대기 중 피해·종료가 생기지 않는 합법적인 시작 상태만 복습합니다."""
        if not hasattr(self, "_safe_wait_cache"):
            self._safe_wait_cache = {}
        if self.start not in self._safe_wait_cache:
            original = deepcopy(self.env.core)
            initial_damage = original.snapshot()["damage_count"]
            candidates = []
            for wait in self.settings["ghost_wait_frames"]:
                self.env.core = deepcopy(original)
                self.env._steps, self.env._finished = 0, False
                safe = initial_damage == 0
                for _ in range(wait):
                    _, _, done, truncated, info = self.env.step(5)
                    if done or truncated or info["snapshot"]["damage_count"] != initial_damage:
                        safe = False
                        break
                if safe:
                    candidates.append((wait, deepcopy(self.env.core)))
            if not candidates:
                raise RuntimeError("무피해로 준비할 수 있는 유령 복습 시작 상태가 없습니다.")
            self._safe_wait_cache[self.start] = candidates
        candidates = self._safe_wait_cache[self.start]
        wait, core = candidates[int(self.env.np_random.integers(len(candidates)))]
        self._ghost_wait_frames = wait
        self.env.core = deepcopy(core)
        self.env._steps, self.env._finished = 0, False
        snapshot = self.env.core.snapshot()
        return self.env.encode(snapshot), self._tag(self.env._info(snapshot))


class Stage2SafeReturnCurriculum(Stage2ThreeStarCurriculum):
    """별 3개와 무피해 귀환을 함께 만족해야 진행 구간을 통과합니다."""

    version = "stage2_safe_return_v1"
    required_damage_free = True
