"""탐사의 수집·충돌·귀환·점수 규칙과 결정적 상태 전이를 구현한다. 화면·저장·강화학습에 의존하지 않는다."""
from __future__ import annotations
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
from numbers import Integral
import random

from .levels import MAX_TICKS, validate_level
from .scoring import score_outcome

OBSERVATION_VERSION = "vector_v5_invincibility"


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    INTERACT = 4
    WAIT = 5


@dataclass(frozen=True)
class StepResult:
    snapshot: dict
    events: list[dict]
    terminated: bool
    outcome: dict | None


class GameCore:
    """캠페인 진행과 분리된 한 번의 탐사. 격자 행동 하나는 한 틱을 소비한다."""

    def __init__(self, config: dict, mode: str = "basic", seed: int = 0,
                 randomize_phase: bool = False) -> None:
        validate_level(config)
        if mode not in ("basic", "relaxed"):
            raise ValueError("mode must be 'basic' or 'relaxed'")
        if not isinstance(seed, Integral) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if not isinstance(randomize_phase, bool):
            raise ValueError("randomize_phase must be a bool")
        self._config = deepcopy(config)
        self.config_hash = hashlib.sha256(json.dumps(
            self._config, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()
        self.mode = mode
        self.seed = int(seed)
        self.randomize_phase = randomize_phase
        self._rng = random.Random(self.seed)
        budget = int(config["time_budget_ticks"])
        self._initial_ticks = budget if mode == "basic" else (3 * budget + 1) // 2
        if self._initial_ticks > MAX_TICKS:
            raise ValueError(f"{mode} time budget exceeds observation ceiling {MAX_TICKS}")
        self._state: dict = {}
        self._actions: list[int] = []
        self._initial_phase: dict = {}
        self._damage_count = 0
        self.reset()

    @property
    def config(self) -> dict:
        """외부에서 진행 중 규칙을 바꾸지 못하도록 복사본을 반환한다."""
        return deepcopy(self._config)

    @property
    def actions(self) -> list[int]:
        return list(self._actions)

    def reset(self, seed: int | None = None, phase_overrides: dict | None = None) -> dict:
        """탐사를 초기화한다. 난수 순찰은 seed=None일 때 이어가며 실제 시작 위상을 재생 기록에 남긴다."""
        if seed is not None and (not isinstance(seed, Integral) or isinstance(seed, bool)):
            raise ValueError("seed must be an integer or None")
        overrides = {} if phase_overrides is None else phase_overrides
        if not isinstance(overrides, dict):
            raise ValueError("phase_overrides must be a dictionary keyed by ghost ID")
        ghost_configs = {ghost["id"]: ghost for ghost in self._config["ghosts"]}
        if set(overrides) - ghost_configs.keys():
            raise ValueError("phase_overrides contains an unknown ghost ID")
        for ghost_id, phase in overrides.items():
            if not isinstance(phase, dict) or set(phase) != {"index", "countdown"}:
                raise ValueError(f"phase override {ghost_id} requires exactly index and countdown")
            index, countdown = phase["index"], phase["countdown"]
            if (not isinstance(index, Integral) or isinstance(index, bool)
                    or not 0 <= index < len(ghost_configs[ghost_id]["route"])):
                raise ValueError(f"phase override {ghost_id} index is out of range")
            if (
                not isinstance(countdown, Integral)
                or isinstance(countdown, bool)
                or not 1 <= countdown <= MAX_TICKS
            ):
                raise ValueError(
                    f"phase override {ghost_id} countdown must be an integer in 1..{MAX_TICKS}"
                )
        if seed is not None:
            self.seed = int(seed)
            self._rng.seed(self.seed)
        ghosts = []
        initial_phase = {}
        for config in self._config["ghosts"]:
            ghost_id = config["id"]
            if ghost_id in overrides:
                index = int(overrides[ghost_id]["index"])
                countdown = int(overrides[ghost_id]["countdown"])
            elif self.randomize_phase:
                index = self._rng.randrange(len(config["route"]))
                countdown = self._rng.randint(1, config["ticks_per_move"])
            else:
                index = config["initial_index"]
                countdown = config["initial_countdown"]
            route = [list(point) for point in config["route"]]
            direction = config["direction"]
            ghosts.append({
                "id": ghost_id, "route": route, "route_index": index,
                "direction": direction, "countdown": countdown,
                "ticks_per_move": config["ticks_per_move"],
                "position": list(route[index]),
                "next_position": list(route[(index + direction) % len(route)]),
            })
            initial_phase[ghost_id] = {"index": index, "countdown": countdown}
        config = self._config
        self._state = {
            "stage_id": config["id"], "stage": config["stage"], "name": config["name"],
            "width": config["width"], "height": config["height"],
            "grid": list(config["grid"]), "exit": list(config["exit"]),
            "position": list(config["start"]), "courage": config["initial_courage"],
            "protection": 0, "carried": 0, "remaining_ticks": self._initial_ticks,
            "initial_ticks": self._initial_ticks, "tick": 0, "target": config["target_value"],
            "mode": self.mode, "terminal": False, "outcome": None,
            "damage_count": 0,
            "quest": {**deepcopy(config["quest"]), "collected": False},
            "items": [{"id": item["id"], "position": list(item["position"]),
                       "value": item["value"], "active": True} for item in config["items"]],
            "ghosts": ghosts,
            "magnet": ({**deepcopy(config["magnet"]), "collected": False,
                        "remaining_ticks": 0} if "magnet" in config else None),
            "invincibility": ({**deepcopy(config["invincibility"]), "collected": False,
                               "remaining_ticks": 0} if "invincibility" in config else None),
            "shortcuts": [{"id": shortcut["id"], "terminal": list(shortcut["terminal"]),
                           "gate": list(shortcut["gate"]), "opened": False}
                          for shortcut in config["shortcuts"]],
        }
        self._state["chaser"] = None
        chaser_config = config.get("chaser")

        if chaser_config is not None:
            self._state["chaser"] = {
                "id": chaser_config["id"],
                "position": list(chaser_config["spawn_position"]),
                "active": False,
                "countdown": chaser_config["initial_countdown"],
                "ticks_per_move": chaser_config["ticks_per_move"],
            }
        self._actions = []
        self._initial_phase = initial_phase
        self._damage_count = 0
        return self.snapshot()

    def snapshot(self) -> dict:
        return deepcopy(self._state)

    def _movement_target(self, action: int) -> tuple[list[int], str | None]:
        state = self._state
        dx, dy = ((0, -1), (0, 1), (-1, 0), (1, 0))[int(action)]
        x, y = state["position"][0] + dx, state["position"][1] + dy
        destination = [x, y]
        if not (0 <= x < state["width"] and 0 <= y < state["height"]):
            return destination, "bounds"
        if state["grid"][y][x] == "#":
            return destination, "wall"
        if any(not shortcut["opened"] and shortcut["gate"] == destination
               for shortcut in state["shortcuts"]):
            return destination, "gate"
        return destination, None

    def can_move(self, action: int) -> bool:
        """시간을 진행하지 않고 지형만 검사한다. 유령은 이동 가능한 칸을 막지 않는다."""
        if (not isinstance(action, Integral) or isinstance(action, bool)
                or not 0 <= action <= 3):
            raise ValueError("movement action must be an integer 0..3")
        return self._movement_target(action)[1] is None

    def _ensure_active(self) -> None:
        if self._state["terminal"]:
            raise RuntimeError("Episode has ended; call reset() before stepping again")

    def _collect_warmth(self, events: list[dict], *, magnetic: bool) -> None:
        """자석은 1점 온기만 끌어오며, 3점 온기는 직접 밟아야 획득한다."""
        state = self._state
        for item in state["items"]:
            magnetic_pickup = magnetic and item["value"] == 1
            radius = 1 if magnetic_pickup else 0
            if (item["active"] and all(
                    abs(item["position"][axis] - state["position"][axis]) <= radius
                    for axis in (0, 1))):
                item["active"] = False
                state["carried"] += item["value"]
                events.append({
                    "type": "collect", "item_id": item["id"],
                    "value": item["value"], "carried": state["carried"],
                    "position": list(state["position"]),
                    "source_position": list(item["position"]),
                    "magnetic": magnetic_pickup,
                })

    def _next_chaser_position(self, origin: list[int]) -> list[int]:
        """벽과 닫힌 문을 피해 고양이에게 가는 최단 경로의 첫 칸을 찾는다."""
        state = self._state
        start = tuple(origin)
        target = tuple(state["position"])

        blocked = {
            tuple(shortcut["gate"])
            for shortcut in state["shortcuts"]
            if not shortcut["opened"]
        }

        queue = deque([start])

        first_steps = {start: start}

        while queue:
            current = queue.popleft()

            if current == target:
                return list(first_steps[current])

            x, y = current

            # 최단 경로 길이가 같으면 이 탐색 순서로 선택해 재생을 결정적으로 유지한다.
            neighbors = (
                (x, y - 1),
                (x, y + 1),
                (x - 1, y),
                (x + 1, y),
            )

            for point in neighbors:
                nx, ny = point

                if point in first_steps:
                    continue

                if not (
                    0 <= nx < state["width"]
                    and 0 <= ny < state["height"]
                ):
                    continue

                if state["grid"][ny][nx] != "." or point in blocked:
                    continue

                first_steps[point] = (
                    point if current == start else first_steps[current]
                )
                queue.append(point)

        return list(start)

    def _update_chaser(self, events: list[dict]) -> None:
        """필수 물건을 획득하면 추격형을 활성화하고 정해진 주기로 이동시킨다."""
        state = self._state
        chaser = state["chaser"]

        if chaser is None or not state["quest"]["collected"]:
            return

        if not chaser["active"]:
            chaser["active"] = True
            events.append({
                "type": "chaser_spawn",
                "ghost_id": chaser["id"],
                "position": list(chaser["position"]),
                "countdown": chaser["countdown"],
            })

            # 추격형은 등장한 틱에 대기 시간을 줄이거나 이동하지 않는다.
            return

        chaser["countdown"] -= 1

        if chaser["countdown"] == 0:
            old_position = list(chaser["position"])
            chaser["position"] = self._next_chaser_position(old_position)
            chaser["countdown"] = chaser["ticks_per_move"]

            if chaser["position"] != old_position:
                events.append({
                    "type": "chaser_move",
                    "ghost_id": chaser["id"],
                    "from": old_position,
                    "to": list(chaser["position"]),
                })

    # 충돌 검사 함수
    def _ghost_contacts(self, previous_position, previous_ghosts, previous_chaser):
        state = self._state
        contacts = []

        for old_ghost, ghost in zip(previous_ghosts, state["ghosts"]):
            final_overlap = state["position"] == ghost["position"]
            swap = (
                state["position"] == old_ghost
                and previous_position == ghost["position"]
            )

            if final_overlap or swap:
                contacts.append({
                    "ghost_id": ghost["id"],
                    "kind": "overlap" if final_overlap else "swap",
                })

        chaser = state["chaser"]

        if chaser is not None and chaser["active"]:
            final_overlap = state["position"] == chaser["position"]
            swap = (
                previous_chaser is not None
                and state["position"] == previous_chaser
                and previous_position == chaser["position"]
            )

            if final_overlap or swap:
                contacts.append({
                    "ghost_id": chaser["id"],
                    "kind": "overlap" if final_overlap else "swap",
                })

                if previous_chaser != previous_position:
                    chaser["countdown"] = 4

        return contacts
    
    def step(self, action: int) -> StepResult:
        return self._transition(action)

    def _transition(self, action: int, *, previous_position=None,
                    advance_world=True, record_action=True) -> StepResult:
        """격자 행동과 연속 이동이 공유하는 규칙. 세계 진행 여부와 재생 행동 기록은 호출자가 지정한다."""
        self._ensure_active()
        if not isinstance(action, Integral) or isinstance(action, bool) or not 0 <= action <= 5:
            raise ValueError("action must be an integer 0..5 from Action")
        chosen = Action(action)
        if record_action:
            self._actions.append(int(chosen))
        state = self._state
        events: list[dict] = []
        previous_position = list(state["position"] if previous_position is None
                                 else previous_position)
        previous_ghosts = [list(ghost["position"]) for ghost in state["ghosts"]]
        chaser = state["chaser"]
        previous_chaser = (
            list(chaser["position"])
            if chaser is not None and chaser["active"]
            else None
        )
        previous_protection = state["protection"]
        requested_return = False

        if chosen <= Action.RIGHT:
            destination, reason = self._movement_target(chosen)
            if reason is not None:
                events.append({"type": "blocked", "from": previous_position,
                               "to": destination, "reason": reason})
            else:
                state["position"] = destination
                events.append({
                    "type": "move",
                    "from": previous_position,
                    "to": list(destination),
                })

        # 격자 이동과 실시간 이동 모두 현재 칸에 도착하면 E 없이 처리한다.
        if state['position'] == state['quest']['position'] and not state['quest']['collected']:
            state['quest']['collected'] = True
            events.append({'type':'quest_collect', 'name':state['quest']['name']})
        for shortcut in state["shortcuts"]:
            if not shortcut["opened"] and shortcut["terminal"] == state["position"]:
                shortcut["opened"] = True
                events.append({"type": "open", "shortcut_id": shortcut["id"],
                               "gate": list(shortcut["gate"]),
                               "position": list(state["position"])})
                break

        invincibility = state["invincibility"]
        if (invincibility is not None and not invincibility["collected"]
                and state["position"] == invincibility["position"]):
            invincibility["collected"] = True
            invincibility["remaining_ticks"] = invincibility["duration_ticks"]
            events.append({"type": "invincibility_collect",
                           "position": list(invincibility["position"]),
                           "duration_ticks": invincibility["duration_ticks"]})
        item_invincible = invincibility is not None and invincibility["remaining_ticks"] > 0

        magnet = state["magnet"]
        if magnet is not None and not magnet["collected"] and state["position"] == magnet["position"]:
            magnet["collected"] = True
            magnet["remaining_ticks"] = magnet["duration_ticks"]
            events.append({"type": "magnet_collect", "position": list(magnet["position"]),
                           "duration_ticks": magnet["duration_ticks"]})
        magnetic = magnet is not None and magnet["remaining_ticks"] > 0
        if magnetic or state["position"] != previous_position:
            self._collect_warmth(events, magnetic=magnetic)
        # 같은 이동에서 획득한 온기도 귀환 조건에 반영한다. 최종 피해 판정은 아래에서 유지한다.
        if state["position"] == state["exit"]:
            requested_return = state['carried'] >= state['target'] and state['quest']['collected']
            if chosen == Action.INTERACT and not requested_return:
                events.append({'type':'return_blocked', 'missing_quest':not state['quest']['collected'], 'missing_heat':max(0,state['target']-state['carried'])})
        # 자석은 획득 틱을 포함해 13틱 지속하며 대기·막힌 이동도 세계 진행 시 한 틱이다.
        if magnetic and advance_world:
            magnet["remaining_ticks"] -= 1
            if magnet["remaining_ticks"] == 0:
                events.append({"type": "magnet_end"})

        # 유령의 이전 위치를 모두 보관한 뒤 이동 결과와 비교해 동시 충돌을 판정한다.
        for ghost in state["ghosts"] if advance_world else ():
            ghost["countdown"] -= 1
            if ghost["countdown"] == 0:
                ghost["route_index"] = (ghost["route_index"] + ghost["direction"]) % len(ghost["route"])
                ghost["countdown"] = ghost["ticks_per_move"]
            ghost["position"] = list(ghost["route"][ghost["route_index"]])
            ghost["next_position"] = list(ghost["route"][(ghost["route_index"] + ghost["direction"]) % len(ghost["route"])])
        
        if advance_world:
            self._update_chaser(events)
            
        contacts = self._ghost_contacts(
            previous_position,
            previous_ghosts,
            previous_chaser,
        )
        
        if contacts:
            events.append({"type": "contact", "ghost_ids": [contact["ghost_id"] for contact in contacts],
                           "contacts": contacts, "protected": previous_protection > 0 or item_invincible,
                           "item_invincible": item_invincible,
                           "position": list(state["position"])})
        if contacts and previous_protection == 0 and not item_invincible:
            state["courage"] = max(0, state["courage"] - 1)
            state["protection"] = self._config["protection_future_ticks"]
            self._damage_count += 1
            state["damage_count"] = self._damage_count
            events.append({"type": "damage", "amount": 1, "courage": state["courage"],
                           "protection": state["protection"], "damage_count": self._damage_count})
        elif advance_world:
            state["protection"] = max(0, previous_protection - 1)

        # 아이템 무적은 획득·마지막 틱에도 피해를 막으며 피격 후 보호 시간을 부여하거나 연장하지 않는다.
        if item_invincible and advance_world:
            invincibility["remaining_ticks"] -= 1
            if invincibility["remaining_ticks"] == 0:
                events.append({"type": "invincibility_end"})

        state["tick"] += int(advance_world)
        state["remaining_ticks"] -= int(advance_world)
        if state["courage"] == 0:
            self._finish("FAILED", "courage", events)
        elif requested_return:
            status = "COMPLETE" if state["carried"] >= state["target"] else "RETURN_LOW"
            events.append({"type": "return", "banked": state["carried"], "status": status})
            self._finish(status, "returned", events)
        elif state["remaining_ticks"] == 0:
            self._finish("FAILED", "timeout", events)
        return self._result(events)

    def _finish(self, status: str, reason: str, events: list[dict]) -> None:
        state = self._state
        state["terminal"] = True
        state["outcome"] = {
            "status": status, "reason": reason,
            "banked": state["carried"] if status in ("COMPLETE", "RETURN_LOW") else 0,
            "carried": state["carried"], "damage_count": self._damage_count,
            "remaining_courage": state["courage"],
            "quest_collected": state["quest"]["collected"],
            "ticks_used": state["tick"], "remaining_ticks": state["remaining_ticks"],
        }
        state["outcome"]["score"] = score_outcome(self._config, state["outcome"])
        events.append({"type": "end", **deepcopy(state["outcome"])})

    def _result(self, events: list[dict]) -> StepResult:
        return StepResult(self.snapshot(), deepcopy(events), self._state["terminal"],
                          deepcopy(self._state["outcome"]))

    def abort(self, reason: str = "user_abort") -> StepResult:
        """시간·점수·행동 기록을 추가하지 않고 탐사를 중단한다."""
        self._ensure_active()
        if not isinstance(reason, str) or not reason:
            raise ValueError("abort reason must be a nonempty string")
        events: list[dict] = []
        self._finish("ABORTED", reason, events)
        return self._result(events)

    def replay_record(self) -> dict:
        """같은 탐사를 재현할 설정·위상·행동 정보의 복사본을 반환한다."""
        return {
            "replay_version": "1.0",
            "rules_version": self._config["rules_version"],
            "observation_version": OBSERVATION_VERSION,
            "stage_id": self._config["id"], "stage": self._config["stage"],
            "config_hash": self.config_hash, "config_hash_algorithm": "sha256-canonical-json-utf8",
            "mode": self.mode, "seed": self.seed, "randomize_phase": self.randomize_phase,
            "initial_phase": deepcopy(self._initial_phase),
            "initial_ticks": self._initial_ticks, "actions": self.actions,
            "terminal": self._state["terminal"], "outcome": deepcopy(self._state["outcome"]),
        }
