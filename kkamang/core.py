"""Deterministic, rendering-independent game state and six-action transition.

No wall clock, UI, input device, filesystem, campaign save, or ML dependency is
consulted by a transition. One accepted action always consumes exactly one tick.
"""
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
    """One exploration episode; campaign progression is deliberately separate."""

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
        """Copy-on-read prevents callers from changing the active rules."""
        return deepcopy(self._config)

    @property
    def actions(self) -> list[int]:
        return list(self._actions)

    def reset(self, seed: int | None = None, phase_overrides: dict | None = None) -> dict:
        """Start a fresh episode, optionally choosing explicit patrol phases.

        With randomize_phase=True, a supplied seed reinitializes the local RNG;
        seed=None continues it. Replay records capture the actual chosen phase,
        so even unseeded consecutive resets can be reproduced exactly.
        """
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
            "shortcuts": [{"id": shortcut["id"], "terminal": list(shortcut["terminal"]),
                           "gate": list(shortcut["gate"]), "opened": False}
                          for shortcut in config["shortcuts"]],
        }
        # 추격형 설정이 없는 스테이지에서는 None입니다.
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

    def _ensure_active(self) -> None:
        if self._state["terminal"]:
            raise RuntimeError("Episode has ended; call reset() before stepping again")

    def _next_chaser_position(self, origin: list[int]) -> list[int]:
        """벽과 닫힌 문을 피해 고양이에게 가는 최단 경로의 첫 칸을 찾습니다."""
        state = self._state
        start = tuple(origin)
        target = tuple(state["position"])

        blocked = {
            tuple(shortcut["gate"])
            for shortcut in state["shortcuts"]
            if not shortcut["opened"]
        }

        queue = deque([start])

        # 각 칸에 도달하려면 출발점에서 첫 번째로 어디로 가야 하는지 기록합니다.
        first_steps = {start: start}

        while queue:
            current = queue.popleft()

            if current == target:
                return list(first_steps[current])

            x, y = current

            # 같은 길이의 경로가 여러 개라면 이 순서로 선택합니다.
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

        # 갈 수 있는 경로가 없으면 현재 칸에 머뭅니다.
        return list(start)

    def _update_chaser(self, events: list[dict]) -> None:
        """필수 물건 획득 후 추격형을 활성화하고 이동시킵니다."""
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

            # 등장한 틱에는 대기 시간을 줄이거나 움직이지 않습니다.
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
    
    def step(self, action: int) -> StepResult:
        self._ensure_active()
        if not isinstance(action, Integral) or isinstance(action, bool) or not 0 <= action <= 5:
            raise ValueError("action must be an integer 0..5 from Action")
        chosen = Action(action)
        self._actions.append(int(chosen))
        state = self._state
        events: list[dict] = []
        previous_position = list(state["position"])
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
            dx, dy = ((0, -1), (0, 1), (-1, 0), (1, 0))[int(chosen)]
            x, y = state["position"][0] + dx, state["position"][1] + dy
            destination = [x, y]
            reason = None
            if not (0 <= x < state["width"] and 0 <= y < state["height"]):
                reason = "bounds"
            elif state["grid"][y][x] == "#":
                reason = "wall"
            elif any(not shortcut["opened"] and shortcut["gate"] == destination
                     for shortcut in state["shortcuts"]):
                reason = "gate"
            if reason is not None:
                events.append({"type": "blocked", "from": previous_position,
                               "to": destination, "reason": reason})
            else:
                # 이동 가능한 칸이면 플레이어 위치를 갱신합니다.
                state["position"] = destination
                events.append({
                    "type": "move",
                    "from": previous_position,
                    "to": list(destination),
                })

                # 이동한 칸에 아직 획득하지 않은 온기가 있으면 수집합니다.
                for item in state["items"]:
                    if (
                        item["active"]
                        and item["position"] == state["position"]
                    ):
                        # 획득한 온기는 다시 수집되지 않도록 비활성화합니다.
                        item["active"] = False
                        state["carried"] += item["value"]

                        # 기존 이벤트를 유지하여 소리와 알림도 연결합니다.
                        events.append({
                            "type": "collect",
                            "item_id": item["id"],
                            "value": item["value"],
                            "carried": state["carried"],
                            "position": list(state["position"]),
                        })
                        break
                    
        elif chosen == Action.INTERACT:
            if state["position"] == state["exit"]:
                if state['carried'] >= state['target'] and state['quest']['collected']:
                    requested_return = True
                else:
                    events.append({'type':'return_blocked', 'missing_quest':not state['quest']['collected'], 'missing_heat':max(0,state['target']-state['carried'])})
            else:
                if state['position'] == state['quest']['position'] and not state['quest']['collected']:
                    state['quest']['collected'] = True
                    events.append({'type':'quest_collect', 'name':state['quest']['name']})
                # for item in state["items"]:
                #     if item["active"] and item["position"] == state["position"]:
                #         item["active"] = False
                #         state["carried"] += item["value"]
                #         events.append({"type": "collect", "item_id": item["id"],
                #                        "value": item["value"], "carried": state["carried"],
                #                        "position": list(state["position"])})
                #         break
                for shortcut in state["shortcuts"]:
                    if not shortcut["opened"] and shortcut["terminal"] == state["position"]:
                        shortcut["opened"] = True
                        events.append({"type": "open", "shortcut_id": shortcut["id"],
                                       "gate": list(shortcut["gate"]),
                                       "position": list(state["position"])})
                        break

        # Patrols are independent of one another; old positions above are kept
        # until all new positions exist, so contact is simultaneous and stable.
        for ghost in state["ghosts"]:
            ghost["countdown"] -= 1
            if ghost["countdown"] == 0:
                ghost["route_index"] = (ghost["route_index"] + ghost["direction"]) % len(ghost["route"])
                ghost["countdown"] = ghost["ticks_per_move"]
            ghost["position"] = list(ghost["route"][ghost["route_index"]])
            ghost["next_position"] = list(ghost["route"][(ghost["route_index"] + ghost["direction"]) % len(ghost["route"])])
        self._update_chaser(events)
        contacts = []
        for old_ghost, ghost in zip(previous_ghosts, state["ghosts"]):
            final_overlap = state["position"] == ghost["position"]
            swap = state["position"] == old_ghost and previous_position == ghost["position"]
            if final_overlap or swap:
                contacts.append({"ghost_id": ghost["id"], "kind": "overlap" if final_overlap else "swap"})
        
        # 추격형도 순찰형과 같은 충돌 기준을 사용합니다.
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
        
        if contacts:
            events.append({"type": "contact", "ghost_ids": [contact["ghost_id"] for contact in contacts],
                           "contacts": contacts, "protected": previous_protection > 0,
                           "position": list(state["position"])})
        if contacts and previous_protection == 0:
            state["courage"] = max(0, state["courage"] - 1)
            state["protection"] = self._config["protection_future_ticks"]
            self._damage_count += 1
            state["damage_count"] = self._damage_count
            events.append({"type": "damage", "amount": 1, "courage": state["courage"],
                           "protection": state["protection"], "damage_count": self._damage_count})
        else:
            state["protection"] = max(0, previous_protection - 1)

        state["tick"] += 1
        state["remaining_ticks"] -= 1
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
        """Cancel without advancing time, awarding points, or adding an action."""
        self._ensure_active()
        if not isinstance(reason, str) or not reason:
            raise ValueError("abort reason must be a nonempty string")
        events: list[dict] = []
        self._finish("ABORTED", reason, events)
        return self._result(events)

    def replay_record(self) -> dict:
        """A copy of the minimum provenance needed to reproduce this episode."""
        return {
            "replay_version": "1.0",
            "rules_version": self._config["rules_version"],
            "observation_version": "vector_v3_chaser",
            "stage_id": self._config["id"], "stage": self._config["stage"],
            "config_hash": self.config_hash, "config_hash_algorithm": "sha256-canonical-json-utf8",
            "mode": self.mode, "seed": self.seed, "randomize_phase": self.randomize_phase,
            "initial_phase": deepcopy(self._initial_phase),
            "initial_ticks": self._initial_ticks, "actions": self.actions,
            "terminal": self._state["terminal"], "outcome": deepcopy(self._state["outcome"]),
        }
