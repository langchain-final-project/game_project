"""Load and validate authored levels without importing the UI or learning stack.

Validation here checks the data contract and static reachability. It deliberately
does not claim a time-feasible, contact-free solution: the recorded
route replay suite verifies those properties for the supplied 12 levels.
"""
from __future__ import annotations

from collections import deque
import json
from numbers import Integral
from pathlib import Path
import re


WIDTH = 16
HEIGHT = 9
MAX_TICKS = 512
RULES_VERSION = "cat_mansion_0_2"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Invalid level: {message}")


def _integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def validate_level(config: dict) -> None:
    """Reject ambiguous, malformed, or unsupported level configurations.

    IDs and slot order are stable because the 602-value observation contract
    relies on them. Modified maps may use fewer patrols or shortcuts; their
    number is not implicitly inferred from the stage number.
    """
    _require(isinstance(config, dict), "configuration must be a dictionary")
    required = (
        "schema_version", "rules_version", "id", "stage", "name", "width",
        "height", "grid", "start", "exit", "items", "ghosts", "shortcuts",
        "quest", "target_value", "total_item_value", "initial_courage",
        "protection_future_ticks", "tick_seconds", "time_budget_ticks",
    )
    missing = [key for key in required if key not in config]
    _require(not missing, f"missing fields: {', '.join(missing)}")
    _require(config["schema_version"] == "1.0", "unsupported schema_version")
    _require(config["rules_version"] == RULES_VERSION, "unsupported rules_version")
    stage = config["stage"]
    _require(_integer(stage) and 1 <= stage <= 12, "stage must be 1..12")
    _require(config["id"] == f"stage_{stage:02d}", "id does not match stage")
    _require(isinstance(config["name"], str) and bool(config["name"].strip()), "name must be nonempty")
    for key, expected in (
        ("width", WIDTH), ("height", HEIGHT), ("target_value", 4),
        ("total_item_value", 14), ("initial_courage", 3),
        ("protection_future_ticks", 3),
    ):
        _require(_integer(config[key]) and config[key] == expected, f"{key} must be {expected}")
    _require(not isinstance(config["tick_seconds"], bool) and config["tick_seconds"] == 0.3,
             "tick_seconds must be 0.3")
    budget = config["time_budget_ticks"]
    _require(_integer(budget) and 1 <= budget <= MAX_TICKS,
             f"time_budget_ticks must be an integer in 1..{MAX_TICKS}")

    grid = config["grid"]
    _require(isinstance(grid, list) and len(grid) == HEIGHT, "grid must have 9 rows")
    _require(all(isinstance(row, str) and len(row) == WIDTH and set(row) <= {"#", "."}
                 for row in grid), "grid rows must contain 16 #/. characters")
    floor = {(x, y) for y, row in enumerate(grid) for x, tile in enumerate(row) if tile == "."}

    def position(value: object, label: str) -> tuple[int, int]:
        _require(isinstance(value, (list, tuple)) and len(value) == 2,
                 f"{label} must contain [x, y]")
        _require(all(_integer(v) for v in value), f"{label} coordinates must be integers")
        x, y = value
        _require(0 <= x < WIDTH and 0 <= y < HEIGHT, f"{label} lies outside the map")
        point = (int(x), int(y))
        _require(point in floor, f"{label} must be on a floor tile")
        return point

    start = position(config["start"], "start")
    exit_position = position(config["exit"], "exit")
    _require(start == exit_position, "this rules version requires start == exit")
    items = config["items"]
    ghosts = config["ghosts"]
    shortcuts = config["shortcuts"]
    _require(isinstance(items, list) and len(items) == 10, "exactly 10 item slots are required")
    _require(isinstance(ghosts, list) and len(ghosts) <= 2, "at most 2 ghosts are allowed")
    _require(isinstance(shortcuts, list) and len(shortcuts) <= 2, "at most 2 shortcuts are allowed")
    occupied = {exit_position}
    item_positions: list[tuple[int, int]] = []
    values: list[int] = []
    for index, item in enumerate(items):
        _require(isinstance(item, dict) and {"id", "position", "value"} <= item.keys(),
                 f"item slot {index} is incomplete")
        _require(item["id"] == f"heat_{index:02d}", "item IDs/order must be heat_00 through heat_09")
        _require(_integer(item["value"]) and item["value"] in (1, 3), f"{item['id']} value must be 1 or 3")
        point = position(item["position"], item["id"])
        _require(point not in occupied, f"{item['id']} overlaps an item or exit")
        occupied.add(point)
        item_positions.append(point)
        values.append(int(item["value"]))
    _require(sorted(values) == [1] * 8 + [3] * 2, "items must be eight +1 and two +3")
    _require(sum(values) == config["total_item_value"], "total_item_value does not match items")

    gate_positions: set[tuple[int, int]] = set()
    terminal_positions: list[tuple[int, int]] = []
    for index, shortcut in enumerate(shortcuts):
        _require(isinstance(shortcut, dict) and {"id", "terminal", "gate", "initially_open"} <= shortcut.keys(),
                 f"shortcut slot {index} is incomplete")
        _require(shortcut["id"] == ("A", "B")[index], "shortcut IDs/order must be A then B")
        _require(shortcut["initially_open"] is False, "shortcuts must initially be closed")
        terminal = position(shortcut["terminal"], f"shortcut {shortcut['id']} terminal")
        gate = position(shortcut["gate"], f"shortcut {shortcut['id']} gate")
        _require(terminal not in occupied and terminal not in gate_positions,
                 f"shortcut {shortcut['id']} terminal overlaps another object")
        _require(gate != terminal and gate not in occupied and gate not in gate_positions,
                 f"shortcut {shortcut['id']} gate overlaps another object")
        occupied.add(terminal)
        gate_positions.add(gate)
        terminal_positions.append(terminal)

    quest = config['quest']
    _require(isinstance(quest, dict) and {'id','name','kind','position','restoration'} <= quest.keys(), 'quest is incomplete')
    _require(all(isinstance(quest[k],str) and quest[k].strip() for k in ('id','name','kind','restoration')), 'quest labels must be nonempty')
    quest_point = position(quest['position'], 'quest')
    _require(quest_point not in occupied | gate_positions, 'quest overlaps another object')
    for index, ghost in enumerate(ghosts):
        fields = {"id", "route", "initial_index", "direction", "initial_countdown", "ticks_per_move"}
        _require(isinstance(ghost, dict) and fields <= ghost.keys(), f"ghost slot {index} is incomplete")
        _require(ghost["id"] == ("A", "B")[index], "ghost IDs/order must be A then B")
        route = ghost["route"]
        _require(isinstance(route, list) and 2 <= len(route) <= 16,
                 f"ghost {ghost['id']} route must contain 2..16 tiles")
        points = [position(point, f"ghost {ghost['id']} route") for point in route]
        # 직선 왕복을 표현할 수 있도록 중복 좌표 허용
        # _require(len(set(points)) == len(points), f"ghost {ghost['id']} route repeats a tile")
        _require(not set(points) & gate_positions and exit_position not in points,
                 f"ghost {ghost['id']} route crosses a gate or exit")
        _require(all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
                     for a, b in zip(points, points[1:] + points[:1])),
                 f"ghost {ghost['id']} route must be an adjacent closed loop")
        _require(_integer(ghost["initial_index"]) and 0 <= ghost["initial_index"] < len(points),
                 f"ghost {ghost['id']} initial_index is outside its route")
        _require(_integer(ghost["direction"]) and ghost["direction"] in (-1, 1),
                 f"ghost {ghost['id']} direction must be -1 or 1")
        # 첫 이동 대기 시간과 이후 이동 간격을 각각 설정할 수 있습니다.
        for key in ("initial_countdown", "ticks_per_move"):
            value = ghost[key]
            _require(
                _integer(value) and 1 <= value <= MAX_TICKS,
                f"ghost {ghost['id']} {key} must be an integer in 1..{MAX_TICKS}",
            )

    def reachable(blocked: set[tuple[int, int]]) -> set[tuple[int, int]]:
        seen = {start}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for point in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)):
                if point in floor and point not in blocked and point not in seen:
                    seen.add(point)
                    queue.append(point)
        return seen

    closed_reachable = reachable(gate_positions)
    closed_value = sum(value for point, value in zip(item_positions, values) if point in closed_reachable)
    _require(closed_value >= config["target_value"], "closed gates leave fewer than 4 statically reachable points")
    _require(quest_point in closed_reachable, 'mandatory quest must be reachable without a shortcut')
    _require(all(p in closed_reachable for p in terminal_positions), 'shortcut controls must be accessible before opening gates')
    all_open_reachable = reachable(set())
    _require(all(point in all_open_reachable for point in item_positions + terminal_positions),
             "an item or terminal is disconnected even with all gates open")
    
    # ── 추격형 유령 설정 검증 ──
    # 설정이 없는 스테이지에서는 이 블록을 실행하지 않습니다.
    if "chaser" in config:
        chaser = config["chaser"]

        _require(
            isinstance(chaser, dict),
            "chaser must be an object",
        )

        # 스키마에 정의한 네 항목이 모두 있어야 합니다.
        fields = {
            "id",
            "spawn_position",
            "initial_countdown",
            "ticks_per_move",
        }
        _require(
            fields <= chaser.keys(),
            "chaser is missing required fields",
        )
        _require(
            chaser.keys() <= fields,
            "chaser contains unknown fields",
        )

        _require(chaser["id"] == "C", "chaser id must be C")

        # bool, 소수, 0, 음수와 상한을 넘는 값은 허용하지 않습니다.
        for key in ("initial_countdown", "ticks_per_move"):
            value = chaser[key]
            _require(
                _integer(value) and 1 <= value <= MAX_TICKS,
                f"chaser {key} must be an integer in 1..{MAX_TICKS}",
            )

        # 기존 position()으로 좌표 형식·범위·바닥 여부를 검사합니다.
        spawn = position(chaser["spawn_position"], "chaser spawn")

        # 기존 물건·장치·귀환점, 샛문, 필수 물건과 겹치지 않게 합니다.
        _require(
            spawn not in occupied | gate_positions | {quest_point},
            "chaser spawn overlaps an object, gate, or quest",
        )

        # 샛문이 닫혀 있어도 고양이와 같은 연결 영역이어야 합니다.
        _require(
            spawn in closed_reachable,
            "chaser spawn must be reachable with shortcuts closed",
        )

def _level_root(root: Path | None) -> Path:
    if root is not None:
        return Path(root)
    from .paths import DATA_DIR
    return DATA_DIR / "levels"


def load_level(stage: int | str, root: Path | None = None) -> dict:
    """Load one stage by integer, '01', or 'stage_01'; paths are not accepted."""
    if _integer(stage):
        number = int(stage)
    elif isinstance(stage, str) and re.fullmatch(r"(?:stage_)?(?:0?[1-9]|1[0-2])", stage):
        number = int(stage.removeprefix("stage_"))
    else:
        raise ValueError("stage must be an integer 1..12 or an ID such as 'stage_01'")
    if not 1 <= number <= 12:
        raise ValueError("stage must be in 1..12")
    path = _level_root(root) / f"stage_{number:02d}.json"
    with path.open("r", encoding="utf-8") as file:
        level = json.load(file)
    validate_level(level)
    if level["stage"] != number:
        raise ValueError(f"Stage ID mismatch in {path.name}")
    return level


def load_levels(root: Path | None = None) -> list[dict]:
    """Load the complete 12-stage campaign in numerical order."""
    folder = _level_root(root)
    return [load_level(stage, folder) for stage in range(1, 13)]
