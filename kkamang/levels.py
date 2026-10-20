"""12개 레벨을 읽고 스키마·배치·도달 가능성을 검사한다. 제한 시간 내 성공 여부는 경로 재생으로 따로 검증한다."""
from __future__ import annotations

from collections import deque
import json
from numbers import Integral
from pathlib import Path
import re


WIDTH = 16
HEIGHT = 9
MAX_TICKS = 512
MAGNET_DURATION_TICKS = 13
INVINCIBILITY_DURATION_TICKS = 10
RULES_VERSION = "cat_mansion_0_2"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Invalid level: {message}")


def _integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def validate_level(config: dict) -> None:
    """지원하는 레벨 계약을 검사한다. 관측 슬롯의 식별자·순서는 고정이며 유령·샛문 수는 방 번호로 추측하지 않는다."""
    _require(isinstance(config, dict), "configuration must be a dictionary")
    required = (
        "schema_version", "rules_version", "id", "stage", "name", "width",
        "height", "grid", "start", "exit", "items", "ghosts", "shortcuts",
        "quest", "target_value", "total_item_value", "initial_courage",
        "protection_future_ticks", "tick_seconds", "time_budget_ticks",
        "star_warmth_cap", "star_fast_ticks",
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
    warmth_cap = config["star_warmth_cap"]
    _require(_integer(warmth_cap) and config['target_value'] < warmth_cap <= config['total_item_value'],
             "star_warmth_cap must be above the return target and within available warmth")
    fast_ticks = config["star_fast_ticks"]
    _require(_integer(fast_ticks) and 0 <= fast_ticks < budget,
             "star_fast_ticks must be from zero up to the time limit")

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
        # 직선 왕복 경로를 표현할 수 있도록 좌표 중복을 허용한다.
        _require(not set(points) & gate_positions and exit_position not in points,
                 f"ghost {ghost['id']} route crosses a gate or exit")
        _require(all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
                     for a, b in zip(points, points[1:] + points[:1])),
                 f"ghost {ghost['id']} route must be an adjacent closed loop")
        _require(_integer(ghost["initial_index"]) and 0 <= ghost["initial_index"] < len(points),
                 f"ghost {ghost['id']} initial_index is outside its route")
        _require(_integer(ghost["direction"]) and ghost["direction"] in (-1, 1),
                 f"ghost {ghost['id']} direction must be -1 or 1")
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
    
    if "chaser" in config:
        chaser = config["chaser"]

        _require(
            isinstance(chaser, dict),
            "chaser must be an object",
        )

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

        for key in ("initial_countdown", "ticks_per_move"):
            value = chaser[key]
            _require(
                _integer(value) and 1 <= value <= MAX_TICKS,
                f"chaser {key} must be an integer in 1..{MAX_TICKS}",
            )

        spawn = position(chaser["spawn_position"], "chaser spawn")

        _require(
            spawn not in occupied | gate_positions | {quest_point},
            "chaser spawn overlaps an object, gate, or quest",
        )

        # 추격형의 출현점은 샛문이 닫혀도 고양이와 같은 연결 영역이어야 한다.
        _require(
            spawn in closed_reachable,
            "chaser spawn must be reachable with shortcuts closed",
        )

    # 자석·무적은 온기 10개와 별개이며 각 종류는 스테이지당 최대 하나다.
    powerup_positions = set()
    for name, duration in (("magnet", MAGNET_DURATION_TICKS),
                           ("invincibility", INVINCIBILITY_DURATION_TICKS)):
        if name not in config:
            continue
        item = config[name]
        _require(isinstance(item, dict), f"{name} must be an object")
        _require(set(item) == {"position", "duration_ticks"},
                 f"{name} requires exactly position and duration_ticks")
        _require(_integer(item["duration_ticks"]) and item["duration_ticks"] == duration,
                 f"{name} duration_ticks must be {duration}")
        point = position(item["position"], name)
        blocked_objects = occupied | gate_positions | {quest_point}
        blocked_objects.update(tuple(p) for ghost in ghosts for p in ghost["route"])
        if "chaser" in config:
            blocked_objects.add(tuple(config["chaser"]["spawn_position"]))
        _require(point not in blocked_objects | powerup_positions,
                 f"{name} overlaps an object or ghost route")
        _require(point in closed_reachable, f"{name} must be reachable with shortcuts closed")
        powerup_positions.add(point)

def _level_root(root: Path | None) -> Path:
    if root is not None:
        return Path(root)
    from .paths import DATA_DIR
    return DATA_DIR / "levels"


def load_level(stage: int | str, root: Path | None = None) -> dict:
    """1~12 정수 또는 01·stage_01 형태의 식별자로 읽는다. 파일 경로는 받지 않는다."""
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
    folder = _level_root(root)
    return [load_level(stage, folder) for stage in range(1, 13)]
