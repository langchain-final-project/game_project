"""게임의 클리어 조건을 유지하며 추가 온기 수집 목표를 선택합니다."""

import math

from kkamang.realtime import FRAMES_PER_TICK
from rl.ppo_path_distance import path_distance, next_waypoint


def collection_target(snapshot, item, settings):
    """1점 온기의 흡수 범위와 남은 시간을 반영한 도달 가능한 목표입니다."""
    target = list(item["position"])
    direct = path_distance(snapshot, target)
    magnet = snapshot.get("magnet") or {}
    if (snapshot["stage"] != 2 or not settings.get("magnet_pickup_targets", False)
            or item["value"] != 1 or magnet.get("remaining_ticks", 0) <= 0):
        return direct, target
    remaining = magnet["remaining_ticks"] * FRAMES_PER_TICK - snapshot["motion"]["frame"] % FRAMES_PER_TICK
    candidates = []
    x, y = target
    grid = snapshot["grid"]
    for cy in range(max(0, y - 1), min(len(grid), y + 2)):
        for cx in range(max(0, x - 1), min(len(grid[0]), x + 2)):
            if grid[cy][cx] == "#":
                continue
            distance = path_distance(snapshot, [cx, cy])
            # 중심까지의 이동 시간으로 보수적으로 추정해 만료 후 흡수를 기대하지 않습니다.
            if distance is not None and math.ceil(distance * FRAMES_PER_TICK) + 1 < remaining:
                candidates.append((distance, [cx, cy]))
    return min(candidates) if candidates else (direct, target)


def bonus_collection_safe(snapshot, star_goal):
    """추가 수집을 계속할 시간·용기 여유가 있는지 판단합니다."""
    required_ticks = star_goal.get('min_remaining_ticks',0)
    if snapshot['stage']==3 and star_goal.get('stage3_route_plan',False):
        distance = path_distance(snapshot,snapshot['exit'])
        if distance is None:
            return False
        required_ticks = max(required_ticks,math.ceil(distance)+star_goal.get('stage3_return_reserve_ticks',3))
    return (
        star_goal.get("enabled", False)
        and (snapshot["stage"] == 2
             or (snapshot["stage"] == 1 and star_goal.get("stage1_three_star", False))
             or (snapshot["stage"] == 3 and star_goal.get("stage3_three_star", False)))
        and snapshot["remaining_ticks"] >= required_ticks
        and snapshot["courage"] >= star_goal["min_courage"]
    )


def select_goal(snapshot, star_goal):
    if star_goal.get("enabled", False) and (
        (snapshot["stage"] == 1 and star_goal.get("stage1_three_star", False))
        or (snapshot["stage"] == 2 and star_goal.get("stage2_three_star", False))
        or (snapshot["stage"] == 3 and star_goal.get("stage3_three_star", False))
    ):
        return _three_star_goal(snapshot, star_goal)
    x, y = snapshot["motion"]["position"]
    enough_warmth = snapshot["carried"] >= snapshot["target"]
    desired_warmth = snapshot["target"]
    if bonus_collection_safe(snapshot, star_goal):
        desired_warmth = max(desired_warmth, star_goal["warmth_target"])

    if not snapshot["quest"]["collected"]:
        return 0, snapshot["quest"]["position"], enough_warmth
    if snapshot["carried"] < desired_warmth:
        active = [item for item in snapshot["items"] if item["active"]]
        if active:
            nearest = min(active, key=lambda item: abs(item["position"][0] - x) + abs(item["position"][1] - y))
            return 1, nearest["position"], enough_warmth
        if not enough_warmth:
            return 1, snapshot["exit"], False
    return 2, snapshot["exit"], enough_warmth


def _three_star_goal(snapshot, settings):
    enough = snapshot["carried"] >= snapshot["target"]
    safe = bonus_collection_safe(snapshot, settings)
    if snapshot['stage'] == 3 and settings.get('stage3_route_plan',False):
        from rl.ppo_stage3_plan import efficient_goal
        return efficient_goal(snapshot, settings, safe)
    desired = settings["warmth_target"] if safe else snapshot["target"]
    invincible = snapshot.get("invincibility")
    magnet = snapshot.get("magnet")

    if (
        snapshot["stage"] == 2
        and safe
        and magnet
        and not magnet["collected"]
        and snapshot["carried"] >= settings.get("magnet_min_warmth", 2)
        and snapshot["carried"] < desired
    ):
        phase, target = 1, magnet["position"]

    elif (
        snapshot["stage"] == 2
        and settings.get("quest_after_magnet", False)
        and not snapshot["quest"]["collected"]
        and (not magnet or magnet["collected"] or not safe)
    ):
        # 새 설정에서만 자석 획득 후 물건을 우선하며, 과거 모델의 순서는 유지합니다.
        phase, target = 0, snapshot["quest"]["position"]

    elif enough and safe and invincible and not invincible["collected"]:
        phase, target = 1, invincible["position"]
    elif snapshot["carried"] < desired:
        reachable = []
        for item in snapshot["items"]:
            if item["active"]:
                distance, pickup = collection_target(snapshot, item, settings)
                if distance is not None:
                    reachable.append((distance / item["value"], distance, pickup))
        if reachable:
            phase, target = 1, min(reachable)[2]
        else:
            phase, target = (0, snapshot["quest"]["position"]) if not snapshot["quest"]["collected"] else (2, snapshot["exit"])
    elif not snapshot["quest"]["collected"]:
        phase, target = 0, snapshot["quest"]["position"]
    else:
        phase, target = 2, snapshot["exit"]
    if settings.get("path_waypoints", False):
        target = next_waypoint(snapshot, target)
    return phase, target, enough
