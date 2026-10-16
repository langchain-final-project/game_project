"""Deterministic score rules for stars and friendship."""
from __future__ import annotations

from numbers import Integral

WARMTH_POINTS = 60
SPEED_POINTS = 25
COURAGE_POINTS = 15
STAR_THRESHOLDS = (45, 65, 85)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f'{name} must be an integer')
    return int(value)


def _round_ratio(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError('score denominator must be positive')
    return (2 * numerator + denominator) // (2 * denominator)


def score_outcome(level: dict, outcome: dict) -> dict:
    """Score one completed run; unsuccessful runs receive zero points."""
    empty = {'warmth': 0, 'speed': 0, 'courage': 0, 'total': 0,
             'stars': 0, 'next_star_points': STAR_THRESHOLDS[0]}
    if not isinstance(level, dict) or not isinstance(outcome, dict):
        raise ValueError('level and outcome must be dictionaries')
    if outcome.get('status') != 'COMPLETE':
        return empty
    target = _integer(level.get('target_value'), 'target_value')
    warmth_cap = _integer(level.get('star_warmth_cap'), 'star_warmth_cap')
    time_limit = _integer(level.get('time_budget_ticks'), 'time_budget_ticks')
    fast_target = _integer(level.get('star_fast_ticks'), 'star_fast_ticks')
    initial_courage = _integer(level.get('initial_courage'), 'initial_courage')
    if not 0 <= target < warmth_cap or not 0 <= fast_target < time_limit or initial_courage <= 0:
        raise ValueError('invalid authored score targets')
    banked = _integer(outcome.get('banked'), 'banked')
    ticks_used = _integer(outcome.get('ticks_used'), 'ticks_used')
    remaining = outcome.get('remaining_courage')
    if remaining is None:
        remaining = initial_courage - _integer(outcome.get('damage_count', 0), 'damage_count')
    remaining = _integer(remaining, 'remaining_courage')
    warmth = max(0, min(WARMTH_POINTS, _round_ratio(WARMTH_POINTS * max(0, banked - target), warmth_cap - target)))
    speed = max(0, min(SPEED_POINTS, _round_ratio(SPEED_POINTS * max(0, time_limit - ticks_used), time_limit - fast_target)))
    courage = max(0, min(COURAGE_POINTS, _round_ratio(COURAGE_POINTS * max(0, remaining), initial_courage)))
    total = warmth + speed + courage
    stars = sum(total >= threshold for threshold in STAR_THRESHOLDS)
    next_threshold = next((threshold for threshold in STAR_THRESHOLDS if total < threshold), None)
    return {'warmth': warmth, 'speed': speed, 'courage': courage, 'total': total,
            'stars': stars, 'next_star_points': 0 if next_threshold is None else next_threshold - total}
