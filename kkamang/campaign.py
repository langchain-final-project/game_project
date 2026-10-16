"""Human campaign progression and defensive, atomic JSON saves.
This module has no Pygame/Gym imports. Episode scores stay in the game core;
settlement only turns a finished human episode into persistent presentation data.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import uuid
from . import likes

SAVE_VERSION = 2
COSMETICS = {
    'scarf': (4, ('default', 'peach', 'lavender')),
    'rug': (8, ('default', 'stars', 'leaves')),
    'frame': (12, ('default', 'leaves', 'stars')),
}


class SaveFormatError(ValueError):
    """The file is not a supported, structurally valid campaign save."""


class UnsupportedSaveVersion(SaveFormatError):
    pass


def _default() -> dict:
    return {
        'save_version': SAVE_VERSION,
        'completed': [], 'photos': [], 'badges': [], 'room_states': {},
        'settings': {'music': 0.55, 'sfx': 0.75, 'shake': True, 'mode': 'basic'},
        'cosmetics': {'scarf': 'default', 'rug': 'default', 'frame': 'default'},
        'intro_seen': False, 'tutorial_seen': False, 'records': {},
        'settled_ids': [], 'favorite_photo': None,
        'likes': likes.empty_state(),
    }


def _stage_number(value) -> int:
    if isinstance(value, str) and value.startswith('stage_'):
        value = value.removeprefix('stage_')
    try:
        stage = int(value)
    except (ValueError, TypeError):
        raise SaveFormatError('구역 번호가 올바르지 않아요.') from None
    if isinstance(value, bool) or not 1 <= stage <= 12 or (isinstance(value, float) and value != stage):
        raise SaveFormatError('구역 번호는 1~12여야 해요.')
    return stage


def _stage_list(value, name: str) -> list[int]:
    if not isinstance(value, list):
        raise SaveFormatError(f'{name} 목록을 읽을 수 없어요.')
    return sorted({_stage_number(v) for v in value})


def _normalized(data: dict) -> dict:
    if not isinstance(data, dict):
        raise SaveFormatError('저장 파일의 구조가 올바르지 않아요.')
    version = data.get('save_version', data.get('version', 0))
    if version not in (0, 1, SAVE_VERSION) or isinstance(version, bool):
        raise UnsupportedSaveVersion(f'지원하지 않는 저장 버전({version})이에요. 원본을 보존했어요.')
    # v0 is the pre-release format. Complete stages and unique badges determine
    # photos, restored rooms and cosmetics once, without granting new currency.
    result = _default()
    completed_raw = data.get('completed', data.get('first_complete', []))
    if isinstance(completed_raw, dict):
        completed_raw = [k for k, v in completed_raw.items() if v is True]
    result['completed'] = _stage_list(completed_raw, '완료 구역')
    photos = _stage_list(data.get('photos', result['completed']), '기억 사진')
    result['photos'] = sorted(set(photos) | set(result['completed']))
    result['room_states'] = {f'stage_{s:02d}': 'restored' for s in result['completed']}
    badges_raw = data.get('badges', data.get('unique_badges', []))
    if not isinstance(badges_raw, list):
        raise SaveFormatError('배지 목록을 읽을 수 없어요.')
    badges = set()
    for badge in badges_raw:
        if not isinstance(badge, str) or ':' not in badge:
            raise SaveFormatError('배지 이름이 올바르지 않아요.')
        stage_name, kind = badge.rsplit(':', 1)
        if kind == 'complete':  # Completion is represented by completed, not optional count.
            continue
        if kind not in ('rich', 'quiet'):
            raise SaveFormatError('알 수 없는 배지 종류예요.')
        badges.add(f'stage_{_stage_number(stage_name):02d}:{kind}')
    result['badges'] = sorted(badges)
    settings = data.get('settings', {})
    if not isinstance(settings, dict):
        raise SaveFormatError('설정 값을 읽을 수 없어요.')
    for kind in ('music', 'sfx'):
        value = settings.get(kind, result['settings'][kind])
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise SaveFormatError('음량 값이 올바르지 않아요.')
        result['settings'][kind] = max(0.0, min(1.0, float(value)))
    result['settings']['shake'] = bool(settings.get('shake', True))
    result['settings']['mode'] = settings.get('mode', 'basic')
    if result['settings']['mode'] not in ('basic', 'relaxed'):
        raise SaveFormatError('알 수 없는 게임 모드예요.')
    cosmetics = data.get('cosmetics', {})
    if not isinstance(cosmetics, dict):
        raise SaveFormatError('꾸미기 값을 읽을 수 없어요.')
    for kind, (threshold, choices) in COSMETICS.items():
        chosen = cosmetics.get(kind, 'default')
        # A broken cosmetic selection cannot grant abilities or unlock a reward.
        if chosen in choices and (chosen == 'default' or len(badges) >= threshold):
            result['cosmetics'][kind] = chosen
    for flag in ('intro_seen', 'tutorial_seen'):
        result[flag] = bool(data.get(flag, False))
    records = data.get('records', {})
    if not isinstance(records, dict):
        raise SaveFormatError('기록 목록을 읽을 수 없어요.')
    # Validate every nesting level before accepting a file: JSON can parse while
    # a damaged record would otherwise crash the next end-of-run settlement.
    for controller, modes in records.items():
        if not isinstance(controller, str) or not controller or not isinstance(modes, dict):
            raise SaveFormatError('플레이어 기록 구조가 올바르지 않아요.')
        for mode, hashes in modes.items():
            if mode not in ('basic', 'relaxed') or not isinstance(hashes, dict):
                raise SaveFormatError('모드별 기록 구조가 올바르지 않아요.')
            for hash_, stages in hashes.items():
                if not isinstance(hash_, str) or not hash_ or not isinstance(stages, dict):
                    raise SaveFormatError('설정별 기록 구조가 올바르지 않아요.')
                for stage_key, entry in stages.items():
                    stage = _stage_number(stage_key)
                    if stage_key != f'stage_{stage:02d}' or not isinstance(entry, dict):
                        raise SaveFormatError('구역별 기록 구조가 올바르지 않아요.')
                    for count in ('episodes', 'complete', 'return_low', 'failed', 'aborted', 'best_banked'):
                        value = entry.get(count)
                        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                            raise SaveFormatError('플레이 횟수와 점수 기록이 올바르지 않아요.')
                    fastest = entry.get('fastest_complete_ticks')
                    if fastest is not None and (isinstance(fastest, bool) or not isinstance(fastest, int) or fastest < 0):
                        raise SaveFormatError('완료 시간 기록이 올바르지 않아요.')
                    last = entry.get('last')
                    if last is not None and not isinstance(last, dict):
                        raise SaveFormatError('최근 플레이 기록이 올바르지 않아요.')
    result['records'] = deepcopy(records)
    ids = data.get('settled_ids', [])
    if not isinstance(ids, list) or any(not isinstance(v, str) or not v for v in ids):
        raise SaveFormatError('정산 식별자 목록을 읽을 수 없어요.')
    result['settled_ids'] = list(dict.fromkeys(ids))
    favorite = data.get('favorite_photo')
    if favorite is not None:
        favorite = _stage_number(favorite)
        if favorite in result['photos']:
            result['favorite_photo'] = favorite
    try:
        result['likes'] = likes.normalize_state(data.get('likes', likes.empty_state()))
        memories = list(result['likes']['representatives'].values())
        if result['likes']['pending'] is not None:
            memories.append(result['likes']['pending'])
        if any(r['stage'] not in result['completed'] for r in memories):
            raise ValueError('memory belongs to an incomplete room')
    except (ValueError, TypeError) as exc:
        raise SaveFormatError(f'호감도 기록을 읽을 수 없어요: {exc}') from exc
    return result


class Campaign:
    def __init__(self, data: dict | None = None):
        self.data = _default() if data is None else _normalized(data)

    def is_unlocked(self, stage: int) -> bool:
        stage = _stage_number(stage)
        return stage == 1 or stage - 1 in self.data['completed']

    def completed(self, stage: int) -> bool:
        return _stage_number(stage) in self.data['completed']

    def optional_badges(self) -> int:
        return len(set(self.data['badges']))

    def unlocked_cosmetics(self, kind: str) -> list[str]:
        if kind not in COSMETICS:
            raise ValueError(f'Unknown cosmetic kind: {kind}')
        threshold, choices = COSMETICS[kind]
        return list(choices if self.optional_badges() >= threshold else choices[:1])

    def set_cosmetic(self, kind: str, value: str) -> bool:
        if kind not in COSMETICS or value not in self.unlocked_cosmetics(kind):
            return False
        self.data['cosmetics'][kind] = value
        return True

    def settle(self, level: dict, outcome: dict, config_hash: str, mode: str,
               settlement_id: str, controller: str = 'human', memory_record=None) -> dict:
        if not isinstance(settlement_id, str) or not settlement_id:
            raise ValueError('settlement_id must be a non-empty string')
        empty = {'first_complete': False, 'new_photo': False,
                 'new_badges': [], 'new_cosmetics': []}
        if settlement_id in self.data['settled_ids']:
            return {**empty, 'already_settled': True}
        if mode not in ('basic', 'relaxed'):
            raise ValueError('Unsupported mode')
        if not isinstance(config_hash, str) or not config_hash or not isinstance(controller, str) or not controller:
            raise ValueError('config_hash and controller must be non-empty strings')
        stage = _stage_number(level.get('stage', level.get('id')))
        status = outcome.get('status')
        if status not in ('COMPLETE', 'RETURN_LOW', 'FAILED', 'ABORTED'):
            raise ValueError('Cannot settle an unfinished or unknown outcome')
        if level.get('quest') and status == 'COMPLETE' and outcome.get('quest_collected') is not True:
            raise ValueError('COMPLETE requires the mandatory object')
        if level.get('quest') and status == 'RETURN_LOW':
            raise ValueError('v0.2 does not settle incomplete returns')
        numbers = {}
        for key in ('banked', 'carried', 'damage_count', 'ticks_used', 'remaining_ticks'):
            val = outcome.get(key, 0)
            if isinstance(val, bool) or not isinstance(val, int) or val < 0:
                raise ValueError(f'Invalid outcome {key}')
            numbers[key] = val
        target = level.get('target_value', 4)
        maximum = level.get('total_item_value', 14)
        if numbers['banked'] > maximum or numbers['carried'] > maximum:
            raise ValueError('Outcome exceeds available warmth')
        if status == 'COMPLETE' and numbers['banked'] < target:
            raise ValueError('COMPLETE requires the return target')
        if status == 'RETURN_LOW' and numbers['banked'] >= target:
            raise ValueError('RETURN_LOW cannot reach the target')
        if status in ('FAILED', 'ABORTED') and numbers['banked'] != 0:
            raise ValueError('Failed or aborted runs cannot bank warmth')
        eligible = controller == 'human' and self.is_unlocked(stage) and status != 'ABORTED'
        if memory_record is not None:
            memory_record = likes.validate_record(memory_record)
            if (memory_record['stage'] != stage or memory_record['run_id'] != settlement_id
                    or memory_record['config_hash'] != config_hash or memory_record['mode'] != mode
                    or memory_record['banked'] != numbers['banked']
                    or memory_record['ticks_used'] != numbers['ticks_used']):
                raise ValueError('memory does not match this settlement')
        before_cosmetics = {f'{k}:{v}' for k in COSMETICS for v in self.unlocked_cosmetics(k)}
        # Build updates in a copy: a malformed record cannot partially pay rewards.
        updated = deepcopy(self.data)
        records = updated['records'].setdefault(controller, {}).setdefault(mode, {}).setdefault(config_hash, {})
        stage_key = f'stage_{stage:02d}'
        record = records.setdefault(stage_key, {
            'episodes': 0, 'complete': 0, 'return_low': 0, 'failed': 0, 'aborted': 0,
            'best_banked': 0, 'fastest_complete_ticks': None, 'last': None,
        })
        record['episodes'] += 1
        record[status.lower()] += 1
        record['best_banked'] = max(record['best_banked'], numbers['banked'])
        if status == 'COMPLETE':
            previous = record['fastest_complete_ticks']
            record['fastest_complete_ticks'] = numbers['ticks_used'] if previous is None else min(previous, numbers['ticks_used'])
        record['last'] = {**deepcopy(outcome), 'settlement_id': settlement_id, 'eligible_for_campaign': eligible}
        result = {**empty, 'eligible': eligible, 'already_settled': False}
        if eligible and status == 'COMPLETE':
            result['first_complete'] = stage not in updated['completed']
            if result['first_complete']:
                updated['completed'] = sorted(set(updated['completed']) | {stage})
                updated['room_states'][stage_key] = 'restored'
            if stage not in updated['photos']:
                updated['photos'] = sorted(set(updated['photos']) | {stage})
                result['new_photo'] = True
                if updated['favorite_photo'] is None:
                    updated['favorite_photo'] = stage
            earned = []
            if numbers['banked'] >= 10:
                earned.append(f'{stage_key}:rich')
            if numbers['damage_count'] == 0:
                earned.append(f'{stage_key}:quiet')
            result['new_badges'] = [b for b in earned if b not in updated['badges']]
            updated['badges'] = sorted(set(updated['badges']) | set(earned))
            if memory_record is not None:
                result['memory_status'] = likes.offer_memory(updated['likes'], memory_record)
        updated['settled_ids'].append(settlement_id)
        self.data = updated
        after_cosmetics = {f'{k}:{v}' for k in COSMETICS for v in self.unlocked_cosmetics(k)}
        result['new_cosmetics'] = sorted(after_cosmetics - before_cosmetics)
        return result


class SaveStore:
    """Atomic primary + previous-good .bak. Memory is never rolled back on error.

    Corrupt files are copied aside before recovery writes. Future versions and
    concurrent disk changes block writing, preventing accidental data destruction.
    """
    def __init__(self, path: Path):
        self.path = Path(path)
        self.backup_path = self.path.with_name(self.path.name + '.bak')
        self.error: str | None = None
        self.read_only = False
        self.has_save = False
        self.recovered_from_backup = False
        self._loaded = False
        self._expected_bytes: bytes | None = None
        self._primary_good = False

    @staticmethod
    def _decode(raw: bytes) -> Campaign:
        def reject_constant(value):
            raise SaveFormatError(f'올바르지 않은 숫자: {value}')
        try:
            data = json.loads(raw.decode('utf-8'), parse_constant=reject_constant)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise SaveFormatError('저장 파일을 읽을 수 없어요.') from exc
        return Campaign(data)

    def load(self) -> Campaign:
        self.error = None
        self.read_only = False
        self.recovered_from_backup = False
        self._primary_good = False
        self._loaded = True
        self._expected_bytes = None
        self.has_save = self.path.exists() or self.backup_path.exists()
        try:
            if self.path.exists():
                self._expected_bytes = self.path.read_bytes()
                campaign = self._decode(self._expected_bytes)
                self._primary_good = True
                return campaign
            if not self.backup_path.exists():
                return Campaign()
        except UnsupportedSaveVersion as exc:
            self.read_only = True
            self.error = str(exc)
            return Campaign()
        except (OSError, SaveFormatError, ValueError, TypeError) as exc:
            self.error = f'저장 파일을 읽지 못했어요: {exc}'
        try:
            if self.backup_path.exists():
                recovered = self._decode(self.backup_path.read_bytes())
                self.recovered_from_backup = True
                self.error = '이전 정상 백업에서 복구했어요. 다시 저장하면 손상된 원본도 별도로 보존해요.'
                return recovered
        except (OSError, SaveFormatError, ValueError, TypeError):
            pass
        self.read_only = True
        self.error = '저장 파일과 백업을 복구하지 못했어요. 원본을 보존하며 이번 진행은 메모리에만 남아요.'
        return Campaign()

    def _atomic_write(self, path: Path, raw: bytes) -> None:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(prefix=path.name + '.', suffix='.tmp', dir=path.parent, delete=False) as tmp:
                temp_path = Path(tmp.name)
                tmp.write(raw)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(temp_path, path)
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def save(self, campaign: Campaign) -> bool:
        if not self._loaded:
            self.load()
        if self.read_only:
            if self.error is None:
                self.error = '기존 저장 파일을 보존하기 위해 저장을 멈췄어요.'
            return False
        try:
            # Validate a copy; never normalize away in-memory rewards on failure.
            normalized = Campaign(campaign.data)
            raw = (json.dumps(normalized.data, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
            current = self.path.read_bytes() if self.path.exists() else None
            if current != self._expected_bytes:
                self.read_only = True
                self.error = '실행 중 다른 곳에서 저장 파일이 변경됐어요. 기존 파일을 보호하려고 저장을 멈췄어요.'
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if current is not None:
                if self._primary_good:
                    self._atomic_write(self.backup_path, current)
                else:
                    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
                    preserved = self.path.with_name(f'{self.path.name}.corrupt-{stamp}-{uuid.uuid4().hex[:8]}')
                    self._atomic_write(preserved, current)
            self._atomic_write(self.path, raw)
            self._expected_bytes = raw
            self._primary_good = True
            self.has_save = True
            self.recovered_from_backup = False
            self.error = None
            return True
        except (OSError, ValueError, TypeError) as exc:
            self.error = f'아직 저장되지 않았어요. 다시 시도할 수 있어요. ({exc})'
            return False
