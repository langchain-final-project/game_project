"""운영체제에 맞게 최신 실행 폴더를 기록하고 찾습니다. Windows에서 심볼릭 링크 권한이 없어도 JSON 경로로 동작합니다."""
from __future__ import annotations

import json
import os
from datetime import timedelta, timezone
from pathlib import Path
import uuid

from .paths import PROJECT_ROOT
from .file_io import replace_with_retry, write_json

# 실행 폴더 시각은 한국 표준시로 통일합니다.
# 시간대 데이터베이스가 없는 Windows에서도 동작하도록 UTC+9를 직접 지정합니다.
KST = timezone(timedelta(hours=9), name='KST')


def set_latest_run(run, *, alias_root=None):
    run = Path(run).resolve(strict=True)
    root = Path(alias_root).resolve() if alias_root else run.parent
    root.mkdir(parents=True, exist_ok=True)
    alias = root / 'latest'
    if alias.exists() and not alias.is_symlink():
        raise ValueError(f'Reserved latest alias is occupied: {alias}')
    # 압축을 풀어 다른 PC로 옮겨도 유효하도록 상대 경로를 저장합니다.
    target = Path(os.path.relpath(run, root))
    write_json(root / 'latest.json', {'run': target.as_posix()})
    link = root / f'.latest_{uuid.uuid4().hex}'
    try:
        link.symlink_to(target, target_is_directory=True)
        replace_with_retry(link, alias)
    except OSError:
        # 심볼릭 링크를 만들 수 없는 환경에서도 위 JSON 경로로 최신 실행을 찾습니다.
        if link.is_symlink():
            link.unlink()
    return run


def resolve_run(value=None):
    path = Path(value) if value is not None else PROJECT_ROOT / 'runs/latest'
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.name == 'latest' and path.with_suffix('.json').is_file():
        pointer = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        candidate = (path.parent / pointer['run']).resolve(strict=True)
        if not candidate.is_relative_to(path.parent.resolve()):
            raise ValueError('Latest pointer must stay inside its run directory.')
        path = candidate
    return path.resolve(strict=True)
