"""파일 잠금 재시도와 원자적 JSON 저장을 제공합니다."""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import tempfile
import time


def replace_with_retry(source, destination, *, retry_timeout=5.):
    """일시적인 접근 거부·공유 잠금만 재시도하며 기존 목적지는 유지합니다."""
    deadline = time.monotonic() + retry_timeout
    delay = .01
    while True:
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            transient = (getattr(error, 'winerror', None) in (5, 32, 33)
                         or error.errno in (errno.EACCES, errno.EPERM, errno.EBUSY))
            remaining = deadline - time.monotonic()
            if not transient or remaining <= 0:
                raise
            time.sleep(min(delay, remaining))
            delay = min(.1, delay * 2)


def write_json(path, value, *, retry_timeout=5.):
    """매번 다른 임시 파일을 같은 폴더에 완성한 뒤 목적지와 교체합니다."""
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp',
                                             dir=path.parent)
    pending = Path(temporary)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(payload)
        replace_with_retry(pending, path, retry_timeout=retry_timeout)
    finally:
        # 백신 등이 임시 파일까지 잠근 경우 원래 오류를 정리 오류로 덮지 않습니다.
        try:
            pending.unlink(missing_ok=True)
        except OSError:
            pass


class LiveJSONWriter:
    """현황 파일이 잠겨도 학습은 계속하고 잠금 해제 뒤 갱신을 재개합니다."""

    def __init__(self, path, *, retry_timeout=.05, cooldown=2.):
        self.path = Path(path)
        self.retry_timeout = retry_timeout
        self.cooldown = cooldown
        self.next_attempt = 0.
        self.warned = False

    def ready(self):
        return time.monotonic() >= self.next_attempt

    def write(self, value):
        if not self.ready():
            return False
        try:
            write_json(self.path, value, retry_timeout=self.retry_timeout)
        except OSError as error:
            self.next_attempt = time.monotonic() + self.cooldown
            if not self.warned:
                print(f'현황 파일 갱신을 건너뜁니다. 학습은 계속됩니다: {error}', flush=True)
                self.warned = True
            return False
        if self.warned:
            print('현황 파일 갱신이 복구되었습니다.', flush=True)
        self.warned = False
        self.next_attempt = 0.
        return True
