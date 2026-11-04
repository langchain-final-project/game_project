"""실제 학습 없이 파일 잠금·현황 갱신·저장 보존을 검증합니다."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import errno
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from a2c_game import file_io


class FileIOTests(unittest.TestCase):
    def setUp(self):
        project = Path(__file__).resolve().parents[1]
        self.temporary = tempfile.TemporaryDirectory(prefix='.io_retry_test_', dir=project)
        self.folder = Path(self.temporary.name).resolve()
        if not self.folder.is_relative_to(project):
            raise AssertionError('Test cleanup must stay within the project.')
        self.addCleanup(self.temporary.cleanup)
        self.path = self.folder / 'live.json'
        self.path.write_text('{"steps": 0}', encoding='utf-8')

    def read(self):
        return json.loads(self.path.read_text(encoding='utf-8'))

    def test_transient_access_denial_recovers(self):
        original_replace = os.replace
        failures = 0

        def replace(source, destination):
            nonlocal failures
            if failures < 2:
                failures += 1
                raise PermissionError(errno.EACCES, 'temporarily locked')
            original_replace(source, destination)

        with patch.object(file_io.os, 'replace', side_effect=replace) as replacement:
            with patch.object(file_io.time, 'sleep') as sleep:
                file_io.write_json(self.path, {'steps': 128})
        self.assertEqual(replacement.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(self.read(), {'steps': 128})
        self.assertEqual(list(self.folder.glob('*.tmp')), [])

    def test_windows_sharing_violation_is_retried(self):
        sharing = OSError(errno.EINVAL, 'sharing violation')
        sharing.winerror = 32
        with patch.object(file_io.os, 'replace', side_effect=[sharing, None]) as replacement:
            with patch.object(file_io.time, 'sleep'):
                file_io.replace_with_retry('source', 'target')
        self.assertEqual(replacement.call_count, 2)

    def test_persistent_lock_preserves_previous_json_and_cleans_temporary(self):
        with patch.object(file_io.os, 'replace', side_effect=PermissionError(errno.EACCES, 'locked')):
            with self.assertRaises(PermissionError):
                file_io.write_json(self.path, {'steps': 128}, retry_timeout=0)
        self.assertEqual(self.read(), {'steps': 0})
        self.assertEqual(list(self.folder.glob('*.tmp')), [])

    def test_retry_deadline_is_bounded(self):
        with patch.object(file_io.os, 'replace', side_effect=PermissionError(errno.EACCES, 'locked')) as replacement:
            with patch.object(file_io.time, 'monotonic', side_effect=[0., .1, .2, 5.1]):
                with patch.object(file_io.time, 'sleep') as sleep:
                    with self.assertRaises(PermissionError):
                        file_io.replace_with_retry('source', 'target')
        self.assertEqual(replacement.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_unrelated_io_errors_are_not_retried(self):
        with patch.object(file_io.os, 'replace', side_effect=OSError(errno.ENOSPC, 'disk full')) as replacement:
            with patch.object(file_io.time, 'sleep') as sleep:
                with self.assertRaises(OSError):
                    file_io.write_json(self.path, {'steps': 128})
        self.assertEqual(replacement.call_count, 1)
        sleep.assert_not_called()
        self.assertEqual(self.read(), {'steps': 0})

    def test_live_output_skips_locked_file_and_recovers_after_cooldown(self):
        writer = file_io.LiveJSONWriter(self.path)
        messages = io.StringIO()
        with redirect_stdout(messages):
            with patch.object(file_io.time, 'monotonic', return_value=10.) as clock:
                with patch.object(file_io, 'write_json', side_effect=PermissionError('locked')) as write:
                    self.assertFalse(writer.write({'steps': 128}))
                    clock.return_value = 11.
                    self.assertFalse(writer.write({'steps': 256}))
                    self.assertEqual(write.call_count, 1)
                    clock.return_value = 12.
                    self.assertFalse(writer.write({'steps': 384}))
                    self.assertEqual(write.call_count, 2)
                clock.return_value = 14.
                with patch.object(file_io, 'write_json'):
                    self.assertTrue(writer.write({'steps': 512}))
        self.assertEqual(len(messages.getvalue().splitlines()), 2)
        self.assertFalse(writer.warned)

    def test_invalid_json_still_raises(self):
        writer = file_io.LiveJSONWriter(self.path)
        with self.assertRaises(ValueError):
            writer.write({'invalid': float('nan')})
        self.assertEqual(self.read(), {'steps': 0})

    def test_concurrent_writers_use_distinct_temporary_files(self):
        def write(worker):
            for index in range(12):
                file_io.write_json(self.path, {'worker': worker, 'index': index})

        with ThreadPoolExecutor(max_workers=2) as executor:
            tasks = [executor.submit(write, worker) for worker in (1, 2)]
            for task in tasks:
                task.result(timeout=10)
        self.assertIn(self.read()['worker'], (1, 2))
        self.assertEqual(self.read()['index'], 11)
        self.assertEqual(list(self.folder.glob('*.tmp')), [])

    @unittest.skipUnless(os.name == 'nt', 'Windows file sharing test')
    def test_real_windows_lock_retries_then_succeeds(self):
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        create = kernel.CreateFileW
        create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                           wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
        create.restype = wintypes.HANDLE
        close = kernel.CloseHandle
        close.argtypes = (wintypes.HANDLE,)
        close.restype = wintypes.BOOL
        # 읽기·쓰기는 공유하지만 교체/삭제는 공유하지 않는 실제 Windows 핸들입니다.
        handle = create(str(self.path), 0x80000000, 3, None, 3, 0x80, None)
        if handle == wintypes.HANDLE(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        denied = threading.Event()
        original_replace = os.replace

        def replace(source, destination):
            try:
                original_replace(source, destination)
            except PermissionError:
                denied.set()
                raise

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                with patch.object(file_io.os, 'replace', side_effect=replace):
                    future = executor.submit(file_io.write_json, self.path, {'steps': 128}, retry_timeout=1.)
                    try:
                        self.assertTrue(denied.wait(timeout=1.), 'Expected a real Windows replacement denial.')
                        self.assertEqual(self.read(), {'steps': 0})
                    finally:
                        close(handle)
                        handle = None
                    future.result(timeout=2.)
        finally:
            if handle is not None:
                close(handle)
        self.assertEqual(self.read(), {'steps': 128})

    def test_training_callback_continues_when_live_output_is_locked(self):
        from types import SimpleNamespace
        from a2c_game.train import LearningLog

        state = {'total_steps': 10_000_000, 'stage_exposure': {'1': {
            'learner_steps': 0, 'physical_frames': 0, 'episodes': 0,
            'returns': 0, 'highscores': 0}}}
        callback = LearningLog(self.folder, state, {}, lambda: False)
        callback.model = SimpleNamespace(n_steps=32)
        callback.n_calls = 8
        callback.num_timesteps = 32
        callback.locals = {'infos': [{'stage': 1, 'executed_frames': 18, 'snapshot': {}}],
                           'n_steps': 7}
        with patch('a2c_game.train.frame_from_snapshot', return_value={}):
            with patch.object(file_io, 'write_json', side_effect=PermissionError('locked')):
                with redirect_stdout(io.StringIO()):
                    self.assertTrue(callback._on_step())
        self.assertEqual(state['stage_exposure']['1']['learner_steps'], 1)


if __name__ == '__main__':
    unittest.main()
