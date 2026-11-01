"""사용자가 제공한 ZIP에서 실행에 필요한 자료만 새 폴더로 복사합니다."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from .config import ROOT, VENDOR_ROOT, SOURCE_ARCHIVE, SOURCE_CHECKPOINT

MODULES = {'__init__.py', 'paths.py', 'policy.py', 'environment.py',
           'observation.py', 'distance.py'}
RECORDS = {'shared_snapshot/config.json', 'shared_snapshot/bundle.json',
           'shared_snapshot/results.json', SOURCE_CHECKPOINT, 'README.md'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(archive=SOURCE_ARCHIVE):
    archive = Path(archive).resolve()
    target = VENDOR_ROOT.resolve()
    if not target.is_relative_to(ROOT.resolve()):
        raise ValueError('복사 위치는 A2cToPpo 폴더 안이어야 합니다.')
    if target.exists():
        raise FileExistsError('이미 준비된 원본이 있습니다. 덮어쓰지 않습니다: ' + str(target))
    with ZipFile(archive) as package:
        chosen = []
        for item in package.infolist():
            path = PurePosixPath(item.filename)
            if item.is_dir() or not path.parts or path.parts[0] != 'a2c':
                continue
            relative = PurePosixPath(*path.parts[1:])
            name = relative.as_posix()
            if not (name.startswith('game_source/') or name in RECORDS
                    or (relative.parent.as_posix() == 'a2c_game' and relative.name in MODULES)):
                continue
            destination = (target / Path(*relative.parts)).resolve()
            if '..' in relative.parts or not destination.is_relative_to(target):
                raise ValueError('ZIP 내부 경로가 복사 위치를 벗어납니다.')
            chosen.append((item, destination))
        required = {SOURCE_CHECKPOINT, 'shared_snapshot/config.json', 'game_source/kkamang/realtime.py',
                    'a2c_game/policy.py', 'a2c_game/environment.py'}
        included = {p.relative_to(target).as_posix() for _, p in chosen}
        if not required <= included:
            raise ValueError('필요한 원본 파일이 ZIP에 없습니다: ' + str(sorted(required - included)))
        hashes = {}
        for item, destination in chosen:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as output:
                output.write(package.read(item))
            hashes[destination.relative_to(target).as_posix()] = sha256(destination)
    manifest = {'source_archive': str(archive), 'archive_sha256': sha256(archive),
                'selected_checkpoint': SOURCE_CHECKPOINT, 'files': hashes}
    (target / 'transfer_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description='A2C 원본을 새 실험 폴더에 준비합니다.')
    parser.add_argument('--archive', type=Path, default=SOURCE_ARCHIVE)
    args = parser.parse_args()
    manifest = prepare(args.archive)
    print(f"원본 준비 완료: {len(manifest['files'])}개 파일 | {VENDOR_ROOT}")
    print('기존 rl·게임 파일·외부 A2C 폴더는 수정하지 않았습니다.')


if __name__ == '__main__':
    main()
