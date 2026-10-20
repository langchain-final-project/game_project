"""명세에 고정된 CC0 피아노 샘플을 복원한다."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import urllib.request

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--check', action='store_true', help='Verify local files without downloading')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    checked = 0
    for entry in manifest['samples']:
        relative = PurePosixPath(entry.get('relative_path', 'VCSL/'+entry['source_path']))
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe sample path')
        path = args.manifest.parent / Path(relative)
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']:
            checked += 1
            continue
        if args.check:
            raise ValueError('Missing or changed sample: '+str(relative))
        url = entry['source_url']
        prefix = 'https://raw.githubusercontent.com/'+manifest['source_repo']+'/'+manifest['commit']+'/'
        if not url.startswith(prefix):
            raise ValueError('Source is not the pinned repository version')
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('Downloaded checksum mismatch: '+str(relative))
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name+'.downloading')
        with temp.open('wb') as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp, path)
        checked += 1
    print(json.dumps({'verified_sample_files': checked, 'check_only': args.check}))

if __name__ == '__main__':
    main()
