"""기존 프로젝트와 독립적으로 설치된 Python을 찾아 실행합니다."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parent
os.chdir(root)
candidates = [root / '.venv/bin/python', root / '.venv/Scripts/python.exe',
              root.parent / 'game_project/.venv/bin/python', Path(sys.executable)]
for name in ('python3.12', 'python3', 'python'):
    path = shutil.which(name)
    if path: candidates.append(Path(path))
for candidate in dict.fromkeys(candidates):
    if not candidate.is_file(): continue
    probe = subprocess.run([str(candidate), '-c', 'import sys,torch,stable_baselines3,gymnasium,pygame; assert sys.version_info >= (3,12)'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if probe.returncode == 0:
        os.execv(str(candidate), [str(candidate), '-u', str(root / 'run.py'), *sys.argv[1:]])
print('Python3.12 이상과 실행 패키지가 필요합니다. setup.command 또는 README 설치 안내를 사용하세요.')
sys.exit(2)
