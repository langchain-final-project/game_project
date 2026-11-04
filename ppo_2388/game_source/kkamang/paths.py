"""소스·배포 환경의 리소스 경로와 운영체제별 사용자 저장 경로를 분리한다."""
from pathlib import Path
import os
import sys

ROOT = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[1]))
DATA_DIR = ROOT / 'data'
ASSET_DIR = ROOT / 'assets'

def user_data_dir() -> Path:
    """설치·번들 경로 밖의 쓰기 가능한 사용자 저장 폴더를 반환한다."""
    if sys.platform == 'win32':
        base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
    elif sys.platform == 'darwin':
        base = Path.home() / 'Library' / 'Application Support'
    else:
        base = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share'))
    return base / 'KkamangManor'
