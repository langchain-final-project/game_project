"""공유 폴더에 포함된 게임 코어를 불러옵니다. 예전 실험 폴더의 게임 코드가 섞이는 것도 방지합니다."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAME_ROOT = PROJECT_ROOT / 'game_source'


def ensure_game_path():
    if not (GAME_ROOT / 'kkamang' / 'realtime.py').is_file():
        raise RuntimeError(f'Copied game engine is missing: {GAME_ROOT}')
    # 같은 이름의 과거 게임 모듈이 먼저 로드됐다면 섞어 쓰지 않고 원인을 알립니다.
    already_loaded = sys.modules.get('kkamang')
    if already_loaded is not None:
        loaded_path = Path(already_loaded.__file__).resolve()
        if not loaded_path.is_relative_to(GAME_ROOT):
            raise RuntimeError(f'A different game engine was imported before a2c_game: {loaded_path}')
    source = str(GAME_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)
    return GAME_ROOT
