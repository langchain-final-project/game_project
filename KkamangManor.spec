# Build on the target operating system. Windows produces KkamangManor.exe.
from pathlib import Path
base = Path(SPECPATH)
a = Analysis([str(base / 'main.py')], pathex=[str(base)],
    binaries=[], datas=[(str(base / 'assets'), 'assets'), (str(base / 'data'), 'data')],
    hiddenimports=[], hookspath=[], hooksconfig={}, runtime_hooks=[],
    excludes=['pytest', 'gymnasium', 'numpy', 'matplotlib', 'PIL'], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='KkamangManor',
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=False, disable_windowed_traceback=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='KkamangManor')
