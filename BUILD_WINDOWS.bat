@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install -r requirements-build.txt
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m PyInstaller --noconfirm KkamangManor.spec
if errorlevel 1 goto fail
echo dist\KkamangGhostManor 폴더 전체를 ZIP으로 배포하세요.
pause
exit /b 0
:fail
echo 빌드 실패. 위 오류를 확인해 주세요.
pause
exit /b 1
