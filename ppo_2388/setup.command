#!/bin/bash
set -euo pipefail
project_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$project_dir"
python_bin="$(command -v python3.12 || command -v python3 || true)"
if [[ -z "$python_bin" ]]; then
    printf 'Python3.12 이상을 설치한 뒤 다시 실행하세요.\n'
    exit 1
fi
"$python_bin" -c 'import sys; assert sys.version_info >= (3,12), "Python3.12 이상이 필요합니다."'
"$python_bin" -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
printf '설치 완료. python run.py prepare 후 python run.py check로 검사하고 python run.py train으로 학습하세요.\n'
