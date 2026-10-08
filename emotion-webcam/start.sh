#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [ ! -x .venv/bin/python ]; then
    python3 -c 'import sys; assert (3,11) <= sys.version_info[:2] < (3,14), "Use Python 3.11-3.13"'
    python3 -m venv .venv
fi
if ! .venv/bin/python -c 'import cv2, numpy, pandas, sklearn, matplotlib' >/dev/null 2>&1; then
    .venv/bin/python -m pip install --only-binary=:all: -r requirements.txt
fi
.venv/bin/python -m emotion_app gui
