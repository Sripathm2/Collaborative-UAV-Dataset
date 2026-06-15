#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2

python3 run_t11.py
python3 run_t12.py
python3 run_t13.py
python3 run_confusion.py
python3 make_tables.py
python3 make_figures.py
