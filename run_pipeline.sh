#!/usr/bin/env bash
set -e
python -m src.download
python -m src.clean
python -m src.train
python -m pytest -q
