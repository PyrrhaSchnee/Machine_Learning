#!/usr/bin/env bash

set -eux -o pipefail

echo "Preparing python3 venv"

python3 -m venv .venv && source ./.venv/bin/activate && \
pip install --upgrade pip && pip install -r requirements.txt

echo "Python3 venv is ready"
