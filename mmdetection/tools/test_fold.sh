#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

for i in ${@}; do

python $(dirname "$0")/test.py model_config.py checkpoints/fold_${i}/latest.pth --out tests/test_fold_${i}_higher_thr.pkl

done