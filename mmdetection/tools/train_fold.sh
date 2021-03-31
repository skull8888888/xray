#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

for i in ${@}; do

python $(dirname "$0")/train.py model_config.py --fold ${i} 

done