#!/bin/sh

# Usefull options:
# --num-gpus
# MODEL.WEIGHTS /path/to/weight.pth

python3 ./src/main.py --config-file configs/Flow-RCNN-3DRPN-v1.yaml
