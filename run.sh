#!/bin/bash
# python3 train.py \
# -c=configs/mura.json \
# -mv="PEN-Net_d23_8k_change_cropping" -mn="pennet" -m="square" -s=512 \
# -dn="d23_8k" -tp="/home/sallylab/min/d23_merge/train/d23_normal_8k" \
# -fsp=5000 -sf=10 -eps=100

python3 train.py \
-c=configs/mura.json \
-mv="PEN-Net_d23_4k_step_5000_change_cropping" -mn="pennet" -m="square" -s=512 \
-dn="d23_4k" -tp="/home/sallylab/min/d23_merge/train/normal_4k" \
-fsp=5000 -sf=10 -eps=100