#!/bin/bash
python3 train.py \
-c configs/mura.json \
-mv mura_d23_4k -mn pennet -m square -s 512 \
-dn mura_d23_4k -tp /hcds_vol/private/howard/mura_data/d23_merge/train/normal_4k \
-sf 5 -eps 100