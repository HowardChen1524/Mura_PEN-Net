#!/bin/bash

declare -a test_type=( 
                        # test_with_supervised_blind.py
                        test_with_supervised.py
                        # test.py
                     )

declare -a measure_list=(
                         "Mask_MSE"
                        )

for type in ${test_type[@]}
do
    for measure in ${measure_list[@]}
    do
        echo $measure
        python3 $type \
        -c configs/mura.json \
        -dn "mura_d24_d25_8k" \
        -mv "mura_d23_8k" \
        -mn pennet -m square -s 512 \
        -t normal -me 100 -as $measure \
        -mm \
        -ut \
        -dp "/hcds_vol/private/howard/mura_data/d25_merge/" \
        -cp "/hcds_vol/private/howard/mura_data/d25_merge/d25_data_merged.csv" \
        -np "/hcds_vol/private/howard/mura_data/d25_merge_8k/d2425_all_normal_8k/" \
        -sp "/hcds_vol/private/howard/mura_data/d25_merge_8k/d2425_all_smura_8k/" 
    done
done

for type in ${test_type[@]}
do
    for measure in ${measure_list[@]}
    do
        echo $measure
        python3 $type \
        -c configs/mura.json \
        -dn "mura_d24_d25_8k" \
        -mv "mura_d23_8k" \
        -mn pennet -m square -s 512 \
        -t normal -me 100 -as $measure \
        -mm \
        -pn \
        -ut \
        -dp "/hcds_vol/private/howard/mura_data/d25_merge/" \
        -cp "/hcds_vol/private/howard/mura_data/d25_merge/d25_data_merged.csv" \
        -np "/hcds_vol/private/howard/mura_data/d25_merge_8k/d2425_all_normal_8k/" \
        -sp "/hcds_vol/private/howard/mura_data/d25_merge_8k/d2425_all_smura_8k/" 
    done
done

# supervised model 資料讀取 -> 先分別讀兩個不同資料夾
# blind test

# 整合 supervised function
# 確認code 
# supervised model 資料讀取統一