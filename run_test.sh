#!/bin/bash

declare -a test_type=( 
                        test_with_supervised.py
                        # test.py
                     )

declare -a measure_list=(
                         "Mask_MSE"
                        )

model_version="mura_d23_8k"
# model_version="mura_d23_4k"

# d23 test
# dataset_name="mura_d23_4k"
# dataset_name="mura_d23_8k"

sup_data_path="/hcds_vol/private/howard/mura_data/d23_merge/" # for supervised model
sup_csv_path="/hcds_vol/private/howard/mura_data/d23_merge/data_merged.csv" # for supervised model

# 4k
# unsup_test_normal_path="/hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_4k/" # for unsupervised model
# unsup_test_smura_path="/hcds_vol/private/howard/mura_data/d23_merge/test/test_smura_4k/" # for unsupervised model

# 8k
# unsup_test_normal_path="/hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_8k/" # for unsupervised model
# unsup_test_smura_path="/hcds_vol/private/howard/mura_data/d23_merge/test/test_smura_8k/" # for unsupervised model

# d24 d25 blind test
dataset_name="mura_d24_d25_8k"
# sup_data_path="/hcds_vol/private/howard/mura_data/d25_merge/"
csv_path="/hcds_vol/private/howard/mura_data/d25_merge/d25_data_merged.csv"
# unsup_test_normal_path="/hcds_vol/private/howard/mura_data/d25_merge_8k/d2425_all_normal_8k/"
# unsup_test_smura_path="/hcds_vol/private/howard/mura_data/d25_merge_8k/d2425_all_smura_8k/"
# normal_num=88
# smura_num=85

# test sup unsup
# for measure in ${measure_list[@]}
# do
#     echo $measure
#     python3 test_sup_unsup.py \
#     -c configs/mura.json \
#     -dn $dataset_name -nn $normal_num -sn $smura_num \
#     -mn pennet -mv $model_version \
#     -me 100 -as $measure \
#     -mm
# done

# unsup testing
# for measure in ${measure_list[@]}
# do
#     echo $measure
#     python3 test.py \
#     -c configs/mura.json \
#     -dn $dataset_name \
#     -mv $model_version \
#     -mn pennet -m square -s 512 \
#     -t normal -me 100 -as $measure \
#     -mm \
#     -dp $sup_data_path \
#     -cp $sup_csv_path \
#     -np $unsup_test_normal_path \
#     -sp $unsup_test_smura_path
# done

# generate conf score 
# for measure in ${measure_list[@]}
# do
#     python3 sup_unsup_gen_res.py \
#     -c configs/mura.json \
#     -mv $model_version -mn pennet -m square -s 512 \
#     -t normal -me 100 -as $measure \
#     -mm \
#     -dn $dataset_name \
#     -dp $sup_data_path \
#     -cp $sup_csv_path \
#     -np $unsup_test_normal_path \
#     -sp $unsup_test_smura_path
# done


# find th or blind test
for measure in ${measure_list[@]}
do
    python3 sup_unsup_find_th_or_test.py \
    -c configs/mura.json \
    -mv $model_version -mn pennet -m square -s 512 \
    -t normal -me 100 -as $measure \
    -mm \
    -dn $dataset_name \
    -ut \
    -cp $csv_path
done