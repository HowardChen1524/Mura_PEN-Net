#!/bin/bash
declare -a measure_list=(
                         "MSE"
                        #  "Mask_MSE"
                        #  "Content_VGG16_sliding"
                        #  "Mask_Content_VGG16_sliding"
                        # "SSIM"
                        # "Mask_SSIM"
                        #  "Discriminator"
                        #  "Pyramid_L1"
                        )


# model_version="PEN-Net_d23_8k"
model_version="PEN-Net_d23_8k_change_cropping"
# model_version="PEN-Net_d23_4k"
# model_version="PEN-Net_d23_4k_step_5000_change_cropping"


# d23 4k test
# dataset_name="d23_4k"
# # sup_data_path="/hcds_vol/private/howard/mura_data/d23_merge/" # for supervised model
# # sup_csv_path="/hcds_vol/private/howard/mura_data/d23_merge/data_merged.csv" # for supervised model
# unsup_test_normal_path="/home/sallylab/min/d23_merge/test/test_normal_4k/" # for unsupervised model
# unsup_test_smura_path="/home/sallylab/min/d23_merge/test/test_smura_4k/" # for unsupervised model
# normal_num=16295
# smura_num=181

# dataset_name="typec_4k"
# # sup_data_path="/hcds_vol/private/howard/mura_data/d23_merge/" # for supervised model
# # sup_csv_path="/hcds_vol/private/howard/mura_data/d23_merge/data_merged.csv" # for supervised model
# unsup_test_normal_path="/home/sallylab/min/d23_merge/test/test_normal_4k/" # for unsupervised model
# unsup_test_smura_path="/home/sallylab/Howard/Mura_ShiftNet/imgs/typec4k/" # for unsupervised model
# normal_num=1
# smura_num=3

# d23 8k test
# dataset_name="d23_8k"
# # sup_data_path="/hcds_vol/private/howard/mura_data/d23_merge/" # for supervised model
# # sup_csv_path="/hcds_vol/private/howard/mura_data/d23_merge/data_merged.csv" # for supervised model
# unsup_test_normal_path="/home/sallylab/min/d23_merge/test/test_normal_8k/" # for unsupervised model
# unsup_test_smura_path="/home/sallylab/min/d23_merge/test/test_smura_8k/" # for unsupervised model
# normal_num=541
# smura_num=143

dataset_name="typecandplus8k"
sup_data_path="/hcds_vol/private/howard/mura_data/d23_merge/" # for supervised model
csv_path="/hcds_vol/private/howard/mura_data/d23_merge/data_merged.csv" # for supervised model
unsup_test_normal_path="/home/sallylab/min/d23_merge/test/test_normal_8k/" # for unsupervised model
unsup_test_smura_path="/home/sallylab/Howard/Mura_ShiftNet/imgs/typecandplus8k/" # for unsupervised model
normal_num=1
smura_num=3

# d24 d25 8k blind test
# dataset_name="mura_d24_d25_8k"
# sup_data_path="/hcds_vol/private/howard/mura_data/d25_merge/"
# # sup_csv_path="/hcds_vol/private/howard/mura_data/d25_merge/d25_data_merged.csv"
# sup_csv_path="/hcds_vol/private/howard/mura_data/d25_merge_8k/d25_data_merged_new.csv"
# unsup_test_normal_path="/hcds_vol/private/howard/mura_data/d25_merge_8k/d2425_all_normal_8k/"
# unsup_test_smura_path="/hcds_vol/private/howard/mura_data/d25_merge_8k/d2425_all_smura_8k/"
# normal_num=87
# smura_num=85

# =====unsup testing=====
for measure in ${measure_list[@]}
do
    echo $measure
    python3 test.py \
    -c configs/mura.json \
    -dn $dataset_name \
    -mv $model_version \
    -mn pennet -m square -s 512 \
    -t normal -me 100 -as $measure \
    -np $unsup_test_normal_path -nn $normal_num \
    -sp $unsup_test_smura_path  -sn $smura_num \
    -gpu 0
done

# for measure in ${measure_list[@]}
# do
#     echo $measure
#     python3 test.py \
#     -c configs/mura.json \
#     -dn $dataset_name \
#     -mv $model_version \
#     -mn pennet -m square -s 512 \
#     -t normal -me 100 -as $measure \
#     -np $unsup_test_normal_path -nn $normal_num \
#     -sp $unsup_test_smura_path  -sn $smura_num \
#     -gpu 0 -pn
# done

# for measure in ${measure_list[@]}
# do
#     echo $measure
#     python3 test.py \
#     -c configs/mura.json \
#     -dn $dataset_name \
#     -mv $model_version \
#     -mn pennet -m square -s 512 \
#     -t normal -me 100 -as $measure \
#     -np $unsup_test_normal_path -nn $normal_num \
#     -sp $unsup_test_smura_path  -sn $smura_num \
#     -gpu 0
# done

# =====generate conf score=====
# for measure in ${measure_list[@]}
# do
#     python3 sup_unsup_gen_res.py \
#     -c configs/mura.json \
#     -mv $model_version -mn pennet -m square -s 512 \
#     -t normal -me 100 -as $measure \
#     -dn $dataset_name \
#     -dp $sup_data_path \
#     -cp $sup_csv_path \
#     -np $unsup_test_normal_path -nn $normal_num \
#     -sp $unsup_test_smura_path -sn $smura_num \
#     -gpu 0
# done

# =====find th or blind test=====
# for measure in ${measure_list[@]}
# do
#     python3 sup_unsup_find_th_or_test.py \
#     -c configs/mura.json \
#     -mv $model_version -mn pennet -m square -s 512 \
#     -t normal -me 100 -as $measure \
#     -dn $dataset_name \
#     -cp $sup_csv_path \
#     -ut \
#     -gpu 0
# done