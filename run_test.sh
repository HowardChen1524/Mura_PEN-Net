#!/bin/bash
# =====normal test=====
# python3 test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as Mask_MSE -t normal -mm 

# =====typec+ position=====
# python3 test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as MSE -t position -sp /hcds_vol/private/howard/mura_data/typecplus/img/ -pn
# python3 test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as Mask_MSE -t position -sp /hcds_vol/private/howard/mura_data/typecplus/img/ -pn
# python3 test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as MAE -t position -sp /hcds_vol/private/howard/mura_data/typecplus/img/ -pn
# python3 test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as Mask_MAE -t position -sp /hcds_vol/private/howard/mura_data/typecplus/img/ -pn
# python3 test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as Pyramid_L1 -t position -sp /hcds_vol/private/howard/mura_data/typecplus/img/ -pn
# python3 test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as Discriminator -t position -sp /hcds_vol/private/howard/mura_data/typecplus/img/ -pn

# =====with supervised model=====
# python3 test_with_supervised.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as Mask_MSE -t normal -mm 

declare -a test_type=( 
                        test_with_supervised.py
                        test.py
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
        -dn "mura_d23_8k" \
        -mn pennet -m square -s 512 \
        -t normal -me 100 -as $measure \
        -mm \
        # -pn \
    done
done

# windows
# E:/CSE/AI/Mura/mura_data/typecplus/ 

# 確認用同樣解析度 -> 沒影響
# 確認用同樣resize演篹法 -> 0.799 -> 0.782
# 確認 minmax 修改是否正確 ok
# 做 minmax 重新跑一次 mask mse test -> 沒影響
# 結合 supervised 
# test th

# 如何印出那個特例點

# dataset ok debug 參數沒用
# tester ok draw 原始圖片size要改成變數