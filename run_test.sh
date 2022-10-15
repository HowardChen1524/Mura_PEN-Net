#!/bin/bash
# =====normal test=====
python3 test_with_supervised.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as Mask_MSE -t normal -mm \
-np /hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_8k/ -sp /hcds_vol/private/howard/mura_data/d23_merge/test/test_smura_8k/
python3 test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as MSE -t normal \
-np /hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_8k/ -sp /hcds_vol/private/howard/mura_data/d23_merge/test/test_smura_8k/
python3 test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as Mask_MSE -t normal \
-np /hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_8k/ -sp /hcds_vol/private/howard/mura_data/d23_merge/test/test_smura_8k/
python3 test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as MAE -t normal \
-np /hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_8k/ -sp /hcds_vol/private/howard/mura_data/d23_merge/test/test_smura_8k/
python3 test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as Mask_MAE -t normal \
-np /hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_8k/ -sp /hcds_vol/private/howard/mura_data/d23_merge/test/test_smura_8k/
python3 test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as Discriminator -t normal \
-np /hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_8k/ -sp /hcds_vol/private/howard/mura_data/d23_merge/test/test_smura_8k/
python3 test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as Pyramid_L1 -t normal \
-np /hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_8k/ -sp /hcds_vol/private/howard/mura_data/d23_merge/test/test_smura_8k/

# =====position test=====
# python3 test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as MSE -t position -sp /hcds_vol/private/howard/mura_data/typecplus/img/ -pn
# python3 test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as Mask_MSE -t position -sp /hcds_vol/private/howard/mura_data/typecplus/img/ -pn
# python3 test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as MAE -t position -sp /hcds_vol/private/howard/mura_data/typecplus/img/ -pn
# python3 test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as Mask_MAE -t position -sp /hcds_vol/private/howard/mura_data/typecplus/img/ -pn
# python3 test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as Discriminator -t position -sp /hcds_vol/private/howard/mura_data/typecplus/img/ -pn
# python3 test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as Pyramid_L1 -t position -sp /hcds_vol/private/howard/mura_data/typecplus/img/ -pn

# windows
# python test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as MSE -t position -sp E:/CSE/AI/Mura/mura_data/typecplus/ 
# python test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as Mask_MSE -t position -sp E:/CSE/AI/Mura/mura_data/typecplus/
# python test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as MSE -t position -sp E:/CSE/AI/Mura/mura_data/typecplus/ -n
# python test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as Mask_MSE -t position -sp E:/CSE/AI/Mura/mura_data/typecplus/ -n
