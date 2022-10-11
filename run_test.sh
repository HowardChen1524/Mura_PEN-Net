# python3 test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as MSE -t normal
# python3 test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as Mask_MSE -t normal
# python3 test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as MSE -t position -sp /hcds_vol/private/howard/mura_data/typecplus/img/
# python3 test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as Mask_MSE -t position -sp /hcds_vol/private/howard/mura_data/typecplus/img/
# python3 test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as MSE -t position -sp /hcds_vol/private/howard/mura_data/typecplus/img/ -pn
# python3 test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as Mask_MSE -t position -sp /hcds_vol/private/howard/mura_data/typecplus/img/ -pn

# windows
python test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as MSE -t position -sp E:/CSE/AI/Mura/mura_data/typecplus/ 
python test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as Mask_MSE -t position -sp E:/CSE/AI/Mura/mura_data/typecplus/
python test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as MSE -t position -sp E:/CSE/AI/Mura/mura_data/typecplus/ -pn
python test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as Mask_MSE -t position -sp E:/CSE/AI/Mura/mura_data/typecplus/ -pn

python test_with_supervised.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as Mask_MSE -t normal -mm
python test.py -c configs/mura.json -mn pennet -m square -s 512 -me 100 -as Mask_MSE -t normal -mm
# MSE
# 7.9499725e-05
# 6.252859e-05
# Smura mean: 7.94997249613516e-05
# Smura std: 6.252859020605683e-05
# Mask_MSE
# 4.726136e-05
# 1.9409596e-05
# Smura mean: 4.726136103272438e-05
# Smura std: 1.9409595552133396e-05

# MSE n
# Mean: 2.6152446269989014
# std: 4.479475975036621
# Smura mean: 2.6152446269989014
# Smura std: 4.479475975036621
# Mask_MSE n
# Mean: 0.30576178431510925
# std: 1.3904967308044434
# Smura mean: 0.30576178431510925
# Smura std: 1.3904967308044434