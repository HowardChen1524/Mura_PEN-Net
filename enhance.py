import os
from glob import glob
from core.utils import enhance_img

path = '/hcds_vol/private/howard/temp/inpainting/'
dir_names = ['7_mura','4_mura_pred_false','109_nomura_pred_true','113_mura']

for dir_name in dir_names:
    print(dir_name)
    dp = f'{path}{dir_name}/'
    fp_list = glob(f'{dp}*png')
    for fp in fp_list:
        en_img = enhance_img(fp, 5)
        print(f'{fp[:-4]}_en.png')
        en_img.save(f'{fp[:-4]}_en.png')
