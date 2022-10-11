from glob import glob
import pandas as pd
from collections import defaultdict 
import xmltodict, json


def create_typecplus_row(info_f, fn, obj):
    info_f['fn'] = fn
    info_f['smura_name'] = obj['name']
    x0 = int(obj['bndbox']['xmin'])
    info_f['x0'] = x0
    y0 = int(obj['bndbox']['ymin'])
    info_f['y0'] = y0
    x1 = int(obj['bndbox']['xmax'])
    info_f['x1'] = x1
    y1 = int(obj['bndbox']['ymax'])
    info_f['y1'] = y1
    info_f['x_center'] = (x0+x1)//2
    info_f['y_center'] = (y0+y1)//2
    info_f['w'] = x1-x0+1
    info_f['h'] = y1-y0+1
    return info_f

path = r"/hcds_vol/private/howard/mura_data/typecplus/xml/"
path_fn_list = glob(f"{path}*xml")

info_fn_list = []
for path_fn in path_fn_list:
    with open(path_fn) as fd:
        json_fd = xmltodict.parse(fd.read())
        if type(json_fd['annotation']['object']) == list:
            for obj in json_fd['annotation']['object']:
                info_f = defaultdict()
                # print(info_f)
                # print(json_fd['annotation']['filename'])
                # print(obj)
                info_fn_list.append(create_typecplus_row(info_f, json_fd['annotation']['filename'], obj))
        else:
            info_f = defaultdict()
            info_fn_list.append(create_typecplus_row(info_f, json_fd['annotation']['filename'], json_fd['annotation']['object']))
df = pd.DataFrame.from_dict(info_fn_list)
df.to_csv (r'./Mura_type_c_plus.csv', index = False, header=True)