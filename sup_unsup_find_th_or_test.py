import os
import argparse
import json

import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn

from opt.option import get_test_parser
from core.utils import set_seed
from core.tester import Tester
from core.utils_howard import mkdir, minmax_scaling, \
                              get_data_info, make_test_dataloader, evaluate, get_line_threshold, \
                              plot_score_distribution, plot_sup_unsup_scatter, plot_line_on_scatter, \
                              sup_unsup_prediction_spec_th, sup_unsup_prediction_spec_multi_th, \
                              sup_unsup_prediction_auto_th, sup_unsup_prediction_auto_multi_th, sup_unsup_SVM, sup_unsup_DT, sup_unsup_SVM_test, sup_unsup_DT_test, \
                              sup_prediction_spec_th, get_value_threshold, find_sup_th, sup_unsup_prediction_spec_th_manual

args = get_test_parser()

def initail_setting(with_sup_model=False):
  
    config = json.load(open(args.config))

    # ===== dataset setting =====
    if args.mask is not None:
        config['data_loader']['mask'] = args.mask

    if args.size is not None:
        config['data_loader']['w'] = config['data_loader']['h'] = args.size

    if args.normal_path is not None:
        config['data_loader']['test_data_root_normal'] = args.normal_path

    if args.smura_path is not None:
        config['data_loader']['test_data_root_smura'] = args.smura_path

    if args.dataset_name is not None:
        config['data_loader']['name'] = args.dataset_name

    if args.dataset_path is not None:
        config['data_loader']['data_root'] = args.dataset_path

    if args.normal_num is not None:
        config['data_loader']['normal_num'] = args.normal_num

    if args.normal_num is not None:
        config['data_loader']['smura_num'] = args.smura_num

    # ===== model setting =====
    config['model_name'] = args.model_name

    if args.model_version is not None:
        config['model']['version'] = args.model_version

    # ===== Test setting =====
    config['model_epoch'] = args.model_epoch
    config['test_type'] = args.test_type
    config['anomaly_score'] = args.anomaly_score
    config['pos_normalized'] = args.pos_normalized
    config['minmax'] = args.minmax
    config['using_record'] = args.using_record
    config['using_threshold'] = args.using_threshold

    # ===== Path setting =====
    if args.csv_path is not None:
        config['data_loader']['csv_path'] = args.csv_path

    config['save_dir'] = os.path.join(config['save_dir'], '{}_{}_{}{}'.format(config['model_name'], 
    config['model']['version'], config['data_loader']['mask'], config['data_loader']['w']))

    result_path = os.path.join(config['save_dir'], '{}_results_{}_{}_with_sup'.format(config['data_loader']['name'], str(config['model_epoch']).zfill(5), config['anomaly_score']))
    
    mkdir(result_path)

    config['result_path'] = result_path

    # ===== GPU setting =====
    gpu = args.gpu_id 
    torch.cuda.set_device(gpu)
    print(f'using GPU device {gpu} for testing ... ')

    # ===== Seed setting =====
    set_seed(config['seed'])

    return config, gpu

def count_data_version(large_smura_name, small_smura_name, csv_path):
    df_large_name = pd.DataFrame(large_smura_name, columns=['name'])
    print(df_large_name)
    df_small_name = pd.DataFrame(small_smura_name, columns=['name'])
    print(df_small_name)

    df_data = pd.read_csv(csv_path)

    df_merge_large = df_data.merge(df_large_name, left_on='name', right_on='name')
    print(df_merge_large['batch'].value_counts())
    df_merge_small = df_data.merge(df_small_name, left_on='name', right_on='name')
    print(df_merge_small['batch'].value_counts())

def show_and_save_result(conf_sup, score_unsup, use_th, path, name):
    all_conf_sup = np.concatenate([conf_sup['conf']['n'], conf_sup['conf']['s']])
    all_score_unsup = np.concatenate([score_unsup['score']['n'], score_unsup['score']['s']])

    true_label = np.concatenate([conf_sup['label']['n'], conf_sup['label']['s']])

    plot_score_distribution(score_unsup['score']['n'], score_unsup['score']['s'], path, f"{name}_unsup")
    plot_sup_unsup_scatter(conf_sup, score_unsup, path, name)

    if use_th:
        sup_unsup_SVM_test(true_label, all_conf_sup, all_score_unsup, path)
        sup_unsup_DT_test(true_label, all_conf_sup, all_score_unsup, path)
        
        # # ===== blind test =====
        # value_th = get_value_threshold(path)
        # one_line_th, two_line_th = get_line_threshold(path)
        # spec_line_th = {"m":32000, "b":2.365}

        # log_name = os.path.join(path, f'{result_name}_blind_test_result_log.txt')
        # msg = ''
        # with open(log_name, "w") as log_file:
        #     msg += f"=============== supervised ===================\n"
        #     msg += sup_prediction_spec_th(true_label, all_conf_sup, value_th, path)
        #     msg += f"=============== unsupervised ===================\n"
        #     msg += f"Normal mean: {score_unsup['all']['n'].mean()}\n"
        #     msg += f"Normal std: {score_unsup['all']['n'].std()}\n"
        #     msg += f"Smura mean: {score_unsup['all']['s'].mean()}\n"
        #     msg += f"Smura std: {score_unsup['all']['s'].std()}\n"
        #     msg += f"=============== Combine both one line ===================\n"
        #     msg += sup_unsup_prediction_spec_th(true_label, all_conf_sup, all_score_unsup, one_line_th, path)
        #     msg += f"=============== Combine both two lines ===================\n"
        #     msg += sup_unsup_prediction_spec_multi_th(true_label, all_conf_sup, all_score_unsup, two_line_th, path)
        #     msg += f"=============== Manual both one lines ===================\n"
        #     msg += sup_unsup_prediction_spec_th_manual(true_label, all_conf_sup, all_score_unsup, spec_line_th, path)
        #     log_file.write(msg)
    else:
        sup_unsup_SVM(true_label, all_conf_sup, all_score_unsup, path)
        # sup_unsup_DT(true_label, all_conf_sup, all_score_unsup, path)

    #     sup_res = find_sup_th(conf_sup, path)
    #     # ===== Auto find threshold line =====
    #     one_res, one_line_time = sup_unsup_prediction_auto_th(true_label, all_conf_sup, all_score_unsup, path)
    #     two_res, two_line_time = sup_unsup_prediction_auto_multi_th(true_label, all_conf_sup, all_score_unsup, path)
    #     sup_unsup_svm(true_label, all_conf_sup, all_score_unsup, path)
    #     log_name = os.path.join(path, f'{result_name}_find_th_log.txt')
    #     msg = ''
    #     with open(log_name, "w") as log_file:
    #         msg += f"=============== supervised ===================\n"
    #         msg += f"tnr0.987 recall: {sup_res['tnr0.987_recall']}\n"
    #         msg += f"tnr0.987 precision: {sup_res['tnr0.987_precision']}\n"
    #         msg += f"tnr0.996 recall: {sup_res['tnr0.996_recall']}\n"
    #         msg += f"tnr0.996 precision: {sup_res['tnr0.996_precision']}\n"
    #         msg += f"tnr0.998 recall: {sup_res['tnr0.998_recall']}\n"
    #         msg += f"tnr0.998 precision: {sup_res['tnr0.998_precision']}\n"
    #         msg += f"=============== one line ===================\n"
    #         msg += f"one line time: {one_line_time}\n"
    #         msg += f"tnr0.987 recall: {one_res['tnr0.987_recall']}\n"
    #         msg += f"tnr0.987 precision: {one_res['tnr0.987_precision']}\n"
    #         msg += f"tnr0.996 recall: {one_res['tnr0.996_recall']}\n"
    #         msg += f"tnr0.996 precision: {one_res['tnr0.996_precision']}\n"
    #         msg += f"tnr0.998 recall: {one_res['tnr0.998_recall']}\n"
    #         msg += f"tnr0.998 precision: {one_res['tnr0.998_precision']}\n"
    #         msg += f"=============== two line ===================\n"
    #         msg += f"two line time: {two_line_time}\n"
    #         msg += f"tnr0.987 recall: {two_res['tnr0.987_recall']}\n"
    #         msg += f"tnr0.987 precision: {two_res['tnr0.987_precision']}\n"
    #         msg += f"tnr0.996 recall: {two_res['tnr0.996_recall']}\n"
    #         msg += f"tnr0.996 precision: {two_res['tnr0.996_precision']}\n"
    #         msg += f"tnr0.998 recall: {two_res['tnr0.998_recall']}\n"
    #         msg += f"tnr0.998 precision: {two_res['tnr0.998_precision']}\n"
    #         log_file.write(msg)

    # plot_line_on_scatter(conf_sup, score_unsup, path)

def model_prediction_using_record(config):
    res_sup = defaultdict(dict)
    for l in ['conf','label','files']:
        for t in ['n','s']:
            res_sup[l][t] = None

    res_unsup = defaultdict(dict)
    for l in ['score','label','files', 'all']:
        for t in ['n','s']:
            res_unsup[l][t] = None

    sup_df = pd.read_csv(os.path.join(config['result_path'], 'sup_conf.csv'))
    unsup_df = pd.read_csv(os.path.join(config['result_path'], 'unsup_score_mean.csv'))
    merge_df = sup_df.merge(unsup_df, left_on='name', right_on='name')
    
    normal_filter = (merge_df['label_x']==0) & (merge_df['label_y']==0)
    smura_filter = (merge_df['label_x']==1) & (merge_df['label_y']==1)
    print(merge_df)
    
    # res_sup = defaultdict(dict)
    for l, c in zip(['conf','label','files'],['conf','label_x','name']):
        for t, f in zip(['n', 's'],[normal_filter,smura_filter]):
            res_sup[l][t] = np.array(merge_df[c][f].tolist())
    # print(res_sup['files']['n'][:10])

    # res_unsup = defaultdict(dict)
    for l, c in zip(['score','label','files'],['score_mean','label_y','name']):
        for t, f in zip(['n', 's'],[normal_filter, smura_filter]):
            res_unsup[l][t] = np.array(merge_df[c][f].tolist())
    # print(res_unsup['files']['n'][:10])
    
    all_df = pd.read_csv(os.path.join(config['result_path'], 'unsup_score_all.csv'))
    normal_filter = (all_df['label']==0)
    smura_filter = (all_df['label']==1)
    res_unsup['all']['n'] = np.array(all_df['score'][normal_filter].tolist())
    res_unsup['all']['s'] = np.array(all_df['score'][smura_filter].tolist())
    
    return res_sup, res_unsup

if __name__ == '__main__':
  
    with_sup_model = True
    config, _ = initail_setting(with_sup_model)  

    res_sup, res_unsup = model_prediction_using_record(config)

    result_name = f"{config['data_loader']['name']}_crop{config['data_loader']['crop_size']}_{config['anomaly_score']}_epoch{config['model_epoch']}_with_seresnext101"
    show_and_save_result(res_sup, res_unsup, config['using_threshold'], config['result_path'], result_name)
    