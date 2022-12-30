import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

# df = pd.read_csv('/home/ldap/sallylin/Howard/Mura_PEN-Net/release_model/pennet_d23_8k_square512/temp/mura_d24_d25_8k_results_00100_Mask_MSE_with_sup/unsup_score.csv')
# normal_filter = df.label == 0
# outlier_filter = df.score > 7e-05
# print(df[normal_filter & outlier_filter])

# d23_sup_path = r'./d23/sup_conf.csv'
# d23_unsup_path = r'./d23/unsup_score.csv'

# d2425_sup_path = r'./d2425/sup_conf.csv'
# d2425_unsup_path = r'./d2425/unsup_score.csv'

# def model_prediction_using_record(sup_path, unsup_path, filter=False):
#     res_sup = defaultdict(dict)
#     for l in ['conf','labels','files']:
#         for t in ['n','s']:
#             res_sup[l][t] = None

#     res_unsup = defaultdict(dict)
#     for l in ['score','labels','files']:
#         for t in ['n','s']:
#             res_unsup[l][t] = None

#     sup_df = pd.read_csv(sup_path)
#     unsup_df = pd.read_csv(unsup_path)
#     merge_df = sup_df.merge(unsup_df, left_on='name', right_on='name')
    
#     normal_filter = (merge_df['label_x']==0) & (merge_df['label_y']==0)
#     smura_filter = (merge_df['label_x']==1) & (merge_df['label_y']==1)
#     # print(merge_df['score_y'])
    
#     if filter:
#         # 拿掉d2425 outlier
#         filt = (merge_df['name'] != '6A2D62V0HAZZ_20220726020448_0_L050P_resize.png') & (merge_df['name'] != '6A2D62V0PAZZ_20220726045602_0_L050P_resize.png')
#         merge_df = merge_df[filt]

#     for l, c in zip(['conf','labels','files'],['score_x','label_x','name']):
#         for t, f in zip(['n', 's'],[normal_filter,smura_filter]):
#             res_sup[l][t] = merge_df[c][f].tolist()
#     # print(res_sup['files']['n'][:10])

#     for l, c in zip(['score','labels','files'],['score_y','label_y','name']):
#         for t, f in zip(['n', 's'],[normal_filter, smura_filter]):
#             res_unsup[l][t] = merge_df[c][f].tolist()
#     # print(res_unsup['files']['n'][:10])
    
#     return res_sup, res_unsup

# def plot_score_scatter(score_1_x, score_1_y, score_2_x, score_2_y, colors, labels, name):
#     # 設定座標軸
#     # all
#     plt.xlim(3e-05, 1.2e-04)
#     plt.xlabel("unsup")
#     plt.ylabel("sup")
#     plt.title('scatter')
#     plt.scatter(score_1_x, score_1_y, s=3, c =colors[0], alpha=0.5, label=labels[0])
#     plt.scatter(score_2_x, score_2_y, s=3, c =colors[1], alpha=0.5, label=labels[1])
#     plt.legend(loc='lower right')
#     plt.savefig(f"./{name}_scatter.png")
#     plt.clf()

# def plot_score_distribution(score_1, score_2, colors, labels, name):
#     plt.xlim(3e-05, 1.2e-04)
#     plt.hist(score_1, bins=50, alpha=0.5, density=True, label=labels[0], color=colors[0])
#     plt.hist(score_2, bins=50, alpha=0.5, density=True, label=labels[1], color=colors[1])
#     plt.xlabel('Anomaly Score')
#     plt.title('Score Distribution')
#     plt.legend(loc='upper right')
#     plt.savefig(f"./{name}_dist_mean.png")
#     plt.clf()

# def plot_score_distribution_all(score_1_n, score_1_s, score_2_n, score_2_s, colors, labels, name):
#     plt.hist(score_1_n, bins=50, alpha=0.5, density=True, label=labels[0], color=colors[0])
#     plt.hist(score_1_s, bins=50, alpha=0.5, density=True, label=labels[1], color=colors[1])
#     plt.hist(score_2_n, bins=50, alpha=0.5, density=True, label=labels[2], color=colors[2])
#     plt.hist(score_2_s, bins=50, alpha=0.5, density=True, label=labels[3], color=colors[3])
#     plt.xlabel('Anomaly Score')
#     plt.title('Score Distribution')
#     plt.legend(loc='upper right')
#     plt.savefig(f"./{name}_dist_mean.png")
#     plt.clf()

# if __name__ == '__main__':
#     d23_sup, d23_unsup = model_prediction_using_record(d23_sup_path, d23_unsup_path)
#     d2425_sup, d2425_unsup = model_prediction_using_record(d2425_sup_path, d2425_unsup_path)
#     d2425_sup_no_outlier, d2425_unsup_no_outlier = model_prediction_using_record(d2425_sup_path, d2425_unsup_path, filter=True)

#     # MSE analysis
#     msg = ''
#     with open('./MSE_result', 'w') as file:
#         msg += "-----normal-----\n"
#         msg += f"d23 normal mean = {np.array(d23_unsup['score']['n']).mean()}, std = {np.array(d23_unsup['score']['n']).std()}\n"
#         msg += f"d2425 normal mean = {np.array(d2425_unsup['score']['n']).mean()}, std = {np.array(d2425_unsup['score']['n']).std()}\n"
#         msg += f"d2425 normal mean (no sumra outlier) = {np.array(d2425_unsup_no_outlier['score']['n']).mean()}, std = {np.array(d2425_unsup_no_outlier['score']['n']).std()}\n"
#         msg += "-----smura-----\n"
#         msg += f"d23 smura mean = {np.array(d23_unsup['score']['s']).mean()}, std = {np.array(d23_unsup['score']['s']).std()}\n"        
#         msg += f"d2425 smura mean = {np.array(d2425_unsup['score']['s']).mean()}, std = {np.array(d2425_unsup['score']['s']).std()}\n"
#         msg += f"d2425 smura mean (no sumra outlier) = {np.array(d2425_unsup_no_outlier['score']['s']).mean()}, std = {np.array(d2425_unsup_no_outlier['score']['s']).std()}\n"
#         file.write(msg)
    
#     # 視覺化分析
#     colors_1 = ['#ff7f0e', '#8B4513'] # smura 比較用
#     colors_2 = ['#1f77b4', '#483D8B'] # normal 比較用
#     colors_3 = ['#1f77b4', '#ff7f0e'] # normal vs. smura 用
#     labels_1 = ['d23', 'd2425']
#     labels_2 = ['normal', 'smura']

#     colors_all = ['#1f77b4', '#ff7f0e', '#483D8B', '#8B4513']
#     labels_all = ['d23_normal', 'd23_smura', 'd2425_normal', 'd2425_smura']
#     # histogram
#     plot_score_distribution(np.array(d23_unsup['score']['n']), np.array(d23_unsup['score']['s']), colors_3, labels_2, 'd23')
#     plot_score_distribution(np.array(d2425_unsup['score']['n']), np.array(d2425_unsup['score']['s']), colors_3, labels_2, 'd2425')
#     plot_score_distribution(np.array(d2425_unsup_no_outlier['score']['n']), np.array(d2425_unsup_no_outlier['score']['s']), colors_3, labels_2, 'd2425_no_outlier')

#     plot_score_distribution(np.array(d23_unsup['score']['n']), np.array(d2425_unsup['score']['n']), colors_2, labels_1, 'normal_d23_vs_d2425')
#     plot_score_distribution(np.array(d23_unsup['score']['s']), np.array(d2425_unsup['score']['s']), colors_1, labels_1, 'smura_d23_vs_d2425')
#     plot_score_distribution(np.array(d23_unsup['score']['n']), np.array(d2425_unsup_no_outlier['score']['n']), colors_2, labels_1, 'normal_d23_vs_d2425_no_outlier')
#     plot_score_distribution(np.array(d23_unsup['score']['s']), np.array(d2425_unsup_no_outlier['score']['s']), colors_1, labels_1, 'smura_d23_vs_d2425_no_outlier')
#     plot_score_distribution_all(np.array(d23_unsup['score']['n']), np.array(d23_unsup['score']['s']), np.array(d2425_unsup['score']['n']), np.array(d2425_unsup['score']['s']), colors_all, labels_all, 'd23_vs_d2425')
#     #scatter
#     plot_score_scatter(np.array(d23_unsup['score']['n']), np.array(d23_sup['conf']['n']), np.array(d23_unsup['score']['s']), np.array(d23_sup['conf']['s']), colors_3, labels_2, 'd23')
#     plot_score_scatter(np.array(d2425_unsup['score']['n']), np.array(d2425_sup['conf']['n']), np.array(d2425_unsup['score']['s']), np.array(d2425_sup['conf']['s']), colors_3, labels_2, 'd2425')
#     plot_score_scatter(np.array(d2425_unsup_no_outlier['score']['n']), np.array(d2425_sup_no_outlier['conf']['n']), np.array(d2425_unsup_no_outlier['score']['s']), np.array(d2425_sup_no_outlier['conf']['s']), colors_3, labels_2, 'd2425_no_outlier')

#     plot_score_scatter(np.array(d23_unsup['score']['n']), np.array(d23_sup['conf']['n']), np.array(d2425_unsup['score']['n']), np.array(d2425_sup['conf']['n']), colors_2, labels_1, 'normal_d23_vs_d2425')
#     plot_score_scatter(np.array(d23_unsup['score']['s']), np.array(d23_sup['conf']['s']), np.array(d2425_unsup['score']['s']), np.array(d2425_sup['conf']['s']), colors_1, labels_1, 'smura_d23_vs_d2425')
#     plot_score_scatter(np.array(d23_unsup['score']['n']), np.array(d23_sup['conf']['n']), np.array(d2425_unsup_no_outlier['score']['n']), np.array(d2425_sup_no_outlier['conf']['n']), colors_2, labels_1, 'normal_d23_vs_d2425_no_outlier')
#     plot_score_scatter(np.array(d23_unsup['score']['s']), np.array(d23_sup['conf']['s']), np.array(d2425_unsup_no_outlier['score']['s']), np.array(d2425_sup_no_outlier['conf']['s']), colors_1, labels_1, 'smura_d23_vs_d2425_no_outlier')



