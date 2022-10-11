import torch
import torch.nn as nn
from torch.optim import Adagrad, SGD, Adam
import random
import json
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
import pandas as pd
import tensorflow as tf
from PIL import ImageFile
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

# from memory_profiler import profile
# import gc
import shutil
########################################
from utils.models import get_model
from utils.wei_dataloader import make_test_dataloader, make_training_dataloader, AI9_Dataset, data_transforms
from utils.loss import WeightedFocalLoss


########################################

def setup():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    
    ImageFile.LOAD_TRUNCATED_IMAGES = False

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)


def init_writer(tbpath):
    os.makedirs(tbpath, exist_ok=True)
    summary_writer = SummaryWriter(tbpath)
    print("$ tensorboard --logdir={} --port 8123 --host 0.0.0.0 \n".format(os.path.dirname(tbpath)))
    print("\nName: {}".format(tbpath.split("/")[-1]))
    return summary_writer

def get_data_info(t, l, image_info):
    res = []
    image_info = image_info[(image_info["train_type"] == t) & (image_info["label"] == l)]
        
    for path, img, label, JND, t in zip(image_info["path"],image_info["name"],image_info["label"],image_info["MULTI_JND"],image_info["train_type"]):
        img_path = os.path.join(os.path.dirname(csv_path), path,img)
        res.append([img_path, label, JND, t, img])
    X = []
    Y = []
    N = []
    
    for d in res:
        # dereference ImageFile obj
        X.append(os.path.join(d[0]))
        Y.append(d[1])
        N.append(d[4])
    dataset = AI9_Dataset(feature=X,
                          target=Y,
                          name=N,
                          transform=data_transforms[t])
    return dataset


def train(model, trainloader, optimizer, loss_function):
    model.train().cuda()
    losses = []
    correct = 0
    total_data_in_loader = 0
    # i =0
    for x, y, n in tqdm(trainloader):
        # print(i)
        x = x.to(torch.float32).cuda()
        y = y.to(torch.float32).cuda()
        # print(f'x = {x}\ny = {y}')
        optimizer.zero_grad()
        output = model(x)
        output = torch.reshape(output, (-1,))
        loss = loss_function(output, y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        preds_tensor = torch.Tensor([1 if x >= 0.5 else 0 for x in output]).cuda()
        correct += preds_tensor.eq(y).sum().item()

        total_data_in_loader += len(y)
        # i +=1

    losses = sum(losses) / len(losses)
    acc = correct / total_data_in_loader
    return losses, acc


def confusion_add(cm, output, target):
    # cm = {"TP":0 ,"TN":0 , "FP":0 ,"FN":0}
    for pre, tar in zip(output, target):
        if pre == tar and int(pre) == 1:
            cm["TP"] += 1
        elif pre == tar and int(pre) == 0:
            cm["TN"] += 1
        elif pre > tar:
            cm["FP"] += 1
        elif tar > pre:
            cm["FN"] += 1


def test(model, testloader, loss_function):
    cm = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    model.eval().cuda()
    losses = []
    correct = 0
    total_data_in_loader = 0
    with torch.no_grad():
        for x, y, n in tqdm(testloader):
            x = x.to(torch.float32).cuda()
            y = y.to(torch.float32).cuda()
            output = model(x)
            output = torch.reshape(output, (-1,))

            loss = loss_function(output, y)
            losses.append(loss.item())

            preds_tensor = torch.Tensor([1 if x >= 0.5 else 0 for x in output]).cuda()

            confusion_add(cm, output=preds_tensor.cpu().tolist(), target=y.cpu().tolist())
            correct += preds_tensor.eq(y).sum().item()
            total_data_in_loader += len(y)

    losses = sum(losses) / len(losses)
    acc = correct / total_data_in_loader
    return losses, acc, cm

def plot_roc_curve(labels_res, preds_res):
    fpr, tpr, threshold = metrics.roc_curve(y_true=labels_res, y_score=preds_res)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    return plt

def calc_metric(labels_res, pred_res, threshold):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true=labels_res, y_pred=(pred_res >= threshold)).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return threshold, tpr, tnr, precision, recall

def get_curve_df(labels_res, preds_res):
    pr_list = []

    for i in tqdm(np.linspace(0, 1, num=10001)):
        pr_result = calc_metric(labels_res, preds_res, i)
        pr_list.append(pr_result)

    curve_df = pd.DataFrame(pr_list, columns=['threshold', 'tpr', 'tnr', 'precision', 'recall'])
    
    return curve_df

def calc_matrix(labels_res, preds_res):
    results = {'accuracy': [],
           'balance_accuracy': [],
           'tpr': [],
           'tnr': [],
           'tnr0.99_precision': [],
           'tnr0.99_recall': [],
           'tnr0.995_precision': [],
           'tnr0.995_recall': [],
           'tnr0.999_precision': [],
           'tnr0.999_recall': [],
           'tnr0.9996_precision': [],
           'tnr0.9996_recall': [],
           'precision': [],
           'recall': []
    }

    tn, fp, fn, tp = metrics.confusion_matrix(y_true=labels_res, y_pred=(preds_res >= 0.5)).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (fp + tn)
    fnr = fn / (tp + fn)
    fpr = fp / (fp + tn)

    results['accuracy'].append((tn + tp) / (tn + fp + fn + tp))
    results['tpr'].append(tpr)
    results['tnr'].append(tnr) 
    results['balance_accuracy'].append(((tp / (tp + fn) + tn / (tn + fp)) / 2))
    results['precision'].append(tp / (tp + fp))
    results['recall'].append(tp / (tp + fn))

    curve_df = get_curve_df(labels_res, preds_res)
    results['tnr0.99_recall'].append((((curve_df[curve_df['tnr'] > 0.99].iloc[0]) + (curve_df[curve_df['tnr'] < 0.99].iloc[-1])) / 2).recall)
    results['tnr0.995_recall'].append((((curve_df[curve_df['tnr'] > 0.995].iloc[0]) + (curve_df[curve_df['tnr'] < 0.995].iloc[-1])) / 2).recall)
    results['tnr0.99_precision'].append((((curve_df[curve_df['tnr'] > 0.99].iloc[0]) + (curve_df[curve_df['tnr'] < 0.99].iloc[-1])) / 2).precision)
    results['tnr0.995_precision'].append((((curve_df[curve_df['tnr'] > 0.995].iloc[0]) + (curve_df[curve_df['tnr'] < 0.995].iloc[-1])) / 2).precision)
    results['tnr0.999_recall'].append((((curve_df[curve_df['tnr'] > 0.999].iloc[0]) + (curve_df[curve_df['tnr'] < 0.999].iloc[-1])) / 2).recall)
    results['tnr0.999_precision'].append((((curve_df[curve_df['tnr'] > 0.999].iloc[0]) + (curve_df[curve_df['tnr'] < 0.999].iloc[-1])) / 2).precision)
    results['tnr0.9996_recall'].append((((curve_df[curve_df['tnr'] > 0.9996].iloc[0]) + (curve_df[curve_df['tnr'] < 0.9996].iloc[-1])) / 2).recall)
    results['tnr0.9996_precision'].append((((curve_df[curve_df['tnr'] > 0.9996].iloc[0]) + (curve_df[curve_df['tnr'] < 0.9996].iloc[-1])) / 2).precision)
    


    # fill empty slot
    for k, v in results.items():
        if len(v) == 0:
            results[k].append(-1)

    model_report = pd.DataFrame(results).T
    
    return model_report, curve_df

def predict_report(preds, labels, names):
    df_res = pd.DataFrame(list(zip(names, preds)), columns=["Img", "Predict"])
    df_res["Label"] = labels
    return df_res

def evaluate(model, testloader, save_path):
    model.eval().cuda()
    preds_res = []
    labels_res = []
    files_res = []

    with torch.no_grad():
        for inputs, labels, names in tqdm(testloader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            preds = model(inputs)
            
            preds = torch.reshape(preds, (-1,)).cpu()
            labels = labels.cpu()
            
            names = list(names)

            files_res.extend(names)
            preds_res.extend(preds)
            labels_res.extend(labels)

    preds_res = np.array(preds_res)
    labels_res = np.array(labels_res)

    model_pred_result = predict_report(preds_res, labels_res, files_res)
    model_pred_result.to_csv(os.path.join(save_path, "model_pred_result.csv"), index=None)
    print("model predict record finished!")

    fig = plot_roc_curve(labels_res, preds_res)
    fig.savefig(os.path.join(save_path, "roc_curve.png"))
    print("roc curve saved!")

    model_report, curve_df = calc_matrix(labels_res, preds_res)
    model_report.to_csv(os.path.join(save_path, "model_report.csv"))
    curve_df.to_csv(os.path.join(save_path, "model_precision_recall_curve.csv"))
    print("model report record finished!")
    


# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="/home/labpc1/Documents/Project/ai9_train/ai9_dataset/data9/data_merged.csv")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--focalloss_alpha', type=float, default=0.75)
    parser.add_argument('--focalloss_gamma', type=float, default=2.0)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--model', type=str, default="resnet50",
                        choices=["vgg16", "resnet50", "cnn", "xception", "inception_v3", "mobilenet_v2", "vit", "seresnext101", "convit"])
    parser.add_argument("--pretrain", type=str2bool, default=False)
    parser.add_argument('--optimizer', type=str, default="SGD", choices=["SGD", "ADAM"])
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--output', type=str, default="./result.json")
    parser.add_argument('--tensorboard_path', type=str, default="./0322")
    parser.add_argument('--log_path', type=str, default="./0322")
    

    args = parser.parse_args()

    writer = init_writer(args.tensorboard_path) if args.tensorboard_path is not None else None
    
    used_files = ['./AI9_train_dvc_wei.py', './utils/models.py',
              './utils/wei_dataloader.py','./utils/loss.py', 
             './baseline_test.sh']
    
    used_files_path = args.log_path + '/used_files/'
    os.makedirs(used_files_path, exist_ok=True) 
    for used in used_files:
        file_name = str(used).split('/')[-1]
        dst = used_files_path + str(file_name)
        shutil.copyfile(used, dst)
    
    ###########################
    set_seed(args.seed)
    batch_size = args.batch_size
    EPOCH = args.epoch
    ###########################
    setup()
    csv_path = args.dataset
    image_info = pd.read_csv(csv_path)
    # print(image_info)
    ds = defaultdict(dict)
    for x in ["train", "test"]:
        for y in ["mura", "normal"]:
            if y == "mura":
                l = 1
            else:
                l = 0
            ds[x][y] = get_data_info(x, l, image_info)

    dataloaders={}
    dataloaders["train"] = make_training_dataloader(ds)
    dataloaders["test"] = make_test_dataloader(ds)

    # trainloader, testloader = crate_dataloader2(datapath=args.dataset, batch_size=args.batch_size, size=args.img_size)
    # kloader = kfold_loader(trainloader, args.kfold_max)

    model = get_model(model=args.model, size=args.img_size, pretrain=args.pretrain)
    model = model.cuda()

    if args.optimizer == "SGD":
        optimizer = SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == "ADAM":
        optimizer = Adam(model.parameters(), lr=args.lr)
    loss_function = WeightedFocalLoss(alpha=args.focalloss_alpha, gamma=args.focalloss_gamma)
    # loss_function = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(EPOCH), eta_min=0)

    record = {
        "train_image_name": [],
        "valid_image_name": [],
        "log": {},
    }

    for ep in range(EPOCH):
        dataloaders["train"] = make_training_dataloader(ds)
        print("\n Epoch: ({}/{})".format(ep, EPOCH))
        train_loss, train_acc = train(model=model,
                                      trainloader=dataloaders["train"],
                                      optimizer=optimizer,
                                      loss_function=loss_function)


        print("Train loss: {}, acc: {}".format(train_loss, train_acc))

        test_loss, test_acc, test_cm = test(model=model,
                                            testloader=dataloaders["test"],
                                            loss_function=loss_function)
        print("Test loss: {}, acc: {}".format(test_loss, test_acc))
        scheduler.step()

        if ep % 50 == 0:
            torch.save(model.state_dict(), os.path.join(args.tensorboard_path, f"model_{ep}.pt"))
            # dataloaders["train"] = make_training_dataloader(ds)

        record["log"][str(ep)] = {"train_loss": train_loss,
                                  "train_acc": train_acc,
                                  "test_loss": test_loss,
                                  "test_acc": test_acc,
                                  "test_precision": 0 if test_cm["TP"] == 0 else test_cm["TP"] / (
                                          test_cm["TP"] + test_cm["FP"]),
                                  "test_recall": 0 if test_cm["TP"] == 0 else test_cm["TP"] / (
                                          test_cm["TP"] + test_cm["FN"]),
                                  "balanced_accuracy": ((test_cm["TP"] / (test_cm["TP"] + test_cm["FN"])) + (
                                          test_cm["TN"] / (test_cm["TN"] + test_cm["FP"]))) / 2,
                                  }
        record["log"][str(ep)]["balanced_accuracy"] = (record["log"][str(ep)]["test_precision"] +
                                                       record["log"][str(ep)]["test_recall"]) / 2
        record["log"][str(ep)]["FDR"] = 1 - record["log"][str(ep)]["test_precision"]
        record["log"][str(ep)]["leakage_rate"] = 0 if (test_cm["FN"] + test_cm["TN"]) == 0 else test_cm["FN"] / (
                    test_cm["FN"] + test_cm["TN"])
        record["log"][str(ep)]["F1_score"] = 0 if (record["log"][str(ep)]["test_precision"] + record["log"][str(ep)][
            "test_recall"]) == 0 else (
                2 * record["log"][str(ep)]["test_precision"] * record["log"][str(ep)]["test_recall"] / (
                record["log"][str(ep)]["test_precision"] + record["log"][str(ep)]["test_recall"]))
        record["log"][str(ep)]["FPR"] = 0 if (test_cm["FP"] + test_cm["TN"]) == 0 else test_cm["FP"] / (
                    test_cm["FP"] + test_cm["TN"])

        if writer is not None:
            for k in record["log"][str(ep)].keys():
                writer.add_scalar(k, record["log"][str(ep)][k], global_step=ep, walltime=None)

        # this will not record by tensorboard
        record["log"][str(ep)]["confusion_matrix"] = test_cm
        

    torch.save(model.state_dict(), os.path.join(args.tensorboard_path, "model.pt"))

    evaluate(model, dataloaders["test"], args.tensorboard_path)

    with open(args.output, 'w') as f:
        json.dump(record, f, indent=4, sort_keys=True)

