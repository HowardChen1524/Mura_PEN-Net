#!/bin/bash

if [ -z "$pyenv" ]
then
  pyenv=$(which python3)
fi

if [ -z "$tb_folder" ]
then
  tb_folder="0801"
fi

if [ -z "$size" ]
then
  size="256"
fi

if [ -z "$pretrain" ]
then
  pretrain="False"
fi

if [ -z "$DATA_VERSION" ]
then
  DATA_VERSION="d23_256_tjwei"
fi

LOOP=3
dataset="/home/sylvia/Documents/AUO/ai9_dataset/d23/data_merged.csv"
epoch=150

echo "***********************"
echo "Python interpreter: $pyenv"
echo "Torch version : $($pyenv -c 'import torch; print(torch.__version__)')"
echo "tb_folder: $tb_folder"
echo "pretrain: $pretrain"
echo "size: $size"
echo "***********************"

echo "This script will benchmark with each model [cnn,vgg16,resnet50,cnn,xception,mobilenet_v2] three time."
sleep 3

##########################################
##########################################
# se-resnext101
model="seresnext101"

for (( i = 0; i < $LOOP; i++ ))
do
$pyenv AI9_train_dvc_wei.py \
--dataset $dataset \
--batch_size 16 --epoch $epoch --lr 6e-4 \
--focalloss_alpha 0.75 --focalloss_gamma 2.0 \
--img_size "$size" \
--model "$model" \
--pretrain "$pretrain" \
--optimizer ADAM \
--output ./"$tb_folder"/"$model"_"$size"_"$DATA_VERSION"_"$i"/result.json \
--tensorboard_path ./"$tb_folder"/"$model"_"$size"_"$DATA_VERSION"_"$i" \
--log_path ./"$tb_folder"/ \
--seed $i
done

# ##########################################
#resnet50
# model="resnet50"

# for (( i = 0; i < 1; i++ ))
# do
# $pyenv AI9_train_dvc_wei.py \
# --dataset $dataset \
# --batch_size 20 --epoch $epoch --lr 5e-5 \
# --focalloss_alpha 0.75 --focalloss_gamma 2.0 \
# --img_size "$size" \
# --model "$model" \
# --pretrain "$pretrain" \
# --optimizer ADAM \
# --output ./"$tb_folder"/"$model"_"$size"_"$DATA_VERSION"_"$i"/result.json \
# --tensorboard_path ./"$tb_folder"/"$model"_"$size"_"$DATA_VERSION"_"$i" \
# --log_path ./"$tb_folder"/ \
# --seed $i
# done

# ##########################################
# ##########################################
# # xception
# model="xception"

# for (( i = 0; i < 1; i++ ))
# do
# $pyenv AI9_train_dvc_wei.py \
# --dataset $dataset \
# --batch_size 20 --epoch $epoch --lr 0.00015 \
# --focalloss_alpha 0.75 --focalloss_gamma 2.0 \
# --img_size "$size" \
# --model "$model" \
# --pretrain "$pretrain" \
# --optimizer ADAM \
# --output ./"$tb_folder"/"$model"_"$size"_"$DATA_VERSION"_"$i"/result.json \
# --tensorboard_path ./"$tb_folder"/"$model"_"$size"_"$DATA_VERSION"_"$i" \
# --log_path ./"$tb_folder"/ \
# --seed $i
# done

# ##########################################
# model="vit"

# for (( i = 0; i < $LOOP; i++ ))
# do
# $pyenv AI9_train_dvc_wei.py \
# --dataset $dataset \
# --batch_size 20 --epoch $epoch --lr 0.001 \
# --img_size "$size" \
# --model "$model" \
# --pretrain "$pretrain" \
# --optimizer ADAM \
# --output ./"$tb_folder"/"$model"_"$size"_"$DATA_VERSION"_"$i"/result.json \
# --tensorboard_path ./"$tb_folder"/"$model"_"$size"_"$DATA_VERSION"_"$i" \
# --seed $i
# done



# ##########################################
# # resnet50
# model="resnet50"

# for (( i = 0; i < $LOOP; i++ ))
# do
# $pyenv AI9_train_dvc_wei.py \
# --dataset $dataset \
# --batch_size 20 --epoch $epoch --lr 5e-5 \
# --img_size "$size" \
# --model "$model" \
# --pretrain "$pretrain" \
# --optimizer ADAM \
# --output ./"$tb_folder"/"$model"_"$size"_"$DATA_VERSION"_"$i"/result.json \
# --tensorboard_path ./"$tb_folder"/"$model"_"$size"_"$DATA_VERSION"_"$i" \
# --seed $i
# done
