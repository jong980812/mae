#!/bin/bash
model=original_vit_small_patch16_224
img_path=/data/dataset/asd/kfold-cleanup-ver2/hand/0/val/ASD/A15-002-002.jpg  
out_path=/data/jongseo/project/asd/imae/result/grad_cam/mae
weight=/data/jongseo/project/asd/imae/result/kfold/pretrain_80_pth/OUT/0/checkpoint-19.pth
#!----------------
python -u /data/jongseo/project/asd/imae/get_gradcam.py \
    --use-cuda \
    --nb_classes 2 \
    --image_path ${img_path}\
    --model ${model} \
    --finetune ${weight} \
    --output_dir ${out_path} \
    --target_classes 0 \




#ASD:'0'
#TD: '1'