#!/bin/bash
model=original_vit_small_patch16_224
img_path=/data/jongseo/project/asd/imae/piano.png
out_path=/data/jongseo/project/asd/imae #/data/jongseo/project/asd/imae/result/grad_cam/small/TD
weight=/data/jongseo/project/asd/imae/result/kfold_cleanup_ver2/pretraining_small_1e-5/OUT/0/checkpoint-19.pth
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

#A11-001-001.jpg  A11-002-002.jpg  A15-001-001.jpg  A15-002-002.jpg  A6-001-001.jpg  A6-002-002.jpg