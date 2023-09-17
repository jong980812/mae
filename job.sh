#!/bin/bash
#SBATCH --job-name replicate 
#SBATCH -w augi3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=35G
#SBATCH --time 6-0
#SBATCH -o logs/asd/replicate/%A.out
#SBATCH -e logs/asd/replicate/%A.err
#SBATCH --partition batch

OUTPUT_DIR=logs/asd/replicate

torchrun --nproc_per_node=1 \
    --master_port=42135 \
    main_finetune_splits.py \
    --output_dir $OUTPUT_DIR \
    --log_dir $OUTPUT_DIR \
    --seed 777 \
    --model efficient \
    --dataset asd \
    --reprob 0.0 \
    --mixup 0.0 \
    --mixup_switch_prob 0.0 \
    --cutmix 0.0 \
    --weight_decay 0.1 \
    --dropout 0.5 \
    --save_ckpt_freq 1 \
    --drop_path 0.1 \
    --dist_eval \
    --warmup_epochs 2 \
    --epochs 30 \
    --num_workers 12 \
    --data_path /local_datasets/asd/All_5split \
    --batch_size 4 \
    --blr 1e-2 \
    --nb_classes 2 \
    --cls_token \
    --max_acc \
    --padding_mode replicate
    # --finetune /data/jong980812/project/mae/result_ai_hub_all/only_body/256_5e-4_no_norm/OUT/01/checkpoint-49.pth