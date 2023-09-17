#!/bin/bash
#SBATCH --job-name per 
#SBATCH -w augi3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=35G
#SBATCH --time 6-0
#SBATCH -o %A.out
#SBATCH -e %A.err
#SBATCH --partition batch

JOB_DIR=/data/geo123/asd_mae/logs/asd/replicate
SAVE_DIR=/data/geo123/asd_mae/logs/asd/replicate

if [ ! -d "$SAVE_DIR" ]; then
  mkdir -p "$SAVE_DIR"
fi 

if [ ! -d "$SAVE_DIR/mask" ]; then
  mkdir -p "$SAVE_DIR/mask"
  mkdir -p "$SAVE_DIR/img"
fi 

python get_torchray.py --job_dir $JOB_DIR --save_dir $SAVE_DIR