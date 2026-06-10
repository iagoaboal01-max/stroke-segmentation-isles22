#!/bin/bash
#SBATCH -J stregm_adamw
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -N 1
#SBATCH --time=12:00:00
#SBATCH -o /home/%u/LOGS/stregm_adamw_%j.out
#SBATCH -e /home/%u/LOGS/stregm_adamw_%j.err

set -euo pipefail
mkdir -p $HOME/LOGS

module load conda
conda activate aabi

echo "============================================================"
echo "Job ID:  $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo "============================================================"

DATASET_PATH=$DATA/ISLES-2022
CODE_DIR=$HOME/stregmentation_adamw
EXP_NAME=adamw_80x128x128_wd5e5_$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=$DATA/runs/$EXP_NAME

mkdir -p $OUTPUT_DIR
cd $CODE_DIR

python train_cluster_adamw.py \
  --base_path $DATASET_PATH/ \
  --csv paths_isles_2022.csv \
  --modalities dwi_path adc_path \
  --image_size 80 128 128 \
  --features 64 128 256 512 \
  --epochs 100 \
  --batch_size 2 \
  --lr 1e-4 \
  --weight_decay 5e-5 \
  --num_workers 8 \
  --threshold 0.5 \
  --output_dir $OUTPUT_DIR \
  --seed 67

mkdir -p $HOME/results
cp -r $OUTPUT_DIR $HOME/results/

echo "Results copied to $HOME/results/$EXP_NAME"
echo "Finished: $(date)"
