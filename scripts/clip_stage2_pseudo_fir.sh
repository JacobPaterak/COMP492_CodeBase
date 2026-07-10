#!/bin/bash
#SBATCH --account=<FILL_IN>          # e.g. def-yourpi or rrg-yourpi
#SBATCH --gres=gpu:1                  # check `sinfo -o "%N %G"` on Fir for the exact gpu type flag if needed, e.g. gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=seal_clip_stage2
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load StdEnv/2023 python/3.11 cuda cudnn      # adjust versions to whatever your venv was built against

source ~/seal_env/bin/activate                       # path to your venv

cd "$SLURM_SUBMIT_DIR/.."                             # run from SEAL/ (script lives in SEAL/scripts/)

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
# Compute nodes have no internet -- CLIP weights/tokenizer must already be
# cached in $HF_HOME, and pseudo_names.json must already exist from
# make_pseudo_names_login.sh (run on a login node beforehand). This job
# itself never calls the Anthropic API, so no API key is needed here.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python train_seal.py \
    --dataset_name aircraft \
    --model_name vit_clip \
    --clip_model_name openai/clip-vit-base-patch16 \
    --align_weight 1.0 \
    --pseudo_names_path pseudo_names.json \
    --pseudo_align_weight 0.5 \
    --conf_thresh 0.7 \
    --pseudo_refresh_epochs 10 \
    --warmup_model_dir dev_outputs/simgcd_baseline/aircraft_seal_clip_stage1_.../checkpoints/model.pt \
    --batch_size 128 \
    --grad_from_block 10 \
    --epochs 200 \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 1 \
    --exp_name aircraft_seal_clip_stage2 \
    --kl_temp 1.0 \
    --update_thd 0 \
    --memax_weight_1 0 \
    --memax_weight_2 0 \
    --unsupervised_smoothing 0.1
