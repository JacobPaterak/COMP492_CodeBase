#!/bin/bash
# Run this directly on a Fir LOGIN node (never via sbatch) -- compute nodes
# have no internet, and this script needs to reach the Anthropic API.
# Wrap it in tmux/screen or nohup since login-node sessions can be cut off:
#   tmux new -s pseudo_names
#   bash scripts/make_pseudo_names_login.sh

set -euo pipefail

module load StdEnv/2023 python/3.11 cuda cudnn
source ~/seal_env/bin/activate

cd "$(dirname "$0")/.."   # SEAL/

# API key lives in a chmod-600 file outside the repo -- see setup note.
source ~/.secrets/anthropic.env

python make_pseudo_names.py \
    --dataset_name aircraft \
    --model_path dev_outputs/simgcd_baseline/aircraft_seal_clip_stage1_.../checkpoints/model.pt \
    --output pseudo_names.json \
    --naming_model claude-opus-4-8
