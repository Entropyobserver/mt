#!/bin/bash -l
#SBATCH -A uppmax2026-1-123
#SBATCH -M pelle
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH -J optuna_en_no
#SBATCH -o logs/optuna_en_no_%j.out
#SBATCH -e logs/optuna_en_no_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /gorilla/proj/uppmax2026-1-123/uppmax2026-1-123/private/yaxj1/conda_envs/mt26

PROJECT_ROOT=/gorilla/proj/uppmax2026-1-123/uppmax2026-1-123/private/yaxj1/mt_oil_no
export HF_CACHE_DIR=/gorilla/proj/uppmax2026-1-123/uppmax2026-1-123/private/yaxj1/hf_cache
export HF_HOME=$HF_CACHE_DIR
export TRANSFORMERS_CACHE=$HF_CACHE_DIR
export HF_DATASETS_CACHE=$HF_CACHE_DIR
export TORCH_HOME=$HF_CACHE_DIR

cd "$PROJECT_ROOT" || exit 1
# Stage 1: Coarse search
python "$PROJECT_ROOT/experiments/en_no_expert/c_optuna_stage1.py"
# Stage 2: Fine-tuning
python "$PROJECT_ROOT/experiments/en_no_expert/c_optuna_stage2.py"

