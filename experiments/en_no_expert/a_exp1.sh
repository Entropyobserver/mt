#!/bin/bash -l
#SBATCH -A uppmax2026-1-123
#SBATCH -M pelle
#SBATCH -p gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -t 04:00:00
#SBATCH --array=0-26
#SBATCH -J exp1_scaling
#SBATCH -o logs/exp1-%A_%a.out
#SBATCH -e logs/exp1-%A_%a.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /gorilla/proj/uppmax2026-1-123/uppmax2026-1-123/private/yaxj1/conda_envs/mt26

PROJECT_ROOT=/gorilla/proj/uppmax2026-1-123/uppmax2026-1-123/private/yaxj1/mt_oil_no
export HF_CACHE_DIR=/gorilla/proj/uppmax2026-1-123/uppmax2026-1-123/private/yaxj1/hf_cache
export HF_HOME=$HF_CACHE_DIR
export TRANSFORMERS_CACHE=$HF_CACHE_DIR
export HF_DATASETS_CACHE=$HF_CACHE_DIR/cache_${SLURM_ARRAY_TASK_ID}
export TORCH_HOME=$HF_CACHE_DIR

cd "$PROJECT_ROOT" || exit 1
mkdir -p logs
python "$PROJECT_ROOT/experiments/en_no_expert/run_exp1.py" --job_id "$SLURM_ARRAY_TASK_ID"
