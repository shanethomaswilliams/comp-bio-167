#!/usr/bin/env bash
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100

module load cuda/12.1
source activate l3d_2024f_cuda12_1

# Env vars expected from launch_experiments.sh:
#   tier          (1|2|3)
#   method        (mean|max|rank_avg|weighted_mean|logreg|xgb|mlp|soft_moe|hard_routing)
#   data_dir
#   results_dir
#   run_name
#   device
#   tune_flag       (empty or "--tune")
#   emb_flag        (empty, or "--embeddings-path PATH [--use-embeddings [--emb-pca N]]")
#   fallback_flag   (empty or "--fallback max|mean")

python -u -m main \
    --tier $tier \
    --method $method \
    --data-dir $data_dir \
    --results-dir $results_dir \
    --run-name $run_name \
    --device $device \
    $tune_flag \
    $emb_flag \
    $fallback_flag
