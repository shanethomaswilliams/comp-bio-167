#!/bin/bash
#
# Usage
# -----
# $ bash launch_experiments.sh ACTION_NAME
# where ACTION_NAME is either 'list' or 'submit' or 'run_here'

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi

# Shared config --------------------------------------------------------------
export data_dir="/Users/shanewilliams/GradSchool/Spring2026/CompBio/final_project/comp-bio-167/data/final_data"
export embeddings_path="/Users/shanewilliams/GradSchool/Spring2026/CompBio/final_project/comp-bio-167/data/embeddings/embeddings.h5"
export results_dir="/Users/shanewilliams/GradSchool/Spring2026/CompBio/final_project/comp-bio-167/results"
export run_name="ensemble_$(date +%Y%m%d_%H%M%S)"
export device="auto"
export fallback="max"   # Tier 3: rule for rows whose protein has no embedding

# Tier 1 ---------------------------------------------------------------------
# tier1_methods=(mean max rank_avg)
tier1_methods=()

# Tier 2 ---------------------------------------------------------------------
# tier2_methods=(weighted_mean logreg xgb mlp)
tier2_methods=()
TUNE_TIER2=1

# Enriched Stacking (Tier 2 + embeddings). Set to 1 to additionally run
# these methods with --use-embeddings. Empty array disables.
# tier2_enriched_methods=(logreg xgb mlp)
tier2_enriched_methods=()
EMB_PCA=128  # None by setting to 0 or ""

# Tier 3 ---------------------------------------------------------------------
# Routing-based combiners. Always use embeddings.
# tier3_methods=(soft_moe hard_routing)
tier3_methods=(soft_moe)
TUNE_TIER3=1 

job_count=0

# Helper: set up per-method output dir and submit/run one job.
launch_one() {
    local method_dir="${results_dir}/${run_name}/tier${tier}/${method}${dir_suffix}"
    mkdir -p "$method_dir"
    export slurm_out="${method_dir}/slurm.out"
    export slurm_err="${method_dir}/slurm.err"
    export job_name="cafa5_t${tier}_${method}${dir_suffix}"

    echo "Job $((++job_count)): tier=$tier method=$method${dir_suffix:+ (variant: $dir_suffix)}"
    echo "  flags: tune=\"$tune_flag\"  emb=\"$emb_flag\"  fb=\"$fallback_flag\""
    echo "  -> $method_dir/"

    if [[ $ACTION_NAME == 'submit' ]]; then
        sbatch \
            --job-name="$job_name" \
            --output="$slurm_out" \
            --error="$slurm_err" \
            ./do_training.slurm
        sleep 0.5
    elif [[ $ACTION_NAME == 'run_here' ]]; then
        bash ./do_training.slurm
    fi
}

# ---- Tier 1 -----------------------------------------------------------------

export tier=1
export tune_flag=""
export emb_flag=""
export fallback_flag=""
export dir_suffix=""

for method in "${tier1_methods[@]}"; do
    export method
    launch_one
done

# ---- Tier 2 (vanilla) -------------------------------------------------------

export tier=2
export emb_flag=""
export fallback_flag=""
export dir_suffix=""

if [[ $TUNE_TIER2 == "1" ]]; then
    export tune_flag="--tune"
else
    export tune_flag=""
fi

for method in "${tier2_methods[@]}"; do
    export method
    launch_one
done

# ---- Tier 2 (enriched: conf + ProtT5 PCA) -----------------------------------

export tier=2
# Enriched stacking runs in its own subdir so results don't clobber the vanilla
# tier 2 outputs above.
export dir_suffix="_enriched"
export fallback_flag=""

if [[ -n $EMB_PCA && $EMB_PCA != "0" ]]; then
    export emb_flag="--embeddings-path $embeddings_path --use-embeddings --emb-pca $EMB_PCA"
else
    export emb_flag="--embeddings-path $embeddings_path --use-embeddings"
fi
# Tuning grids may not know about embeddings; leave tune off for enriched runs
# unless you've extended src/finetune.py accordingly.
export tune_flag=""

for method in "${tier2_enriched_methods[@]}"; do
    export method
    launch_one
done

# ---- Tier 3 (routing) ------------------------------------------------------

export tier=3
if [[ $TUNE_TIER3 == "1" ]]; then
    export tune_flag="--tune"
else
    export tune_flag=""
fi
export emb_flag="--embeddings-path $embeddings_path"
export fallback_flag="--fallback $fallback"

for method in "${tier3_methods[@]}"; do
    export method
    launch_one
done

echo "Total jobs: $job_count"