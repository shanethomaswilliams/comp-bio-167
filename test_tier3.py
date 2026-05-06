from src.dataloaders import EnsembleData
from src.models.tier3 import SoftMoEMethod, HardRoutingMethod

PROJECT = "/Users/shanewilliams/GradSchool/Spring2026/CompBio/final_project/comp-bio-167"

data = EnsembleData.from_merged_dir(
    "data/final_data",
    embeddings_path=f"{PROJECT}/data/embeddings/embeddings.h5",
)
train_df = data.get_train_raw()
test_df  = data.get_test_raw()

# ---- smoke test: subsample proteins that have embeddings ----
# Sample whole proteins, not rows, so HardRouting still has enough
# annotations per protein to compute best-team labels.
import numpy as np
rng = np.random.default_rng(42)
covered_proteins = set(data._load_embeddings().keys())
train_proteins = train_df['protein_id'].unique()
eligible = np.array([p for p in train_proteins if p in covered_proteins])
sample = rng.choice(eligible, size=min(2000, len(eligible)), replace=False)

train_small = train_df[train_df['protein_id'].isin(sample)].reset_index(drop=True)
test_small  = test_df.head(50_000).reset_index(drop=True)  # arbitrary cap
print(f"smoke train: {len(train_small):,} rows / {len(sample):,} proteins")

# ---- tiny, fast models ----
fast_kwargs = dict(
    hidden=(64, 32),
    epochs=5,          # was 30/50
    patience=2,        # was 5
    batch_size=1024,   # smaller = more frequent feedback on MPS
    val_frac=0.1,
    fallback='max',
)

soft = SoftMoEMethod(entropy_reg=0.01, **fast_kwargs)
soft.fit(train_small, data.train_terms, data.ia_weights, embeddings=data)
preds_soft = soft.predict(test_small, embeddings=data)

hard = HardRoutingMethod(min_annotations=3, **fast_kwargs)
hard.fit(train_small, data.train_terms, data.ia_weights, embeddings=data)
preds_hard = hard.predict(test_small, embeddings=data)

print(soft.train_coverage_)
print(hard._team_distribution_)

# Quick sanity on outputs
print(preds_soft['confidence'].describe())
print(preds_hard['confidence'].describe())