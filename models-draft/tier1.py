import numpy as np
import pandas as pd
import sys

# SEED = 42
PATH = "cafa-5-protein-function-prediction/"

"""
The main function reads in files from filename on command line
Files are parsed by tabs and read into padas Dataframe
the confidence columns will be merged by protein and GO label index.
"""

def tier1_mean(merged_conf):
    out = merged_conf[['protein_id', 'GO_term']].copy()
    out['confidence'] = merged_conf.filter(like='conf_').mean(axis=1)
    print(out.head())
    return out


# I feel this one has very similar appraoch to logistics regression, which is
# in tier2, and to choose weights we would need to train the model and do cv.
# should we just leave it to logistics regression in tier2 methods?
def tier1_weighted_mean(merged_conf):
    def weighted_mean(weights):
        out = merged_conf[['protein_id', 'GO_term']].copy()
        out['confidence'] = np.average(merged_conf.filter(like='conf_'), weights=weights, axis=1)
        print(out.head())
        return out
    best_score = -1
    best_weights = [1/3, 1/3, 1/3]
    step = 0.1
    for i in range(11):
        for j in range(11 - i):
            w_1 = round(i * step, 2)
            w_2 = round(j * step, 2)
            w_3 = round(1.0 - w_1 - w_2, 2)
            weighted_conf = weighted_mean([w_1, w_2, w_3])
            score = fmax(weighted_conf)
            if score > best_score:
                best_score = score
                best_weights = [w_1, w_2, w_3]
    return weighted_mean(best_weights)


def tier1_max(merged_conf):
    out = merged_conf[['protein_id', 'GO_term']].copy()
    out['confidence'] = merged_conf.filter(like='conf_').max(axis=1)
    print(out.head())
    return out


def tier1_rank_avg(merged_conf):
    def rank_confidence(conf_col):
        vals = np.asarray(conf_col, dtype=float)
        sorted_i = np.argsort(-vals)
        rank = np.empty(len(conf_col), dtype=float)
        rank[sorted_i] = np.arange(1, len(conf_col) + 1)
        sorted_vals = vals[sorted_i]
        i = 0
        while i < len(sorted_vals):
            j = i + 1
            while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
                j += 1
            rank[sorted_i[i:j]] = (i + 1 + j) / 2.0
            i = j
        return rank

    protein_preds = []
    for protein, group in merged_conf.groupby('protein_id'):
        worst_rank = len(group) + 1
        rank_cols = []
        for col in merged_conf.filter(like='conf_').columns:
            conf_col = group[col].values  # from group, not terms
            rank = rank_confidence(conf_col)
            rank_cols.append(np.where(conf_col > 0, rank, worst_rank))

        out = group[['protein_id', 'GO_term']].copy()
        mean_rank = np.mean(rank_cols, axis=0)  # axis=0, not 1
        print(f"{protein} mean_rank{mean_rank}")
        # force the ranks into confidence levels
        out['confidence'] = np.clip(1.0 - (mean_rank - 1.0) / worst_rank, 0.01, 0.99)
        protein_preds.append(out)

    protein_preds = pd.concat(protein_preds, ignore_index=True)
    print(protein_preds.head())
    return protein_preds

"""usage: python tier1.py [test terms predictioin files]"""
if __name__ == "__main__":
    dfs = []
    for i, filename in enumerate(sys.argv[1:]):
        df = pd.read_csv(PATH + filename,
                         sep='\t',
                         header=None,
                         names=['protein_id', 'GO_term', 'conf_'+str(i)])
        df = df.set_index(['protein_id', 'GO_term'])
        dfs.append(df)

    merged_conf = pd.concat(dfs, axis=1, join='outer').fillna(0.0).reset_index()

    print("mean")
    mean_conf = tier1_mean(merged_conf[:6])
    # print("weighted mean")
    # weighted_mean_conf = tier1_weighted_mean(merged_conf[:6])
    print("max")
    max_conf = tier1_max(merged_conf[:6])
    print("rank_avg")
    rank_avg_conf = tier1_rank_avg(merged_conf[:6])



