import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SEED = 42
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


# Weight search runs on merged_fit (so the chosen weights never saw merged_eval).
# The best weights are then applied to merged_eval to produce eval predictions.
def tier1_weighted_mean(merged_fit, merged_eval, fit_terms, ia_weights):
    def apply_weights(merged, weights):
        out = merged[['protein_id', 'GO_term']].copy()
        out['confidence'] = np.average(merged.filter(like='conf_'), weights=weights, axis=1)
        return out

    best_score = -1
    best_weights = [1/3, 1/3, 1/3]
    step = 0.1
    for i in range(11):
        for j in range(11 - i):
            w_1 = round(i * step, 2)
            w_2 = round(j * step, 2)
            w_3 = round(1.0 - w_1 - w_2, 2)
            preds = apply_weights(merged_fit, [w_1, w_2, w_3])
            score = fmax(preds, fit_terms, ia_weights)['mean']
            if score > best_score:
                best_score = score
                best_weights = [w_1, w_2, w_3]

    print(f"Best weights: {best_weights}  fit F-max: {best_score:.4f}")
    return apply_weights(merged_eval, best_weights)


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

def fmax(preds_df, ground_truth_df, ia_weights, thresholds=None):
    """
    Compute weighted F-max (the official CAFA-5 metric) averaged across MFO, BPO, CCO.

    Args:
        preds_df        : DataFrame[protein_id, GO_term, confidence]
        ground_truth_df : DataFrame[EntryID, term, aspect]
        ia_weights      : dict {GO_term: ia_value}
        thresholds      : array of cutoffs to sweep; defaults to 0.01..0.99 step 0.02

    Returns:
        dict with keys 'MFO', 'BPO', 'CCO', 'mean'
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.02)

    true_map = {}
    for _, row in ground_truth_df.iterrows():
        true_map.setdefault(row['EntryID'], {}).setdefault(row['aspect'], set()).add(row['term'])

    pred_map = (
        preds_df.groupby('protein_id')
        .apply(lambda g: dict(zip(g['GO_term'], g['confidence'])))
        .to_dict()
    )

    scores = {}
    for aspect in ['MFO', 'BPO', 'CCO']:
        gt_proteins = [p for p in true_map if aspect in true_map[p]]
        if not gt_proteins:
            scores[aspect] = 0.0
            continue
        best_f = 0.0
        for t in thresholds:
            sp_num = sp_den = sr_num = sr_den = 0.0
            for p in gt_proteins:
                true_terms = true_map[p].get(aspect, set())
                pred_terms = {g for g, c in pred_map.get(p, {}).items() if c >= t}
                tp_ia   = sum(ia_weights.get(g, 0.0) for g in true_terms & pred_terms)
                pre_den = sum(ia_weights.get(g, 0.0) for g in pred_terms)
                rec_den = sum(ia_weights.get(g, 0.0) for g in true_terms)
                sp_num += tp_ia;  sp_den += pre_den if pre_den > 0 else 1e-9
                sr_num += tp_ia;  sr_den += rec_den if rec_den > 0 else 1e-9
            prec = sp_num / sp_den
            rec  = sr_num / sr_den
            f = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            if f > best_f:
                best_f = f
        scores[aspect] = best_f

    scores['mean'] = float(np.mean([scores['MFO'], scores['BPO'], scores['CCO']]))
    return scores


"""usage: python tier1.py -tr [train prediction files] -te [test prediction files]"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', nargs='+', required=True)
    parser.add_argument('-te', nargs='+', required=True)
    args = parser.parse_args()

    # load training set predictions
    tr_dfs = []
    for i, filename in enumerate(args.tr):
        df = pd.read_csv(PATH + filename,
                         sep='\t',
                         header=None,
                         names=['protein_id', 'GO_term', 'conf_'+str(i)])
        df = df.set_index(['protein_id', 'GO_term'])
        tr_dfs.append(df)
    merged_tr_conf = pd.concat(tr_dfs, axis=1, join='outer').fillna(0.0).reset_index()

    # load true labels and ia weights
    train_terms = pd.read_csv(PATH + "train_terms.tsv", sep='\t')
    ia_df = pd.read_csv(f'{PATH}/IA.txt', sep='\t', header=None, names=['term', 'ia'])
    IA_WEIGHTS = dict(zip(ia_df['term'], ia_df['ia']))

    # split data into training and evaluation sets
    # we are doing this really just for weighted mean method to have training and eval sets
    # the other three methods have no training
    tr_ev_splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    fit_idx, eval_idx = next(
        tr_ev_splitter.split(merged_tr_conf, groups=merged_tr_conf['protein_id'])
    )
    merged_tr_fit  = merged_tr_conf.iloc[fit_idx].reset_index(drop=True)
    merged_tr_eval = merged_tr_conf.iloc[eval_idx].reset_index(drop=True)

    fit_proteins  = set(merged_tr_fit['protein_id'])
    eval_proteins = set(merged_tr_eval['protein_id'])
    fit_terms  = train_terms[train_terms['EntryID'].isin(fit_proteins)]
    eval_terms = train_terms[train_terms['EntryID'].isin(eval_proteins)]

    # evaluate methods on eval set
    eval_scores = {}
    eval_preds  = {}

    for name, preds in [
        ('mean',          tier1_mean(merged_tr_eval)),
        ('max',           tier1_max(merged_tr_eval)),
        ('rank_avg',      tier1_rank_avg(merged_tr_eval)),
        ('weighted_mean', tier1_weighted_mean(merged_tr_fit, merged_tr_eval, fit_terms, IA_WEIGHTS)),
    ]:
        score = fmax(preds, eval_terms, IA_WEIGHTS)
        eval_scores[name] = score
        eval_preds[name]  = preds
        print(f"  {name:<15}: MFO={score['MFO']:.4f}  BPO={score['BPO']:.4f}  "
              f"CCO={score['CCO']:.4f}  mean={score['mean']:.4f}")

    # pick best method
    best_method = max(eval_scores, key=lambda k: eval_scores[k]['mean'])
    print(f"\nBest method: {best_method}  "
          f"(eval F-max mean = {eval_scores[best_method]['mean']:.4f})")

    # load test predictions
    te_dfs = []
    for i, filename in enumerate(args.te):
        df = pd.read_csv(PATH + filename,
                         sep='\t',
                         header=None,
                         names=['protein_id', 'GO_term', 'conf_'+str(i)])
        df = df.set_index(['protein_id', 'GO_term'])
        te_dfs.append(df)
    merged_test_conf = pd.concat(te_dfs, axis=1, join='outer').fillna(0.0).reset_index()

    # need to rerun training set if doing weighted mean
    if best_method == 'weighted_mean':
        final_preds = tier1_weighted_mean(
            merged_tr_conf, merged_test_conf, train_terms, IA_WEIGHTS
        )
    # else the methods are ready to use
    elif best_method == 'mean':
        final_preds = tier1_mean(merged_test_conf)
    elif best_method == 'max':
        final_preds = tier1_max(merged_test_conf)
    else:
        final_preds = tier1_rank_avg(merged_test_conf)

    out_path = f"tier1_{best_method}_output.tsv"
    submission = final_preds[final_preds['confidence'] > 0.0].copy()
    submission['confidence'] = submission['confidence'].round(3)
    submission.to_csv(out_path, sep='\t', header=False, index=False)


