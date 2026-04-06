import argparse
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
import xgboost

SEED = 32
PATH = "cafa-5-protein-function-prediction/"


"""Currently only searching for C"""
def logistic(x_merged_conf, y_labels, x_test):
    splitter = sklearn.model_selection.GroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED)
    classifier = sklearn.linear_model.LogisticRegression(
        C = 0.001,
        max_iter = 1000,
        solver = 'lbfgs',
        class_weight = 'balanced',
        random_state = SEED,
    )
    param_grid = {'C':np.logspace(-4,2,5),}
    search = sklearn.model_selection.GridSearchCV(
        classifier,
        param_grid=param_grid,
        cv=splitter,
        scoring='roc_auc'
    )
    x_feature = x_merged_conf.filter(like='conf_')
    search.fit(x_feature, y_labels['label'], groups=x_merged_conf['protein_id'])

    estimator = search.best_estimator_

    out = x_test[['protein_id', 'GO_term']].copy()
    out['confidence'] = estimator.predict_proba(x_test.filter(like='conf_'))[:, 1]

    return out


"""
I'm using early stopping for this one instead of a cv because there are too
many parameters for xgbclassifier and i'm not sure which ones are worth tuning.
Early stopping 'searches' for n_estimators.
"""
def xgboost_stacking(x_merged_conf, y_labels, x_test):
    x_feature = x_merged_conf.filter(like='conf_')
    y = y_labels['label']
    splitter = sklearn.model_selection.GroupKFold(n_splits=5)
    train_idx, val_idx = next(
        splitter.split(
            x_feature,
            y,
            groups=x_merged_conf['protein_id']))

    x_tr, x_val = x_feature.iloc[train_idx], x_feature.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    pos_rate = y_tr.mean()
    xgb_model = xgboost.XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        scale_pos_weight=(1 - pos_rate) / pos_rate,
        random_state=SEED,
    )

    xgb_model.fit(x_tr,
                  y_tr,
                  eval_set=[(x_val, y_val)],
                  early_stopping_rounds=20,
                  verbose=False,
    )

    importance = dict(zip(x_feature.columns, xgb_model.feature_importances_))
    print("\nXGBoost feature importance (gain — higher = more useful):")
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {feat:18s}: {imp:.4f}")

    out = x_test[['protein_id', 'GO_term']].copy()
    out['confidence'] = xgb_model.predict_proba(x_test.filter(like='conf_'))[:, 1]
    return out


def mlp_stacking(x_merged_conf, y_labels, x_test):
    x_feature = x_merged_conf.filter(like='conf_')
    y = y_labels['label']

    splitter = sklearn.model_selection.GroupKFold(n_splits=5)

    classifier = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        max_iter=500,
        random_state=SEED,
    )

    param_grid = {
        'alpha': [0.0001, 0.001, 0.01],
        'hidden_layer_sizes': [(64, 32), (32, 16)],
    }

    search = sklearn.model_selection.GridSearchCV(
        classifier,
        param_grid=param_grid,
        cv=splitter,
        scoring='roc_auc',
    )
    search.fit(x_feature, y, groups=x_merged_conf['protein_id'])

    print(f"Best alpha: {search.best_params_['alpha']}  |  Best CV AUC: {search.best_score_:.4f}")

    out = x_test[['protein_id', 'GO_term']].copy()
    out['confidence'] = search.best_estimator_.predict_proba(x_test.filter(like='conf_'))[:, 1]
    return out

"""Claude made this fmax function; looks right to me, needs further validation"""
def fmax(preds_df, ground_truth_df, ia_weights, thresholds=None):
    """
    Compute weighted F-max (the official CAFA-5 metric) averaged across MFO, BPO, CCO.

    For each sub-ontology and each confidence threshold t:
      - Predict GO term g for protein p iff confidence(p,g) >= t
      - IA-weighted precision = sum(IA(g) for true positives) / sum(IA(g) for all predicted)
      - IA-weighted recall    = sum(IA(g) for true positives) / sum(IA(g) for all true terms)
      - F(t) = 2 * precision * recall / (precision + recall)
    F-max = max over all t; final score = mean(F-max_MFO, F-max_BPO, F-max_CCO).

    Args:
        preds_df        : DataFrame[protein_id, GO_term, confidence]
        ground_truth_df : DataFrame[EntryID, term, aspect]  (filtered to eval proteins)
        ia_weights      : dict {GO_term: ia_value}
        thresholds      : array of cutoffs to sweep; defaults to 0.01..0.99 step 0.02

    Returns:
        dict with keys 'MFO', 'BPO', 'CCO', 'mean'
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.02)

    # Build nested lookup: {protein_id: {aspect: set(true_terms)}}
    true_map = {}
    for _, row in ground_truth_df.iterrows():
        true_map.setdefault(row['EntryID'], {}).setdefault(row['aspect'], set()).add(row['term'])

    # Build prediction lookup: {protein_id: {GO_term: confidence}}
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
                tp_ia  = sum(ia_weights.get(g, 0.0) for g in true_terms & pred_terms)
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


"""usage: python tier2.py -tr [train term prediction files] -te [test term prediction filenames]"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', nargs='+', required=True)
    parser.add_argument('-te', nargs='+', required=True)
    args = parser.parse_args()

    # Load training predictions
    tr_dfs = []
    for i, filename in enumerate(args.tr):
        df = pd.read_csv(PATH + filename,
                         sep='\t',
                         header=None,
                         names=['protein_id', 'GO_term', 'conf_'+str(i)])
        df = df.set_index(['protein_id', 'GO_term'])
        tr_dfs.append(df)
    merged_tr_conf = pd.concat(tr_dfs, axis=1, join='outer').fillna(0.0).reset_index()

    # Load true labels and IA weights
    train_terms = pd.read_csv(PATH + "train_terms.tsv", sep='\t')
    true_pairs = set(zip(train_terms['EntryID'], train_terms['term']))
    labels = merged_tr_conf[['protein_id', 'GO_term']].copy()
    labels['label'] = [
        int((row.protein_id, row.GO_term) in true_pairs)
        for row in labels.itertuples(index=False)
    ]

    ia_df = pd.read_csv(f'{PATH}/IA.txt', sep='\t', header=None, names=['term', 'ia'])
    IA_WEIGHTS = dict(zip(ia_df['term'], ia_df['ia']))


    # Split training data into train and evaluation sets
    # test_size=0.15 holds out 15% of proteins to compare the three methods
    tr_ev_splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    fit_idx, eval_idx = next(
        tr_ev_splitter.split(merged_tr_conf, groups=merged_tr_conf['protein_id'])
    )

    merged_tr_fit  = merged_tr_conf.iloc[fit_idx].reset_index(drop=True)
    merged_tr_eval = merged_tr_conf.iloc[eval_idx].reset_index(drop=True)
    labels_fit     = labels.iloc[fit_idx].reset_index(drop=True)

    # filter out the true labels for the eval proteins
    eval_proteins = set(merged_tr_eval['protein_id'])
    eval_terms    = train_terms[train_terms['EntryID'].isin(eval_proteins)]

    methods = {
        'logistic': logistic,
        'xgboost' : xgboost_stacking,
        'mlp'     : mlp_stacking,
    }
    eval_scores = {}
    eval_preds  = {}

    for name, fn in methods.items():
        print(f"\n--- {name} ---")
        preds = fn(merged_tr_fit, labels_fit, merged_tr_eval)
        score = fmax(preds, eval_terms, IA_WEIGHTS)
        eval_scores[name] = score
        eval_preds[name]  = preds
        print(f"  F-max: MFO={score['MFO']:.4f}  BPO={score['BPO']:.4f}  "
              f"CCO={score['CCO']:.4f}  mean={score['mean']:.4f}")

    # pick best method
    best_name = max(eval_scores, key=lambda k: eval_scores[k]['mean'])
    print(f"\nBest method: {best_name}  "
          f"(eval F-max mean = {eval_scores[best_name]['mean']:.4f})")

    # load test predictions
    test_dfs = []
    for i, filename in enumerate(args.te):
        df = pd.read_csv(PATH + filename,
                         sep='\t',
                         header=None,
                         names=['protein_id', 'GO_term', 'conf_'+str(i)])
        df = df.set_index(['protein_id', 'GO_term'])
        test_dfs.append(df)
    merged_test_conf = pd.concat(test_dfs, axis=1, join='outer').fillna(0.0).reset_index()

    # retrain the best method on all training data and predict for test data
    final_preds = methods[best_name](merged_tr_conf, labels, merged_test_conf)
    print(final_preds.head())

    out_path = f"tier2_{best_name}_output.tsv"
    submission = final_preds[final_preds['confidence'] > 0.0].copy()
    submission['confidence'] = submission['confidence'].round(3)
    submission.to_csv(out_path, sep='\t', header=False, index=False)