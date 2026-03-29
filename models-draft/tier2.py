import argparse
import numpy as np
import pandas as pd
import sklearn
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


 """
 TODO： implement the fmax function
 do we want to use fmax as the metrics for tuning or is auroc fine?
 """
#def fmax (confidence, y):


"""usage: python tier2.py -tr [train term prediction filenames] -te [test term prediction filenames]"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', nargs='+', required=True)
    parser.add_argument('-te', nargs='+', required=True)
    args = parser.parse_args()

    tr_dfs = []
    for i, filename in enumerate(args.tr):
        df = pd.read_csv(PATH + filename,
                         sep='\t',
                         header=None,
                         names=['protein_id', 'GO_term', 'conf_'+str(i)])
        df = df.set_index(['protein_id', 'GO_term'])
        tr_dfs.append(df)
    merged_tr_conf = pd.concat(tr_dfs, axis=1, join='outer').fillna(0.0).reset_index()

    labels = merged_tr_conf[['protein_id', 'GO_term']].copy()
    train_terms = pd.read_csv(PATH + "train_terms.tsv", sep='\t')
    true_pairs = set(zip(train_terms['EntryID'], train_terms['term']))
    labels['label'] = [
        int((row.protein_id, row.GO_term) in true_pairs)
        for row in labels.itertuples(index=False)
    ]

    test_dfs = []
    for i, filename in enumerate(args.te):
        df = pd.read_csv(PATH + filename,
                         sep='\t',
                         header=None,
                         names=['protein_id', 'GO_term', 'conf_'+str(i)])
        df = df.set_index(['protein_id', 'GO_term'])
        test_dfs.append(df)
    merged_test_conf = pd.concat(test_dfs, axis=1, join='outer').fillna(0.0).reset_index()

    logistic_conf = logistic(merged_tr_conf, labels, merged_test_conf)
    print(logistic_conf.head())
    xgb_conf = xgboost_stacking(merged_tr_conf, labels, merged_test_conf)
    print(xgb_conf.head())