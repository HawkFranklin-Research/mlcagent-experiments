import mlca_runtime, json, pickle, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             balanced_accuracy_score, confusion_matrix, f1_score)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(Path(inp['_protocol']).read_text())
task = protocol['task']; primary = task['primary_metric']
targets = protocol['targets']; excluded = list(protocol.get('excluded_columns', []))
gcol = protocol.get('group_column')
train = pd.read_parquet(inp['train']); test = pd.read_parquet(inp['test'])

drop = set(targets) | set(excluded) | ({gcol} if gcol else set())
def keylike(s):
    if len(s) < 3: return False
    code = pd.api.types.is_integer_dtype(s) or pd.api.types.is_string_dtype(s) or str(s.dtype) == 'object'
    return code and s.nunique(dropna=False) == len(s)
feat_cols = [c for c in train.columns if c not in drop and not keylike(train[c])]

with open(inp['latest_model'], 'rb') as f: model = pickle.load(f)
classes = list(getattr(model, 'classes_', [0, 1]))

def predict_proba_on(df):
    cols = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else feat_cols
    if all(c in df.columns for c in cols):
        try: return np.asarray(model.predict_proba(df[cols]))
        except Exception: pass
    with open(inp['05_preprocess_fit_preprocessing_6GF5BX9F/preprocessor.pkl'], 'rb') as f: pre = pickle.load(f)
    with open(inp['06_features_select_features_1B1C3SBH/selector.pkl'], 'rb') as f: sel = pickle.load(f)
    pcols = list(pre.feature_names_in_) if hasattr(pre, 'feature_names_in_') else feat_cols
    Xt = pre.transform(df[pcols])
    if not isinstance(Xt, pd.DataFrame): Xt = pd.DataFrame(Xt, index=df.index)
    Xs = sel.transform(Xt)
    if not isinstance(Xs, pd.DataFrame): Xs = pd.DataFrame(Xs, index=df.index)
    mcols = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else list(Xs.columns)
    return np.asarray(model.predict_proba(Xs[mcols]))

def get_threshold():
    try:
        m = json.loads(Path(inp['09_validate_score_validation_50QNR0X8/manifest.json']).read_text())
        if isinstance(m, dict):
            for k in ('threshold', 'binary_threshold', 'decision_threshold'):
                if m.get(k) is not None: return float(m[k])
    except Exception: pass
    try:
        vm = pd.read_csv(inp['09_validate_score_validation_50QNR0X8/validation_metrics.csv'])
        if 'metric' in vm.columns and 'value' in vm.columns:
            hit = vm[vm['metric'].astype(str).str.contains('threshold', case=False, na=False)]
            if len(hit) and pd.notna(hit['value'].iloc[0]): return float(hit['value'].iloc[0])
    except Exception: pass
    return 0.5

target = targets[0]
proba = predict_proba_on(test)
y_true_orig = test[target].values
n = len(y_true_orig); rng = np.random.RandomState(42)
threshold = get_threshold()

is_bin = len(classes) == 2
if is_bin:
    y_bin = (y_true_orig == classes[1]).astype(int); p_pos = proba[:, 1]
    y_pred_bin = (p_pos >= threshold).astype(int)
    y_pred_out = np.where(y_pred_bin == 1, classes[1], classes[0])
    def _sens(y, yp):
        tn, fp, fn_, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
        return tp / (tp + fn_) if (tp + fn_) else np.nan
    def _spec(y, yp):
        tn, fp, fn_, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
        return tn / (tn + fp) if (tn + fp) else np.nan
    metric_fns = {
        'roc_auc': lambda i: roc_auc_score(y_bin[i], p_pos[i]),
        'pr_auc': lambda i: average_precision_score(y_bin[i], p_pos[i]),
        'brier_score': lambda i: brier_score_loss(y_bin[i], p_pos[i]),
        'balanced_accuracy': lambda i: balanced_accuracy_score(y_bin[i], y_pred_bin[i]),
        'sensitivity': lambda i: _sens(y_bin[i], y_pred_bin[i]),
        'specificity': lambda i: _spec(y_bin[i], y_pred_bin[i]),
    }
    primary_key = 'roc_auc'
else:
    y_pred_bin = np.asarray(classes)[proba.argmax(axis=1)]; y_pred_out = y_pred_bin
    present = [c for c in classes if c in set(y_true_orig)]
    pidx = [classes.index(c) for c in present]
    metric_fns = {
        'roc_auc_ovr_macro': lambda i: roc_auc_score(y_true_orig[i], proba[i][:, pidx],
                                                      labels=present, multi_class='ovr', average='macro'),
        'macro_f1': lambda i: f1_score(y_true_orig[i], y_pred_out[i], labels=classes,
                                       average='macro', zero_division=0),
    }
    primary_key = 'roc_auc_ovr_macro'

all_idx = np.arange(n); results = {}
for name, fn in metric_fns.items():
    try: v = float(fn(all_idx))
    except Exception: v = float('nan')
    vals = []
    for _ in range(1000):
        idx = rng.randint(0, n, n)
        try:
            x = float(fn(idx))
            if np.isfinite(x): vals.append(x)
        except Exception: pass
    lo, hi = tuple(np.percentile(vals, [2.5, 97.5])) if vals else (float('nan'), float('nan'))
    results[name] = (v, float(lo), float(hi))

dfm = pd.DataFrame([{'metric': k,
                     'value': round(v[0], 6),
                     'ci_low': round(v[1], 6) if np.isfinite(v[1]) else np.nan,
                     'ci_high': round(v[2], 6) if np.isfinite(v[2]) else np.nan}
                    for k, v in results.items()])
mlca_runtime.save_table(dfm, 'test_metrics', caption='Test metrics with 95% bootstrap CI')

pred_df = pd.DataFrame({'row_id': test.index, 'partition': 'test',
                        f'y_true__{target}': y_true_orig, f'y_pred__{target}': y_pred_out})
for i, c in enumerate(classes): pred_df[f'proba__{c}'] = proba[:, i]
mlca_runtime.save_row_outputs(pred_df, 'test_predictions')

pk = primary_key if primary_key in results else primary if primary in results else list(results.keys())[0]
v, lo, hi = results[pk]
mlca_runtime.record_metric(primary, float(v), 'test', ci=(float(lo), float(hi)))
if primary == 'roc_auc' and 'roc_auc' in results:
    mlca_runtime.record_fact_candidate('test_roc_auc', float(results['roc_auc'][0]), source='test_metrics.csv')

print(f"feat_cols={len(feat_cols)} test_rows={n} threshold={threshold:.4f} classes={classes}")
print(f"{primary}={results[pk][0]:.4f} CI=({results[pk][1]:.4f},{results[pk][2]:.4f})")
