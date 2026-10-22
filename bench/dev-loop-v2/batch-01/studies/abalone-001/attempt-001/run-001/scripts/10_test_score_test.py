import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs['_protocol'].read_text())
for k in ('targets', 'excluded_columns', 'group_column', 'task'):
    if k not in proto:
        raise KeyError(f"Missing _protocol[{k!r}]; available top-level keys: {sorted(proto)}")
targets = list(proto['targets'])
excluded = list(proto['excluded_columns'] or [])
group = proto['group_column']
task = proto['task']
if 'primary_metric' not in task:
    raise KeyError(f"Missing _protocol['task']['primary_metric']; task keys: {sorted(task)}")
primary = task['primary_metric']

train = pd.read_parquet(inputs['train'])
test = pd.read_parquet(inputs['test'])

def is_key_like(s):
    if len(s) <= 1:
        return False
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype) or pd.api.types.is_integer_dtype(s.dtype):
        return s.nunique(dropna=False) == len(s)
    return False

skip = set(targets) | set(excluded)
if group is not None:
    skip.add(group)
features = [c for c in train.columns if c not in skip and not is_key_like(train[c])]

try:
    vdf = pd.read_csv(inputs['09_validate_score_validation_77M5Y6B8/validation_metrics.csv'])
except Exception:
    vdf = pd.read_parquet(inputs['09_validate_score_validation_77M5Y6B8/validation_metrics.parquet'])

def find_col(df, names):
    for c in df.columns:
        if str(c).lower() in names:
            return c
    return None

mc = find_col(vdf, {'metric', 'name', 'measure'})
vc = find_col(vdf, {'value', 'score', 'metric_value'})
mcol = find_col(vdf, {'model', 'candidate', 'model_name', 'estimator'})
tc = find_col(vdf, {'target', 'outcome', 'target_name'})
if mc is None or vc is None or mcol is None:
    raise KeyError(f"Validation metrics columns not found: {list(vdf.columns)}")
v = vdf.copy()
v['_m'] = v[mc].astype(str).str.lower()
sub = v[v['_m'] == str(primary).lower()]
if sub.empty:
    sub = v[v['_m'].str.contains(str(primary).lower(), regex=False)]
if sub.empty:
    raise ValueError(f"No validation rows for primary metric {primary!r}")
if tc is not None:
    mac = sub[sub[tc].astype(str).str.lower().str.contains('macro|mean|avg|average', regex=True)]
    sub = mac if not mac.empty else sub.groupby(mcol, as_index=False)[vc].mean()
else:
    sub = sub.groupby(mcol, as_index=False)[vc].mean()
higher = str(primary).lower() in {'r2', 'r2_score', 'accuracy', 'roc_auc', 'auc', 'f1', 'precision', 'recall'} or 'auc' in str(primary).lower() or 'r2' in str(primary).lower()
sub = sub.sort_values(vc, ascending=not higher)
best_model = str(sub.iloc[0][mcol])
name = best_model.lower()
if 'baseline' in name or 'base' in name or '07_model_train_baseline' in name or name in {'model', 'model.pkl'}:
    cand = 'baseline'
elif 'alternative' in name or 'alt' in name or 'latest' in name or '08_model_train_alternative' in name:
    cand = 'alternative'
else:
    uniq = sub[mcol].astype(str).tolist()
    if len(uniq) == 1:
        cand = 'baseline' if 'base' in uniq[0].lower() else 'alternative'
    else:
        raise ValueError(f"Cannot map validation model {best_model!r} to a saved model")

pre = pickle.load(open(inputs['05_preprocess_fit_preprocessing_EQ66FFY1/preprocessor.pkl'], 'rb'))
sel = pickle.load(open(inputs['06_features_retain_features_4S51ZGZN/selector_all.pkl'], 'rb'))
model_path = inputs['07_model_train_baseline_4EB5V15N/model.pkl'] if cand == 'baseline' else inputs['08_model_train_alternative_1TXETTYN/alternative_model.pkl']
model = pickle.load(open(model_path, 'rb'))

def take(df, obj):
    cols = list(getattr(obj, 'feature_names_in_', features))
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing model input columns: {missing}")
    return df[cols]

try:
    preds = model.predict(take(test, model))
except Exception:
    def frontend(df):
        if isinstance(sel, Pipeline):
            return sel.transform(take(df, sel))
        xt = pre.transform(take(df, pre))
        if hasattr(sel, 'feature_names_in_') and isinstance(xt, pd.DataFrame):
            cols = list(sel.feature_names_in_)
            if all(c in xt.columns for c in cols):
                xt = xt[cols]
        return sel.transform(xt)
    preds = model.predict(frontend(test))

preds = np.asarray(preds)
if preds.ndim == 1:
    preds = preds.reshape(-1, 1)
if preds.shape[1] != len(targets):
    raise ValueError(f"Prediction shape {preds.shape} does not match targets {targets}")

rows, per = [], {}
for i, t in enumerate(targets):
    yt, yp = test[t].to_numpy(), preds[:, i]
    vals = {
        'rmse': float(np.sqrt(mean_squared_error(yt, yp))),
        'mae': float(mean_absolute_error(yt, yp)),
        'r2': float(r2_score(yt, yp)),
    }
    per[t] = vals
    for m, val in vals.items():
        rows.append({'model': best_model, 'metric': f'{m}__{t}', 'value': val})

macro = {}
for m in ('rmse', 'mae', 'r2'):
    macro[m] = float(np.mean([per[t][m] for t in targets]))
    rows.append({'model': best_model, 'metric': m, 'value': macro[m]})

table = pd.DataFrame(rows)
table['value'] = table['value'].map(lambda x: float(f"{x:.4g}"))
mlca_runtime.save_table(table, 'test_metrics.csv', caption='Test metrics')

out = pd.DataFrame({'row_id': test.index, 'partition': 'test'})
if len(targets) == 1:
    out['y_true'] = test[targets[0]].to_numpy()
    out['y_pred'] = preds[:, 0]
else:
    for i, t in enumerate(targets):
        out[f'y_true__{t}'] = test[t].to_numpy()
        out[f'y_pred__{t}'] = preds[:, i]
mlca_runtime.save_row_outputs(out, 'test_predictions.parquet')

pval = macro.get(str(primary).lower())
if pval is None:
    raise ValueError(f"Declared primary metric {primary!r} is not among computed metrics {sorted(macro)}")
mlca_runtime.record_metric(str(primary), pval, split='test')
for m in ('mae', 'r2'):
    if m != str(primary).lower():
        mlca_runtime.record_metric(m, macro[m], split='test')
mlca_runtime.record_fact_candidate('test_score', pval, source='test_metrics.csv')
print(f"selected={best_model} test_rows={len(test)} primary={primary}:{pval:.4g}")
