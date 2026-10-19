import mlca_runtime
import pandas as pd, numpy as np, pickle, json
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
test = pd.read_parquet(inp['test'])

try:
    targets = list(protocol['targets'])
    excluded = list(protocol['excluded_columns'])
    task = protocol['task']
    primary_metric = task['primary_metric']
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; top-level keys: {list(protocol.keys())}")
if task.get('type') != 'multiclass':
    raise ValueError('This script handles the declared multiclass task')
if len(targets) != 1:
    raise ValueError(f'Expected one target, got {targets}')
target_col = targets[0]

def load_pkl(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

preprocessor = load_pkl(inp['05_preprocess_fit_preprocessing_H4RCZJ73/preprocessor.pkl'])
selector = load_pkl(inp['06_features_select_features_F6NPFNPS/selector.pkl'])

def keylike(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=False) == len(s)
    return False

group_col = protocol.get('group_column')
drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feature_cols = [c for c in train.columns if c not in drop and not keylike(train[c])]

def classes_of(pipe):
    if hasattr(pipe, 'classes_'):
        return np.asarray(pipe.classes_)
    return np.asarray(list(pipe.named_steps.values())[-1].classes_)

def X_for(pipe, df):
    cols = list(pipe.feature_names_in_) if hasattr(pipe, 'feature_names_in_') else feature_cols
    return df.loc[:, cols]

def ovr_auc(y, proba, classes):
    classes = list(classes)
    vals = []
    for c in classes:
        if not np.any(y == c):
            continue
        yb = (y == c).astype(int)
        if yb.min() == yb.max():
            continue
        j = classes.index(c)
        vals.append(roc_auc_score(yb, proba[:, j]))
    return float(np.mean(vals)) if vals else np.nan

def macro_f1(y, proba, classes):
    classes = np.asarray(classes)
    pred = classes[np.argmax(proba, axis=1)]
    return float(f1_score(y, pred, labels=classes, average='macro', zero_division=0))

def build_options(m):
    opts = []
    if isinstance(m, Pipeline) and hasattr(m, 'predict_proba'):
        opts.append(m)
    opts.append(Pipeline([('preprocessor', preprocessor), ('selector', selector), ('model', m)]))
    opts.append(Pipeline([('selector', selector), ('model', m)]))
    opts.append(Pipeline([('preprocessor', preprocessor), ('model', m)]))
    opts.append(Pipeline([('model', m)]))
    return opts

cand_keys = ['07_model_train_logistic_Q9WJ6BHH/model.pkl', '08_model_train_alternative_3WAJAX78/alternative_model.pkl']
best = None
for key in cand_keys:
    model = load_pkl(inp[key])
    for opt in build_options(model):
        try:
            Xv = X_for(opt, val)
            proba = opt.predict_proba(Xv)
            cls = classes_of(opt)
            yv = val[target_col].to_numpy()
            score = ovr_auc(yv, proba, cls) if primary_metric == 'roc_auc_ovr' else macro_f1(yv, proba, cls)
            if np.isfinite(score) and (best is None or score > best[0]):
                best = (score, key, opt, cls)
            break
        except Exception:
            continue
if best is None:
    raise RuntimeError('No candidate could be scored on validation')

_, best_key, pipe, classes = best
X_test = X_for(pipe, test)
y_test = test[target_col].to_numpy()
proba_test = pipe.predict_proba(X_test)
primary_val = ovr_auc(y_test, proba_test, classes)
f1_val = macro_f1(y_test, proba_test, classes)

def bootstrap(y, proba, cls, fn, seed=42, n=1000):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    N = len(y)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        vals.append(fn(y[idx], proba[idx], cls))
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return (np.nan, np.nan)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return float(lo), float(hi)

primary_ci = bootstrap(y_test, proba_test, classes, ovr_auc)
f1_ci = bootstrap(y_test, proba_test, classes, macro_f1)
rows = [{'metric': primary_metric, 'value': primary_val, 'ci_low': primary_ci[0], 'ci_high': primary_ci[1]}]
if primary_metric != 'f1_macro':
    rows.append({'metric': 'f1_macro', 'value': f1_val, 'ci_low': f1_ci[0], 'ci_high': f1_ci[1]})
df = pd.DataFrame(rows, columns=['metric', 'value', 'ci_low', 'ci_high']).round(4)
mlca_runtime.save_table(df, 'test_metrics.csv', caption='Test metrics with 95% bootstrap CI')
mlca_runtime.record_metric(primary_metric, primary_val, 'test', ci=primary_ci)
if primary_metric != 'f1_macro':
    mlca_runtime.record_metric('f1_macro', f1_val, 'test', ci=f1_ci)
if 'roc_auc' in primary_metric:
    mlca_runtime.record_fact_candidate('test_roc_auc', primary_val, source='test_metrics.csv')

print(f"features={len(feature_cols)} selected={best_key}")
print(f"test {primary_metric}={primary_val:.4f} ci=({primary_ci[0]:.4f},{primary_ci[1]:.4f})")
print('wrote test_metrics.csv')
