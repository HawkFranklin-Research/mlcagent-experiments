import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
targets = proto['targets']
excluded = proto['excluded_columns']
try:
    group_col = proto['group_column']
except KeyError as e:
    raise KeyError(f"Missing group_column; available top-level keys: {list(proto.keys())}") from e
task = proto['task']
primary = task['primary_metric']
task_type = task.get('type')
if task_type != 'multiclass':
    raise ValueError(f"Expected multiclass task, got {task_type}")

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
test = pd.read_parquet(inp['test'])
drop_cols = set(targets) | set(excluded) | ({group_col} if group_col else set())
feature_cols = [c for c in train.columns if c not in drop_cols]
feature_cols = [c for c in feature_cols if not (train[c].dtype.kind in 'OSU' and train[c].nunique(dropna=True) == len(train))]

def load_pkl(key):
    return pickle.loads(inp[key].read_bytes())

pre = load_pkl('05_preprocess_fit_preprocessing_RMJEH56H/preprocessor.pkl')
sel = load_pkl('06_features_select_features_4X00PG4B/selector.pkl')
mdl = load_pkl('07_model_train_logistic_7T5DR438/model.pkl')
alt = load_pkl('08_model_train_alternative_T24P90EK/alternative_model.pkl')

def input_names(o):
    if isinstance(o, Pipeline):
        first = o.named_steps[list(o.named_steps)[0]]
        return input_names(first)
    names = getattr(o, 'feature_names_in_', None)
    return list(names) if names is not None else None

def complete_pipeline(o):
    if isinstance(o, Pipeline) and hasattr(o, 'predict_proba'):
        names = input_names(o)
        if names is None or set(names) == set(feature_cols):
            return o
    steps = []
    pre_names = input_names(pre) if pre is not None else None
    sel_names = input_names(sel) if sel is not None else None
    raw = set(feature_cols)
    if pre is not None and (pre_names is None or set(pre_names) == raw):
        steps.append(('preprocessor', pre))
    if sel is not None:
        if steps and sel_names is not None and set(sel_names) == raw:
            steps = [('selector', sel)]
        else:
            steps.append(('selector', sel))
    steps.append(('model', o))
    return Pipeline(steps)

pipelines = {'07': complete_pipeline(mdl), '08': complete_pipeline(alt)}

def get_classes(pipe):
    if hasattr(pipe, 'classes_'):
        return pipe.classes_
    if isinstance(pipe, Pipeline):
        est = pipe.named_steps[list(pipe.named_steps)[-1]]
        if hasattr(est, 'classes_'):
            return est.classes_
    return np.unique(train[targets[0]].values)

def input_cols(pipe):
    names = input_names(pipe)
    return names if names is not None else feature_cols

def prepare_X(df, pipe):
    return df[input_cols(pipe)]

def macro_ovr_auc(y, proba, classes):
    present = [i for i, c in enumerate(classes) if np.any(y == c)]
    if len(present) < 2:
        return np.nan
    if len(present) == len(classes):
        return float(roc_auc_score(y, proba, multi_class='ovr', average='macro', labels=classes))
    vals = [roc_auc_score((y == classes[i]).astype(int), proba[:, i]) for i in present]
    return float(np.mean(vals))

lower_better = {'brier_score', 'log_loss', 'mae', 'mse', 'rmse'}
scores = {}
for key, pipe in pipelines.items():
    proba = pipe.predict_proba(prepare_X(val, pipe))
    classes = get_classes(pipe)
    if primary == 'roc_auc_ovr':
        scores[key] = macro_ovr_auc(val[targets[0]].values, proba, classes)
    elif primary == 'f1_macro':
        pred = classes[np.argmax(proba, axis=1)]
        scores[key] = float(f1_score(val[targets[0]].values, pred, average='macro', labels=classes, zero_division=0))
    else:
        raise ValueError(f"Unsupported primary metric: {primary}")
selected_key = (min if primary in lower_better else max)(scores, key=scores.get)
selected_pipe = pipelines[selected_key]
classes = get_classes(selected_pipe)

X_test = prepare_X(test, selected_pipe)
proba_test = selected_pipe.predict_proba(X_test)
y_test = test[targets[0]].values

def metric_value(name, y, proba):
    if name == 'roc_auc_ovr':
        return macro_ovr_auc(y, proba, classes)
    if name == 'f1_macro':
        pred = classes[np.argmax(proba, axis=1)]
        return float(f1_score(y, pred, average='macro', labels=classes, zero_division=0))
    raise ValueError(name)

metric_names = list(dict.fromkeys([primary, 'f1_macro']))
metrics = []
rng = np.random.default_rng(42)
n = len(y_test)
for name in metric_names:
    value = metric_value(name, y_test, proba_test)
    vals = []
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        vals.append(metric_value(name, y_test[idx], proba_test[idx]))
    vals = np.array(vals, dtype=float)
    ci_low, ci_high = np.nanpercentile(vals, [2.5, 97.5])
    metrics.append({'metric': name, 'value': value, 'ci_low': ci_low, 'ci_high': ci_high})

df = pd.DataFrame(metrics)
for c in ['value', 'ci_low', 'ci_high']:
    df[c] = df[c].apply(lambda x: float(f"{x:.4g}") if pd.notna(x) else np.nan)
mlca_runtime.save_table(df, 'test_metrics.csv', caption='Test metrics')

primary_row = df[df['metric'] == primary].iloc[0]
mlca_runtime.record_metric(primary, float(primary_row['value']), 'test',
                           ci=(float(primary_row['ci_low']), float(primary_row['ci_high'])))
if primary in ('roc_auc', 'roc_auc_ovr'):
    mlca_runtime.record_fact_candidate('test_roc_auc', float(primary_row['value']), source='test_metrics.csv')
for _, row in df.iterrows():
    if row['metric'] != primary:
        mlca_runtime.record_metric(row['metric'], float(row['value']), 'test',
                                   ci=(float(row['ci_low']), float(row['ci_high'])))

print(f"test rows={len(test)} features={len(feature_cols)} selected={selected_key}")
print(f"{primary}={primary_row['value']:.4f} CI=({primary_row['ci_low']:.4f},{primary_row['ci_high']:.4f})")
