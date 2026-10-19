import mlca_runtime, json, pickle, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
out = mlca_runtime.output_dir()

protocol = json.loads(inp['_protocol'].read_text())
for key in ['task', 'targets', 'excluded_columns']:
    if key not in protocol:
        raise KeyError(f"Missing protocol key '{key}'. Available: {list(protocol.keys())}")
task = protocol['task']
if 'primary_metric' not in task:
    raise KeyError(f"Missing task['primary_metric']. Available: {list(task.keys())}")
primary_metric = task['primary_metric']
targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column')

train = pd.read_parquet(inp['train'])
test = pd.read_parquet(inp['test'])

feature_cols = []
for col in train.columns:
    if col in targets or col in excluded or col == group_col:
        continue
    if train[col].dtype == 'object' or str(train[col].dtype).startswith('string'):
        if train[col].nunique() == len(train):
            continue
    elif pd.api.types.is_integer_dtype(train[col]):
        if train[col].nunique() == len(train):
            continue
    feature_cols.append(col)

val_manifest = json.loads(inp['09_validate_score_validation_0PB13SD0/manifest.json'].read_text())
def find_key(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None

selected_name = find_key(val_manifest, ['selected_model', 'best_model', 'chosen_model', 'model'])
threshold = find_key(val_manifest, ['threshold', 'best_threshold', 'frozen_threshold'])

if selected_name is None:
    df_val = pd.read_csv(inp['09_validate_score_validation_0PB13SD0/validation_metrics.csv'])
    if 'model' in df_val.columns:
        if 'metric' in df_val.columns and 'value' in df_val.columns:
            sub = df_val[df_val['metric'] == primary_metric]
            if not sub.empty:
                selected_name = sub.loc[sub['value'].idxmax()]['model']
        elif primary_metric in df_val.columns:
            selected_name = df_val.loc[df_val[primary_metric].idxmax()]['model']

if threshold is None:
    for mkey in ['07_model_train_logistic_12XC5SW7/manifest.json', '08_model_train_alternative_0NB2T98C/manifest.json']:
        m = json.loads(inp[mkey].read_text())
        threshold = find_key(m, ['threshold', 'best_threshold', 'frozen_threshold'])
        if threshold is not None:
            break
if threshold is None:
    threshold = 0.5

model_path = None
if selected_name:
    for key in inp:
        if selected_name in key or selected_name in inp[key].name:
            model_path = inp[key]
            break
if model_path is None:
    model_path = inp['latest_model']

with open(model_path, 'rb') as f:
    model = pickle.load(f)

if isinstance(model, Pipeline):
    pipe = model
else:
    with open(inp['05_preprocess_fit_preprocessing_5G7E554W/preprocessor.pkl'], 'rb') as f:
        pre = pickle.load(f)
    with open(inp['06_features_select_features_Y1ATFPQ6/selector.pkl'], 'rb') as f:
        sel = pickle.load(f)
    pipe = Pipeline([('preprocessor', pre), ('selector', sel), ('model', model)])

input_cols = list(pipe.feature_names_in_) if hasattr(pipe, 'feature_names_in_') else feature_cols
X_test = test[input_cols].copy()
y_test = test[targets[0]].values

proba = pipe.predict_proba(X_test)
classes = pipe.classes_ if hasattr(pipe, 'classes_') else pipe.named_steps['model'].classes_
positive_label = task.get('positive_label')
if positive_label is None:
    positive_label = classes[1]
pos_idx = list(classes).index(positive_label)
y_prob = proba[:, pos_idx]
y_true_bin = (y_test == positive_label).astype(int)

def compute_metrics(yt, yp, th):
    roc = roc_auc_score(yt, yp)
    pr = average_precision_score(yt, yp)
    brier = brier_score_loss(yt, yp)
    ypred = (yp >= th).astype(int)
    ba = balanced_accuracy_score(yt, ypred)
    tn, fp, fn, tp = confusion_matrix(yt, ypred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return {'roc_auc': roc, 'pr_auc': pr, 'balanced_accuracy': ba, 'brier_score': brier, 'sensitivity': sens, 'specificity': spec}

metrics = compute_metrics(y_true_bin, y_prob, threshold)

n_boot = 1000
rng = np.random.RandomState(42)
boot = {k: [] for k in metrics}
for _ in range(n_boot):
    idx = rng.randint(0, len(y_true_bin), len(y_true_bin))
    yt, yp = y_true_bin[idx], y_prob[idx]
    if len(np.unique(yt)) < 2:
        continue
    m = compute_metrics(yt, yp, threshold)
    for k in metrics:
        boot[k].append(m[k])

rows = []
for k in metrics:
    vals = [v for v in boot[k] if not np.isnan(v)]
    ci_low = np.percentile(vals, 2.5) if vals else np.nan
    ci_high = np.percentile(vals, 97.5) if vals else np.nan
    rows.append({'metric': k, 'value': metrics[k], 'ci_low': ci_low, 'ci_high': ci_high})

df_metrics = pd.DataFrame(rows)
mlca_runtime.save_table(df_metrics, 'test_metrics.csv')

prim = df_metrics[df_metrics['metric'] == primary_metric].iloc[0]
mlca_runtime.record_metric(primary_metric, prim['value'], 'test', ci=[prim['ci_low'], prim['ci_high']])
mlca_runtime.record_fact_candidate('test_roc_auc', prim['value'], source='test_metrics.csv')

print(f"Test {primary_metric}: {prim['value']:.4f} (CI {prim['ci_low']:.4f}-{prim['ci_high']:.4f})")
print(f"Threshold: {threshold}, positive label: {positive_label}")
