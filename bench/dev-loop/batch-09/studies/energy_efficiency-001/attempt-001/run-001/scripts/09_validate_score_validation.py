import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

targets = list(protocol['targets'])
excluded = list(protocol['excluded_columns'])
group_col = protocol.get('group_column')
primary = protocol['task']['primary_metric']
lower_better = primary.lower() in ('rmse', 'mae', 'mse', 'log_loss', 'mape')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

def feature_columns(df):
    drop = set(targets) | set(excluded) | ({group_col} if group_col else set())
    keep = []
    for c in df.columns:
        if c in drop:
            continue
        s = df[c]
        if s.dtype.kind in 'iu' and s.nunique() == len(s) and s.nunique() > 1:
            continue
        if (pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s)) and s.nunique() == len(s):
            continue
        keep.append(c)
    return keep

features = feature_columns(train)
Xval = val[features]

def load(key):
    with open(inputs[key], 'rb') as f:
        return pickle.load(f)

pre = load('05_preprocess_fit_preprocessing_0CQ66EDA/preprocessor.pkl')
sel = load('06_features_retain_features_JTARV5X3/selector_all.pkl')
base = load('07_model_train_baseline_Q7Z8BHZ9/model.pkl')
alt = load('08_model_train_alternative_5G4S54TV/alternative_model.pkl')

def candidates(model):
    out = []
    if isinstance(model, Pipeline):
        out.append(model)
    out.append(Pipeline([('pre', pre), ('sel', sel), ('model', model)]))
    out.append(Pipeline([('pre', pre), ('model', model)]))
    out.append(Pipeline([('sel', sel), ('model', model)]))
    return out

def predict_pipeline(model):
    last = None
    for c in candidates(model):
        try:
            p = np.asarray(c.predict(Xval))
            if p.ndim == 1 and len(targets) == 1:
                p = p.reshape(-1, 1)
            if p.shape[0] == len(Xval) and p.ndim == 2:
                return c, p
        except Exception as e:
            last = e
    raise RuntimeError(f"no working pipeline: {last}")

y_val = val[targets].to_numpy()

def metrics(y_true, y_pred):
    per = []
    for i in range(len(targets)):
        yt, yp = y_true[:, i], y_pred[:, i]
        per.append({
            'rmse': float(np.sqrt(mean_squared_error(yt, yp))),
            'mae': float(mean_absolute_error(yt, yp)),
            'r2': float(r2_score(yt, yp)),
        })
    macro = {k: float(np.mean([p[k] for p in per])) for k in ('rmse', 'mae', 'r2')}
    return per, macro

results = {}
rows = []
for name, mdl in [('baseline', base), ('alternative', alt)]:
    pipe, preds = predict_pipeline(mdl)
    per, macro = metrics(y_val, preds)
    results[name] = (per, macro)
    for i, t in enumerate(targets):
        for k in ('rmse', 'mae', 'r2'):
            rows.append({'model': name, 'metric': f'{t}_{k}', 'value': per[i][k]})
    for k in ('rmse', 'mae', 'r2'):
        rows.append({'model': name, 'metric': f'macro_{k}', 'value': macro[k]})

vdf = pd.DataFrame(rows)
vdf['value'] = vdf['value'].round(6)
mlca_runtime.save_table(vdf, 'validation_metrics.csv', 'Validation metrics per model/target')

if lower_better:
    best_name = min(results, key=lambda n: results[n][1][primary])
else:
    best_name = max(results, key=lambda n: results[n][1][primary])
best_val = float(results[best_name][1][primary])

for i, t in enumerate(targets):
    mlca_runtime.record_metric(f'{t}_{primary}', results[best_name][0][i][primary], 'val')
mlca_runtime.record_metric(primary, best_val, 'val')
mlca_runtime.record_fact_candidate('best_model', best_name, source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_validation_score', best_val, source='validation_metrics.csv')
print(f"selected={best_name} val_{primary}={best_val:.4f}")
