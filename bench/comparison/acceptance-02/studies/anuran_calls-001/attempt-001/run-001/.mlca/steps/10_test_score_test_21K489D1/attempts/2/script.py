import mlca_runtime
import json, pickle
import pandas as pd, numpy as np
from sklearn.metrics import f1_score, hamming_loss
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
targets = proto['targets']
primary_metric = proto['task']['primary_metric']

val = pd.read_csv(inp['09_validate_score_validation_Z1N606V7/validation_metrics.csv'])
if 'model' in val.columns and primary_metric in val.columns:
    best_name = val.loc[val[primary_metric].idxmax(), 'model']
elif 'model' in val.columns and 'metric' in val.columns:
    cand = val[val['metric'] == primary_metric]
    best_name = cand.loc[cand['value'].idxmax(), 'model']
else:
    best_name = val[primary_metric].idxmax() if primary_metric in val.columns else 'baseline'

if 'alternative' in best_name.lower():
    model_path = inp['08_model_train_alternative_025XEA6J/alternative_model.pkl']
elif 'baseline' in best_name.lower():
    model_path = inp['07_model_train_baseline_R0Q02H91/model.pkl']
else:
    model_path = inp['latest_model']

model = pickle.loads(model_path.read_bytes())
test = pd.read_parquet(inp['test'])

def get_features(df, obj):
    if hasattr(obj, 'feature_names_in_'):
        return df[obj.feature_names_in_]
    drop = set(targets) | set(proto.get('excluded_columns', []))
    if proto.get('group_column'):
        drop.add(proto['group_column'])
    cols = [c for c in df.columns if c not in drop]
    for c in list(cols):
        if df[c].dtype == object and df[c].nunique() == len(df):
            cols.remove(c)
    return df[cols]

if isinstance(model, Pipeline):
    X_test = get_features(test, model)
    y_pred = model.predict(X_test)
else:
    pre = pickle.loads(inp['05_preprocess_fit_preprocessing_WHDXCN7B/preprocessor.pkl'].read_bytes())
    sel = pickle.loads(inp['06_features_retain_features_J1HGBJXK/selector_all.pkl'].read_bytes())
    X_test = get_features(test, pre)
    X_pre = pre.transform(X_test)
    if not isinstance(X_pre, pd.DataFrame):
        if hasattr(pre, 'get_feature_names_out'):
            X_pre = pd.DataFrame(X_pre, columns=pre.get_feature_names_out(), index=test.index)
        else:
            X_pre = pd.DataFrame(X_pre, index=test.index)
    X_sel = sel.transform(X_pre)
    if not isinstance(X_sel, pd.DataFrame):
        if hasattr(sel, 'get_feature_names_out'):
            X_sel = pd.DataFrame(X_sel, columns=sel.get_feature_names_out(), index=test.index)
        else:
            X_sel = pd.DataFrame(X_sel, index=test.index)
    y_pred = model.predict(X_sel)

y_true = test[targets].values
y_pred = np.asarray(y_pred)
if y_pred.ndim == 1:
    y_pred = y_pred.reshape(-1, 1)

f1s = [f1_score(y_true[:, i], y_pred[:, i], average='macro') for i in range(len(targets))]
macro_f1 = float(np.mean(f1s))
hls = [hamming_loss(y_true[:, i], y_pred[:, i]) for i in range(len(targets))]
avg_hl = float(np.mean(hls))
subset_acc = float(np.mean(np.all(y_true == y_pred, axis=1)))

metrics_df = pd.DataFrame({
    'model': [best_name] * 3,
    'metric': ['macro_f1', 'hamming_loss', 'subset_accuracy'],
    'value': [macro_f1, avg_hl, subset_acc]
})
mlca_runtime.save_table(metrics_df, 'test_metrics.csv')
mlca_runtime.record_metric(primary_metric, macro_f1, split='test')
mlca_runtime.record_fact_candidate('test_score', macro_f1, source='test_metrics.csv')

print(f"Selected model: {best_name}")
print(f"Test {primary_metric}: {macro_f1:.4f}")
