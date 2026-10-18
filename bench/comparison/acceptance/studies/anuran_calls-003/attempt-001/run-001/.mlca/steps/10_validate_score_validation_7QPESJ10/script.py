import mlca_runtime
import pickle, json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column')
primary_metric = protocol['task']['primary_metric']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

drop_cols = set(targets + excluded + ([group] if group else []))
feature_cols = []
for c in train.columns:
    if c in drop_cols:
        continue
    if train[c].nunique() == len(train) and (train[c].dtype == object or pd.api.types.is_integer_dtype(train[c])):
        continue
    feature_cols.append(c)

candidates = {
    'baseline': inputs['08_model_train_baseline_SJFKVA7H/model.pkl'],
    'alternative': inputs['09_model_train_alternative_GFWY19ZA/alternative_model.pkl']
}

pre_path = inputs['06_preprocess_fit_preprocessing_D9KP7ADR/preprocessor.pkl']
sel_path = inputs['07_features_retain_features_11DAHWEE/selector_all.pkl']

results = []
best_score = -np.inf
best_model = None

for name, path in candidates.items():
    with open(path, 'rb') as f:
        model = pickle.load(f)
    if hasattr(model, 'feature_names_in_'):
        input_cols = list(model.feature_names_in_)
    else:
        input_cols = feature_cols
    if not set(input_cols).issubset(val.columns):
        with open(pre_path, 'rb') as f:
            pre = pickle.load(f)
        with open(sel_path, 'rb') as f:
            sel = pickle.load(f)
        if isinstance(sel, Pipeline):
            model = Pipeline(sel.steps + [('model', model)])
        else:
            model = Pipeline([('preprocessor', pre), ('selector', sel), ('model', model)])
        if hasattr(model, 'feature_names_in_'):
            input_cols = list(model.feature_names_in_)
        else:
            input_cols = feature_cols

    X_val = val[input_cols].copy()
    preds = model.predict(X_val)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)

    macro_f1s = []
    hamming_losses = []
    for i, t in enumerate(targets):
        y_true = val[t].values
        y_pred = preds[:, i]
        macro_f1s.append(f1_score(y_true, y_pred, average='macro', zero_division=0))
        hamming_losses.append(hamming_loss(y_true, y_pred))

    mean_macro_f1 = np.mean(macro_f1s)
    mean_hamming = np.mean(hamming_losses)
    subset_acc = np.mean(np.all(preds == val[targets].values, axis=1))

    results.append({
        'model': name,
        'macro_f1': mean_macro_f1,
        'hamming_loss': mean_hamming,
        'subset_accuracy': subset_acc
    })

    if mean_macro_f1 > best_score:
        best_score = mean_macro_f1
        best_model = name

df = pd.DataFrame(results)
df_long = df.melt(id_vars='model', var_name='metric', value_name='value')
mlca_runtime.save_table(df_long, 'validation_metrics.csv')

mlca_runtime.record_metric(primary_metric, best_score, split='val')
mlca_runtime.record_fact_candidate('best_model', best_model, source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_validation_score', best_score, source='validation_metrics.csv')

print(f"Best model: {best_model} with {primary_metric}={best_score:.4f}")
