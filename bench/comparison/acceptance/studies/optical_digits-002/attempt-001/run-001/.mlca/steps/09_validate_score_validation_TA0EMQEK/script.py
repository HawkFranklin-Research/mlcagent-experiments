import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

try:
    task = protocol["task"]
    primary_metric = task["primary_metric"]
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
except KeyError as e:
    raise ValueError(f"Missing protocol key {e}; top-level keys: {list(protocol)}")
group_col = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def feature_columns(df, targets, excluded, group_col):
    drop = set(targets) | set(excluded)
    if group_col is not None:
        drop.add(group_col)
    cols = []
    for c in df.columns:
        if c in drop:
            continue
        s = df[c]
        if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
                or isinstance(s.dtype, pd.CategoricalDtype)):
            if len(s) > 1 and s.nunique(dropna=False) == len(s):
                continue
        if pd.api.types.is_integer_dtype(s) and len(s) > 1 and s.is_unique:
            continue
        cols.append(c)
    return cols

feature_cols = feature_columns(train, targets, excluded, group_col)
X_val = val[feature_cols].copy()

with open(inputs["05_preprocess_fit_preprocessing_XD81KYMB/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_retain_features_75C8096C/selector_all.pkl"], "rb") as f:
    selector = pickle.load(f)

def make_global_front(preprocessor, selector):
    if selector is None:
        return preprocessor
    if isinstance(selector, Pipeline) and len(selector.named_steps) > 1:
        return selector
    if preprocessor is None:
        return selector
    return Pipeline([("preprocessor", preprocessor), ("selector", selector)])

global_front = make_global_front(preprocessor, selector)

def get_front_final(model):
    if isinstance(model, Pipeline):
        steps = list(model.steps)
        final = steps[-1][1]
        front = model[:-1] if len(steps) > 1 else None
        return front, final
    return None, model

def transform_front(front, X):
    if front is None:
        raise ValueError("No frontend available")
    Z = front.transform(X)
    if hasattr(Z, "toarray"):
        Z = Z.toarray()
    return np.asarray(Z)

def score_candidate(model, X):
    front, final = get_front_final(model)
    if front is None:
        front = global_front
    Z = transform_front(front, X)
    if isinstance(final, PCA):
        recon = final.inverse_transform(final.transform(Z))
    elif isinstance(final, MLPRegressor):
        recon = final.predict(Z)
    else:
        raise TypeError(f"Unsupported final estimator: {type(final)}")
    recon = np.asarray(recon)
    if Z.shape != recon.shape:
        raise ValueError(f"Reconstruction shape {recon.shape} != frontend shape {Z.shape}")
    mse = float(np.mean((Z - recon) ** 2))
    return mse, Z.shape[1], type(final).__name__

models = [
    ("baseline", inputs["07_model_train_baseline_93EYVTD0/model.pkl"]),
    ("alternative", inputs["08_model_train_alternative_NZ42FXM6/alternative_model.pkl"]),
]
results = []
for key, path in models:
    with open(path, "rb") as f:
        model = pickle.load(f)
    mse, dim, name = score_candidate(model, X_val)
    results.append((name, mse, dim, key))

def sig4(x):
    return float(f"{x:.4g}")

metrics_df = pd.DataFrame([
    {"model": name, "metric": primary_metric, "value": sig4(mse)}
    for name, mse, _, _ in results
])
mlca_runtime.save_table(metrics_df, "validation_metrics.csv", caption="Validation reconstruction error by candidate")

best = min(results, key=lambda r: r[1])
best_name, best_mse = best[0], best[1]
best_saved = sig4(best_mse)
mlca_runtime.record_metric(primary_metric, best_mse, "val")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_validation_score", best_saved, source="validation_metrics.csv")

print(f"train/val shapes: {train.shape} {val.shape}")
print(f"features used: {len(feature_cols)}")
print(f"best {best_name}: {best_saved:.6g}")
