import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())

for k in ["targets", "excluded_columns", "task"]:
    if k not in proto:
        raise KeyError(f"MLCA: _protocol missing required key '{k}'. Available: {list(proto.keys())}")

targets = proto["targets"]
excluded = proto["excluded_columns"]
group = proto.get("group_column", None)

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def is_keylike(s):
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s) and s.nunique() == len(s):
        return True
    return False

feat_cols = [c for c in train.columns
             if c not in targets and c not in excluded and c != group
             and not is_keylike(train[c])]

with open(inp["05_preprocess_fit_preprocessing_78KYAZ91/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_retain_features_01V2VW4T/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def input_cols(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.named_steps.items():
            c = input_cols(step)
            if c:
                return c
    if hasattr(obj, "mlp_"):
        return input_cols(obj.mlp_)
    return None

def first_estimator(obj):
    if isinstance(obj, Pipeline):
        for _, step in obj.named_steps.items():
            return first_estimator(step)
    return obj

def get_n_features(obj):
    obj = first_estimator(obj)
    if hasattr(obj, "n_features_in_"):
        return obj.n_features_in_
    if hasattr(obj, "mlp_") and hasattr(obj.mlp_, "n_features_in_"):
        return obj.mlp_.n_features_in_
    return None

sel_cols = input_cols(sel)
if sel_cols and set(sel_cols).issubset(set(feat_cols)):
    front = sel
else:
    front = Pipeline([("pre", pre), ("sel", sel)])

front_cols = input_cols(front)
raw_cols = front_cols if front_cols else feat_cols
raw_cols = [c for c in raw_cols if c in val.columns]

Xval_raw = val[raw_cols]
Xval_trans = front.transform(Xval_raw)
X_trans_arr = np.asarray(Xval_trans)

candidates = {
    "baseline": inp["07_model_train_baseline_XDV836RH/model.pkl"],
    "alternative": inp["08_model_train_alternative_X9VQCMW4/alternative_model.pkl"],
}

results = []
for name, path in candidates.items():
    with open(path, "rb") as f:
        model = pickle.load(f)
    n_feat = get_n_features(model)
    n_raw = len(raw_cols)
    n_trans = X_trans_arr.shape[1]
    if n_feat is not None and n_feat == n_trans:
        pred = model.predict(Xval_trans)
    else:
        pred = model.predict(Xval_raw)
    pred = np.asarray(pred)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    mse = float(np.mean((pred - X_trans_arr) ** 2))
    results.append({"model": name, "metric": "reconstruction_error", "value": round(mse, 6)})

metrics = pd.DataFrame(results)
mlca_runtime.save_table(metrics, "validation_metrics.csv", caption="Validation reconstruction error (scaled space)")

best = min(results, key=lambda x: x["value"])
mlca_runtime.record_metric("reconstruction_error", best["value"], "val")
mlca_runtime.record_fact_candidate("best_validation_score", best["value"], source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_model", best["model"], source="validation_metrics.csv")

print(f"features={len(raw_cols)} val={Xval_raw.shape}")
print(f"best_model={best['model']} val_mse={best['value']:.6f}")
