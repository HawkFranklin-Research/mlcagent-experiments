import json
import pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available: {list(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"] or [])
group = protocol["group_column"]
task = protocol["task"]
primary_metric = task.get("primary_metric", "")

train_df = pd.read_parquet(inputs["train"])
val_df = pd.read_parquet(inputs["val"])

def key_like(s: pd.Series) -> bool:
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
        return len(s) > 0 and s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s.dtype):
        v = s.dropna().to_numpy()
        if len(v) == len(s) and len(v) > 1 and len(np.unique(v)) == len(v):
            vs = np.sort(v)
            return bool(np.all(np.diff(vs) == 1) and vs[0] in (0, 1))
    return False

excl = set(targets) | set(excluded)
if group is not None:
    excl.add(group)
feature_cols = [c for c in train_df.columns if c not in excl and not key_like(train_df[c])]

def load_pkl(key):
    with open(inputs[key], "rb") as f:
        return pickle.load(f)

pre = load_pkl("05_preprocess_fit_preprocessing_XJYNH9KV/preprocessor.pkl")
sel = load_pkl("06_features_select_features_61DSRHRD/selector.pkl")
model = load_pkl("07_model_train_logistic_61NSXR0X/model.pkl")
alt = load_pkl("08_model_train_alternative_K9A47CPR/alternative_model.pkl")
latest = load_pkl("latest_model")

candidates = [("latest", latest), ("alternative", alt), ("logistic", model)]
final_pipeline = None
for _, obj in candidates:
    if hasattr(obj, "predict_proba"):
        try:
            obj.predict_proba(val_df[feature_cols].iloc[:1])
            final_pipeline = obj
            break
        except Exception:
            pass

if final_pipeline is None:
    def raw_front(obj):
        return hasattr(obj, "feature_names_in_") and all(
            c in obj.feature_names_in_ for c in feature_cols
        )
    front = sel if hasattr(sel, "transform") and raw_front(sel) else Pipeline([
        ("preprocessor", pre), ("selector", sel)
    ])
    final_pipeline = Pipeline([("frontend", front), ("model", model)])

if hasattr(final_pipeline, "feature_names_in_"):
    names = list(final_pipeline.feature_names_in_)
    missing_cols = [c for c in names if c not in val_df.columns]
    if missing_cols:
        raise KeyError(f"Missing model input columns: {missing_cols}")
    feature_cols = names

target = targets[0]
X_val = val_df[feature_cols].copy()
y_val = val_df[target].copy()

if primary_metric in ("roc_auc", "roc_auc_ovr", "roc_auc_ovr_weighted", "roc_auc_ovo", "roc_auc_ovo_weighted"):
    scoring = primary_metric
else:
    scoring = "roc_auc_ovr" if y_val.nunique() > 2 else "roc_auc"

result = permutation_importance(
    final_pipeline, X_val, y_val, scoring=scoring,
    n_repeats=20, random_state=42, n_jobs=1
)

importance = pd.DataFrame({
    "feature": feature_cols,
    "importance_mean": result.importances_mean,
    "importance_sd": result.importances_std,
}).sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)

def round_sig(x, sig=4):
    if pd.isna(x) or x == 0:
        return x
    return float(f"{x:.{sig}g}")

importance["importance_mean"] = importance["importance_mean"].map(round_sig)
importance["importance_sd"] = importance["importance_sd"].map(round_sig)

top_feature = str(importance.loc[0, "feature"])
mlca_runtime.save_table(importance, "interpretation.csv", caption="Permutation importance on validation rows")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"train rows={len(train_df)} val rows={len(val_df)}")
print(f"features={len(feature_cols)} scoring={scoring}")
print(f"pipeline={type(final_pipeline).__name__}")
print(f"top_feature={top_feature}")
