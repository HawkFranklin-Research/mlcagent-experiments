import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing required key {k!r}; available top-level keys: {sorted(d.keys())}")
    return d[k]

protocol = json.loads(inputs["_protocol"].read_text())
targets = req(protocol, "targets")
excluded = req(protocol, "excluded_columns")
group_col = req(protocol, "group_column")
task = req(protocol, "task")
primary_metric = req(task, "primary_metric")

train_df = pd.read_parquet(inputs["train"])
val_df = pd.read_parquet(inputs["val"])

def is_keylike(s):
    if s.nunique(dropna=False) < len(s):
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        return True
    return False

exclude = set(targets) | set(excluded) | {group_col}
feature_cols = [c for c in train_df.columns if c not in exclude and not is_keylike(train_df[c])]

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

model_path = inputs.get("latest_model") or inputs["08_model_train_baseline_6RRDC1E6/model.pkl"]
model = load_pkl(model_path)

def final_est(obj):
    if hasattr(obj, "steps") and obj.steps:
        return obj.steps[-1][1]
    return obj

def base_est(obj):
    est = final_est(obj)
    if hasattr(est, "estimator"):
        return est.estimator
    if hasattr(est, "estimators_") and est.estimators_:
        return est.estimators_[0]
    return est

model_type = type(model).__name__
if hasattr(model, "steps"):
    model_type = f"Pipeline[{','.join(n + ':' + type(s).__name__ for n, s in model.steps)}]"

est = final_est(model)
be = base_est(model)
try:
    hp = "; ".join(f"{k}={v}" for k, v in be.get_params(deep=False).items()
                   if isinstance(v, (str, int, float, bool, type(None))))
except Exception:
    hp = "unavailable"
hp = hp[:500]

sel_features = None
sel_path = inputs.get("07_features_retain_features_Y0FEJMPH/selector_all.pkl")
if sel_path and sel_path.exists():
    sel = load_pkl(sel_path)
    sel_f = final_est(sel)
    if hasattr(sel_f, "get_support"):
        support = np.asarray(sel_f.get_support())
        names = None
        if hasattr(sel_f, "feature_names_in_"):
            names = list(sel_f.feature_names_in_)
        elif hasattr(sel_f, "get_feature_names_out"):
            try:
                names = list(sel_f.get_feature_names_out())
            except Exception:
                names = None
        if names is None and len(support) == len(feature_cols):
            names = feature_cols
        if names is not None and len(names) == len(support):
            sel_features = [n for n, keep in zip(names, support) if keep]

if not sel_features:
    sel_features = list(getattr(model, "feature_names_in_", feature_cols))
if not sel_features:
    sel_features = list(feature_cols)

imps = []
if hasattr(est, "estimators_"):
    for e in est.estimators_:
        if hasattr(e, "coef_"):
            imps.append(np.abs(np.asarray(e.coef_)).mean(axis=0).ravel())
        elif hasattr(e, "feature_importances_"):
            imps.append(np.asarray(e.feature_importances_).ravel())
elif hasattr(est, "coef_"):
    imps.append(np.abs(np.asarray(est.coef_)).mean(axis=0).ravel())
elif hasattr(est, "feature_importances_"):
    imps.append(np.asarray(est.feature_importances_).ravel())

top_feature = sel_features[0]
if imps and all(len(i) == len(imps[0]) for i in imps):
    imp = np.mean(imps, axis=0)
    if len(imp) == len(sel_features):
        top_feature = sel_features[int(np.argmax(imp))]

vmetrics = pd.read_csv(inputs["10_validate_score_validation_E0F8K8XX/validation_metrics.csv"])

def find_metric(df, name):
    if name in df.columns:
        s = pd.to_numeric(df[name], errors="coerce").dropna()
        if len(s):
            return float(s.iloc[0])
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object:
            mask = df[c].astype(str).str.lower().eq(name.lower())
            if mask.any():
                row = df.loc[mask].iloc[0]
                for vc in df.columns:
                    if vc == c:
                        continue
                    try:
                        v = float(row[vc])
                        if not np.isnan(v):
                            return v
                    except Exception:
                        pass
    return np.nan

val_metric = find_metric(vmetrics, primary_metric)

def r4(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    return float(f"{float(x):.4g}")

rows = [
    {"aspect": "task", "detail": "primary_metric", "value_text": primary_metric, "value_numeric": np.nan, "source": "_protocol"},
    {"aspect": "model", "detail": "class", "value_text": model_type, "value_numeric": np.nan, "source": "latest_model"},
    {"aspect": "model", "detail": "selected_hyperparameters", "value_text": hp, "value_numeric": np.nan, "source": "latest_model"},
    {"aspect": "inputs", "detail": "raw_feature_count", "value_text": np.nan, "value_numeric": r4(len(feature_cols)), "source": "train"},
    {"aspect": "inputs", "detail": "selected_feature_count", "value_text": np.nan, "value_numeric": r4(len(sel_features)), "source": "07_features_retain_features_Y0FEJMPH/selector_all.pkl"},
    {"aspect": "inputs", "detail": "selected_feature_sample", "value_text": ", ".join(map(str, sel_features[:5])), "value_numeric": np.nan, "source": "07_features_retain_features_Y0FEJMPH/selector_all.pkl"},
    {"aspect": "inputs", "detail": "train_rows", "value_text": np.nan, "value_numeric": r4(len(train_df)), "source": "train"},
    {"aspect": "inputs", "detail": "val_rows", "value_text": np.nan, "value_numeric": r4(len(val_df)), "source": "val"},
    {"aspect": "validation", "detail": primary_metric, "value_text": np.nan, "value_numeric": r4(val_metric), "source": "10_validate_score_validation_E0F8K8XX/validation_metrics.csv"},
    {"aspect": "interpretation", "detail": "top_feature", "value_text": top_feature, "value_numeric": np.nan, "source": "interpretation.csv"},
    {"aspect": "limitations", "detail": "scope", "value_text": "Frozen model only; no refitting. Validation evidence is not a test result. Multilabel fault indicators overlap, and coefficient/importance ranking is descriptive, not causal.", "value_numeric": np.nan, "source": "interpretation.csv"},
]
interp = pd.DataFrame(rows, columns=["aspect", "detail", "value_text", "value_numeric", "source"])
mlca_runtime.save_table(interp, "interpretation.csv", caption="Frozen model interpretation")

if np.isfinite(val_metric):
    mlca_runtime.record_metric(primary_metric, float(val_metric), "val")
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print("train/val rows:", len(train_df), len(val_df))
print("raw/selected features:", len(feature_cols), len(sel_features))
print("model:", model_type[:100])
print("top_feature:", top_feature)
print("val", primary_metric + ":", val_metric)
