import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, poisson_deviance

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
need = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in need if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available top-level keys: {sorted(protocol.keys())}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol.get("group_column")
primary_metric = protocol["task"]["primary_metric"]

def read_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
pre = read_pkl(inputs["05_preprocess_fit_preprocessing_Q23GTBD5/preprocessor.pkl"])
sel = read_pkl(inputs["06_features_retain_features_R3TMNHH4/selector_all.pkl"])
base = read_pkl(inputs["07_model_train_baseline_4E4949CT/model.pkl"])
alt = read_pkl(inputs["08_model_train_alternative_4C73HMKJ/alternative_model.pkl"])

def is_key_like(s, name):
    if len(s) == 0 or s.nunique(dropna=False) != len(s):
        return False
    if s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
        return True
    if pd.api.types.is_integer_dtype(s.dtype):
        low = name.lower()
        return bool(s.is_monotonic_increasing or s.is_monotonic_decreasing or
                    any(t in low for t in ["id", "index", "instant", "code"]))
    return False

exclude_cols = set(targets) | set(excluded) | ({group_col} if group_col else set())
feature_cols = [c for c in train.columns
                if c not in exclude_cols and not is_key_like(train[c], c)]

def assemble(model_obj):
    if isinstance(model_obj, Pipeline):
        return model_obj
    if isinstance(sel, Pipeline):
        return Pipeline(list(sel.steps) + [("model", model_obj)])
    steps = []
    if pre is not None:
        steps.append(("preprocessor", pre))
    if sel is not None:
        steps.append(("selector", sel))
    steps.append(("model", model_obj))
    return Pipeline(steps)

def expected_input_cols(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

def metric_value(name, yt, yp):
    if name == "poisson_deviance":
        return poisson_deviance(yt, np.clip(yp, 1e-12, None))
    if name == "rmse":
        return float(np.sqrt(mean_squared_error(yt, yp)))
    if name == "mae":
        return mean_absolute_error(yt, yp)
    if name == "r2":
        return r2_score(yt, yp)
    raise ValueError(f"Unsupported primary metric: {name}")

candidates = {"baseline": base, "alternative": alt}
rows, candidate_primary = [], {}
for name, obj in candidates.items():
    pipe = assemble(obj)
    cols = expected_input_cols(pipe) or feature_cols
    bad = set(cols) & exclude_cols
    if bad:
        raise ValueError(f"{name}: targets/excluded columns reached model: {sorted(bad)}")
    miss = [c for c in cols if c not in val.columns]
    if miss:
        raise KeyError(f"{name}: missing validation columns: {miss}")
    X = val[cols]
    pred = pipe.predict(X)
    pred = pred.reshape(len(val), -1) if np.ndim(pred) == 1 else np.asarray(pred)
    y = val[targets].to_numpy()
    primary_vals = []
    for j, t in enumerate(targets):
        yt, yp = y[:, j], pred[:, j]
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mae = mean_absolute_error(yt, yp)
        r2 = r2_score(yt, yp)
        pv = metric_value(primary_metric, yt, yp)
        primary_vals.append(pv)
        rows += [
            {"model": name, "metric": f"rmse_{t}", "value": rmse},
            {"model": name, "metric": f"mae_{t}", "value": mae},
            {"model": name, "metric": f"r2_{t}", "value": r2},
            {"model": name, "metric": f"{primary_metric}_{t}", "value": pv},
        ]
    macro_rmse = np.mean([r["value"] for r in rows if r["model"] == name and r["metric"].startswith("rmse_")])
    macro_mae = np.mean([r["value"] for r in rows if r["model"] == name and r["metric"].startswith("mae_")])
    macro_r2 = np.mean([r["value"] for r in rows if r["model"] == name and r["metric"].startswith("r2_")])
    macro_primary = float(np.mean(primary_vals))
    candidate_primary[name] = macro_primary
    rows += [
        {"model": name, "metric": "macro_rmse", "value": macro_rmse},
        {"model": name, "metric": "macro_mae", "value": macro_mae},
        {"model": name, "metric": "macro_r2", "value": macro_r2},
        {"model": name, "metric": primary_metric, "value": macro_primary},
    ]

higher_better = primary_metric in {"r2", "explained_variance", "accuracy", "f1", "f1_score",
                                   "roc_auc", "average_precision", "balanced_accuracy",
                                   "precision", "recall"}
finite = {k: v for k, v in candidate_primary.items() if np.isfinite(v)}
if not finite:
    raise ValueError("No finite validation primary-metric scores")
best_model = max(finite, key=finite.get) if higher_better else min(finite, key=finite.get)
best_score = float(finite[best_model])

def rs(v):
    return float(f"{float(v):.4g}") if np.isfinite(v) else v

out = pd.DataFrame(rows).sort_values(["model", "metric"]).reset_index(drop=True)
out["value"] = out["value"].map(rs)
mlca_runtime.save_table(out, "validation_metrics.csv", caption="Validation metrics per candidate model.")
mlca_runtime.record_metric(primary_metric, best_score, split="val")
mlca_runtime.record_fact_candidate("best_model", best_model, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_validation_score", best_score, source="validation_metrics.csv")

print(f"train/val shapes: {train.shape}, {val.shape}")
print(f"targets: {targets}")
print(f"feature columns: {len(feature_cols)}")
print(f"best model: {best_model}")
print(f"best {primary_metric}: {best_score:.4g}")
