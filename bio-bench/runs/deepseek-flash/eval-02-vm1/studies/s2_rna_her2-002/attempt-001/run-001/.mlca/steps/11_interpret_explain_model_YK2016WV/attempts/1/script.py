import mlca_runtime
import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(Path(inp["_protocol"]).read_text())
for k in ("task", "targets", "excluded_columns"):
    if k not in protocol:
        raise KeyError(f"Missing {k}; available top-level keys: {list(protocol.keys())}")

targets = list(protocol["targets"])
target_col = targets[0]
primary_metric = protocol["task"].get("primary_metric", "roc_auc")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
y = val[target_col].to_numpy()

with open(inp["latest_model"], "rb") as f:
    model = pickle.load(f)

raw_cols = None
if hasattr(model, "feature_names_in_"):
    raw_cols = [str(c) for c in model.feature_names_in_]
elif hasattr(model, "named_steps"):
    for step in model.named_steps.values():
        if hasattr(step, "feature_names_in_"):
            raw_cols = [str(c) for c in step.feature_names_in_]
            break
if raw_cols is None:
    exc = protocol.get("excluded_columns", [])
    exc = set(exc) if isinstance(exc, list) else set()
    exc |= set(targets)
    if protocol.get("group_column"):
        exc.add(protocol["group_column"])
    raw_cols = [c for c in val.columns if c not in exc]

final_est = model
if hasattr(model, "named_steps") and model.named_steps:
    final_est = list(model.named_steps.values())[-1]

def final_importances(est):
    if hasattr(est, "coef_"):
        c = np.asarray(est.coef_)
        if c.ndim == 1:
            c = c.reshape(1, -1)
        return np.abs(c).mean(axis=0), "coef_abs_mean"
    if hasattr(est, "feature_importances_"):
        return np.asarray(est.feature_importances_), "feature_importances_"
    return None, None

def clean_name(s):
    s = str(s)
    for p in ("num__", "cat__", "remainder__"):
        if s.startswith(p):
            return s[len(p):]
    return s

def importance_names(model, importances, raw_cols):
    if hasattr(model, "named_steps") and model.named_steps:
        try:
            names = model[:-1].get_feature_names_out(input_features=raw_cols)
            if len(names) == len(importances):
                return [clean_name(n) for n in names]
        except Exception:
            pass
    if len(importances) == len(raw_cols):
        return [clean_name(c) for c in raw_cols]
    try:
        with open(inp["06_features_select_features_FJMHG7YC/selector.pkl"], "rb") as f:
            selector = pickle.load(f)
        if hasattr(selector, "get_feature_names_out"):
            names = selector.get_feature_names_out(input_features=raw_cols)
            if len(names) == len(importances):
                return [clean_name(n) for n in names]
        if hasattr(selector, "get_support"):
            support = selector.get_support()
            if len(support) == len(raw_cols):
                return [clean_name(c) for c, s in zip(raw_cols, support) if s]
    except Exception:
        pass
    return None

def score(m, X, y):
    proba = m.predict_proba(X)
    if proba.ndim == 2 and proba.shape[1] == 2:
        return roc_auc_score(y, proba[:, 1])
    if len(np.unique(y)) > 2:
        return roc_auc_score(y, proba, multi_class="ovr", average="macro")
    return roc_auc_score(y, proba.ravel())

imp, method = final_importances(final_est)
if imp is not None:
    names = importance_names(model, imp, raw_cols)
    if names is not None:
        df = pd.DataFrame({"feature": names, "importance_mean": imp, "importance_sd": np.nan})
        df = df.sort_values("importance_mean", ascending=False).head(15).round(4)
        mlca_runtime.save_table(df, "interpretation.csv", caption="Top 15 frozen-model features")
        top = str(df.iloc[0]["feature"])
        mlca_runtime.record_fact_candidate("top_feature", top, source="interpretation.csv")
        mlca_runtime.record_metric("n_features_ranked", float(len(imp)), "val")
        mlca_runtime.record_metric("n_features_permuted", 0.0, "val")
        print(f"train rows={len(train)} val rows={len(val)} method={method} ranked={len(imp)} top={top}")
    else:
        df = pd.DataFrame({"feature": ["LIMITATION: no reliable feature-name mapping"], "importance_mean": [np.nan], "importance_sd": [np.nan]})
        mlca_runtime.save_table(df, "interpretation.csv", caption="Feature ranking unavailable")
        mlca_runtime.record_fact_candidate("top_feature", "unavailable", source="interpretation.csv")
        print(f"train rows={len(train)} val rows={len(val)} method=unavailable")
elif len(raw_cols) > 500:
    df = pd.DataFrame({"feature": ["LIMITATION: >500 input features and no reliable importances"], "importance_mean": [np.nan], "importance_sd": [np.nan]})
    mlca_runtime.save_table(df, "interpretation.csv", caption="Feature ranking unavailable")
    mlca_runtime.record_fact_candidate("top_feature", "unavailable", source="interpretation.csv")
    print(f"train rows={len(train)} val rows={len(val)} method=unavailable_large")
else:
    n_repeats = 20
    X = val[raw_cols].copy()
    base = score(model, X, y)
    rng = np.random.default_rng(42)
    rows = []
    for col in raw_cols:
        scores = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            scores.append(score(model, Xp, y))
        scores = np.asarray(scores)
        rows.append((col, base - scores.mean(), scores.std(ddof=1) if n_repeats > 1 else np.nan))
    df = pd.DataFrame(rows, columns=["feature", "importance_mean", "importance_sd"])
    df = df.sort_values("importance_mean", ascending=False).head(15).round(4)
    mlca_runtime.save_table(df, "interpretation.csv", caption="Top 15 permutation importances")
    top = str(df.iloc[0]["feature"])
    mlca_runtime.record_fact_candidate("top_feature", top, source="interpretation.csv")
    mlca_runtime.record_metric("n_features_permuted", float(len(raw_cols)), "val")
    mlca_runtime.record_metric("n_repeats", float(n_repeats), "val")
    print(f"train rows={len(train)} val rows={len(val)} method=permutation n={len(raw_cols)} repeats={n_repeats} top={top}")
