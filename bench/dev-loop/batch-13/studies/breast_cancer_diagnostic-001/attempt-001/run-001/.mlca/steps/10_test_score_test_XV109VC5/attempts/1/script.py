import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss, recall_score, f1_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
for k in ["targets", "excluded_columns", "task"]:
    if k not in protocol:
        raise KeyError(f"Missing {k}; available keys: {list(protocol)}")
targets = protocol["targets"]
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column")
task = protocol["task"]
primary_metric = task["primary_metric"]
train = pd.read_parquet(inp["train"])
test = pd.read_parquet(inp["test"])

def keylike(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=True) == len(s)
    return False

feature_cols = [c for c in train.columns if c not in set(targets) and c not in excluded and c != group and not keylike(train[c])]

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

model = load_pkl(inp["latest_model"])
if isinstance(model, dict):
    model = next(model[k] for k in ["model", "estimator", "pipeline"] if k in model)

def find_num(obj, needle):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if needle in str(k).lower() and isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
            r = find_num(v, needle)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_num(v, needle)
            if r is not None:
                return r
    return None

thr = None
for key in ["09_validate_score_validation_XQ95W1BE/manifest.json",
            "07_model_train_logistic_QHAMT8ZK/manifest.json",
            "08_model_train_alternative_SCMRPCK3/manifest.json"]:
    if key in inp:
        thr = find_num(json.loads(inp[key].read_text()), "threshold")
        if thr is not None:
            break

def build_predictor(m):
    if hasattr(m, "named_steps") and hasattr(m, "predict_proba"):
        def pred(df):
            cols = list(m.feature_names_in_) if hasattr(m, "feature_names_in_") else feature_cols
            return m.predict_proba(df[cols])
        return pred, list(m.classes_)
    pre = load_pkl(inp["05_preprocess_fit_preprocessing_XHNDRHCS/preprocessor.pkl"])
    sel = load_pkl(inp["06_features_select_features_4TDFQMJC/selector.pkl"])
    sel_has_pre = any("preprocess" in k.lower() for k in getattr(sel, "named_steps", {}))
    def pred(df):
        X = df[feature_cols]
        if sel_has_pre:
            Xs = sel.transform(X)
        else:
            Xp = pre.transform(X) if hasattr(pre, "transform") else X
            Xs = sel.transform(Xp) if hasattr(sel, "transform") else Xp
        return m.predict_proba(Xs)
    return pred, list(m.classes_)

predict_proba, classes = build_predictor(model)

if len(classes) == 2:
    pos_label = task.get("positive_label", task.get("positive_class"))
    if pos_label is None:
        pos_label = classes[1]
    pos_idx = classes.index(pos_label)
    if thr is None:
        val = pd.read_parquet(inp["val"])
        yv = val[targets[0]].to_numpy()
        pv = predict_proba(val)[:, pos_idx]
        from sklearn.metrics import roc_curve
        fpr, tpr, ths = roc_curve((yv == pos_label).astype(int), pv)
        thr = float(ths[np.argmax(tpr - fpr)])
    proba = predict_proba(test)
    y = test[targets[0]].to_numpy()
    p = proba[:, pos_idx]
    neg_label = [c for c in classes if c != pos_label][0]
    y_pred = np.where(p >= thr, pos_label, neg_label)

    def calc(idx=None):
        yy = y if idx is None else y[idx]
        pp = p if idx is None else p[idx]
        yp = y_pred if idx is None else y_pred[idx]
        yb = (yy == pos_label).astype(int)
        out = {
            "balanced_accuracy": balanced_accuracy_score(yy, yp),
            "brier_score": brier_score_loss(yb, pp),
            "sensitivity": recall_score(yy, yp, pos_label=pos_label, zero_division=0),
            "specificity": recall_score(yy, yp, pos_label=neg_label, zero_division=0),
        }
        out["roc_auc"] = roc_auc_score(yb, pp) if len(np.unique(yb)) > 1 else np.nan
        out["pr_auc"] = average_precision_score(yb, pp) if len(np.unique(yb)) > 1 else np.nan
        return out
else:
    proba = predict_proba(test)
    y = test[targets[0]].to_numpy()
    y_pred = np.array(classes)[np.argmax(proba, axis=1)]

    def calc(idx=None):
        yy = y if idx is None else y[idx]
        pp = proba if idx is None else proba[idx]
        yp = y_pred if idx is None else y_pred[idx]
        vals = [roc_auc_score(yy == c, pp[:, j]) for j, c in enumerate(classes)
                if (yy == c).any() and (yy != c).any()]
        return {
            "roc_auc_ovr_macro": float(np.mean(vals)) if vals else np.nan,
            "f1_macro": f1_score(yy, yp, average="macro", zero_division=0),
        }

base = calc()
rng = np.random.default_rng(42)
boots = {k: [] for k in base}
for _ in range(1000):
    idx = rng.integers(0, len(y), len(y))
    vals = calc(idx)
    for k, v in vals.items():
        if np.isfinite(v):
            boots[k].append(v)
rows = []
for k, v in base.items():
    arr = np.array(boots[k])
    lo, hi = np.percentile(arr, [2.5, 97.5]) if len(arr) else (np.nan, np.nan)
    rows.append({"metric": k, "value": v, "ci_low": lo, "ci_high": hi})

def sig4(x):
    return float(f"{x:.4g}") if pd.notna(x) and np.isfinite(x) else np.nan

metrics_df = pd.DataFrame(rows)
for c in ["value", "ci_low", "ci_high"]:
    metrics_df[c] = metrics_df[c].map(sig4)
mlca_runtime.save_table(metrics_df, "test_metrics.csv", "Test metrics with 95% bootstrap CI")
primary_row = metrics_df[metrics_df["metric"] == primary_metric]
if primary_row.empty:
    raise KeyError(f"Primary metric {primary_metric} not computed")
for _, r in metrics_df.iterrows():
    mlca_runtime.record_metric(r["metric"], r["value"], "test", ci=(r["ci_low"], r["ci_high"]))
if "roc_auc" in set(metrics_df["metric"]):
    v = metrics_df.loc[metrics_df["metric"] == "roc_auc", "value"].iloc[0]
    mlca_runtime.record_fact_candidate("test_roc_auc", v, source="test_metrics.csv")

pred_df = pd.DataFrame({"row_id": test.index, "partition": "test", "y_true": y, "y_pred": y_pred})
for j, c in enumerate(classes):
    pred_df[f"proba__{c}"] = proba[:, j]
mlca_runtime.save_row_outputs(pred_df, "test_predictions.parquet")
print("test rows", len(test), "features", len(feature_cols))
print("primary", primary_metric, float(primary_row["value"].iloc[0]))
