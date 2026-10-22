import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
try:
    targets = list(proto["targets"])
    excluded = list(proto.get("excluded_columns", []))
    task = proto["task"]
    primary = task["primary_metric"]
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available top-level keys: {list(proto)}")
target_col = targets[0]
if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")
group_col = proto.get("group_column")
train = pd.read_parquet(inp["train"])
test = pd.read_parquet(inp["test"])

raw_features = []
for c in train.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    s = train[c]
    is_text = pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)
    is_int = pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s)
    all_distinct = s.notna().all() and s.nunique(dropna=False) == len(train)
    if is_text and all_distinct:
        continue
    if is_int and all_distinct and (s.min() in (0, 1) and s.max() - s.min() + 1 == len(s)):
        continue
    if is_int and all_distinct and (c.lower().endswith("id") or c.lower() == "id"):
        continue
    raw_features.append(c)

def load_pkl(k):
    with open(inp[k], "rb") as f:
        return pickle.load(f)

pre = load_pkl("06_preprocess_fit_preprocessing_Q5T3SXM3/preprocessor.pkl")
sel = load_pkl("07_features_select_features_Y03WXMRF/selector.pkl")
model = load_pkl("08_model_train_logistic_MMDVH86M/model.pkl")
alt = load_pkl("09_model_train_alternative_4GY8FEXD/alternative_model.pkl")

def load_json_key(k):
    return json.loads(inp[k].read_text())

mman = load_json_key("08_model_train_logistic_MMDVH86M/manifest.json")
aman = load_json_key("09_model_train_alternative_4GY8FEXD/manifest.json")
vman = load_json_key("10_validate_score_validation_6GADJVGB/manifest.json")
vmet = pd.read_csv(inp["10_validate_score_validation_6GADJVGB/validation_metrics.csv"])

selected = None
for k in ("selected_model", "best_model", "selected_candidate", "chosen_model", "model"):
    if isinstance(vman, dict) and k in vman:
        selected = str(vman[k])
        break
if selected is None and "model" in vmet.columns:
    sub = vmet
    if "metric" in vmet.columns:
        sub = vmet[vmet["metric"].astype(str).str.lower() == primary.lower()]
    if len(sub) and "value" in sub.columns:
        lower_better = any(x in primary.lower() for x in ("loss", "brier", "error", "mae", "mse", "rmse"))
        selected = str(sub.sort_values("value", ascending=lower_better).iloc[0]["model"])
if selected is None:
    selected = "alternative"
sl = selected.lower()
if "alternative" in sl or "09" in sl:
    eval_model, model_manifest = alt, aman
elif "logistic" in sl or "08" in sl:
    eval_model, model_manifest = model, mman
else:
    matches = [(m, j) for m, j in ((model, mman), (alt, aman)) if sl in json.dumps(j).lower()]
    eval_model, model_manifest = matches[0] if len(matches) == 1 else (alt, aman)

def feats(est):
    if hasattr(est, "feature_names_in_"):
        return list(est.feature_names_in_)
    if hasattr(est, "named_steps"):
        first = next(iter(est.named_steps.values()))
        if hasattr(first, "feature_names_in_"):
            return list(first.feature_names_in_)
    return None

def classes(est):
    if hasattr(est, "classes_"):
        return np.asarray(est.classes_)
    if hasattr(est, "named_steps"):
        return np.asarray(list(est.named_steps.values())[-1].classes_)
    raise AttributeError("classes_")

if feats(eval_model) is None or set(feats(eval_model)) != set(raw_features):
    if feats(sel) and set(feats(sel)) == set(raw_features):
        eval_model = Pipeline([("sel", sel), ("clf", eval_model)])
    elif feats(pre) and set(feats(pre)) == set(raw_features):
        eval_model = Pipeline([("pre", pre), ("sel", sel), ("clf", eval_model)])
    else:
        raise ValueError("Cannot assemble frozen preprocessing/selector/model accepting raw features")

cols = feats(eval_model) or raw_features
bad = set(cols) & (set(targets) | set(excluded))
if bad:
    raise ValueError(f"Feature input contains target/excluded columns: {bad}")
X_test = test[list(cols)].copy()
proba = eval_model.predict_proba(X_test)
cls = classes(eval_model)
y_true = test[target_col].to_numpy()

def find_threshold():
    if "metric" in vmet.columns and "value" in vmet.columns:
        m = vmet["metric"].astype(str).str.lower().str.contains("threshold")
        if m.any():
            return float(vmet.loc[m, "value"].iloc[0])
    for manifest in (vman, model_manifest):
        for k in ("threshold", "selected_threshold", "decision_threshold", "best_threshold", "frozen_threshold"):
            if isinstance(manifest, dict) and k in manifest:
                return float(manifest[k])
    val = pd.read_parquet(inp["val"])
    pv = eval_model.predict_proba(val[list(cols)].copy())
    pos = task.get("positive_label", task.get("positive_class", None))
    if pos is None:
        pos = cls[1]
    ppos = pv[:, list(cls).index(pos)]
    yb = (val[target_col].to_numpy() == pos).astype(int)
    best_t, best_s = 0.5, -1.0
    for t in np.unique(ppos):
        s = balanced_accuracy_score(yb, (ppos >= t).astype(int))
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t

if len(cls) == 2 or str(task.get("type", "")).lower() == "binary":
    pos = task.get("positive_label", task.get("positive_class", None))
    pos = cls[1] if pos is None or pos not in cls else pos
    j = list(cls).index(pos)
    other = cls[0] if pos != cls[0] else cls[1]
    ppos = proba[:, j]
    yb = (y_true == pos).astype(int)
    thr = find_threshold()
    ypred_bin = (ppos >= thr).astype(int)
    ypred = np.where(ypred_bin == 1, pos, other)

    def point_metrics(yy, pp, yp):
        return {
            "roc_auc": roc_auc_score(yy, pp) if len(np.unique(yy)) > 1 else np.nan,
            "pr_auc": average_precision_score(yy, pp) if len(np.unique(yy)) > 1 else np.nan,
            "balanced_accuracy": balanced_accuracy_score(yy, yp),
            "brier_score": np.mean((pp - yy) ** 2),
            "sensitivity": yp[yy == 1].mean() if (yy == 1).any() else np.nan,
            "specificity": 1 - yp[yy == 0].mean() if (yy == 0).any() else np.nan,
        }

    vals = point_metrics(yb, ppos, ypred_bin)
    rng = np.random.default_rng(42)
    n = len(yb)
    boot = {k: [] for k in vals}
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        ybi, ppi, ypi = yb[idx], ppos[idx], ypred_bin[idx]
        if len(np.unique(ybi)) > 1:
            boot["roc_auc"].append(roc_auc_score(ybi, ppi))
            boot["pr_auc"].append(average_precision_score(ybi, ppi))
        boot["balanced_accuracy"].append(balanced_accuracy_score(ybi, ypi))
        boot["brier_score"].append(np.mean((ppi - ybi) ** 2))
        if (ybi == 1).any():
            boot["sensitivity"].append(ypi[ybi == 1].mean())
        if (ybi == 0).any():
            boot["specificity"].append(1 - ypi[ybi == 0].mean())

    rows = []
    for k, v in vals.items():
        arr = np.asarray(boot[k], float)
        arr = arr[np.isfinite(arr)]
        lo, hi = np.percentile(arr, [2.5, 97.5]) if len(arr) else (np.nan, np.nan)
        rows.append({"metric": k, "value": v, "ci_low": lo, "ci_high": hi})
    metrics_df = pd.DataFrame(rows)
    pr = metrics_df[metrics_df["metric"].str.lower() == primary.lower()].iloc[0]
    mlca_runtime.record_metric(primary, float(pr["value"]), "test", ci=(float(pr["ci_low"]), float(pr["ci_high"])))
    if primary.lower() == "roc_auc":
        mlca_runtime.record_fact_candidate("test_roc_auc", float(pr["value"]), source="test_metrics.csv")
    mlca_runtime.save_table(metrics_df.round(4), "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")
    out_df = pd.DataFrame({"row_id": test.index, "partition": "test", f"y_true__{target_col}": y_true, f"y_pred__{target_col}": ypred})
    for j, c in enumerate(cls):
        out_df[f"proba__{c}"] = proba[:, j]
    mlca_runtime.save_row_outputs(out_df, "test_predictions.parquet")
    print(f"test rows={len(test)} features={len(cols)} classes={len(cls)}")
    print(f"{primary}={pr['value']:.4g} CI=({pr['ci_low']:.4g},{pr['ci_high']:.4g})")
else:
    ypred = cls[np.argmax(proba, axis=1)]
    present = [c for c in cls if c in set(y_true)]

    def macro_ovr(y, P):
        vals = []
        for c in present:
            j = list(cls).index(c)
            yb = (y == c).astype(int)
            if len(np.unique(yb)) > 1:
                vals.append(roc_auc_score(yb, P[:, j]))
        return float(np.mean(vals)) if vals else np.nan

    vals = {"roc_auc_ovr_macro": macro_ovr(y_true, proba), "f1_macro": f1_score(y_true, ypred, average="macro")}
    rng = np.random.default_rng(42)
    n = len(y_true)
    boot = {k: [] for k in vals}
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        yt, P, yp = y_true[idx], proba[idx], ypred[idx]
        boot["roc_auc_ovr_macro"].append(macro_ovr(yt, P))
        boot["f1_macro"].append(f1_score(yt, yp, average="macro"))
    rows = []
    for k, v in vals.items():
        arr = np.asarray(boot[k], float)
        arr = arr[np.isfinite(arr)]
        lo, hi = np.percentile(arr, [2.5, 97.5]) if len(arr) else (np.nan, np.nan)
        rows.append({"metric": k, "value": v, "ci_low": lo, "ci_high": hi})
    metrics_df = pd.DataFrame(rows)
    if primary.lower() not in metrics_df["metric"].str.lower().values:
        src_key = "roc_auc_ovr_macro" if "roc_auc" in primary.lower() else "f1_macro"
        src = metrics_df[metrics_df["metric"] == src_key].iloc[0]
        metrics_df = pd.concat([metrics_df, pd.DataFrame([{"metric": primary, "value": src["value"], "ci_low": src["ci_low"], "ci_high": src["ci_high"]}])], ignore_index=True)
    pr = metrics_df[metrics_df["metric"].str.lower() == primary.lower()].iloc[0]
    mlca_runtime.record_metric(primary, float(pr["value"]), "test", ci=(float(pr["ci_low"]), float(pr["ci_high"])))
    mlca_runtime.save_table(metrics_df.round(4), "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")
    out_df = pd.DataFrame({"row_id": test.index, "partition": "test", f"y_true__{target_col}": y_true, f"y_pred__{target_col}": ypred})
    for j, c in enumerate(cls):
        out_df[f"proba__{c}"] = proba[:, j]
    mlca_runtime.save_row_outputs(out_df, "test_predictions.parquet")
    print(f"test rows={len(test)} features={len(cols)} classes={len(cls)}")
    print(f"{primary}={pr['value']:.4g} CI=({pr['ci_low']:.4g},{pr['ci_high']:.4g})")
