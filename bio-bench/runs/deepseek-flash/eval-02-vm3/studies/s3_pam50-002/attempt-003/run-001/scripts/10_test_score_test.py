import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
task = protocol["task"]
primary = task["primary_metric"]
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
target = targets[0]

train = pd.read_parquet(inputs["train"])
test = pd.read_parquet(inputs["test"])
external = pd.read_parquet(inputs["external"])

def is_key_like(s):
    if pd.api.types.is_integer_dtype(s) and s.is_unique:
        vals = np.sort(s.dropna().unique())
        return len(vals) > 1 and np.all(np.diff(vals) == 1)
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.is_unique and s.notna().all()
    return False

feature_cols = [c for c in train.columns
                if c not in targets and c not in excluded and c != group
                and not is_key_like(train[c])]

val = pd.read_csv(inputs["09_validate_score_validation_AHZRMGCJ/validation_metrics.csv"])
def higher_is_better(m):
    m = m.lower()
    return not any(k in m for k in ["brier", "log_loss", "loss", "error", "rmse", "mae", "mse"])

candidate = None
cand_cols = ["candidate", "model", "model_name", "candidate_name", "model_id", "name", "estimator", "variant"]
if primary in val.columns:
    ccol = next((c for c in cand_cols if c in val.columns), None)
    if "selected" in val.columns and val["selected"].any():
        candidate = val.loc[val.index[val["selected"]][0], ccol] if ccol else val.index[val["selected"]][0]
    elif ccol:
        g = val.groupby(ccol)[primary].mean()
        candidate = g.idxmax() if higher_is_better(primary) else g.idxmin()
    else:
        candidate = val[primary].idxmax() if higher_is_better(primary) else val[primary].idxmin()
else:
    mcol = next((c for c in ["metric", "metric_name", "name", "measure"] if c in val.columns), None)
    vcol = next((c for c in ["value", "score"] if c in val.columns), None)
    if mcol and vcol:
        sub = val[val[mcol].astype(str) == primary]
        ccol = next((c for c in cand_cols if c in sub.columns), None)
        if ccol:
            g = sub.groupby(ccol)[vcol].mean()
            candidate = g.idxmax() if higher_is_better(primary) else g.idxmin()
        elif len(sub):
            candidate = sub.index[0]

def resolve(cand):
    s = str(cand).lower()
    if "logistic" in s or "e9c4bbrq" in s:
        return inputs["07_model_train_logistic_E9C4BBRQ/model.pkl"]
    if "alternative" in s or "8t0dw9qp" in s or "latest" in s:
        return inputs["08_model_train_alternative_8T0DW9QP/alternative_model.pkl"]
    for k, p in inputs.items():
        if k.endswith(".pkl") and (s in k.lower() or k.lower().endswith(s)):
            return p
    return inputs["latest_model"]

model = pickle.load(open(resolve(candidate) if candidate is not None else inputs["latest_model"], "rb"))
if isinstance(model, Pipeline):
    predictor = model
else:
    pre = pickle.load(open(inputs["05_preprocess_fit_preprocessing_FS0JB8RV/preprocessor.pkl"], "rb"))
    sel = pickle.load(open(inputs["06_features_select_features_5R6M6WZH/selector.pkl"], "rb"))
    sel_names = getattr(sel, "feature_names_in_", None)
    if sel_names is not None and set(map(str, sel_names)).issubset(set(feature_cols)):
        predictor = Pipeline([("sel", sel), ("model", model)])
    else:
        predictor = Pipeline([("pre", pre), ("sel", sel), ("model", model)])

def X(df):
    first = predictor.steps[0][1] if isinstance(predictor, Pipeline) else predictor
    names = getattr(first, "feature_names_in_", None)
    return df[list(names)] if names is not None else df[feature_cols]

final = predictor.steps[-1][1] if isinstance(predictor, Pipeline) else predictor
classes = final.classes_

def auc_macro(y, p, cs):
    present = [j for j, c in enumerate(cs) if np.any(y == c)]
    if not present:
        return np.nan
    return float(np.mean([roc_auc_score((y == cs[j]).astype(int), p[:, j]) for j in present]))

def f1_macro(y, yp, cs):
    return float(f1_score(y, yp, labels=list(cs), average="macro", zero_division=0))

def boot(y, yp, p, fn, seed=42, n=1000):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        try:
            v = fn(y[idx], yp[idx], p[idx], classes)
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 20:
        return (np.nan, np.nan)
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return (float(lo), float(hi))

def evaluate(df):
    Xd = X(df)
    y = df[target].to_numpy()
    yp = predictor.predict(Xd)
    p = predictor.predict_proba(Xd)
    a = auc_macro(y, p, classes)
    f = f1_macro(y, yp, classes)
    aci = boot(y, yp, p, lambda yy, yp2, pp, cc: auc_macro(yy, pp, cc))
    fci = boot(y, yp, p, lambda yy, yp2, pp, cc: f1_macro(yy, yp2, cc))
    return dict(y=y, yp=yp, p=p, auc=a, f1=f, aci=aci, fci=fci)

test_res = evaluate(test)
ext_res = evaluate(external)

def sig(x):
    return float(f"{x:.4g}") if np.isfinite(x) else np.nan

rows = [
    {"metric": primary, "value": sig(test_res["auc"]), "ci_low": sig(test_res["aci"][0]), "ci_high": sig(test_res["aci"][1])},
    {"metric": "macro_f1", "value": sig(test_res["f1"]), "ci_low": sig(test_res["fci"][0]), "ci_high": sig(test_res["fci"][1])},
]
mlca_runtime.save_table(pd.DataFrame(rows), "test_metrics.csv", caption="Test metrics")
mlca_runtime.record_metric(primary, test_res["auc"], "test", ci=test_res["aci"])
mlca_runtime.record_metric("macro_f1", test_res["f1"], "test", ci=test_res["fci"])
mlca_runtime.record_fact_candidate("test_roc_auc", test_res["auc"], source="test_metrics.csv")
mlca_runtime.record_metric(primary, ext_res["auc"], "external", ci=ext_res["aci"])
mlca_runtime.record_metric("macro_f1", ext_res["f1"], "external", ci=ext_res["fci"])

out = pd.DataFrame({
    "row_id": test.index,
    "partition": "test",
    "y_true": test_res["y"],
    "y_pred": test_res["yp"],
})
for j, c in enumerate(classes):
    out[f"proba__{c}"] = test_res["p"][:, j]
mlca_runtime.save_row_outputs(out, "test_predictions.parquet")

missing = [c for c in classes if not np.any(ext_res["y"] == c)]
print(f"selected={candidate} n_test={len(test)} {primary}={test_res['auc']:.4f} n_external={len(external)} external_{primary}={ext_res['auc']:.4f}")
if missing:
    print(f"external missing classes: {missing}")
