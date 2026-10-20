import mlca_runtime, json, pickle
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
P = mlca_runtime.inputs()

def J(k):
    return json.loads(P[k].read_text())

proto = J("_protocol")
need = ["targets", "excluded_columns", "group_column", "task"]
miss = [k for k in need if k not in proto]
if miss:
    raise KeyError(f"Missing _protocol keys {miss}; available: {list(proto.keys())}")

targets = proto["targets"]
excluded = proto["excluded_columns"]
group = proto["group_column"]
task = proto["task"]
primary = task["primary_metric"]

train = pd.read_parquet(P["train"])
val = pd.read_parquet(P["val"])

def key_like(s):
    if s.nunique(dropna=True) != len(s):
        return False
    return bool(pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s))

feat_cols = [
    c for c in train.columns
    if c not in targets and c not in excluded and c != group and not key_like(train[c])
]

base_man = J("07_model_train_baseline_2DXYNCH8/manifest.json")
alt_man = J("08_model_train_alternative_QF453D1M/manifest.json")
model = pickle.loads(P["latest_model"].read_bytes())
base = pickle.loads(P["07_model_train_baseline_2DXYNCH8/model.pkl"].read_bytes())

def params(m):
    if isinstance(m, dict):
        for k in ("params", "hyperparameters", "best_params", "selected_hyperparameters", "model_params", "estimator_params"):
            if k in m:
                return m[k]
        for v in m.values():
            if isinstance(v, dict):
                p = params(v)
                if p is not None:
                    return p
    return None

def imp_vec(est):
    if hasattr(est, "feature_importances_"):
        return np.asarray(est.feature_importances_, float)
    if hasattr(est, "coef_"):
        a = est.coef_
        if isinstance(a, list):
            vs = []
            for x in a:
                x = np.abs(np.asarray(x, float))
                vs.append(x.mean(axis=0) if x.ndim > 1 else x)
            return np.mean(vs, axis=0) if vs else None
        a = np.abs(np.asarray(a, float))
        return a.mean(axis=0) if a.ndim > 1 else a
    return None

def imp_of(est):
    if hasattr(est, "estimators_"):
        vs = [imp_vec(e) for e in est.estimators_]
        vs = [v for v in vs if v is not None]
        if vs:
            return np.mean(vs, axis=0)
    return imp_vec(est)

def top_feat(m, names):
    steps = list(m.named_steps.values()) if hasattr(m, "named_steps") else [m]
    imp = imp_of(steps[-1])
    if imp is None:
        return None
    for st in reversed(steps):
        if hasattr(st, "get_support"):
            try:
                sup = np.asarray(st.get_support())
            except Exception:
                continue
            if int(sup.sum()) != len(imp):
                continue
            sn = np.asarray(st.feature_names_in_) if hasattr(st, "feature_names_in_") else np.asarray(names)
            if len(sn) == len(sup):
                return str(sn[sup][int(np.argmax(imp))])
    for obj in [steps[-1], m]:
        if hasattr(obj, "feature_names_in_") and len(obj.feature_names_in_) == len(imp):
            return str(obj.feature_names_in_[int(np.argmax(imp))])
    if len(names) == len(imp):
        return str(np.asarray(names)[int(np.argmax(imp))])
    return None

top = top_feat(model, feat_cols)
if top is None:
    top = top_feat(base, feat_cols) or "unavailable"

def metric_val():
    for k in (
        "09_validate_score_validation_FQ87WBM2/validation_metrics.csv",
        "09_validate_score_validation_FQ87WBM2/validation_metrics.parquet",
    ):
        try:
            df = pd.read_csv(P[k]) if k.endswith(".csv") else pd.read_parquet(P[k])
        except Exception:
            continue
        if "metric" in df.columns and "value" in df.columns:
            hit = df[df["metric"].astype(str).str.lower().str.contains(str(primary).lower(), regex=False)]
            if len(hit):
                return float(hit["value"].iloc[0]), k
        if primary in df.columns:
            return float(df[primary].iloc[0]), k
        num = df.select_dtypes("number")
        if num.shape[1] == 1 and len(num):
            return float(num.iloc[0, 0]), k
    return None, None

vmet, vfile = metric_val()
inp_names = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else feat_cols

rows = [
    {"item": "frozen_model", "detail": f"{type(model).__name__} loaded from latest_model", "evidence_file": "latest_model"},
    {"item": "selected_hyperparameters", "detail": json.dumps(params(alt_man), default=str)[:1200], "evidence_file": "08_model_train_alternative_QF453D1M/manifest.json"},
    {"item": "baseline_hyperparameters", "detail": json.dumps(params(base_man), default=str)[:1200], "evidence_file": "07_model_train_baseline_2DXYNCH8/manifest.json"},
    {"item": "input_features", "detail": f"{len(inp_names)} columns: {', '.join(map(str, inp_names))}"[:1200], "evidence_file": "latest_model"},
    {"item": "excluded_columns", "detail": ", ".join(map(str, excluded)), "evidence_file": "_protocol"},
    {"item": "targets", "detail": ", ".join(map(str, targets)), "evidence_file": "_protocol"},
    {"item": "primary_metric", "detail": str(primary), "evidence_file": "_protocol"},
]
if vmet is not None:
    rows.append({"item": "validation_primary_metric", "detail": f"{primary}={vmet:.6g}", "evidence_file": vfile})

rows += [
    {"item": "validation_evidence", "detail": f"validation metrics and manifest; task={task.get('type')}", "evidence_file": "09_validate_score_validation_FQ87WBM2/manifest.json"},
    {"item": "limitation", "detail": "Interpretation only; no estimator was refit and no validation/test rows were used for fitting.", "evidence_file": vfile or "09_validate_score_validation_FQ87WBM2/manifest.json"},
    {"item": "limitation", "detail": "Y_Minimum/Y_Maximum are near-unique record identifiers and excluded; model inputs are the saved pipeline's declared feature columns.", "evidence_file": "_protocol"},
    {"item": "limitation", "detail": "Multilabel targets are seven binary fault indicators; macro F1 is averaged per target, not a single multiclass score.", "evidence_file": "_protocol"},
    {"item": "top_feature", "detail": str(top), "evidence_file": "latest_model"},
]

out = pd.DataFrame(rows)
mlca_runtime.save_table(out, "interpretation.csv", caption="Frozen multilabel model interpretation summary")
mlca_runtime.record_fact_candidate("top_feature", str(top), source="interpretation.csv")
if vmet is not None:
    mlca_runtime.record_metric(f"validation_{primary}", float(vmet), "val")

print(f"train rows={len(train)} val rows={len(val)}")
print(f"model={type(model).__name__} features={len(inp_names)}")
print(f"top_feature={top}")
print(f"validation_{primary}={vmet}")
print("saved interpretation.csv")
