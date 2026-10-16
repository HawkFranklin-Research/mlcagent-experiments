import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
targets = proto["targets"]
excluded = list(proto.get("excluded_columns", []))
group = proto.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def keylike(s):
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_integer_dtype(s):
        return False
    return s.nunique(dropna=True) == len(s)

feat_cols = [c for c in train.columns
             if c not in targets and c not in excluded and c != group
             and not keylike(train[c])]

X_val = val[feat_cols]
Y_val = val[targets]

with open(inputs["05_preprocess_fit_preprocessing_Z1YWJ829/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_M3TS3EMK/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def full_pipeline(model):
    if isinstance(model, Pipeline):
        names = list(model.named_steps)
        has_pre = any("preprocess" in n.lower() for n in names)
        has_sel = any("select" in n.lower() for n in names)
        if has_pre and has_sel:
            return model
        parts = []
        if not has_pre:
            parts.append(("preprocessor", pre))
        if not has_sel:
            parts.append(("selector", sel))
        parts.extend(model.steps)
        return Pipeline(parts)
    return Pipeline([("preprocessor", pre), ("selector", sel), ("model", model)])

def get_encoders_from_model(model):
    if isinstance(model, Pipeline):
        est = model.steps[-1][1]
    else:
        est = model
    if hasattr(est, 'estimators_'):
        return [e.classes_ for e in est.estimators_]
    elif hasattr(est, 'classes_'):
        if isinstance(est.classes_, list) and len(est.classes_) > 0 and hasattr(est.classes_[0], '__len__'):
            return est.classes_
        return [est.classes_]
    return None

def get_encoders(model, manifest_path):
    try:
        with open(manifest_path) as f:
            man = json.load(f)
        for key in ['target_encoders', 'encoders', 'label_encoders', 'target_label_encoders']:
            if key in man:
                enc = man[key]
                if isinstance(enc, dict):
                    return [enc[t] for t in targets]
                return enc
    except Exception:
        pass
    return get_encoders_from_model(model)

cands = {
    "baseline": (inputs["07_model_train_baseline_JFX1KK5N/model.pkl"],
                 inputs["07_model_train_baseline_JFX1KK5N/manifest.json"]),
    "alternative": (inputs["08_model_train_alternative_CTC7G59X/alternative_model.pkl"],
                    inputs["08_model_train_alternative_CTC7G59X/manifest.json"]),
}

rows = []
best_name, best_score = None, -1.0
for name, (path, man_path) in cands.items():
    with open(path, "rb") as f:
        model = pickle.load(f)
    pipe = full_pipeline(model)
    pred = np.asarray(pipe.predict(X_val))
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    encoders = get_encoders(model, man_path)
    if encoders is None:
        raise ValueError("No encoders found")
    Y_enc = np.empty_like(pred, dtype=object)
    for i, t in enumerate(targets):
        classes = encoders[i]
        mapping = {c: j for j, c in enumerate(classes)}
        Y_enc[:, i] = Y_val[t].map(mapping).values
    per = [f1_score(Y_enc[:, i], pred[:, i], average="macro") for i in range(len(targets))]
    mf1 = float(np.mean(per))
    rows.append({"model": name, "metric": "macro_f1", "value": mf1})
    for i, t in enumerate(targets):
        rows.append({"model": name, "metric": f"macro_f1_{t}", "value": per[i]})
    hamming = float((pred != Y_enc).mean())
    subset = float((pred == Y_enc).all(axis=1).mean())
    rows.append({"model": name, "metric": "hamming_loss", "value": hamming})
    rows.append({"model": name, "metric": "subset_accuracy", "value": subset})
    if mf1 > best_score:
        best_score, best_name = mf1, name

df = pd.DataFrame(rows)
df["value"] = df["value"].round(4)
mlca_runtime.save_table(df, "validation_metrics.csv", "Validation metrics per candidate")
mlca_runtime.record_metric("macro_f1", best_score, "val")
mlca_runtime.record_fact_candidate("best_validation_score", best_score, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
print(f"features={len(feat_cols)} val_rows={len(val)}")
print(f"best={best_name} macro_f1={best_score:.4f}")
