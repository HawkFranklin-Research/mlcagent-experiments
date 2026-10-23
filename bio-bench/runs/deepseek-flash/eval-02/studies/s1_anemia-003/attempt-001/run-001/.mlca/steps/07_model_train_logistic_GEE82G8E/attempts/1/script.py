import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def get_key(d, key):
    if key not in d:
        raise KeyError(f"Missing {key!r}; available top-level keys: {sorted(d.keys())}")
    return d[key]

targets = get_key(protocol, "targets")
excluded = get_key(protocol, "excluded_columns")
group = protocol.get("group_column")
train = pd.read_parquet(inputs["train"])

with open(inputs["05_preprocess_fit_preprocessing_KCWWRC66/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_53QNTHXM/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def input_features(obj):
    if obj is None:
        return None
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            feats = input_features(step)
            if feats is not None:
                return feats
    return None

def derive_features(df, targets, excluded, group):
    drop = set(targets) | set(excluded)
    if group:
        drop.add(group)
    keep = []
    for c in df.columns:
        if c in drop:
            continue
        s = df[c]
        if pd.api.types.is_string_dtype(s.dtype) and s.nunique(dropna=True) == len(df) and len(df) > 5:
            continue
        keep.append(c)
    return keep

raw_cols = input_features(pre)
if raw_cols is None:
    raw_cols = derive_features(train, targets, excluded, group)
else:
    forbidden = set(targets) | set(excluded) | ({group} if group else set())
    bad = set(raw_cols) & forbidden
    if bad:
        raise ValueError(f"Saved preprocessor expects forbidden columns: {sorted(bad)}")
    missing = [c for c in raw_cols if c not in train.columns]
    if missing:
        raise KeyError(f"Missing feature columns in train: {missing}")

sel_feats = input_features(sel)
if sel_feats is not None and set(sel_feats) == set(raw_cols):
    base = sel
    raw_cols = sel_feats
else:
    base = Pipeline([("preprocessor", pre), ("selector", sel)])

clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
if isinstance(base, Pipeline):
    final = Pipeline(base.steps + [("classifier", clf)])
else:
    final = Pipeline([("base", base), ("classifier", clf)])

y = train[targets[0]]
mlca_runtime.fit_record(final, train[raw_cols], "train", y=y)
mlca_runtime.save_model(final, "model.pkl")
print(f"train rows: {len(train)}, features: {len(raw_cols)}")
print("saved model.pkl")
