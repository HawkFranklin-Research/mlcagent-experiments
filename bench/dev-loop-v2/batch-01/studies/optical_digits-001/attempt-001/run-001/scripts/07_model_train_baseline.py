import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

def pg(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available top-level keys: {list(d.keys())}")
    return d[k]

targets = pg(protocol, "targets")
excluded = pg(protocol, "excluded_columns")
group = protocol.get("group_column")
pg(protocol, "task")

train = pd.read_parquet(paths["train"])
with open(paths["05_preprocess_fit_preprocessing_KTEB41J1/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(paths["06_features_retain_features_K0P2E6DF/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

for obj in (pre, sel):
    if hasattr(obj, "set_output"):
        try:
            obj.set_output(transform="pandas")
        except Exception:
            pass

def first_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

pre_in, sel_in = first_names(pre), first_names(sel)
drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)

def key_like(c):
    s = train[c]
    if s.isna().all():
        return False
    if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
        return len(s) > 1 and s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return len(s) > 1 and s.nunique(dropna=True) == len(s)
    return False

feature_cols = [c for c in train.columns if c not in drop and not key_like(c)]
for expected in (pre_in, sel_in):
    if expected is not None and set(expected) == set(feature_cols):
        feature_cols = expected
        break
X = train[feature_cols].copy()

def includes_pre(obj):
    if not isinstance(obj, Pipeline):
        return False
    steps = list(obj.named_steps.values())
    if any(isinstance(s, ColumnTransformer) for s in steps[:-1]):
        return True
    return bool(steps) and hasattr(steps[0], "get_feature_names_out") and not hasattr(steps[0], "get_support")

if includes_pre(sel):
    frontend = sel
elif pre_in is not None and list(pre_in) == list(feature_cols):
    frontend = Pipeline([("preprocessor", pre), ("selector", sel)])
else:
    frontend = sel

def out_width(obj):
    if hasattr(obj, "get_feature_names_out"):
        try:
            return len(obj.get_feature_names_out())
        except Exception:
            pass
    if hasattr(obj, "get_support"):
        try:
            return int(obj.get_support().sum())
        except Exception:
            pass
    if isinstance(obj, Pipeline):
        try:
            return out_width(list(obj.named_steps.values())[-1])
        except Exception:
            pass
    return None

w = out_width(sel)
k = min(8, max(1, w if w is not None else len(feature_cols)))
model = Pipeline([("frontend", frontend), ("pca", PCA(n_components=k))])
if hasattr(model, "set_output"):
    try:
        model.set_output(transform="pandas")
    except Exception:
        pass

model = mlca_runtime.fit_record(model, X, "train")
mlca_runtime.save_model(model, "model.pkl")
print(f"train_rows={len(X)} features={len(feature_cols)} pca_components={k}")
