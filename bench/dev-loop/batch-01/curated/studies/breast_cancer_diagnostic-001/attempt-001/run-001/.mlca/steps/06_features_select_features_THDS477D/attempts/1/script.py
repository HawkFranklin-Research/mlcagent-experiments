import functools
import json
import pickle

import pandas as pd

import mlca_runtime
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
ins = mlca_runtime.inputs()
protocol = json.loads(ins["_protocol"].read_text())

for req in ("targets", "excluded_columns"):
    if req not in protocol:
        raise KeyError(f"_protocol is missing '{req}'; available keys: {sorted(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns") or [])
group = protocol.get("group_column")
task = protocol.get("task", {})

train = pd.read_parquet(ins["train"])
declared = targets + excluded + ([group] if group else [])
absent = [c for c in declared if c not in train.columns]
if absent:
    raise KeyError(f"declared columns absent from train: {absent}")

def is_key_like(col):
    s = train[col]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return bool(s.is_unique)
    if pd.api.types.is_integer_dtype(s):
        return bool(s.notna().all() and s.is_unique)
    return False

blocked = set(targets) | set(excluded) | ({group} if group else set())
features = [c for c in train.columns if c not in blocked and not is_key_like(c)]
X = train[features]
y = train[targets[0]]

with open(ins["05_preprocess_fit_preprocessing_YS7QE43Y/preprocessor.pkl"], "rb") as fh:
    pre = pickle.load(fh)

steps = list(pre.steps) if isinstance(pre, Pipeline) and pre.steps else [("preprocess", pre)]

try:
    width = Pipeline(steps).transform(X.head(3)).shape[1]
except Exception:
    width = len(features)

k = max(1, min(20, len(features), int(width)))
score = functools.partial(mutual_info_classif, random_state=42)
pipe = Pipeline(steps + [("select", SelectKBest(score_func=score, k=k))])
try:
    pipe.set_output(transform="pandas")
except Exception:
    pass

fitted = mlca_runtime.fit_record(pipe, X, "train", y=y)
n_sel = int(fitted.named_steps["select"].get_support().sum())
mlca_runtime.save_model(fitted, "selector.pkl")
mlca_runtime.record_metric("n_selected_features", float(n_sel), "train")

print(f"train rows={len(X)}, feature columns={len(features)}")
print(f"preprocessing output width={int(width)}, k={k}")
print(f"selected features={n_sel}")
