import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
for key in ["targets", "excluded_columns", "group_column", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available top-level keys: {list(protocol)}")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
with open(inp["05_preprocess_fit_preprocessing_XT8NJ07Q/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

targets = list(protocol["targets"])
if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")
target = targets[0]
excluded = list(protocol["excluded_columns"])
group_col = protocol["group_column"]
blocked = set(targets) | set(excluded)
if group_col is not None:
    blocked.add(group_col)

fallback = []
for c in train.columns:
    if c in blocked:
        continue
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=False) == len(s):
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s):
        lc = c.lower()
        if any(tok in lc for tok in ("id", "key", "code", "index", "row")):
            continue
    fallback.append(c)

def expected_columns(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return fallback

cols = expected_columns(pre)
bad = set(cols) & blocked
if bad:
    raise ValueError(f"Preprocessor expects blocked columns: {sorted(bad)}")

X_train = train[cols]
y_train = train[target]
if not pd.api.types.is_numeric_dtype(y_train):
    y_train = pd.factorize(y_train, sort=True)[0]

sel_name = None
sel_obj = None
if isinstance(pre, Pipeline):
    for name, step in pre.named_steps.items():
        if isinstance(step, SelectKBest):
            sel_name, sel_obj = name, step
            break

if sel_obj is not None:
    n_before = int(getattr(sel_obj, "n_features_in_", 0) or 0)
    if n_before <= 0:
        idx = list(pre.named_steps).index(sel_name)
        sub = Pipeline(pre.steps[:idx])
        n_before = int(sub.transform(X_train.head(min(5, len(X_train)))).shape[1])
    k = min(20, n_before)
    pre.set_params(**{f"{sel_name}__k": k})
    pipeline = pre
    n_input = n_before
else:
    sample_t = pre.transform(X_train.head(min(5, len(X_train))))
    n_input = int(sample_t.shape[1])
    if n_input <= 300 and hasattr(pre, "set_output"):
        try:
            pre.set_output(transform="pandas")
        except Exception:
            pass
    k = min(20, n_input)
    pipeline = Pipeline([
        ("preprocessor", pre),
        ("selector", SelectKBest(score_func=mutual_info_classif, k=k)),
    ])

mlca_runtime.fit_record(pipeline, X_train, "train", y=y_train)

sel = None
for step in pipeline.named_steps.values():
    if isinstance(step, SelectKBest):
        sel = step
        break

if sel is not None:
    mlca_runtime.record_metric("n_input_features", int(sel.n_features_in_), "train")
    mlca_runtime.record_metric("n_selected_features", int(sel.get_support().sum()), "train")

mlca_runtime.save_model(pipeline, "selector.pkl")
print(f"train rows={len(train)}, feature columns={X_train.shape[1]}, transformed={n_input}, k={k}")
print(f"selected={int(sel.get_support().sum()) if sel is not None else 'NA'}")
