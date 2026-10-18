import mlca_runtime
import json
import pickle
import inspect
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier

SEED = 42
mlca_runtime.set_seed(SEED)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "group_column", "task"):
    if key not in protocol:
        raise KeyError(f"missing {key}; available top-level keys: {sorted(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol["group_column"]
if len(targets) != 1:
    raise ValueError(f"expected one target, got {targets}")
target = targets[0]

train = pd.read_parquet(inputs["train"])
non_features = set(targets) | set(excluded)
if group_col is not None:
    non_features.add(group_col)

feature_cols = []
key_like = set()
for col in train.columns:
    if col in non_features:
        continue
    s = train[col]
    is_text = (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    )
    if is_text and s.nunique(dropna=False) == len(train):
        key_like.add(col)
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(train):
        key_like.add(col)
        continue
    feature_cols.append(col)


def feature_names(obj):
    fni = getattr(obj, "feature_names_in_", None)
    if fni is not None:
        return list(fni)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            cols = feature_names(step)
            if cols is not None:
                return cols
    return None


def contains_ct(obj):
    if isinstance(obj, ColumnTransformer):
        return True
    if isinstance(obj, Pipeline):
        return any(contains_ct(step) for _, step in obj.steps)
    return False


def input_cols(obj, default):
    cols = feature_names(obj)
    if cols is None:
        cols = default
    bad = sorted(set(cols) & (non_features | key_like))
    if bad:
        raise ValueError(f"input columns include target/excluded/group/key-like: {bad}")
    missing = [c for c in cols if c not in train.columns]
    if missing:
        raise KeyError(f"missing input columns: {missing}")
    return list(cols)


with open(inputs["06_preprocess_fit_preprocessing_4H3435T3/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["07_features_select_features_D255Z34M/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

if isinstance(sel, Pipeline) and contains_ct(sel):
    frontend = sel
else:
    frontend = Pipeline([("preprocess", pre), ("select", sel)])


def set_pandas(obj):
    if hasattr(obj, "set_output"):
        try:
            obj.set_output(transform="pandas")
        except Exception:
            pass


for obj in (pre, sel, frontend):
    set_pandas(obj)

cols = input_cols(frontend, feature_cols)
X = train[cols].copy()
y = train[target]

params = {"random_state": SEED}
counts = y.value_counts()
if len(counts) and counts.min() / len(y) < 0.10:
    if "class_weight" in inspect.signature(HistGradientBoostingClassifier).parameters:
        params["class_weight"] = "balanced"

clf = HistGradientBoostingClassifier(**params)
full = Pipeline([("frontend", frontend), ("model", clf)])
full = mlca_runtime.fit_record(full, X, "train", y=y)
mlca_runtime.save_model(full, "alternative_model.pkl")
print(f"train rows={len(X)} features={X.shape[1]} classes={len(full.named_steps['model'].classes_)}")
print("saved alternative_model.pkl")
