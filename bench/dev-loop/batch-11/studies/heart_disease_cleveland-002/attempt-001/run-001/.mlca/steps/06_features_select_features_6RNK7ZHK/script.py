import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "task"):
    if key not in proto:
        raise ValueError(f"_protocol missing {key!r}; available top-level keys: {list(proto)}")

targets = proto["targets"]
excluded = proto["excluded_columns"]
group_col = proto.get("group_column")

train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_FQWKV8QW/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)


def is_keylike(s: pd.Series) -> bool:
    n = len(s)
    if n == 0:
        return False
    nn = s.nunique(dropna=True)
    if s.dtype == object or str(s.dtype).startswith(("string", "category")):
        return nn == n
    if s.dtype.kind in "iu":
        name = str(s.name).lower()
        return ("id" in name or name.endswith("_id")) and nn == n
    return False


feature_cols = [
    c for c in train.columns
    if c not in targets and c not in excluded and c != group_col
]
feature_cols = [c for c in feature_cols if not is_keylike(train[c])]

if hasattr(preprocessor, "feature_names_in_"):
    expected = [str(c) for c in preprocessor.feature_names_in_]
    missing = [c for c in expected if c not in train.columns]
    if missing:
        raise ValueError(f"preprocessor expects missing columns: {missing}")
    forbidden = [c for c in expected if c in targets or c in excluded or c == group_col]
    if forbidden:
        raise ValueError(f"preprocessor expects forbidden columns: {forbidden}")
    feature_cols = expected

target_col = targets[0]
if target_col not in train.columns:
    raise ValueError(f"target {target_col!r} not in training table")

X = train[feature_cols]
y = train[target_col]

Xt = preprocessor.transform(X)
n_features = int(Xt.shape[1])
k = min(20, n_features)

selector = SelectKBest(score_func=mutual_info_classif, k=k)
fitted_selector = mlca_runtime.fit_record(selector, Xt, "train", y=y)

if isinstance(preprocessor, Pipeline):
    steps = list(preprocessor.steps)
    names = {name for name, _ in steps}
    sel_name = "selectkbest"
    i = 1
    while sel_name in names:
        sel_name = f"selectkbest_{i}"
        i += 1
    steps.append((sel_name, fitted_selector))
    pipeline = Pipeline(steps)
else:
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("selectkbest", fitted_selector),
    ])

mlca_runtime.save_model(pipeline, "selector.pkl")
print(f"train_rows={len(train)} input_features={len(feature_cols)} preprocessed_features={n_features} k={k}")
