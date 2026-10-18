import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

with open(inputs["_protocol"], "r", encoding="utf-8") as f:
    protocol = json.load(f)

required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(
        f"Missing required protocol keys {missing}; "
        f"available top-level keys: {list(protocol.keys())}"
    )

targets = protocol["targets"]
targets = [targets] if isinstance(targets, str) else list(targets)
excluded = protocol["excluded_columns"] or []
excluded = [excluded] if isinstance(excluded, str) else list(excluded)
group_col = protocol["group_column"]

train = pd.read_parquet(inputs["train"])
base_cols = [
    c for c in train.columns
    if c not in targets and c not in excluded and c != group_col
]

def is_key_like(s: pd.Series) -> bool:
    if (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    ):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=False) == len(s)
    return False

feature_cols = [c for c in base_cols if not is_key_like(train[c])]

with open(inputs["05_preprocess_fit_preprocessing_X2VH4MDF/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

if hasattr(pre, "feature_names_in_"):
    expected = list(pre.feature_names_in_)
    if set(expected) == set(feature_cols):
        feature_cols = expected
    else:
        raise ValueError("Preprocessor input columns do not match derived feature columns")

X_train = train.loc[:, feature_cols]
y_train = train.loc[:, targets]

if isinstance(pre, Pipeline):
    steps = list(pre.steps)
else:
    steps = [("preprocessor", pre)]

name = "selector_all"
existing = {n for n, _ in steps}
while name in existing:
    name += "_"
steps.append((name, "passthrough"))

final = Pipeline(steps)
final = mlca_runtime.fit_record(final, X_train, "train", y=y_train)

try:
    width = final[:-1].transform(X_train).shape[1]
except Exception:
    width = None

if width is not None and width <= 300:
    try:
        final.set_output(transform="pandas")
    except Exception:
        pass

mlca_runtime.save_model(final, "selector_all.pkl")
print(f"selector_all fitted: rows={len(X_train)}, features={len(feature_cols)}, width={width}")
