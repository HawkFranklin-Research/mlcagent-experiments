import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()


def req(d, key):
    if key not in d:
        raise KeyError(f"_protocol missing {key!r}; available top-level keys: {sorted(d.keys())}")
    return d[key]


protocol = json.loads(inputs["_protocol"].read_text())
targets = list(req(protocol, "targets"))
excluded = list(req(protocol, "excluded_columns"))
group = req(protocol, "group_column")
_ = req(protocol, "task")

train = pd.read_parquet(inputs["train"])
with open(inputs["06_preprocess_fit_preprocessing_DB5T0R19/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

forbidden = set(targets) | set(excluded) | {group}
feature_cols = []
for c in train.columns:
    if c in forbidden:
        continue
    s = train[c]
    is_text = (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    )
    if is_text and s.nunique(dropna=False) == len(s):
        continue
    if (
        pd.api.types.is_integer_dtype(s)
        and s.nunique(dropna=False) == len(s)
        and (str(c).lower().endswith(("id", "key", "code")) or "id" in str(c).lower())
    ):
        continue
    feature_cols.append(c)


def fitted_input_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None


pre_features = fitted_input_names(pre)
if pre_features is not None:
    bad = set(pre_features) & forbidden
    if bad:
        raise ValueError(f"Preprocessor input contains forbidden columns: {sorted(bad)}")
    missing = [c for c in pre_features if c not in train.columns]
    if missing:
        raise ValueError(f"Preprocessor input columns missing from train: {missing}")
    feature_cols = pre_features

X_train = train.loc[:, feature_cols]

selector = FunctionTransformer(feature_names_out="one-to-one")
step_name = "passthrough_selector"
if isinstance(pre, Pipeline):
    while step_name in pre.named_steps:
        step_name += "_"
    selector_all = Pipeline(list(pre.steps) + [(step_name, selector)])
else:
    selector_all = Pipeline([("preprocessor", pre), (step_name, selector)])

mlca_runtime.fit_record(selector_all, X_train, "train", y=None)
mlca_runtime.save_model(selector_all, "selector_all.pkl")

print(f"train_rows={len(X_train)} features={len(feature_cols)}")
print("saved selector_all.pkl")
