import mlca_runtime
import json, pickle
from functools import partial
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

train = pd.read_parquet(inputs["train"])
protocol = json.loads(inputs["_protocol"].read_text())
for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}. Available keys: {list(protocol.keys())}")
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")


def is_key_like(series, name):
    n = len(series)
    if n == 0 or series.nunique(dropna=False) < n:
        return False
    if series.dtype == object or str(series.dtype).startswith("string") or isinstance(series.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_integer_dtype(series.dtype):
        name_l = name.lower()
        if any(tok in name_l for tok in ("id", "code", "key", "index")):
            return True
        if series.min() in (0, 1) and series.max() == series.min() + n - 1 and series.is_monotonic_increasing:
            return True
    return False


feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
feature_cols = [c for c in feature_cols if not is_key_like(train[c], c)]

with open(inputs["05_preprocess_fit_preprocessing_M7A7QE51/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)


def infer_n_features(pp, X):
    if hasattr(pp, "get_feature_names_out"):
        try:
            return len(pp.get_feature_names_out())
        except Exception:
            pass
    return pp.transform(X.head(1)).shape[1]


n_features = infer_n_features(preprocessor, train[feature_cols])
k = min(20, n_features)
score_func = partial(mutual_info_classif, random_state=42)
selector = SelectKBest(score_func=score_func, k=k)

if isinstance(preprocessor, Pipeline):
    has_sel = any(isinstance(step, SelectKBest) for _, step in preprocessor.steps)
    final_pipeline = preprocessor if has_sel else Pipeline(list(preprocessor.steps) + [("selectkbest", selector)])
else:
    final_pipeline = Pipeline([("preprocessor", preprocessor), ("selectkbest", selector)])

if n_features <= 300:
    for obj in (preprocessor, final_pipeline):
        if hasattr(obj, "set_output"):
            try:
                obj.set_output(transform="pandas")
            except Exception:
                pass

X_train = train[feature_cols]
y_train = train[targets[0]] if len(targets) == 1 else train[targets]
fitted = mlca_runtime.fit_record(final_pipeline, X_train, "train", y=y_train)

last = fitted.named_steps.get("selectkbest") if isinstance(fitted, Pipeline) else fitted
if last is None and isinstance(fitted, Pipeline):
    last = fitted.steps[-1][1]
n_selected = int(last.get_support().sum()) if hasattr(last, "get_support") else n_features
mlca_runtime.record_metric("n_features_after_preprocessing", int(n_features), "train")
mlca_runtime.record_metric("n_features_selected", n_selected, "train")
mlca_runtime.save_model(fitted, "selector.pkl")

print(f"train rows: {len(train)}")
print(f"feature columns: {len(feature_cols)}")
print(f"features after preprocessing: {n_features}")
print(f"selected features: {n_selected}")
print("saved selector.pkl")
