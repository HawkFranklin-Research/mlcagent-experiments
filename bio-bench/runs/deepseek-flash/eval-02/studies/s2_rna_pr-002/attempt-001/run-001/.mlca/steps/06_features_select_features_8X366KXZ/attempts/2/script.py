import json
import pickle
import mlca_runtime
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol["group_column"]

train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_WM2D925J/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)


def input_cols(est):
    if hasattr(est, "feature_names_in_"):
        return list(est.feature_names_in_)
    if isinstance(est, Pipeline):
        for _, step in est.steps:
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None


feature_cols = input_cols(preprocessor) or [
    c for c in train.columns
    if c not in targets and c not in excluded and c != group_col
]

X = train[feature_cols]
y = train[targets[0]]

sample = X.iloc[: min(50, len(X))]
n_out = preprocessor.transform(sample).shape[1]
if n_out <= 300:
    try:
        preprocessor.set_output(transform="pandas")
    except Exception:
        pass
k = min(20, n_out, len(feature_cols))

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("select", SelectKBest(mutual_info_classif, k=k)),
])

mlca_runtime.fit_record(pipe, X, "train", y=y)
mlca_runtime.save_model(pipe, "selector.pkl")
print(f"selector fitted: k={k}, n_in={len(feature_cols)}, n_out={n_out}")
