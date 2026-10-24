import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])

with open(inputs["05_preprocess_fit_preprocessing_BE6QCTJ4/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_select_features_KYA9N72M/selector.pkl"], "rb") as f:
    selector = pickle.load(f)

preproc_in = getattr(preprocessor, "feature_names_in_", None)
sel_in = getattr(selector, "feature_names_in_", None)
preproc_in = list(preproc_in) if preproc_in is not None else []
sel_in = list(sel_in) if sel_in is not None else []

# Selector is "raw-frontend" (self-contained pipeline incl. preprocessing, or sibling
# raw frontend) when every column it expects exists in the frozen train frame.
sel_is_raw = bool(sel_in) and all(c in train.columns for c in sel_in)

if sel_is_raw:
    frontend = selector
    feature_in = sel_in
else:
    frontend = Pipeline([("preprocessor", preprocessor), ("selector", selector)])
    if preproc_in and all(c in train.columns for c in preproc_in):
        feature_in = preproc_in
    else:
        excl = set(protocol.get("excluded_columns") or [])
        excl |= set(protocol.get("targets") or [])
        grp = protocol.get("group_column")
        if grp:
            excl.add(grp)
        feature_in = [c for c in train.columns if c not in excl]

target = protocol["targets"][0]
X = train[feature_in]
y = train[target]

clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
pipeline = Pipeline([("frontend", frontend), ("clf", clf)])
pipeline = mlca_runtime.fit_record(pipeline, X, "train", y=y)
mlca_runtime.save_model(pipeline, "model.pkl")

print(f"frontend={'selector' if sel_is_raw else 'preproc+selector'} n_features={len(feature_in)} n_train={len(train)}")
mlca_runtime.record_metric("n_features_in", float(len(feature_in)), "train")
mlca_runtime.record_metric("positive_rate", float(pd.Series(y).astype(str).value_counts(normalize=True).iloc[0]), "train")
