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

def _has_preproc(obj):
    return isinstance(obj, Pipeline) and any('preprocess' in str(n).lower() for n in obj.named_steps)

def _feature_in(obj):
    if hasattr(obj, 'feature_names_in_') and obj.feature_names_in_ is not None:
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline) and obj.steps:
        first = obj.steps[0][1]
        if hasattr(first, 'feature_names_in_') and first.feature_names_in_ is not None:
            return list(first.feature_names_in_)
    return None

if _has_preproc(selector):
    frontend = selector
else:
    frontend = Pipeline([('preprocessor', preprocessor), ('selector', selector)])

clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
pipeline = Pipeline([('frontend', frontend), ('clf', clf)])

feature_in = _feature_in(preprocessor) or _feature_in(selector)
if feature_in is None:
    excl = protocol["excluded_columns"]
    excl_set = set(excl if isinstance(excl, list) else excl.get("examples", []))
    excl_set |= set(protocol["targets"])
    grp = protocol.get("group_column")
    if grp:
        excl_set.add(grp)
    feature_in = [c for c in train.columns if c not in excl_set]

target = protocol["targets"][0]
X = train[feature_in]
y = train[target]

pipeline = mlca_runtime.fit_record(pipeline, X, 'train', y=y)
mlca_runtime.save_model(pipeline, "model.pkl")

print(f"steps={list(pipeline.named_steps)}")
print(f"n_features_in={len(feature_in)} n_train={len(train)} target={target} pos_rate={float(y.mean()):.4f}")
mlca_runtime.record_metric("n_features_in", float(len(feature_in)), "train")
mlca_runtime.record_metric("positive_rate", float(y.mean()), "train")
