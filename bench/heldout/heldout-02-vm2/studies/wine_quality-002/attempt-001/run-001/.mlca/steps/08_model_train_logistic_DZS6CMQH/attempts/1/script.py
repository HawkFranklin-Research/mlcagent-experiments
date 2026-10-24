import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])

targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column")

with open(inputs["06_preprocess_fit_preprocessing_HTY1VTC0/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["07_features_select_features_Z274ACYD/selector.pkl"], "rb") as f:
    selector = pickle.load(f)

pre_in = getattr(preprocessor, "feature_names_in_", None)
if pre_in is not None:
    raw_feats = list(pre_in)
else:
    drop = set(targets) | set(excluded) | ({group} if group else set())
    raw_feats = [c for c in train.columns if c not in drop]

X = train[raw_feats]
y = train[targets[0]]

clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)

sel_in = getattr(selector, "feature_names_in_", None)
pre_out = None
try:
    probe = preprocessor.transform(X.head(5))
    if hasattr(probe, "columns"):
        pre_out = list(probe.columns)
except Exception:
    pre_out = None

if sel_in is not None and pre_out is not None and list(sel_in) == pre_out:
    steps = [("pre", preprocessor), ("sel", selector), ("clf", clf)]
elif sel_in is not None and pre_in is not None and list(sel_in) == raw_feats:
    steps = [("sel", selector), ("clf", clf)]
else:
    steps = [("pre", preprocessor), ("sel", selector), ("clf", clf)]

pipe = Pipeline(steps)
pipe = mlca_runtime.fit_record(pipe, X, "train", y=y)
mlca_runtime.save_model(pipe, "model.pkl")
print(f"fitted steps={len(steps)} features_in={len(raw_feats)} classes={len(clf.classes_)}")
