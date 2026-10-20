import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
targets = proto["targets"]; excluded = proto["excluded_columns"]
group = proto.get("group_column")
train = pd.read_parquet(inputs["train"])
target = targets[0]

with open(inputs["05_preprocess_fit_preprocessing_6KJVP4ST/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_select_features_TX9DR60B/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

names = lambda o: list(getattr(o, "feature_names_in_", []))
pre_in, sel_in = names(pre), names(sel)

skip = set(targets) | set(excluded) | ({group} if group else set())
raw = []
for c in train.columns:
    if c in skip:
        continue
    s = train[c]
    if (s.dtype == object or str(s.dtype).startswith("string")) and s.nunique() == len(s):
        continue
    raw.append(c)

labels = sorted(train[target].dropna().unique())
pos = "yes" if "yes" in labels else labels[-1]
y = (train[target] == pos).astype(int)
minor = y.value_counts(normalize=True).min()
cw = "balanced" if minor < 0.10 else None
clf = HistGradientBoostingClassifier(random_state=42, class_weight=cw)

if sel_in and pre_in and set(sel_in) == set(pre_in) and len(sel_in) == len(pre_in):
    feat_cols = sel_in
    pipe = Pipeline([("selector", sel), ("model", clf)])
elif sel_in and sel_in and set(sel_in).issubset(set(raw)):
    feat_cols = sel_in
    pipe = Pipeline([("selector", sel), ("model", clf)])
elif pre_in:
    feat_cols = pre_in
    pipe = Pipeline([("preprocessor", pre), ("selector", sel), ("model", clf)])
else:
    feat_cols = raw
    pipe = Pipeline([("preprocessor", pre), ("selector", sel), ("model", clf)])

X = train[feat_cols]
mlca_runtime.fit_record(pipe, X, "train", y)
mlca_runtime.save_model(pipe, "alternative_model.pkl")
print(f"rows={len(X)} feats={len(feat_cols)} pos={pos} cw={cw} pipeline_steps={list(pipe.named_steps)}")
