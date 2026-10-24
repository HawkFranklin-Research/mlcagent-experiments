import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = protocol["targets"]
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column")

train = pd.read_parquet(inp["train"])
with open(inp["06_preprocess_fit_preprocessing_Z6Z720YG/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["07_features_select_features_Y69JECM8/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    s = train[c]
    if (s.dtype == object or str(s.dtype).startswith("string")) and s.nunique(dropna=False) == len(train):
        continue
    if pd.api.types.is_integer_dtype(s.dtype) and s.nunique(dropna=False) == len(train):
        vals = set(s.dropna().unique())
        if not vals.issubset({0, 1}):
            continue
    feature_cols.append(c)

if hasattr(pre, "feature_names_in_"):
    raw_cols = [c for c in pre.feature_names_in_ if c in feature_cols]
else:
    raw_cols = feature_cols

X = train[raw_cols].copy()
y = train[targets[0]].copy()
clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)

sel_has_pre = isinstance(sel, Pipeline) and any("pre" in name.lower() for name in sel.named_steps)
if sel_has_pre:
    model = Pipeline([("selector", sel), ("clf", clf)])
else:
    model = Pipeline([("preprocessor", pre), ("selector", sel), ("clf", clf)])

mlca_runtime.fit_record(model, X, "train", y=y)
mlca_runtime.save_model(model, "model.pkl")
print(f"train rows: {len(train)}, features: {len(raw_cols)}")
print("saved model.pkl")
