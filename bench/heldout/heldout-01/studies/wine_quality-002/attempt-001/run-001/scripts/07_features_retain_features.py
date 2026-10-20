import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest

inputs = mlca_runtime.inputs()
mlca_runtime.set_seed(42)

protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column", None)
task = protocol["task"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
drop_key = []
for c in feature_cols:
    s = train[c]
    if s.dtype == object and s.nunique() == len(s):
        drop_key.append(c)
    elif pd.api.types.is_integer_dtype(s) and s.nunique() == len(s) and "id" in c.lower():
        drop_key.append(c)
feature_cols = [c for c in feature_cols if c not in drop_key]

with open(inputs["06_preprocess_fit_preprocessing_2FS9KME6/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

if hasattr(pre, "feature_names_in_"):
    input_cols = [c for c in pre.feature_names_in_ if c not in targets and c not in excluded and c != group]
else:
    input_cols = feature_cols

X = train[input_cols]
y = train[targets[0]] if len(targets) == 1 else train[targets]

n_out = None
try:
    n_out = len(pre.get_feature_names_out())
except Exception:
    pass

steps = [("preprocess", pre)]
sel_name = "selector" if not isinstance(pre, Pipeline) or "selector" not in pre.named_steps else "selector_passthrough"
steps.append((sel_name, SelectKBest(k="all")))
pipeline = Pipeline(steps)

if n_out is not None and n_out <= 300:
    for obj in (pre, pipeline):
        try:
            obj.set_output(transform="pandas")
        except Exception:
            pass

fitted = mlca_runtime.fit_record(pipeline, X, "train", y=y)
mlca_runtime.save_model(fitted, "selector_all.pkl")

print(f"train shape {train.shape}; features {len(input_cols)}; output width {n_out}")
print("saved selector_all.pkl")
print(f"target {targets[0] if len(targets)==1 else targets}")
