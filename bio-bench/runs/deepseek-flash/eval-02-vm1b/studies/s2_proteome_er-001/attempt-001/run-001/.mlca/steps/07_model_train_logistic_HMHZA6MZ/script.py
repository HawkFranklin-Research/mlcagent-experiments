import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for k in ("targets", "task", "excluded_columns"):
    if k not in protocol:
        raise KeyError(f"missing {k}; available: {list(protocol.keys())}")

targets = protocol["targets"]
group_col = protocol.get("group_column")
excl_obj = protocol.get("excluded_columns", [])
if isinstance(excl_obj, dict):
    excluded = set(excl_obj.get("columns") or excl_obj.get("examples") or [])
elif isinstance(excl_obj, list):
    excluded = set(excl_obj)
else:
    excluded = set()

train = pd.read_parquet(inputs["train"])

def input_cols(obj):
    if obj is None:
        return None
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            c = input_cols(step)
            if c is not None:
                return c
    return None

with open(inputs["05_preprocess_fit_preprocessing_5K21TQY3/preprocessor.pkl"], "rb") as f:
    prep = pickle.load(f)
with open(inputs["06_features_select_features_RDJRDVNY/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

prep_in = input_cols(prep)
sel_in = input_cols(sel)

n = len(train)
candidates = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
guessed = []
for c in candidates:
    s = train[c]
    nuniq = s.nunique(dropna=True) if hasattr(s, "nunique") else n
    if nuniq == n and (s.dtype == object or pd.api.types.is_integer_dtype(s)):
        continue
    guessed.append(c)
raw_set = set(guessed)

sel_has_raw = sel_in is not None and set(sel_in) == raw_set
prep_has_raw = prep_in is not None and set(prep_in) == raw_set

if sel_has_raw:
    feat_cols = [c for c in sel_in if c in train.columns and c not in targets and c not in excluded]
    frontend_steps = [('sel', sel)]
elif prep_has_raw:
    feat_cols = [c for c in prep_in if c in train.columns and c not in targets and c not in excluded]
    frontend_steps = [('prep', prep), ('sel', sel)]
elif prep_in is not None:
    feat_cols = [c for c in prep_in if c in train.columns and c not in targets and c not in excluded]
    frontend_steps = [('prep', prep), ('sel', sel)]
elif sel_in is not None:
    feat_cols = [c for c in sel_in if c in train.columns and c not in targets and c not in excluded]
    frontend_steps = [('sel', sel)]
else:
    feat_cols = guessed
    frontend_steps = []

clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
pipeline = Pipeline(frontend_steps + [('clf', clf)])

X = train[feat_cols]
y = train[targets[0]]
fitted = mlca_runtime.fit_record(pipeline, X, "train", y)
mlca_runtime.save_model(fitted, "model.pkl")
print(f"n_features={len(feat_cols)} n_train={len(X)}")
print("saved model.pkl")
