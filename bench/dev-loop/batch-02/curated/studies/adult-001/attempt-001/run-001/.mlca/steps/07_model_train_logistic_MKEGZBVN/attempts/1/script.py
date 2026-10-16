import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing {k!r} in _protocol; available keys: {list(d.keys())}")
    return d[k]

targets = req(protocol, "targets")
excluded = req(protocol, "excluded_columns")
group_col = req(protocol, "group_column")
task = req(protocol, "task")
if isinstance(targets, str):
    targets = [targets]
if isinstance(excluded, str):
    excluded = [excluded]
target = targets[0]

train = pd.read_parquet(inputs["train"])

with open(inputs["05_preprocess_fit_preprocessing_CZ0KAB5J/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_select_features_X4RMHKJ0/selector.pkl"], "rb") as f:
    selector = pickle.load(f)

def is_key_like(s):
    n = len(s)
    if s.nunique(dropna=False) < n:
        return False
    if pd.api.types.is_numeric_dtype(s):
        if pd.api.types.is_integer_dtype(s):
            vals = s.dropna()
            if len(vals) == n and (set(vals) == set(range(n)) or set(vals) == set(range(1, n + 1))):
                return True
        return False
    return True

raw_cols = []
for c in train.columns:
    if c in targets or c in excluded or (group_col is not None and c == group_col):
        continue
    if is_key_like(train[c]):
        continue
    raw_cols.append(c)

def feature_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline) and obj.named_steps:
        return feature_names(list(obj.named_steps.values())[0])
    return None

def has_preprocessing_step(obj):
    if isinstance(obj, Pipeline):
        first = list(obj.named_steps.values())[0]
        return isinstance(first, (ColumnTransformer, Pipeline))
    return False

if has_preprocessing_step(selector):
    input_cols = feature_names(selector) or raw_cols
    pipeline = Pipeline([
        ('selector', selector),
        ('model', LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000))
    ])
else:
    input_cols = feature_names(preprocessor) or raw_cols
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('model', LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000))
    ])

bad = set(input_cols) & (set(targets) | set(excluded) | ({group_col} if group_col is not None else set()))
if bad:
    raise ValueError(f"Target/excluded/group columns leaked into features: {sorted(bad)}")

X_train = train.loc[:, input_cols].copy()
y_train = train[target].copy()

mlca_runtime.fit_record(pipeline, X_train, partition='train', y=y_train)
mlca_runtime.save_model(pipeline, "model.pkl")
print(f"train shape: {train.shape}")
print(f"features: {len(input_cols)}")
print(f"classes: {list(pipeline.named_steps['model'].classes_)}")
