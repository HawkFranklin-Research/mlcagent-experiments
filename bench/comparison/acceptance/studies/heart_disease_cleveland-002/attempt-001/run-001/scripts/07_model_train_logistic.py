import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectorMixin
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
protocol = json.loads(inputs["_protocol"].read_text())

try:
    targets = protocol['targets']
    excluded = protocol['excluded_columns']
    task = protocol['task']
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available keys: {list(protocol.keys())}") from e
group = protocol.get('group_column', None)
target_col = targets[0]

with open(inputs["05_preprocess_fit_preprocessing_02TSADV2/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_select_features_BS9MRTV1/selector.pkl"], "rb") as f:
    sel = pickle.load(f)


def includes_preprocessor(selector, preproc):
    if isinstance(selector, ColumnTransformer):
        return True
    if not isinstance(selector, Pipeline):
        return False
    for step in selector.named_steps.values():
        if isinstance(step, ColumnTransformer):
            return True
        if isinstance(step, Pipeline) and any(isinstance(s, ColumnTransformer) for s in step.named_steps.values()):
            return True
    non_selectors = [s for s in selector.named_steps.values() if not isinstance(s, SelectorMixin)]
    return any(not isinstance(s, str) for s in non_selectors)


raw_cols = None
for obj in (sel, pre):
    cols = getattr(obj, 'feature_names_in_', None)
    if cols is not None and all(c in train.columns for c in cols):
        if not any(c in cols for c in targets + excluded + ([group] if group else [])):
            raw_cols = list(cols)
            break

if raw_cols is None:
    raw_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]

feature_cols = [
    c for c in raw_cols
    if c in train.columns
    and c not in targets
    and c not in excluded
    and (group is None or c != group)
]
X_train = train[feature_cols].copy()
y_train = train[target_col]

log = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
if includes_preprocessor(sel, pre):
    pipe = Pipeline([('frontend', sel), ('model', log)])
else:
    pipe = Pipeline([('preprocessor', pre), ('selector', sel), ('model', log)])

fitted = mlca_runtime.fit_record(pipe, X_train, 'train', y=y_train)
if fitted is None:
    fitted = pipe
mlca_runtime.save_model(fitted, "model.pkl")

print(f"train rows: {len(train)}")
print(f"features: {len(feature_cols)}")
print(f"steps: {list(fitted.named_steps)}")
print("saved model.pkl")
