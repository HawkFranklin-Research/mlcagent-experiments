import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())

train = pd.read_parquet(inp['train'])

targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column')

with open(inp['05_preprocess_fit_preprocessing_8XQPADGG/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inp['06_features_select_features_NKRWNGEQ/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

if hasattr(pre, 'feature_names_in_'):
    input_cols = list(pre.feature_names_in_)
else:
    input_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]

X = train[input_cols]
y = train[targets[0]]

raw_cols = list(input_cols)
sel_includes_pre = False
if isinstance(sel, Pipeline) and hasattr(sel, 'feature_names_in_'):
    if list(sel.feature_names_in_) == raw_cols:
        sel_includes_pre = True

clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)

if sel_includes_pre:
    pipeline = Pipeline([('selector', sel), ('clf', clf)])
else:
    pipeline = Pipeline([('preprocessor', pre), ('selector', sel), ('clf', clf)])

mlca_runtime.fit_record(pipeline, X, 'train', y=y)
mlca_runtime.save_model(pipeline, 'model.pkl')
print(f"fitted pipeline on {X.shape[0]} rows, {X.shape[1]} raw features; classes={list(pipeline.named_steps['clf'].classes_)}")
