import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
with open(inp['_protocol']) as f:
    protocol = json.load(f)
targets = protocol['targets']
excluded = protocol.get('excluded_columns') or []
group = protocol.get('group_column')

train = pd.read_parquet(inp['train'])
with open(inp['05_preprocess_fit_preprocessing_G1EGCAW6/preprocessor.pkl'], 'rb') as f:
    prep = pickle.load(f)
with open(inp['06_features_select_features_CZZQV6ME/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

def input_cols(obj):
    if isinstance(obj, Pipeline) and obj.steps:
        for cand in (obj.steps[0][1], obj):
            c = getattr(cand, 'feature_names_in_', None)
            if c is not None:
                return [str(x) for x in c]
    c = getattr(obj, 'feature_names_in_', None)
    if c is not None:
        return [str(x) for x in c]
    return None

cols = input_cols(prep) or input_cols(sel)
if cols is None:
    cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
cols = [c for c in cols if c not in targets and c not in excluded and c != group]
X, y = train[cols], train[targets[0]]

def contains_transform(obj):
    if isinstance(obj, Pipeline):
        return any(contains_transform(s) for _, s in obj.steps)
    return hasattr(obj, 'transformers') or obj.__class__.__name__ == 'ColumnTransformer'

clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
pipe = Pipeline([('selector', sel), ('clf', clf)]) if contains_transform(sel) \
    else Pipeline([('prep', prep), ('selector', sel), ('clf', clf)])

def out_width(obj):
    if obj is None:
        return None
    if hasattr(obj, 'n_features_out_'):
        return obj.n_features_out_
    if isinstance(obj, Pipeline):
        return out_width(obj.steps[-1][1])
    if hasattr(obj, 'transformers_'):
        total = 0
        for name, trans, _ in obj.transformers_:
            if name == 'remainder' and trans == 'drop':
                continue
            try:
                total += len(trans.get_feature_names_out())
            except Exception:
                return None
        return total
    return None

width = out_width(prep) or out_width(sel)
if width is not None and width <= 300:
    for o in (prep, sel, pipe):
        if hasattr(o, 'set_output'):
            try:
                o.set_output(transform='pandas')
            except Exception:
                pass

pipe = mlca_runtime.fit_record(pipe, X, 'train', y=y)
mlca_runtime.save_model(pipe, "model.pkl")
print(f"train_rows={X.shape[0]} features={len(cols)}")
