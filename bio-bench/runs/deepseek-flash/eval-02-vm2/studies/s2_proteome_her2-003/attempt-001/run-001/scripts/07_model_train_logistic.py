import mlca_runtime, json, pickle, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
train = pd.read_parquet(inputs['train'])

targets = protocol['targets']
if not isinstance(targets, list):
    targets = [targets]
excluded = protocol['excluded_columns']
if isinstance(excluded, dict):
    for k in ('columns', 'excluded_columns', 'examples'):
        if k in excluded:
            excluded = excluded[k]
            break
    else:
        raise ValueError(f"excluded_columns dict keys: {list(excluded)}")
excluded = set(excluded)
group = protocol.get('group_column', None)
if group in (None, '', 'None'):
    group = None

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

pre = load_pkl(inputs['05_preprocess_fit_preprocessing_D50P13FR/preprocessor.pkl'])
sel = load_pkl(inputs['06_features_select_features_CJ8Y4QR9/selector.pkl'])

def input_cols(obj):
    if obj is None:
        return None
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            if hasattr(step, 'feature_names_in_'):
                return list(step.feature_names_in_)
    return None

pre_cols = input_cols(pre)
if pre_cols is not None:
    raw_cols = [c for c in pre_cols if c in train.columns and c not in targets and c not in excluded and c != group]
else:
    raw_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
    keep = []
    for c in raw_cols:
        s = train[c]
        if ((s.dtype == object or pd.api.types.is_string_dtype(s.dtype))
                and s.nunique(dropna=True) == len(s) and s.notna().all()):
            continue
        if (pd.api.types.is_integer_dtype(s.dtype)
                and s.nunique(dropna=True) == len(s) and len(s) > 1
                and not (set(s.dropna().unique()) <= {0, 1})):
            continue
        keep.append(c)
    raw_cols = keep

raw_cols = list(dict.fromkeys(raw_cols))
X = train[raw_cols]
target_col = targets[0]
y = train[target_col]

def wraps_raw(p):
    if not isinstance(p, Pipeline):
        return False
    for step in p.named_steps.values():
        if hasattr(step, 'feature_names_in_'):
            cols = list(step.feature_names_in_)
            if len(cols) == len(raw_cols) and set(cols) == set(raw_cols):
                return True
    return False

sel_cols = input_cols(sel)
sel_wraps = (isinstance(sel, Pipeline)
             and (wraps_raw(sel) or (sel_cols is not None
                                     and len(sel_cols) == len(raw_cols)
                                     and set(sel_cols) == set(raw_cols))))

clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
if sel_wraps:
    steps = [('selector_pipeline', sel), ('clf', clf)]
else:
    steps = []
    if pre is not None:
        steps.append(('preprocessor', pre))
    if sel is not None:
        steps.append(('selector', sel))
    steps.append(('clf', clf))

model = Pipeline(steps)
model = mlca_runtime.fit_record(model, X, 'train', y=y)
mlca_runtime.save_model(model, 'model.pkl')

print(f"train rows={len(X)} raw features={len(raw_cols)}")
print(f"pre input={len(pre_cols) if pre_cols else 'n/a'} sel input={len(sel_cols) if sel_cols else 'n/a'} selector_wraps={sel_wraps}")
print(f"saved model.pkl steps={list(model.named_steps)}")
