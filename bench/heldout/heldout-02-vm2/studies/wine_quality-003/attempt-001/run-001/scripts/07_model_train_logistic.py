import mlca_runtime, json, pickle, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
train = pd.read_parquet(inputs['train'])

with open(inputs['05_preprocess_fit_preprocessing_HXN3H5DR/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inputs['06_features_select_features_ANJ3XYKV/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

def input_names(obj):
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline) and len(obj.named_steps):
        first = next(iter(obj.named_steps.values()))
        if hasattr(first, 'feature_names_in_'):
            return list(first.feature_names_in_)
    return None

def output_names(obj):
    if hasattr(obj, 'get_feature_names_out'):
        try:
            return list(obj.get_feature_names_out())
        except Exception:
            pass
    if isinstance(obj, Pipeline) and len(obj.named_steps):
        last = list(obj.named_steps.values())[-1]
        if hasattr(last, 'get_feature_names_out'):
            try:
                return list(last.get_feature_names_out())
            except Exception:
                pass
    return None

pre_in = input_names(pre)
pre_out = output_names(pre)
sel_in = input_names(sel)

if sel_in is None:
    use_sel_alone = False
elif pre_in is not None and set(sel_in) == set(pre_in):
    use_sel_alone = True
elif pre_out is not None and set(sel_in) == set(pre_out):
    use_sel_alone = False
else:
    use_sel_alone = isinstance(sel, Pipeline)

lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
if use_sel_alone:
    model = Pipeline([('sel', sel), ('clf', lr)])
    feats = sel_in if sel_in is not None else pre_in
else:
    model = Pipeline([('pre', pre), ('sel', sel), ('clf', lr)])
    feats = pre_in

if feats is None:
    exclude = set(targets) | set(protocol.get('excluded_columns', []))
    group = protocol.get('group_column')
    if group:
        exclude.add(group)
    feats = [c for c in train.columns if c not in exclude]

X = train[feats]
y = train[targets[0]]

mlca_runtime.fit_record(model, X, 'train', y=y)
mlca_runtime.save_model(model, 'model.pkl')

print(f"train shape: {train.shape}")
print(f"features: {len(feats)}")
print(f"model steps: {list(model.named_steps)}")
