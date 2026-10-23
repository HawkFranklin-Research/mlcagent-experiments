import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', []) or []
group = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
target = targets[0]
feature_cols = [c for c in train.columns
                if c not in targets and c not in excluded and c != group]

X_train, y_train = train[feature_cols], train[target]
X_val, y_val = val[feature_cols], val[target]

with open(inputs['05_preprocess_fit_preprocessing_W6YRPZJC/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inputs['06_features_select_features_GPEMXWYC/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

sel_in = getattr(sel, 'feature_names_in_', None)
pre_out = None
if hasattr(pre, 'get_feature_names_out'):
    try:
        pre_out = list(pre.get_feature_names_out())
    except Exception:
        pre_out = None

sel_alone = False
if sel_in is not None:
    if pre_out is not None and list(sel_in) == list(pre_out):
        sel_alone = False
    elif set(sel_in) <= set(feature_cols) and len(sel_in) == len(feature_cols):
        sel_alone = True

clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
if sel_alone:
    pipe = Pipeline([('selector', sel), ('clf', clf)])
else:
    pipe = Pipeline([('pre', pre), ('selector', sel), ('clf', clf)])

mlca_runtime.fit_record(pipe, X_train, 'train', y=y_train)

classes = list(pipe.classes_)
pos_label = classes[1]
proba = pipe.predict_proba(X_val)[:, 1]
y_bin = (y_val == pos_label).astype(int).to_numpy()
auc = float(roc_auc_score(y_bin, proba))

mlca_runtime.save_model(pipe, 'model.pkl')
mlca_runtime.record_metric('val_roc_auc', auc, 'val')

print(f'features={len(feature_cols)} train_rows={len(X_train)} sel_alone={sel_alone}')
print(f'pos_label={pos_label} classes={classes} val_auc={auc:.4f}')
