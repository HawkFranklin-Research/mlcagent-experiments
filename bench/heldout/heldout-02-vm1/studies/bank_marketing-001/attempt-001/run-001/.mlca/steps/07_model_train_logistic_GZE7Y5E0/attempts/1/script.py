import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])

with open(inp['05_preprocess_fit_preprocessing_3K40DCMA/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inp['06_features_select_features_XBX6AN5A/selector.pkl'], 'rb') as f:
    selector = pickle.load(f)

def front_cols(o):
    return list(getattr(o, 'feature_names_in_', []))

prep_cols = front_cols(preprocessor)
sel_cols = front_cols(selector)
print('prep front', len(prep_cols), '| sel front', len(sel_cols))

for bad in list(targets) + list(excluded):
    if bad in prep_cols or bad in sel_cols:
        raise ValueError(f'leak column {bad} reached a saved frontend')

clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)

if prep_cols and set(sel_cols) == set(prep_cols):
    feature_cols = sel_cols
    pipe = Pipeline([('sel', selector), ('clf', clf)])
elif prep_cols:
    feature_cols = prep_cols
    pipe = Pipeline([('prep', preprocessor), ('sel', selector), ('clf', clf)])
else:
    raise ValueError(f'cannot determine frontend; sel has {len(sel_cols)} cols')

tgt = targets[0]
X_train, y_train = train[feature_cols], train[tgt]
X_val, y_val = val[feature_cols], val[tgt]

pipe = mlca_runtime.fit_record(pipe, X_train, 'train', y_train)

classes = pipe.named_steps['clf'].classes_
proba = pipe.predict_proba(X_val)
pos_idx = 1
y_bin = (y_val == classes[pos_idx]).astype(int)
auc = float(roc_auc_score(y_bin, proba[:, pos_idx]))
mlca_runtime.record_metric('roc_auc', auc, 'val')
mlca_runtime.save_model(pipe, 'model.pkl')
print('val roc_auc', round(auc, 4), '| classes', list(classes))
print('features', len(feature_cols))
