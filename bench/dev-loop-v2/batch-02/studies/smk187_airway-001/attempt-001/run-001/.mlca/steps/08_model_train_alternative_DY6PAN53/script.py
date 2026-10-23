import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']; excluded = protocol['excluded_columns']
group = protocol.get('group_column')
target = targets[0]

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]

pre = pickle.loads(inputs['05_preprocess_fit_preprocessing_4ZKXP498/preprocessor.pkl'].read_bytes())
sel = pickle.loads(inputs['06_features_select_features_E9CTG9EJ/selector.pkl'].read_bytes())
logit = pickle.loads(inputs['07_model_train_logistic_KVA6214M/model.pkl'].read_bytes())

if isinstance(logit, Pipeline) and hasattr(logit.steps[-1][1], 'predict_proba'):
    prefix = list(logit.steps[:-1])
elif isinstance(sel, Pipeline):
    prefix = list(sel.steps)
else:
    prefix = [('preprocessor', pre), ('selector', sel)]

clf = HistGradientBoostingClassifier(random_state=42)
pipe = Pipeline(prefix + [('clf', clf)])

X_train = train[feature_cols]; y_train = train[target]
pipe = mlca_runtime.fit_record(pipe, X_train, 'train', y=y_train)

X_val = val[feature_cols]; y_val = val[target]
proba = pipe.predict_proba(X_val)[:, 1]
auc = roc_auc_score((y_val == pipe.classes_[1]).astype(int), proba)
mlca_runtime.record_metric('val_roc_auc', float(auc), 'val')
mlca_runtime.save_model(pipe, 'alternative_model.pkl')
print('features', len(feature_cols), 'steps', [n for n, _ in pipe.steps], 'val_auc', round(auc, 4))
