import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
targets = list(protocol['targets'])
excluded = list(protocol.get('excluded_columns') or [])
group_col = protocol.get('group_column')

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])

with open(inp['06_preprocess_fit_preprocessing_HH50E3RG/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inp['07_features_retain_features_Y0FEJMPH/selector_all.pkl'], 'rb') as f:
    selector = pickle.load(f)

# Feature contract: prefer preprocessor's fitted input names (already curated upstream)
if hasattr(preprocessor, 'feature_names_in_'):
    raw_cols = [c for c in preprocessor.feature_names_in_ if c not in targets]
else:
    raw_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
assert not (set(raw_cols) & set(targets)), "target column leaked into feature set"

clf = MultiOutputClassifier(LogisticRegression(max_iter=2000))
sel_in = list(selector.feature_names_in_) if hasattr(selector, 'feature_names_in_') else None

# If selector's inputs equal raw features, it already wraps preprocessing -> use it as sole frontend.
# Otherwise chain preprocessor -> selector -> classifier.
if sel_in is not None and set(sel_in) == set(raw_cols):
    pipe = Pipeline([('sel', selector), ('clf', clf)])
else:
    pipe = Pipeline([('pre', preprocessor), ('sel', selector), ('clf', clf)])

X_train = train[raw_cols]
X_val = val[raw_cols]
y_train = train[targets].astype(int)
y_val = val[targets].astype(int)

fitted = mlca_runtime.fit_record(pipe, X_train, 'train', y=y_train)

y_pred = np.asarray(fitted.predict(X_val))
per_f1 = [float(f1_score(y_val.iloc[:, i].to_numpy(), y_pred[:, i], average='macro'))
          for i in range(len(targets))]
macro_f1 = float(np.mean(per_f1))
subset_acc = float(np.mean(np.all(y_pred == y_val.to_numpy(), axis=1)))

mlca_runtime.record_metric('macro_f1', macro_f1, 'val')
mlca_runtime.record_metric('subset_accuracy', subset_acc, 'val')
mlca_runtime.save_model(fitted, 'model.pkl')

print(f"n_features={len(raw_cols)} n_targets={len(targets)}")
print(f"val_macro_f1={macro_f1:.4f} val_subset_acc={subset_acc:.4f}")
