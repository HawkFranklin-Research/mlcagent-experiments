import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.utils.validation import check_is_fitted

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
targets = list(proto['targets'])
excluded = set(proto['excluded_columns'])
group_col = proto.get('group_column')

train = pd.read_parquet(inp['train'])
with open(inp['05_preprocess_fit_preprocessing_ABA216ED/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)

feat_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]

def is_keylike(s):
    if s.dtype == object:
        return len(s) > 0 and s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        vals = set(s.dropna().unique())
        return s.nunique(dropna=True) == len(s) and not vals <= {0, 1}
    return False

feat_cols = [c for c in feat_cols if not is_keylike(train[c])]
X = train[feat_cols].copy()
y_raw = train[targets[0]]
if pd.api.types.is_numeric_dtype(y_raw):
    y = y_raw.astype(int)
else:
    classes = sorted(y_raw.astype(str).dropna().unique())
    y = pd.Series((y_raw.astype(str) == classes[-1]).astype(int),
                  index=y_raw.index, name=targets[0])

try:
    check_is_fitted(pre)
    n_features = pre.transform(X.head(5)).shape[1]
except Exception:
    pre = mlca_runtime.fit_record(pre, X, 'train', y)
    n_features = pre.transform(X.head(5)).shape[1]

k = min(20, n_features)
pipe = Pipeline([('preprocessor', pre),
                 ('select', SelectKBest(mutual_info_classif, k=k))])
full = mlca_runtime.fit_record(pipe, X, 'train', y)
mlca_runtime.save_model(full, 'selector.pkl')
mlca_runtime.record_metric('n_features_in', float(len(feat_cols)), 'train')
mlca_runtime.record_metric('n_features_selected', float(k), 'train')
print('features', len(feat_cols), 'encoded', n_features, 'selected', k)
