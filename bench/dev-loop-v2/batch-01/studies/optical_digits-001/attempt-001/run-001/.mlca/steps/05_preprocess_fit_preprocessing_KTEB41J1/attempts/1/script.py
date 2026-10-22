import mlca_runtime, json
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp['train'])
protocol = json.loads(inp['_protocol'].read_text())

for k in ('targets', 'excluded_columns', 'task'):
    if k not in protocol:
        raise ValueError(f"Missing {k}. Keys: {list(protocol.keys())}")
targets = protocol['targets']
excluded = list(protocol['excluded_columns'])
group = protocol.get('group_column')

cand = [c for c in train.columns if c not in targets and c not in excluded and c != group]
id_drop = [c for c in cand if train[c].dtype.kind in 'Oiu'
           and train[c].nunique() == len(train) and train[c].nunique() > 10]
feats = [c for c in cand if c not in id_drop]
X = train[feats].copy()

num_cols, cat_cols, dt_cols = [], [], []
for c in feats:
    s = X[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        dt_cols.append(c)
    elif s.dtype.kind in 'iuf':
        num_cols.append(c)
    else:
        cat_cols.append(c)

for c in dt_cols:
    d = pd.to_datetime(X[c], errors='coerce')
    X[c + '_dow'] = d.dt.dayofweek
    X[c + '_mon'] = d.dt.month
    num_cols.extend([c + '_dow', c + '_mon'])

high_card = [c for c in cat_cols if X[c].nunique() > max(10, 0.05 * len(X))]
for c in high_card:
    cat_cols.remove(c)
    X.drop(columns=[c], inplace=True)

ph = []
for c in cat_cols:
    s = X[c].astype(str)
    mask = s.str.strip().isin(['?', '-', 'NA', 'N/A', ''])
    if mask.any():
        ph.append(c)
        X[c] = s.str.strip().replace({'?': np.nan, '-': np.nan, 'NA': np.nan, 'N/A': np.nan, '': np.nan})

print(f"features={len(feats)} id_drop={len(id_drop)} high_card={len(high_card)} ph_cols={len(ph)}")

num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                     ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
parts = ([('num', num_pipe, num_cols)] if num_cols else []) + \
        ([('cat', cat_pipe, cat_cols)] if cat_cols else [])
pre = ColumnTransformer(parts, remainder='drop')
pre.set_output(transform='pandas')
pre = mlca_runtime.fit_record(pre, X, partition='train')
mlca_runtime.save_model(pre, 'preprocessor.pkl')
print("saved preprocessor.pkl with", pre.get_feature_names_out().shape[0], "output cols")
