import json, itertools
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pandas.api import types as ptypes

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
protocol = json.loads(inp['_protocol'].read_text())

targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
try:
    group_col = protocol['group_column']
except KeyError as e:
    raise KeyError(f"group_column missing; available keys: {list(protocol.keys())}") from e
if group_col is None:
    group_col = '__none__'

def is_key_like(s):
    n = len(s)
    if ptypes.is_object_dtype(s) or ptypes.is_string_dtype(s):
        return s.nunique(dropna=False) == n
    if ptypes.is_integer_dtype(s) and not ptypes.is_bool_dtype(s):
        return s.nunique(dropna=False) == n
    return False

feature_cols = [c for c in train.columns
                if c not in targets and c not in excluded and c != group_col
                and not is_key_like(train[c])]

excluded_feats = {}
for t in targets:
    if t not in train.columns:
        continue
    for c in feature_cols:
        if train[c].equals(train[t]):
            excluded_feats[c] = f"exact equality with {t}"
    if ptypes.is_numeric_dtype(train[t]):
        nonneg = [c for c in feature_cols
                  if ptypes.is_numeric_dtype(train[c]) and (train[c] >= 0).all()]
        for r in (1, 2, 3):
            for combo in itertools.combinations(nonneg, r):
                if (train[list(combo)].sum(axis=1) == train[t]).all():
                    for c in combo:
                        excluded_feats[c] = f"row sum equals {t}"
        for c in feature_cols:
            if ptypes.is_numeric_dtype(train[c]):
                corr = train[c].corr(train[t])
                if abs(corr) >= 0.98:
                    print(f"corr warning {c} {corr:.3f} vs {t}")

feature_cols = [c for c in feature_cols if c not in excluded_feats]
for c, r in excluded_feats.items():
    print(f"excluded {c}: {r}")
if not excluded_feats:
    print("no target-derived features excluded")

placeholders = {'?', '-', 'NA', 'N/A', ''}
text_cols = [c for c in feature_cols
             if ptypes.is_object_dtype(train[c]) or ptypes.is_string_dtype(train[c])]
ph_cols = []
for c in text_cols:
    s = train[c].astype(str).str.strip()
    if s.isin(placeholders).any() or (s == '').any():
        ph_cols.append(c)
print(f"placeholder tokens in {len(ph_cols)} text columns")

numeric_cols = [c for c in feature_cols if ptypes.is_numeric_dtype(train[c])]
cat_cols = [c for c in feature_cols if c not in numeric_cols]

num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('sc', StandardScaler())])
cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                     ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
num_pipe.set_output(transform='pandas')
cat_pipe.set_output(transform='pandas')

pre = ColumnTransformer(
    [('num', num_pipe, numeric_cols), ('cat', cat_pipe, cat_cols)],
    remainder='drop'
)
pre.set_output(transform='pandas')

X_train = train[feature_cols]
X_val = val[feature_cols]
y_train = train[targets[0]] if len(targets) == 1 else train[targets]

pre = mlca_runtime.fit_record(pre, X_train, 'train', y_train)
Xt_val = pre.transform(X_val)
print(f"fit train {X_train.shape}; transformed val {Xt_val.shape}")
mlca_runtime.save_model(pre, "preprocessor.pkl")
