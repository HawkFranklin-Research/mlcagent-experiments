import mlca_runtime, json, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
targets = proto['targets']
excluded = set(proto.get('excluded_columns', []))
group_col = proto.get('group_column', None)
train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
n = len(train)
features = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]

key_cols = []
for c in features:
    s = train[c]
    if s.dtype == object and s.nunique() == n and n > 1:
        key_cols.append(c)
    elif pd.api.types.is_integer_dtype(s) and s.nunique() == n and n > 1 and (s.max() - s.min()) == n - 1:
        key_cols.append(c)
features = [c for c in features if c not in key_cols]

y = train[targets[0]]
deriv = []
if pd.api.types.is_numeric_dtype(y):
    yv = y.values
    for c in features:
        if np.array_equal(train[c].values, yv):
            deriv.append(c)
    warn = []
    for c in features:
        if c in deriv: continue
        if pd.api.types.is_numeric_dtype(train[c]):
            corr = train[c].corr(y)
            if abs(corr) >= 0.98:
                warn.append(f"{c}({corr:.2f})")
    print("Correlation warnings: " + (", ".join(warn) if warn else "none"))
features = [c for c in features if c not in deriv]
if deriv: print("Excluded derivations: " + ", ".join(deriv))

numeric, cat_low, cat_high, dt_cols = [], [], [], []
for c in features:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        dt_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        numeric.append(c)
    elif s.dtype == object or pd.api.types.is_string_dtype(s):
        if s.nunique() <= 0.05 * n:
            cat_low.append(c)
        else:
            cat_high.append(c)
    else:
        numeric.append(c)
features = [c for c in features if c not in dt_cols]

ph = ['?', '-', 'NA', 'N/A', '']
ph_cols = sum(1 for c in features if train[c].dtype == object
              and train[c].astype(str).str.strip().isin(ph).any())
print(f"Features:{len(features)} num:{len(numeric)} cat_low:{len(cat_low)} "
      f"cat_high_dropped:{len(cat_high)} dt_dropped:{len(dt_cols)} ph_cols:{ph_cols}")

num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                     ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
pre = ColumnTransformer([('num', num_pipe, numeric), ('cat', cat_pipe, cat_low)],
                        remainder='drop')
pre.set_output(transform='pandas')

clean = Pipeline([
    ('strip', FunctionTransformer(pd.DataFrame.replace,
                                  kw_args={'to_replace': r'^\s+|\s+$', 'value': '', 'regex': True})),
    ('ph', FunctionTransformer(pd.DataFrame.replace,
                               kw_args={'to_replace': ph, 'value': np.nan})),
    ('pre', pre)
])

X_train = train[features].copy()
mlca_runtime.fit_record(clean, X_train, 'train', y=None)
Xv = clean.transform(val[features].copy())
print(f"Transformed val shape: {Xv.shape}")
mlca_runtime.save_model(clean, 'preprocessor.pkl')
