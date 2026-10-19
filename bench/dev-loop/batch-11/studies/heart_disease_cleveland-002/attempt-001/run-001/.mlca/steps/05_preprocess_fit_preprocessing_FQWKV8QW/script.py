import mlca_runtime
import json
import itertools
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = list(protocol.get('excluded_columns', []))
group_col = protocol.get('group_column') or None
target = targets[0]

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    s = train[c]
    if pd.api.types.is_integer_dtype(s) and s.notna().all() and s.is_unique:
        vals = sorted(s.unique())
        if vals and vals[0] in (0, 1) and vals[-1] - vals[0] + 1 == len(vals):
            print(f"Key-like int drop: {c}")
            continue
    if s.dtype == object and s.notna().all() and s.nunique() == len(train):
        print(f"Key-like text drop: {c}")
        continue
    feature_cols.append(c)

placeholders = {'?', '-', 'NA', 'N/A'}
ph_cols = [c for c in feature_cols
           if train[c].dtype == object
           and train[c].dropna().astype(str).str.strip().isin(placeholders).any()]
print(f"Placeholders found in {len(ph_cols)} columns")

y_tr = train[target]
nonneg = [c for c in feature_cols
          if pd.api.types.is_numeric_dtype(train[c]) and (train[c].fillna(0) >= 0).all()]
derived = set()
for c in nonneg:
    if (train[c] == y_tr).all():
        derived.add(c)
        print(f"Exclude {c}: equals target")
for k in (2, 3):
    for combo in itertools.combinations(nonneg, k):
        if (train[list(combo)].sum(axis=1) == y_tr).all():
            for c in combo:
                derived.add(c)
            print(f"Exclude {list(combo)}: sum equals target")
            break
feature_cols = [c for c in feature_cols if c not in derived]

if pd.api.types.is_numeric_dtype(y_tr) and y_tr.nunique() > 2:
    high = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(train[c]):
            corr = train[c].corr(y_tr)
            if pd.notna(corr) and abs(corr) >= 0.98:
                high.append(c)
    if high:
        print(f"High-corr warn: {high}")

num_cols, cat_cols = [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_numeric_dtype(s):
        if pd.api.types.is_integer_dtype(s) and 2 < s.nunique() <= 10:
            cat_cols.append(c)
        else:
            num_cols.append(c)
    else:
        if s.nunique() / max(len(train), 1) > 0.05 and s.nunique() > 5:
            print(f"Drop high-card: {c}")
            continue
        cat_cols.append(c)

num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('sc', StandardScaler())])
cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                     ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
ct = ColumnTransformer([('num', num_pipe, num_cols),
                        ('cat', cat_pipe, cat_cols)], remainder='drop')
ct.set_output(transform='pandas')

X_train = train[feature_cols]
X_val = val[feature_cols]
ct = mlca_runtime.fit_record(ct, X_train, 'train')
Xv = ct.transform(X_val)
print(f"train={X_train.shape} val={X_val.shape} encoded={Xv.shape}")
mlca_runtime.save_model(ct, 'preprocessor.pkl')
