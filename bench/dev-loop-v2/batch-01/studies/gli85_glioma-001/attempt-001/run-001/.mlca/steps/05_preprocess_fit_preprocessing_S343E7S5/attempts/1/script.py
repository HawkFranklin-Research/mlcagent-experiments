import json
import mlca_runtime
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
protocol = json.loads(inp['_protocol'].read_text())
targets, excluded = protocol['targets'], protocol['excluded_columns']
group = protocol.get('group_column')
feat = [c for c in train.columns if c not in targets and c not in excluded and c != group]

# Derivation screen (target is categorical -> exact-equality only)
derived = []
tcol = targets[0]
if pd.api.types.is_numeric_dtype(train[tcol]):
    for c in feat:
        if pd.api.types.is_numeric_dtype(train[c]) and train[c].equals(train[tcol]):
            derived.append(c)
else:
    tstr = train[tcol].astype(str)
    for c in feat:
        if train[c].astype(str).equals(tstr):
            derived.append(c)
for c in derived:
    print(f"Excluded derivation: {c}")
feat = [c for c in feat if c not in derived]

numeric_cols = [c for c in feat
                if pd.api.types.is_numeric_dtype(train[c]) and not pd.api.types.is_bool_dtype(train[c])]
cat_cols = [c for c in feat if c not in numeric_cols
            and train[c].nunique() <= max(5, int(0.05 * len(train)))]

for c in list(numeric_cols):
    if pd.api.types.is_datetime64_any_dtype(train[c]):
        print(f"Datetime column {c} dropped")
        numeric_cols.remove(c)

ph = {'?', '-', 'NA', 'N/A'}
ph_cols = [c for c in cat_cols
           if train[c].dropna().astype(str).str.strip().isin(ph).any()]
print(f"Placeholder token columns: {len(ph_cols)}")

num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('sc', StandardScaler())])
cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                     ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
trs = []
if numeric_cols:
    trs.append(('num', num_pipe, numeric_cols))
if cat_cols:
    trs.append(('cat', cat_pipe, cat_cols))
pre = ColumnTransformer(trs, remainder='drop')

print(f"features={len(feat)} numeric={len(numeric_cols)} cat={len(cat_cols)}")

mlca_runtime.fit_record(pre, train[feat], 'train', y=train[tcol])
Xt = pre.transform(val[feat])
print(f"val transformed shape: {Xt.shape}")
mlca_runtime.save_model(pre, 'preprocessor.pkl')
