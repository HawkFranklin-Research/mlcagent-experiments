import mlca_runtime, json
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])

targets = protocol['targets']
excluded = protocol.get('excluded_columns', []) or []
group = protocol.get('group_column')

feat_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]

key_like = []
for c in feat_cols:
    s = train[c]
    nn = int(s.notna().sum())
    if nn <= 1 or s.nunique(dropna=True) != nn:
        continue
    if pd.api.types.is_float_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        continue
    if pd.api.types.is_integer_dtype(s) or not pd.api.types.is_numeric_dtype(s):
        key_like.append(c)
feat_cols = [c for c in feat_cols if c not in key_like]

n_rows = len(train)
numeric_cols, cat_cols, date_cols = [], [], []
for c in feat_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        date_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    else:
        nun = s.nunique(dropna=True)
        if 0 < nun <= max(1, int(0.05 * n_rows)):
            cat_cols.append(c)

def date_feats(df):
    out = {}
    for c in df.columns:
        d = pd.to_datetime(df[c], errors='coerce')
        out[f'{c}_year'] = d.dt.year
        out[f'{c}_month'] = d.dt.month
        out[f'{c}_day'] = d.dt.day
        out[f'{c}_dow'] = d.dt.dayofweek
    return pd.DataFrame(out, index=df.index).fillna(-1)

tfs = []
if numeric_cols:
    tfs.append(('num', SimpleImputer(strategy='median'), numeric_cols))
if cat_cols:
    tfs.append(('cat', Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ]), cat_cols))
if date_cols:
    tfs.append(('date', FunctionTransformer(date_feats, validate=False), date_cols))

pre = ColumnTransformer(tfs, remainder='drop')
pre.set_output(transform='pandas')

clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
pipe = Pipeline([('pre', pre), ('clf', clf)])

X = train[feat_cols]
y = train[targets]
model = mlca_runtime.fit_record(pipe, X, 'train', y=y)
mlca_runtime.save_model(model, 'alternative_model.pkl')

print(f"train_rows={n_rows} feat_cols={len(feat_cols)} numeric={len(numeric_cols)} cat={len(cat_cols)} date={len(date_cols)}")
print(f"key_like_dropped={key_like} targets={targets}")
