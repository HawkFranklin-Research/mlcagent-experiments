import mlca_runtime, json
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())

missing = [k for k in ['targets', 'excluded_columns', 'task'] if k not in proto]
if missing:
    raise KeyError(f"Missing protocol keys {missing}. Available: {list(proto.keys())}")
targets, excluded, task = proto['targets'], proto['excluded_columns'], proto['task']
group_col = proto.get('group_column', None)
target_col = targets[0]

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
print(f"Train={train.shape} Val={val.shape} target={target_col}")

def is_id(df, c):
    s = df[c]
    if s.nunique(dropna=False) != len(df): return False
    if pd.api.types.is_bool_dtype(s): return False
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_integer_dtype(s): return False
    return True

feature_cols = [c for c in train.columns
                if c not in targets and c not in excluded and c != group_col
                and not is_id(train, c)]

# RL-0008 derivation screen + correlation warning (numeric outcome only)
if pd.api.types.is_numeric_dtype(train[target_col]):
    y = train[target_col].astype(float).values
    for c in list(feature_cols):
        s = train[c]
        if not pd.api.types.is_numeric_dtype(s) or s.isna().any(): continue
        if np.allclose(s.values, y):
            print(f"Drop {c}: exact match with target")
            feature_cols.remove(c)
    for c in feature_cols:
        s = train[c]
        if not pd.api.types.is_numeric_dtype(s): continue
        try:
            corr = abs(np.corrcoef(s.fillna(s.mean()).values, y)[0, 1])
            if corr >= 0.98: print(f"WARN corr>=0.98: {c} ({corr:.3f})")
        except Exception: pass

# RL-0023 placeholder / whitespace screen
ph_cols = 0
for c in feature_cols:
    s = train[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        stripped = s.astype(str).str.strip()
        if stripped.isin(['?', '-', 'NA', 'N/A']).any() or (s.astype(str) != stripped).any():
            ph_cols += 1
print(f"Placeholder columns: {ph_cols}")

# RL-0010 type classification
num_cols, low_card, high_card, dt_cols = [], [], [], []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        dt_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        num_cols.append(c)
    elif s.nunique() <= max(1, int(0.05 * len(train))):
        low_card.append(c)
    else:
        high_card.append(c)

if high_card or dt_cols:
    print(f"Dropped high_card={len(high_card)} dt={len(dt_cols)}")

keep_cols = num_cols + low_card
if not keep_cols:
    raise ValueError("No feature columns remain after filtering")

# RL-0009 width decision
enc_width = len(num_cols) + sum(int(train[c].nunique()) for c in low_card)
use_pandas = enc_width <= 300

num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                     ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=not use_pandas))])
transformers = []
if num_cols: transformers.append(('num', num_pipe, num_cols))
if low_card: transformers.append(('cat', cat_pipe, low_card))

pre = ColumnTransformer(transformers, remainder='drop')
if use_pandas:
    pre.set_output(transform='pandas')

X_train = train[keep_cols]
X_val = val[keep_cols]

fitted = mlca_runtime.fit_record(pre, X_train, 'train')
if fitted is None:
    fitted = pre

Xt = fitted.transform(X_val)
print(f"Encoded val shape: {Xt.shape}")

mlca_runtime.save_model(fitted, 'preprocessor.pkl')
