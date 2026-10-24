import mlca_runtime, json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
train = pd.read_parquet(inputs['train'])
targets = protocol['targets']
excluded = list(protocol['excluded_columns'])
group_col = protocol.get('group_column')
n = len(train)

num_cols, cat_cols, feature_cols = [], [], []
for c in train.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    s = train[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        continue
    if pd.api.types.is_numeric_dtype(s):
        num_cols.append(c); feature_cols.append(c)
    else:
        nun = s.nunique(dropna=True)
        if nun == n and nun > 0:
            continue
        if nun <= max(1, int(0.05 * n)):
            cat_cols.append(c); feature_cols.append(c)

pre_steps = []
if num_cols:
    pre_steps.append(('num', Pipeline([('imp', SimpleImputer(strategy='median'))]), num_cols))
if cat_cols:
    pre_steps.append(('cat', Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ]), cat_cols))

pre = ColumnTransformer(pre_steps, remainder='drop')
pre.set_output(transform='pandas')

model = Pipeline([
    ('pre', pre),
    ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))),
])

mlca_runtime.fit_record(model, train[feature_cols], 'train', y=train[targets])
mlca_runtime.save_model(model, 'alternative_model.pkl')
