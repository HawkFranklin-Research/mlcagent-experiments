import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

SEED = 42
mlca_runtime.set_seed(SEED)

inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
target = protocol["targets"][0]

train = pd.read_parquet(inp["train"])
with open(inp["07_model_train_logistic_BS7D3474/model.pkl"], "rb") as f:
    model = pickle.load(f)

def get_raw_columns(obj, frame):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    drop = set(protocol["targets"]) | set(protocol.get("excluded_columns", []))
    if protocol.get("group_column"):
        drop.add(protocol["group_column"])
    n = len(frame)
    cols = []
    for c in frame.columns:
        if c in drop:
            continue
        s = frame[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            cols.append(c)
        elif s.nunique(dropna=True) <= max(5, int(0.05 * n)):
            cols.append(c)
    return cols

cols = get_raw_columns(model, train)
X = train.loc[:, cols]
y = train[target]

hgb = HistGradientBoostingClassifier(random_state=SEED)

if isinstance(model, Pipeline):
    alt = clone(model)
    last_name = list(alt.named_steps)[-1]
    alt.set_params(**{last_name: hgb})
else:
    with open(inp["05_preprocess_fit_preprocessing_AP3P14P9/preprocessor.pkl"], "rb") as f:
        pre = pickle.load(f)
    with open(inp["06_features_select_features_NBH1R9WT/selector.pkl"], "rb") as f:
        sel = pickle.load(f)
    if isinstance(pre, Pipeline) and any(k in pre.named_steps for k in ("selector", "select")):
        alt = Pipeline([("preprocessor", pre), ("model", hgb)])
    else:
        alt = Pipeline([("preprocessor", pre), ("selector", sel), ("model", hgb)])

fitted = mlca_runtime.fit_record(alt, X, "train", y=y)
mlca_runtime.save_model(fitted, "alternative_model.pkl")
print(f"train rows={len(train)} features={len(cols)} target={target}")
