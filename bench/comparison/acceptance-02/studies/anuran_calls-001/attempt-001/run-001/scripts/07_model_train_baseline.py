import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol.get("group_column")

train = pd.read_parquet(inp["train"])
y = train[targets].copy()

pre = pickle.loads(inp["05_preprocess_fit_preprocessing_WHDXCN7B/preprocessor.pkl"].read_bytes())
sel = pickle.loads(inp["06_features_retain_features_J1HGBJXK/selector_all.pkl"].read_bytes())

raw = set(train.columns)

def _fn(o):
    v = getattr(o, "feature_names_in_", None)
    return list(v) if v is not None else []

sel_cols = _fn(sel)
pre_cols = _fn(pre)
sel_on_raw = bool(sel_cols) and all(c in raw for c in sel_cols)
sel_has_pre = isinstance(sel, Pipeline) and any(
    k in sel.named_steps for k in ("preprocessor", "prep", "preprocess")
)
drop = set(targets + excluded + ([group_col] if group_col else []))

clf = MultiOutputClassifier(LogisticRegression(max_iter=2000))

if sel_has_pre:
    candidates = sel_cols if sel_cols else list(train.columns)
    fcols = [c for c in candidates if c in raw and c not in drop]
    model = Pipeline(list(sel.steps) + [("clf", clf)])
elif sel_on_raw:
    fcols = [c for c in sel_cols if c not in drop]
    model = Pipeline([("selector", sel), ("clf", clf)])
else:
    base = [c for c in pre_cols if c in raw and c not in drop]
    if not base:
        base = [c for c in train.columns if c not in drop]
    fcols = base
    for obj in (pre, sel):
        try:
            obj.set_output(transform="pandas")
        except Exception:
            pass
    model = Pipeline([("preprocessor", pre), ("selector", sel), ("clf", clf)])

X = train[fcols].copy()
mlca_runtime.fit_record(model, X, "train", y=y)
mlca_runtime.save_model(model, "model.pkl")
mlca_runtime.record_metric("n_train_rows", float(len(X)), "train")
mlca_runtime.record_metric("n_features", float(len(fcols)), "train")
print(f"trained rows={len(X)} features={len(fcols)} targets={len(targets)}")
print("saved model.pkl")
