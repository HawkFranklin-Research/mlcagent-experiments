import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, RegressorMixin

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

targets = proto.get("targets", [])
excluded = proto.get("excluded_columns", [])
group = proto.get("group_column", None)

def is_keylike(s):
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s) and s.nunique() == len(s):
        return True
    return False

feat_cols = [c for c in train.columns
             if c not in targets and c not in excluded and c != group
             and not is_keylike(train[c])]

with open(inp["05_preprocess_fit_preprocessing_78KYAZ91/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_retain_features_01V2VW4T/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def input_cols(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.named_steps.items():
            c = input_cols(step)
            if c:
                return c
    return None

pc = input_cols(pre)
if pc:
    feat_cols = [c for c in pc if c in train.columns and c not in targets
                 and c not in excluded and c != group]

Xtr = train[feat_cols]
Xva = val[feat_cols]

def contains(outer, inner):
    if not isinstance(outer, Pipeline):
        return False
    for _, step in outer.named_steps.items():
        if step is inner or contains(step, inner):
            return True
    return False

front = sel if contains(sel, pre) else Pipeline([("pre", pre), ("sel", sel)])

class AEWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, hidden=(32, 16, 32), max_iter=500, random_state=42):
        self.hidden = hidden
        self.max_iter = max_iter
        self.random_state = random_state
    def fit(self, X, y=None):
        self.mlp_ = MLPRegressor(hidden_layer_sizes=self.hidden, max_iter=self.max_iter,
                                 random_state=self.random_state, activation="relu", solver="adam")
        self.mlp_.fit(X, X)
        return self
    def predict(self, X):
        return self.mlp_.predict(X)

pipe = Pipeline([("front", front), ("ae", AEWrapper())])
mlca_runtime.fit_record(pipe, Xtr, partition="train")

Xva_t = pipe.named_steps["front"].transform(Xva)
rec = pipe.named_steps["ae"].predict(Xva_t)
mse = float(np.mean((rec - Xva_t) ** 2))

mlp = pipe.named_steps["ae"].mlp_
H1 = np.maximum(Xva_t @ mlp.coefs_[0] + mlp.intercepts_[0], 0)
Z = np.maximum(H1 @ mlp.coefs_[1] + mlp.intercepts_[1], 0)

mlca_runtime.save_model(pipe, "autoencoder_pipeline")

metrics = pd.DataFrame([{"model": "autoencoder_mlp", "metric": "reconstruction_error",
                          "value": round(mse, 6)}])
mlca_runtime.save_table(metrics, "validation_metrics.csv",
                        caption="Validation reconstruction error (scaled space)")

mlca_runtime.record_metric("reconstruction_error", mse, "val")
mlca_runtime.record_fact_candidate("best_validation_score", mse, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_model", "autoencoder_mlp", source="validation_metrics.csv")

print(f"features={len(feat_cols)} train={Xtr.shape} val={Xva.shape}")
print(f"bottleneck_dim={Z.shape[1]} val_mse={mse:.6f}")
