import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])

targets = protocol["targets"]
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column")
drop = excluded | set(targets) | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in drop]

with open(inputs["05_preprocess_fit_preprocessing_GK4CB9FZ/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(inputs["06_features_select_features_5542Z2W2/selector.pkl"], "rb") as f:
    selector = pickle.load(f)


def fn_in(obj):
    return list(getattr(obj, "feature_names_in_", []))


# If selector already takes raw feature columns, it embeds preprocessing; use it alone.
if fn_in(selector) == feature_cols:
    frontend = selector
else:
    frontend = Pipeline([("prep", preprocessor), ("select", selector)])

target = targets[0]
y = train[target]
minority = y.value_counts(normalize=True).min()
cw = "balanced" if minority < 0.1 else None

clf = HistGradientBoostingClassifier(random_state=42, class_weight=cw)
model = Pipeline([("frontend", frontend), ("clf", clf)])
X = train[feature_cols]
mlca_runtime.fit_record(model, X, "train", y=y)
mlca_runtime.save_model(model, "alternative_model.pkl")
print("features:", len(feature_cols), "rows:", len(train), "minority_frac:", round(float(minority), 3))
