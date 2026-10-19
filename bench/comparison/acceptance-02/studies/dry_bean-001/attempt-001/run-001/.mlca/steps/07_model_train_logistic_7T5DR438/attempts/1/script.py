import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

required = ["targets", "excluded_columns", "task", "group_column"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available: {list(protocol)}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol["group_column"]
task = protocol["task"]
target_col = targets[0]

train = pd.read_parquet(inputs["train"])
y = train[target_col]

with open(inputs["05_preprocess_fit_preprocessing_RMJEH56H/preprocessor.pkl"], "rb") as f:
    preproc = pickle.load(f)
with open(inputs["06_features_select_features_4X00PG4B/selector.pkl"], "rb") as f:
    selector = pickle.load(f)

if isinstance(selector, Pipeline) and len(selector.steps) > 1:
    base = selector
else:
    base = Pipeline([("preprocess", preproc), ("selector", selector)])

if hasattr(base, "feature_names_in_"):
    raw_cols = list(base.feature_names_in_)
elif hasattr(preproc, "feature_names_in_"):
    raw_cols = list(preproc.feature_names_in_)
else:
    raw_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
raw_cols = [c for c in raw_cols if c not in targets and c not in excluded and c != group]
X = train[raw_cols]

model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
steps = list(base.steps) if hasattr(base, "steps") else [("pipeline", base)]
name = "logreg"
if name in {s[0] for s in steps}:
    name = "logreg_model"
final = Pipeline(steps + [(name, model)])

fitted = mlca_runtime.fit_record(final, X, "train", y=y)
if fitted is None:
    fitted = final
mlca_runtime.save_model(fitted, "model.pkl")

print("train shape", train.shape)
print("features", len(raw_cols))
print("classes", len(train[target_col].unique()))
