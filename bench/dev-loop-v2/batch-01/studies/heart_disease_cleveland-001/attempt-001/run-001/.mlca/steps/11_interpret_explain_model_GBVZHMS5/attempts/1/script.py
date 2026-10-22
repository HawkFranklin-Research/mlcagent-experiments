import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

mlca_runtime.set_seed(42)
SEED = 42

inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []))
group = protocol.get("group_column")

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def load_pkl(key):
    with open(inp[key], "rb") as f:
        return pickle.load(f)

preprocessor = load_pkl("05_preprocess_fit_preprocessing_7578M7XP/preprocessor.pkl")
selector = load_pkl("06_features_select_features_J83AE8FG/selector.pkl")
model = load_pkl("07_model_train_logistic_X39FZFWK/model.pkl")
latest = load_pkl("latest_model")

drop_set = set(targets) | set(excluded)
if group:
    drop_set.add(group)

key_like = []
n = len(train)
for c in train.columns:
    if c in drop_set:
        continue
    s = train[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=False) == n:
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(s.dtype):
        if s.is_unique and (s.sort_values().diff().dropna() == 1).all():
            key_like.append(c)

feature_cols = [c for c in train.columns if c not in drop_set and c not in key_like]

def first_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline) and obj.named_steps:
        return first_names(next(iter(obj.named_steps.values())))
    return None

def matches(obj, cols):
    names = first_names(obj)
    return names is not None and list(names) == list(cols)

complete = None
for cand in (latest, model):
    if cand is None:
        continue
    names = first_names(cand)
    if names is not None and set(names).issubset(set(feature_cols)):
        complete = cand
        break

if complete is None:
    steps = []
    if preprocessor is not None:
        steps.append(("preprocessor", preprocessor))
    if selector is not None:
        if matches(selector, feature_cols):
            steps = [("selector", selector)]
        else:
            steps.append(("selector", selector))
    model_step = latest if latest is not None else model
    if matches(model_step, feature_cols):
        complete = model_step
    else:
        steps.append(("model", model_step))
        complete = Pipeline(steps)

expected = first_names(complete)
use_cols = expected if expected is not None and set(expected).issubset(set(val.columns)) else feature_cols
X_val = val[use_cols].copy()
y_val = val[targets[0]].copy()

task = protocol.get("task", {})
primary = task.get("primary_metric")
if task.get("type") == "binary" or primary in ("roc_auc", "auc"):
    scoring = "roc_auc"
elif task.get("type") == "multiclass":
    scoring = "roc_auc_ovr"
else:
    scoring = primary

result = permutation_importance(
    complete,
    X_val,
    y_val,
    scoring=scoring,
    n_repeats=20,
    random_state=SEED,
    n_jobs=1,
)

imp = pd.DataFrame(
    {
        "feature": list(X_val.columns),
        "importance_mean": result.importances_mean,
        "importance_sd": result.importances_std,
    }
)
imp = imp.sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
imp["importance_mean"] = imp["importance_mean"].map(lambda v: float(f"{v:.4g}"))
imp["importance_sd"] = imp["importance_sd"].map(lambda v: float(f"{v:.4g}"))

mlca_runtime.save_table(
    imp,
    "interpretation.csv",
    caption="Permutation importance on validation rows (top 15 features).",
)

top_feature = imp.iloc[0]["feature"]
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"train rows={len(train)}, val rows={len(val)}")
print(f"features={len(feature_cols)}, pipeline={type(complete).__name__}")
print(f"top_feature={top_feature}")
