import mlca_runtime, pickle, json
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]; excluded = protocol["excluded_columns"]
group = protocol.get("group_column"); task = protocol["task"]
metric = task["primary_metric"]

train = pd.read_parquet(inputs["train"]); val = pd.read_parquet(inputs["val"])
print(f"train rows={train.shape[0]} val rows={val.shape[0]} primary_metric={metric}")

def _load(p):
    with open(p, "rb") as f: return pickle.load(f)

model = _load(inputs["08_model_train_logistic_1JG6Z9GW/model.pkl"])
pre   = _load(inputs["06_preprocess_fit_preprocessing_XWW5WTFN/preprocessor.pkl"])
sel   = _load(inputs["07_features_select_features_3JX8FX9S/selector.pkl"])

if isinstance(model, Pipeline):
    pipe = model
else:
    steps = list(pre.steps) if isinstance(pre, Pipeline) else [("pre", pre)]
    steps += list(sel.steps) if isinstance(sel, Pipeline) else [("sel", sel)]
    steps.append(("frozen_model", model))
    pipe = Pipeline(steps)

raw_features = None
for _name, step in pipe.steps:
    if hasattr(step, "feature_names_in_"):
        raw_features = list(step.feature_names_in_); break
if raw_features is None:
    raw_features = [c for c in train.columns
                    if c not in targets and c not in excluded
                    and (group is None or c != group)]
print(f"raw feature schema size={len(raw_features)}")

final = pipe.steps[-1][1]
coef = None
if hasattr(final, "coef_"):
    arr = np.asarray(final.coef_)
    coef = arr.ravel() if arr.ndim > 1 else arr
elif hasattr(final, "feature_importances_"):
    coef = np.asarray(final.feature_importances_)

imp = None
if coef is not None:
    if len(coef) == len(raw_features):
        imp = pd.Series(coef, index=raw_features)
    else:
        cur = coef
        for _name, step in pipe.steps[-2::-1]:
            if hasattr(step, "get_support"):
                mask = np.asarray(step.get_support())
                if len(cur) != int(mask.sum()):
                    break
                full = np.zeros(len(mask)); full[mask] = cur
                if len(full) == len(raw_features):
                    imp = pd.Series(full, index=raw_features); break
                cur = full

if imp is not None:
    imp_abs = imp.abs().sort_values(ascending=False).head(15)
    interp = pd.DataFrame({
        "feature": imp_abs.index.astype(str),
        "importance_mean": imp_abs.values.astype(float),
        "importance_sd": np.nan,
    })
    mlca_runtime.save_table(interp, "interpretation.csv",
        caption="Top-15 features by |coef| from the frozen logistic pipeline (>500-feature regime: coefficients used instead of full permutation)")
    top_feature = str(interp.iloc[0]["feature"])
    mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
    mlca_runtime.save_json({
        "method": "frozen_logistic_coef_abs",
        "reason": ">500 input features; using model coefficients instead of full-width permutation",
        "n_features_ranked": int(len(imp)),
        "n_repeats": 0,
        "top_n_reported": 15,
    }, "interpretation_manifest.json")
    print(f"top feature(by |coef|): {top_feature}")
    print(f"ranked {len(imp)} features (coefficients, 0 permutation repeats)")
else:
    interp = pd.DataFrame({
        "feature": pd.Series(dtype="object"),
        "importance_mean": pd.Series(dtype="float"),
        "importance_sd": pd.Series(dtype="float"),
    })
    mlca_runtime.save_table(interp, "interpretation.csv",
        caption="No reliable feature importance ranking could be extracted from the frozen model; full-width permutation not performed (>500 features).")
    mlca_runtime.record_fact_candidate("top_feature", "unavailable", source="interpretation.csv")
    mlca_runtime.save_json({
        "limitation": "No reliable importance ranking (no coef_/feature_importances_ mappable to raw schema) and >500 features preclude full permutation; reporting limitation instead."
    }, "interpretation_manifest.json")
    print("no reliable importance ranking available")
