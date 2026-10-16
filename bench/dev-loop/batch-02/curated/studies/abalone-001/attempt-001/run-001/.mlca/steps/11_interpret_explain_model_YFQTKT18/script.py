import mlca_runtime
import json
import pickle
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

required = ['targets', 'excluded_columns', 'group_column', 'task']
for k in required:
    if k not in protocol:
        raise KeyError(f"Missing key {k} in _protocol. Available keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol['group_column']
task = protocol['task']

with open(inputs['05_preprocess_fit_preprocessing_JT5JG4F1/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_retain_features_6CDSH3YD/selector_all.pkl'], 'rb') as f:
    selector = pickle.load(f)
with open(inputs['07_model_train_baseline_F3SAKS84/model.pkl'], 'rb') as f:
    baseline_model = pickle.load(f)
with open(inputs['latest_model'], 'rb') as f:
    latest_model = pickle.load(f)

with open(inputs['07_model_train_baseline_F3SAKS84/manifest.json'], 'r') as f:
    baseline_manifest = json.load(f)
with open(inputs['08_model_train_alternative_G7VTXE7A/manifest.json'], 'r') as f:
    alt_manifest = json.load(f)
with open(inputs['09_validate_score_validation_WEK9FDVR/manifest.json'], 'r') as f:
    val_manifest = json.load(f)

val_metrics = pd.read_csv(inputs['09_validate_score_validation_WEK9FDVR/validation_metrics.csv'])

def get_final_estimator(model):
    if hasattr(model, 'named_steps'):
        return list(model.named_steps.values())[-1]
    return model

def get_hyperparams(manifest):
    for key in ['params', 'hyperparameters', 'best_params', 'model_params']:
        if key in manifest:
            return manifest[key]
    return {}

def get_feature_names(obj):
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if hasattr(obj, 'get_feature_names_out'):
        try:
            return list(obj.get_feature_names_out())
        except:
            pass
    return []

def get_top_feature(model, selector, preprocessor):
    final = get_final_estimator(model)
    if hasattr(final, 'feature_importances_'):
        importances = final.feature_importances_
    elif hasattr(final, 'coef_'):
        importances = np.abs(final.coef_).flatten()
    else:
        return "unknown"
    if hasattr(selector, 'get_support'):
        support = selector.get_support()
        all_names = get_feature_names(preprocessor)
        if not all_names:
            all_names = [f"f{i}" for i in range(len(support))]
        selected_names = np.array(all_names)[support]
    else:
        selected_names = [f"f{i}" for i in range(len(importances))]
    if len(importances) != len(selected_names):
        return "unknown"
    idx = np.argmax(importances)
    return str(selected_names[idx])

rows = []
rows.append({"section": "Model", "item": "Type", "value": type(get_final_estimator(latest_model)).__name__})
rows.append({"section": "Model", "item": "Hyperparameters", "value": json.dumps(get_hyperparams(alt_manifest))})
rows.append({"section": "Inputs", "item": "Target", "value": str(targets)})
feature_names = get_feature_names(selector) or get_feature_names(preprocessor)
rows.append({"section": "Inputs", "item": "Features", "value": ", ".join(feature_names[:20]) + ("..." if len(feature_names) > 20 else "")})
rows.append({"section": "Validation", "item": "Primary metric", "value": f"{task['primary_metric']}: {val_metrics[task['primary_metric']].iloc[0] if task['primary_metric'] in val_metrics.columns else 'N/A'}"})
for col in val_metrics.columns:
    if col != task['primary_metric']:
        rows.append({"section": "Validation", "item": col, "value": str(val_metrics[col].iloc[0])})
rows.append({"section": "Limitations", "item": "Notes", "value": "Frozen model, not fitted in this script. Trained on abalone_uci dataset. No repeated subjects. May not generalize to other populations."})

df = pd.DataFrame(rows)
mlca_runtime.save_table(df, "interpretation.csv", caption="Model interpretation summary")

top_feature = get_top_feature(latest_model, selector, preprocessor)
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"Saved interpretation.csv with {len(df)} rows")
print(f"Top feature: {top_feature}")
