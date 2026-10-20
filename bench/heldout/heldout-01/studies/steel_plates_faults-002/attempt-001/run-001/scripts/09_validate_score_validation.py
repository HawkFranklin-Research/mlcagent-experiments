import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = list(protocol["task"]["targets"])
primary = protocol["task"]["primary_metric"]

val = pd.read_parquet(inp["val"])
y_true = val[targets]

model_paths = {
    "baseline": inp["07_model_train_baseline_2DXYNCH8/model.pkl"],
    "alternative": inp["08_model_train_alternative_QF453D1M/alternative_model.pkl"],
}

def score(yt, yp):
    if not isinstance(yp, pd.DataFrame):
        yp = pd.DataFrame(yp, columns=targets, index=yt.index)
    else:
        yp = yp[targets]
    f1s = [f1_score(yt[t].to_numpy(), yp[t].to_numpy(), average='macro') for t in targets]
    hls = [hamming_loss(yt[t].to_numpy(), yp[t].to_numpy()) for t in targets]
    match = np.all(yt[targets].to_numpy() == yp[targets].to_numpy(), axis=1)
    return {"macro_f1": float(np.mean(f1s)),
            "hamming_loss": float(np.mean(hls)),
            "subset_accuracy": float(np.mean(match))}

rows, results = [], {}
for name, path in model_paths.items():
    with open(path, "rb") as f:
        model = pickle.load(f)
    cols = list(model.feature_names_in_)
    pred = model.predict(val[cols])
    s = score(y_true, pred)
    results[name] = s
    for m, v in s.items():
        rows.append({"model": name, "metric": m, "value": round(float(v), 4)})

mlca_runtime.save_table(pd.DataFrame(rows), "validation_metrics.csv",
                        "Validation metrics per candidate model")
best_name = max(results, key=lambda k: results[k][primary])
best_val = float(results[best_name][primary])
mlca_runtime.record_metric(primary, best_val, "val")
mlca_runtime.record_fact_candidate("best_validation_score", best_val, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")
print(f"val_rows={len(val)} models={list(results)}")
print(f"best={best_name} {primary}={best_val:.4f}")
