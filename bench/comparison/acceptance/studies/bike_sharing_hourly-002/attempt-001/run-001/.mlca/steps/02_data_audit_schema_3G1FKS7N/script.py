import mlca_runtime
import pandas as pd
import json

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["bike_sharing_hourly_uci"])
_ = json.loads(inputs["_protocol"].read_text())

n_rows = len(df)
records = []
for i, col in enumerate(df.columns):
    s = df.iloc[:, i]
    n_missing = int(s.isna().sum())
    pct = 100.0 * n_missing / n_rows if n_rows else 0.0
    records.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": float(f"{pct:.4g}"),
    })

schema = pd.DataFrame(
    records,
    columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"],
)
mlca_runtime.save_table(schema, "schema.json", caption="Per-column schema summary")

missing_total = int(schema["n_missing"].sum())
for key, value in [
    ("n_rows", n_rows),
    ("n_columns", int(df.shape[1])),
    ("missing.total", missing_total),
]:
    mlca_runtime.record_fact_candidate(key, value, source="schema.json")

print(f"Rows {n_rows}, columns {df.shape[1]}, missing total {missing_total}")
