import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

for key in ("targets", "excluded_columns"):
    if key not in protocol:
        raise ValueError(f"_protocol missing {key!r}; available keys: {sorted(protocol)}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]

df = pd.read_parquet(paths["smk187_airway_openml"])
n_rows = len(df)
records = []

for i, col in enumerate(df.columns):
    s = df.iloc[:, i]
    n_missing = int(s.isna().sum())
    pct_missing = float(f"{(n_missing / n_rows) * 100.0:.4g}") if n_rows else float("nan")
    records.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })

schema = pd.DataFrame.from_records(
    records,
    columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"],
)
mlca_runtime.save_table(
    schema,
    "schema.json",
    caption="Column schema: name, dtype, unique count, missing count and percent missing.",
)

missing_total = int(schema["n_missing"].sum()) if len(schema) else 0
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", len(df.columns), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"rows={n_rows} columns={len(df.columns)} missing={missing_total}")
