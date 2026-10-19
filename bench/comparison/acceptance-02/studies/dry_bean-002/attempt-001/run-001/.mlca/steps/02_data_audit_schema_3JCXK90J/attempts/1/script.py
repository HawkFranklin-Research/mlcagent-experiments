import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
df = pd.read_parquet(paths["dry_bean_uci"])
protocol = json.loads(paths["_protocol"].read_text())
for key in ("targets", "excluded_columns"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {sorted(protocol)}")

n_rows = int(df.shape[0])
schema = pd.DataFrame({
    "name": [str(c) for c in df.columns],
    "dtype": [str(t) for t in df.dtypes],
    "n_unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
    "n_missing": [int(df[c].isna().sum()) for c in df.columns],
})
if n_rows:
    schema["pct_missing"] = (schema["n_missing"] / n_rows * 100).map(lambda x: float(f"{x:.4g}"))
else:
    schema["pct_missing"] = float("nan")

mlca_runtime.save_table(schema, "schema.json", caption="Per-column schema summary")
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", int(schema["n_missing"].sum()), source="schema.json")
print(f"rows={n_rows} cols={df.shape[1]} missing_total={int(schema['n_missing'].sum())}")
