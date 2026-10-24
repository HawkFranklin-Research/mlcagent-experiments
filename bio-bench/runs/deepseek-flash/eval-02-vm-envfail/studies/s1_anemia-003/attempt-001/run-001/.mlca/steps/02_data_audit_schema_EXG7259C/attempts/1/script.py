import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
_ = protocol["task"]["primary_metric"]

df = pd.read_parquet(inputs["anemia"])
n_rows = len(df)
rows = []
for c in df.columns:
    s = df[c]
    miss = int(s.isna().sum())
    rows.append({
        "name": c,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": miss,
        "pct_missing": round(100.0 * miss / n_rows, 4) if n_rows else float("nan"),
    })

placeholders = {"?", "-", "NA", "N/A"}
hits = 0
for c in df.columns:
    s = df[c]
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        vals = s.dropna().astype(str).str.strip()
        if (vals.isin(placeholders) | (vals == "")).any():
            hits += 1

schema_df = pd.DataFrame(rows)
total_missing = int(schema_df["n_missing"].sum())
mlca_runtime.save_json(rows, "schema.json", kind="table")
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", len(df.columns), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", total_missing, source="schema.json")
print(f"placeholder columns: {hits}")
print(f"rows={n_rows} cols={len(df.columns)} missing={total_missing}")
