import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
df = pd.read_parquet(inp["rna_pr"])
protocol = json.loads(inp["_protocol"].read_text())
targets = list(protocol["targets"])

rows = []
for c in df.columns:
    s = df[c]
    n_miss = int(s.isna().sum())
    rows.append({
        "name": c,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_miss,
        "pct_missing": round(float(s.isna().mean() * 100.0), 4),
    })
schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_table(schema, "schema.json", caption="Per-column schema of the frozen RNA-seq table")

n_rows = int(df.shape[0])
n_columns = int(df.shape[1])
missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_columns, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"table rows={n_rows} columns={n_columns}")
print(f"saved schema.json with {len(schema)} column rows")
print(f"missing_total={missing_total}")
