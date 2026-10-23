import mlca_runtime
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["gcm_tumour_types_openml"])

placeholders = {"?", "-", "NA", "N/A", ""}
text_cols = df.select_dtypes(include=["object", "string"]).columns
ph_cols = []
for col in text_cols:
    s = df[col].astype("string").str.strip()
    mask = s.isin(placeholders)
    if mask.any():
        ph_cols.append(col)
        df.loc[mask, col] = pd.NA

records = []
n = len(df)
for col in df.columns:
    s = df[col]
    miss = int(s.isna().sum())
    pct = float(f"{miss / n * 100:.4g}") if n else float("nan")
    records.append({
        "name": col,
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": miss,
        "pct_missing": pct,
    })

schema = pd.DataFrame(records)
mlca_runtime.save_table(schema, "schema.json", caption="Column schema")

missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", int(df.shape[0]), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"placeholders: {len(ph_cols)} text columns contain placeholder tokens")
print(f"rows={df.shape[0]} cols={df.shape[1]} missing_total={missing_total}")
