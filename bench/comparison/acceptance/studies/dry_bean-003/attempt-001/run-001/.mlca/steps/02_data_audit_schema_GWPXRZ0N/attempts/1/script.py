import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

needed = ["targets", "excluded_columns", "group_column", "task"]
missing_keys = [k for k in needed if k not in protocol]
if missing_keys:
    raise ValueError(f"Missing protocol keys {missing_keys}; available keys: {list(protocol.keys())}")

df = pd.read_parquet(paths["dry_bean_uci"])
n_rows = int(len(df))

rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    rows.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": int(s.nunique(dropna=True)),
        "n_missing": n_missing,
        "pct_missing": round(100.0 * n_missing / n_rows, 4) if n_rows else float("nan"),
    })

schema = pd.DataFrame(rows, columns=["name", "dtype", "n_unique", "n_missing", "pct_missing"])
mlca_runtime.save_table(schema, "schema.json", caption="Column schema summary")

tokens = {"?", "-", "NA", "N/A", ""}
cols_with_placeholders = 0
found_tokens = set()
for col in df.columns:
    s = df[col]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        stripped = s.astype("string").str.strip()
        hit = stripped.isin(tokens)
        if bool(hit.any()):
            cols_with_placeholders += 1
            found_tokens |= set(stripped[hit].dropna().unique().tolist())
print(f"placeholder tokens: {sorted(found_tokens) if found_tokens else 'none'}; text columns containing them: {cols_with_placeholders}")

n_cols = int(df.shape[1])
missing_total = int(schema["n_missing"].sum())
mlca_runtime.record_fact_candidate("n_rows", n_rows, source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", n_cols, source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")
print(f"rows={n_rows} columns={n_cols} missing_total={missing_total}")
