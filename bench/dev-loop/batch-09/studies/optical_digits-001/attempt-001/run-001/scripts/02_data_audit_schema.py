import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs["optical_digits_uci"])
protocol = json.loads(inputs["_protocol"].read_text())

for key in ["targets", "excluded_columns", "task"]:
    if key not in protocol:
        raise KeyError(
            f"Missing '{key}' in _protocol; available top-level keys: {list(protocol.keys())}"
        )
targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
task = protocol["task"]

# RL-0023 placeholder scan (also strips whitespace for the check)
placeholders = {"?", "-", "NA", "N/A", ""}
found_tokens = set()
placeholder_cols = 0
for col in df.columns:
    s = df[col]
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        stripped = s.astype("string").str.strip()
        hit = False
        for tok in placeholders:
            if bool((stripped == tok).fillna(False).any()):
                found_tokens.add(tok)
                hit = True
        if hit:
            placeholder_cols += 1

n = len(df)
rows = []
for col in df.columns:
    s = df[col]
    n_missing = int(s.isna().sum())
    pct_missing = round(100.0 * n_missing / n, 4) if n else 0.0
    try:
        n_unique = int(s.nunique(dropna=True))
    except TypeError:
        n_unique = int(s.astype(str).nunique(dropna=True))
    rows.append({
        "name": str(col),
        "dtype": str(s.dtype),
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct_missing,
    })

mlca_runtime.save_json(rows, "schema.json", kind="table")

missing_total = int(sum(r["n_missing"] for r in rows))
mlca_runtime.record_fact_candidate("n_rows", int(n), source="schema.json")
mlca_runtime.record_fact_candidate("n_columns", int(df.shape[1]), source="schema.json")
mlca_runtime.record_fact_candidate("missing.total", missing_total, source="schema.json")

print(f"shape={df.shape}")
print(f"placeholder tokens found: {sorted(found_tokens)}; columns affected: {placeholder_cols}")
print(f"missing.total={missing_total}")
