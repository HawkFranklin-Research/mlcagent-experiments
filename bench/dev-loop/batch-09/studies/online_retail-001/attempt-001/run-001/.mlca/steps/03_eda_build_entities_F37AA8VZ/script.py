import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["online_retail_uci"])

def need(obj, key):
    if key not in obj:
        raise KeyError(f"Missing {key!r} in _protocol; available top-level keys: {sorted(obj.keys())}")
    return obj[key]

group_col = need(protocol, "group_column")
need(protocol, "targets")
need(protocol, "excluded_columns")
need(protocol, "task")

if group_col not in df.columns:
    raise ValueError("Declared group_column is not present in the input table")

def find_col(name):
    target = name.lower()
    for c in df.columns:
        if str(c).lower() == target:
            return c
    return None

qty_col = find_col("Quantity")
price_col = find_col("UnitPrice")
inv_col = find_col("InvoiceNo")
if qty_col is None or price_col is None:
    raise ValueError("Quantity and UnitPrice columns are required for monetary aggregation")

date_candidates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
if not date_candidates:
    date_candidates = [c for c in df.columns if "date" in str(c).lower() or "time" in str(c).lower()]
if not date_candidates:
    raise ValueError("No datetime/date column found for recency")
date_col = date_candidates[0]
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
if df[date_col].notna().sum() == 0:
    raise ValueError("No valid dates found for recency")
global_end = df[date_col].max()

raw_n = len(df)
entity = df[group_col]
mask = entity.notna()
if entity.dtype == object or pd.api.types.is_string_dtype(entity):
    mask = mask & entity.astype("string").str.strip().ne("")
df = df.loc[mask].copy()
dropped = raw_n - len(df)
if df.empty:
    raise ValueError("No rows with a nonmissing entity key")

qty = pd.to_numeric(df[qty_col], errors="coerce")
price = pd.to_numeric(df[price_col], errors="coerce")
df["_amount"] = qty * price
df["_is_return"] = (qty < 0) | (df["_amount"] < 0)
return_n = int(df["_is_return"].sum())

def recency_days(s):
    m = s.max()
    return (global_end - m).days if pd.notna(m) else float("nan")

g = df.groupby(group_col, dropna=False, sort=False)
sizes = g.size()
freq_spec = (inv_col, "nunique") if inv_col is not None else ("_amount", "size")
entities = g.agg(
    recency_days=(date_col, recency_days),
    frequency=freq_spec,
    monetary_total=("_amount", "sum"),
    monetary_mean=("_amount", "mean"),
    n_lines=("_amount", "size"),
    n_returns=("_is_return", "sum"),
    last_purchase=(date_col, "max"),
    first_purchase=(date_col, "min"),
).reset_index()
entities = entities.sort_values(group_col, kind="mergesort").reset_index(drop=True)

out_path = mlca_runtime.output_dir() / "entities.parquet"
entities.to_parquet(out_path, index=False)

src = "entities.parquet"
mlca_runtime.record_fact_candidate("entity_count", int(len(entities)), source=src)
mlca_runtime.record_fact_candidate("dropped_missing_entity_rows", int(dropped), source=src)
mlca_runtime.record_fact_candidate("return_rows", int(return_n), source=src)
mlca_runtime.record_fact_candidate("repeated_entities", int((sizes > 1).sum()), source=src)

mlca_runtime.save_json(
    {
        "group_column": group_col,
        "time_column": date_col,
        "reference_time": str(global_end),
        "rows_in": int(raw_n),
        "rows_out": int(len(entities)),
        "dropped_missing_entity_rows": int(dropped),
        "return_rows": int(return_n),
        "columns": list(entities.columns),
    },
    "entities_summary.json",
    kind="report",
)

print(f"rows={raw_n}, entities={len(entities)}, dropped_missing_entity={dropped}, repeated_entities={int((sizes > 1).sum())}")
print("Return rule: negative-quantity or negative-amount rows retained with signed monetary values.")
print(f"Saved entities.parquet rows={len(entities)} cols={entities.shape[1]}")
