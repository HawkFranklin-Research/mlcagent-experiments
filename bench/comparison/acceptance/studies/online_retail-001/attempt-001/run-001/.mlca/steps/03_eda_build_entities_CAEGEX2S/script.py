import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
out_dir = mlca_runtime.output_dir()
proto = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "group_column", "task"):
    if key not in proto:
        raise KeyError(f"missing _protocol key {key!r}; top-level keys={list(proto)}")

group_col = proto["group_column"]
targets = proto["targets"]
excluded = proto["excluded_columns"]
time_cols = proto["task"].get("time_columns", [])

df = pd.read_parquet(inputs["online_retail_uci"])
text_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
found = {}
for c in text_cols:
    s = df[c].astype("string").str.strip()
    mask = s.isin(["?", "-", "NA", "N/A"]) | (s == "")
    if mask.any():
        found[c] = int(mask.sum())
        df[c] = s.mask(mask, pd.NA)
    else:
        df[c] = s
if found:
    print(f"placeholders found in {len(found)} text columns: {sorted(found)[:5]}")

n0 = len(df)
df = df[df[group_col].notna()].copy()

if "Quantity" not in df or "UnitPrice" not in df:
    raise ValueError("Quantity and UnitPrice are required for monetary aggregation")
df["_amount"] = pd.to_numeric(df["Quantity"], errors="coerce") * pd.to_numeric(df["UnitPrice"], errors="coerce")
is_return = pd.to_numeric(df["Quantity"], errors="coerce") < 0
print("Return rule: rows with Quantity<0 are kept and netted into monetary totals; frequency counts non-return invoices only.")

if "InvoiceDate" not in df:
    raise ValueError("InvoiceDate is required to compute recency")
df["_dt"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
last_date = df["_dt"].max()
if pd.isna(last_date):
    raise ValueError("InvoiceDate has no valid values; cannot compute recency")

gb = df.groupby(group_col, dropna=True, observed=True)
agg = gb.agg(
    monetary_total=("_amount", "sum"),
    monetary_mean=("_amount", "mean"),
    quantity_total=("Quantity", "sum"),
    quantity_mean=("Quantity", "mean"),
    n_lines=("_amount", "size"),
    last_purchase=("_dt", "max"),
)
agg["recency_days"] = (last_date - agg["last_purchase"]).dt.days

if "InvoiceNo" in df:
    freq = df.loc[~is_return].groupby(group_col, dropna=True, observed=True)["InvoiceNo"].nunique()
    agg["frequency"] = freq.reindex(agg.index).fillna(0).astype(int)
else:
    agg["frequency"] = agg["n_lines"]

if "StockCode" in df:
    agg["n_products"] = gb["StockCode"].nunique()
if "Country" in df:
    agg["n_countries"] = gb["Country"].nunique()

for t in time_cols:
    if t not in df:
        raise ValueError(f"declared time column missing: {t}")
    dt = pd.to_datetime(df[t], errors="coerce")
    agg[t] = dt.groupby(df[group_col], observed=True).max().reindex(agg.index)

entities = agg.reset_index()
entities = entities[[group_col] + [c for c in entities.columns if c != group_col]]
entities.to_parquet(out_dir / "entities.parquet", index=False)

summary = {
    "group_column": group_col,
    "n_input_rows": int(n0),
    "n_rows_used": int(len(df)),
    "n_entities": int(len(entities)),
    "time_columns_preserved": time_cols,
    "return_rule": "Quantity<0 kept and netted in monetary totals; frequency counts non-return invoices only",
    "output": "entities.parquet",
}
mlca_runtime.save_json(summary, "entities_summary.json")
mlca_runtime.record_metric("n_entities", int(len(entities)), split="train")
mlca_runtime.record_metric("n_transactions_used", int(len(df)), split="train")
print(f"rows_in={n0} rows_used={len(df)} entities={len(entities)} saved entities.parquet")
