import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("group_column", "targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available top-level keys: {sorted(protocol)}")

group_col = protocol["group_column"]
task = protocol["task"]
time_cols = list(task.get("time_columns", []))

df = pd.read_parquet(inputs["online_retail_uci"])
if group_col not in df.columns:
    raise ValueError(f"group_column {group_col!r} not in data columns: {list(df.columns)}")

if "Quantity" not in df.columns or "UnitPrice" not in df.columns:
    raise ValueError("Need Quantity and UnitPrice for monetary summaries")

df["__amount"] = df["Quantity"].fillna(0).astype(float) * df["UnitPrice"].fillna(0).astype(float)
neg_rows = int((df["Quantity"] < 0).sum())

date_col = "InvoiceDate" if "InvoiceDate" in df.columns else next(
    (c for c in df.columns if "date" in c.lower() or "time" in c.lower()), None
)
if date_col is None:
    raise ValueError(f"No date column found for recency; columns: {list(df.columns)}")
df["__date"] = pd.to_datetime(df[date_col], errors="coerce")
max_date = df["__date"].max()

invoice_col = "InvoiceNo" if "InvoiceNo" in df.columns else None
df2 = df[df[group_col].notna()].copy()
dropped = len(df) - len(df2)

work = df2[[group_col, "__amount", "__date"]].copy()
agg_spec = {
    "monetary_total": ("__amount", "sum"),
    "monetary_mean": ("__amount", "mean"),
    "line_count": ("__amount", "size"),
    "last_date": ("__date", "max"),
}
if invoice_col is not None:
    work["__invoice"] = df2[invoice_col]
    agg_spec["frequency"] = ("__invoice", "nunique")
else:
    agg_spec["frequency"] = ("__amount", "size")

entities = work.groupby(group_col, sort=True).agg(**agg_spec).reset_index()
entities["recency_days"] = (max_date - entities["last_date"]).dt.days
entities = entities.drop(columns=["last_date"])

if time_cols:
    missing = [c for c in time_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Declared time_columns missing from data: {missing}")
    for c in time_cols:
        df2[c] = pd.to_datetime(df2[c], errors="coerce")
    time_entities = df2.groupby(group_col, sort=True)[time_cols].max().reset_index()
    entities = entities.merge(time_entities, on=group_col, how="left", validate="one_to_one")

out_path = mlca_runtime.output_dir() / "entities.parquet"
entities.to_parquet(out_path, index=False)

mlca_runtime.record_metric("input_rows", float(len(df)), "train")
mlca_runtime.record_metric("entity_rows", float(len(entities)), "train")
mlca_runtime.record_metric("dropped_missing_entity_rows", float(dropped), "train")
mlca_runtime.record_metric("negative_quantity_rows", float(neg_rows), "train")

print(f"input_rows={len(df)} entities={len(entities)} dropped_missing_key={dropped}")
print(f"returns: {neg_rows} negative-quantity rows kept as signed amounts; frequency uses distinct invoices when available")
print(f"wrote entities.parquet shape={entities.shape}")
