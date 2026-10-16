import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["online_retail_uci"])

cols = {c.lower(): c for c in df.columns}

def find(*names):
    for n in names:
        if n.lower() in cols:
            return cols[n.lower()]
    return None

group = proto.get("group_column") or find("CustomerID", "customer_id", "entity_id")
if group is None:
    raise KeyError(f"no entity column; columns: {list(df.columns)}")

qty = find("Quantity", "quantity")
price = find("UnitPrice", "unit_price", "price")
inv = find("InvoiceNo", "invoice_no", "invoice")
date = find("InvoiceDate", "invoice_date", "date")

df = df.dropna(subset=[group]).copy()
df[group] = df[group].astype(str).str.strip()
df = df[df[group] != ""]

if qty is not None and price is not None:
    df["Amount"] = df[qty] * df[price]
else:
    df["Amount"] = 1.0

if qty is not None:
    neg = int((df[qty] < 0).sum())
    print(f"reversing rows (negative qty): {neg}")

if date is not None:
    df[date] = pd.to_datetime(df[date], errors="coerce")
    last_date = df[date].max()
    recency = df.groupby(group)[date].max().apply(lambda x: (last_date - x).days)
else:
    recency = pd.Series(0, index=df.groupby(group).size().index)

if inv is not None:
    freq = df.groupby(group)[inv].nunique()
else:
    freq = df.groupby(group).size()

mon_total = df.groupby(group)["Amount"].sum()
mon_mean = df.groupby(group)["Amount"].mean()
n_lines = df.groupby(group).size()

agg = pd.DataFrame({
    "entity_id": recency.index,
    "recency": recency.values,
    "frequency": freq.values,
    "monetary_total": mon_total.values,
    "monetary_mean": mon_mean.values,
    "n_lines": n_lines.values,
})

out = mlca_runtime.output_dir()
agg.to_parquet(out / "entities.parquet", index=False)
mlca_runtime.save_json(
    {"n_entities": int(len(agg)), "n_rows_in": int(len(df))},
    "entities_summary.json",
)
mlca_runtime.record_metric("n_entities", float(len(agg)), "train")
print(f"entities: {len(agg)}")
