import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
df = pd.read_parquet(inputs["online_retail_uci"])

group = proto.get("group_column")
if group is None:
    raise KeyError(f"missing group_column; top-level keys: {list(proto.keys())}")

df = df.dropna(subset=[group]).copy()
df[group] = df[group].astype(str).str.strip()
df = df[df[group] != ""]

neg = int((df["Quantity"] < 0).sum())
print(f"reversing rows (Quantity<0): {neg}; kept as negative amounts")

df["Amount"] = df["Quantity"] * df["UnitPrice"]
last_date = df["InvoiceDate"].max()

agg = (
    df.groupby(group)
    .agg(
        recency=("InvoiceDate", lambda s: (last_date - s.max()).days),
        frequency=("InvoiceNo", "nunique"),
        monetary_total=("Amount", "sum"),
        monetary_mean=("Amount", "mean"),
        n_lines=("Amount", "size"),
    )
    .reset_index()
    .rename(columns={group: "entity_id"})
)

out = mlca_runtime.output_dir()
agg.to_parquet(out / "entities.parquet", index=False)

mlca_runtime.save_json(
    {"n_entities": int(len(agg)), "n_rows_in": int(len(df))},
    "entities_summary.json",
)
mlca_runtime.record_metric("n_entities", float(len(agg)), "train")
print(f"entities: {len(agg)}")
