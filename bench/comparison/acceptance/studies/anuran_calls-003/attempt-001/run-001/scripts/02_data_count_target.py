import json
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

try:
    targets = [str(t) for t in protocol["targets"]]
except KeyError:
    raise KeyError(f"Missing 'targets'. Available top-level keys: {sorted(protocol.keys())}") from None
if not targets:
    raise ValueError("_protocol['targets'] is empty.")

df = pd.read_parquet(paths["anuran_calls_uci"])
missing = [t for t in targets if t not in df.columns]
if missing:
    raise KeyError(f"Target columns missing from data: {missing}")

n_rows = len(df)
rows = []
for t in targets:
    vc = df[t].dropna().astype(str).value_counts()
    for cls, c in vc.items():
        rows.append({
            "class": f"{t}::{cls}",
            "n": int(c),
            "share": round(float(c) / n_rows, 6) if n_rows else float("nan"),
        })

full = pd.DataFrame(rows, columns=["class", "n", "share"])
flagged = int((full["share"] < 0.05).sum())
out = full if len(full) <= 50 else full.sort_values("share").head(50).reset_index(drop=True)

mlca_runtime.save_table(
    out, "target_counts.csv",
    caption="Per-target class counts and shares; classes under 5% flagged in share column.",
)

print(f"rows={n_rows} targets={len(targets)} classes_total={len(full)}")
print(f"flagged_under_5pct={flagged} saved_rows={len(out)}")
print("saved target_counts.csv")
