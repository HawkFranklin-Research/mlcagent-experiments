import mlca_runtime
import json
import pandas as pd

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

if "targets" not in protocol:
    raise KeyError(
        "Missing _protocol['targets']. Available top-level keys: "
        + str(sorted(protocol.keys()))
    )

targets = protocol["targets"]
if not isinstance(targets, list) or len(targets) != 1:
    raise ValueError(f"Expected exactly one target in _protocol['targets'], got {targets!r}")
target = targets[0]

df = pd.read_parquet(inputs["proteome_her2"])
if target not in df.columns:
    raise KeyError(f"Target column {target!r} not found in proteome_her2 columns")

total = len(df)
vc = df[target].value_counts(dropna=False)
counts = pd.DataFrame({"class": vc.index.astype(str), "n": vc.to_numpy(dtype=int)})
counts["share"] = (counts["n"] / total).round(4) if total else float("nan")
counts = counts.sort_values("n", ascending=False).reset_index(drop=True)

mlca_runtime.save_table(counts, "target_counts.csv", caption="Target class counts and shares")

flagged = counts.loc[counts["share"] < 0.05, "class"].tolist() if total else []

print(f"rows={total}, target={target}")
print("counts=" + ", ".join(f"{r['class']}:{int(r['n'])}" for _, r in counts.iterrows()))
print("flagged_under_5pct=" + (",".join(flagged) if flagged else "none"))
print(f"classes_flagged={len(flagged)}")
print("saved target_counts.csv")
