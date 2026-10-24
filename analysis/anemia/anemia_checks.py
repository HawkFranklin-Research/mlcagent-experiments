"""S1 anemia table checks for R2-5 and R1-7d (no model calls; reads the prepared table only).

1. Duplicates: rows that are exact copies, and how often a random 80/20 split of all 1,421 rows
   (as in the submitted analysis) puts a copy of a test row into training.
2. Label vs haemoglobin: how well a single haemoglobin cut-off per sex reproduces the label.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data_prep/hf-111de16/prepared/S1/anemia.csv"


def main():
    d = pd.read_csv(DATA)
    out = {"rows": len(d), "unique_rows": int(len(d.drop_duplicates())),
           "label_counts": d["Result"].value_counts().to_dict(),
           "label_counts_unique_rows": d.drop_duplicates()["Result"].value_counts().to_dict()}
    rng = np.random.default_rng(0)
    shares = []
    keys = d.astype(str).agg("|".join, axis=1).to_numpy()
    for _ in range(200):
        order = rng.permutation(len(d))
        test, train = order[: int(0.2 * len(d))], order[int(0.2 * len(d)):]
        train_keys = set(keys[train])
        shares.append(np.mean([k in train_keys for k in keys[test]]))
    out["random_80_20_split_of_all_rows"] = {
        "test_rows_with_exact_copy_in_train_mean": round(float(np.mean(shares)), 3),
        "min": round(float(np.min(shares)), 3), "max": round(float(np.max(shares)), 3), "repeats": 200}
    u = d.drop_duplicates()
    rules = {}
    for g in sorted(u["Gender"].unique()):
        s = u[u["Gender"] == g]
        best = max(((t, float(((s["Hemoglobin"] < t).astype(int) == s["Result"]).mean()))
                    for t in np.arange(8.0, 16.0, 0.05)), key=lambda x: x[1])
        rules[f"gender_{g}"] = {"rows": int(len(s)), "best_cutoff_g_dl": round(float(best[0]), 2),
                                "accuracy_of_Hb_below_cutoff": round(best[1], 4)}
        # overlap check: anemic max Hb vs non-anemic min Hb
        rules[f"gender_{g}"]["max_Hb_anemic"] = float(s.loc[s["Result"] == 1, "Hemoglobin"].max())
        rules[f"gender_{g}"]["min_Hb_not_anemic"] = float(s.loc[s["Result"] == 0, "Hemoglobin"].min())
    out["label_vs_haemoglobin_unique_rows"] = rules
    # The separating value per sex: anemic exactly when Hb is below the lowest non-anemic Hb.
    pred = np.zeros(len(u), int)
    for g, r in rules.items():
        m = (u["Gender"] == int(g.split("_")[1])).to_numpy()
        r["rule"] = f"Result = 1 exactly when Hemoglobin < {r['min_Hb_not_anemic']}"
        pred[m] = (u.loc[m, "Hemoglobin"] < r["min_Hb_not_anemic"]).astype(int)
    out["sex_specific_Hb_rule_accuracy_unique_rows"] = round(float((pred == u["Result"].to_numpy()).mean()), 4)
    full = np.zeros(len(d), int)
    for g, r in rules.items():
        m = (d["Gender"] == int(g.split("_")[1])).to_numpy()
        full[m] = (d.loc[m, "Hemoglobin"] < r["min_Hb_not_anemic"]).astype(int)
    out["sex_specific_Hb_rule_accuracy_all_rows"] = round(float((full == d["Result"].to_numpy()).mean()), 4)
    (HERE / "anemia_checks.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
