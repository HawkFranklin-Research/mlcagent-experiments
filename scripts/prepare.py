"""Deterministic cohort assembly; no imputation, scaling, filtering by data or AI."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(study, config_path, destination):
    config = yaml.safe_load(Path(config_path).read_text())
    source = Path(config["SOURCE_DATA"]).expanduser().resolve()
    out = Path(destination).resolve() / study
    out.mkdir(parents=True, exist_ok=True)
    inputs, outputs = [], []
    discarded = {}

    def read(relative, **kw):
        p = source / relative
        inputs.append({"file": relative, "sha256": digest(p)})
        return pd.read_csv(p, **kw)

    def save(name, frame):
        path = out / name
        frame.to_csv(path, index=False)
        outputs.append(
            {
                "file": name,
                "sha256": digest(path),
                "rows": len(frame),
                "columns": len(frame.columns),
                "labels": {
                    str(k): int(v) for k, v in frame["label"].value_counts().items()
                }
                if "label" in frame
                else None,
            }
        )

    if study == "S1":
        relative = "final results/hfr/Anemia/anemia.csv"
        p = source / relative
        inputs.append({"file": relative, "sha256": digest(p)})
        f = pd.read_csv(p)
        outputs.append(
            {
                "file": "source-only",
                "sha256": digest(p),
                "rows": len(f),
                "columns": len(f.columns),
                "labels": f.Result.value_counts().to_dict(),
            }
        )
    elif study == "S2":
        clinical = read(
            "brca_cptac/HS_CPTAC_BRCA_2018_CLI.tsi", sep="\t", index_col=0
        ).iloc[1:]
        columns = {
            "ER": ["ER.Updated.Clinical.Status", "ER"],
            "PR": ["PR"],
            "HER2": [
                "ERBB2.Proteogenomic.Status",
                "Her2.Updated.Clinical.Status",
                "HER2.Amplified",
            ],
        }
        filenames = {
            "rna": "HS_CPTAC_BRCA_2018_RNA_GENE.cct",
            "proteome": "HS_CPTAC_BRCA_2018_Proteome_Ratio_Norm_gene_Median.cct",
        }
        expected = {"ER": {1: 81, 0: 39}, "PR": {1: 68, 0: 47}, "HER2": {1: 15, 0: 107}}
        for modality, name in filenames.items():
            matrix = read("brca_cptac/" + name, sep="\t", index_col=0).T
            if matrix.columns.duplicated().any():
                raise ValueError(
                    "Duplicate feature symbols require a recorded preparation rule"
                )
            for receptor, choices in columns.items():
                column = next(c for c in choices if c in clinical)
                labels = (
                    pd.to_numeric(clinical[column], errors="coerce")
                    .dropna()
                    .astype(int)
                )
                if (
                    not set(labels.unique()) <= {0, 1}
                    or labels.value_counts().to_dict() != expected[receptor]
                ):
                    raise ValueError("Unexpected receptor labels/counts: " + receptor)
                labels = labels.loc[labels.index.intersection(matrix.index, sort=False)]
                frame = matrix.loc[labels.index].copy()
                frame.insert(0, "sample_id", labels.index)
                frame["label"] = labels.values
                save(modality + "_" + receptor.lower() + ".csv", frame)
    elif study == "S3":
        raw = read("brca_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem.txt", sep="\t")
        raw = (
            raw.drop_duplicates("Hugo_Symbol", keep="first")
            .set_index("Hugo_Symbol")
            .drop(columns=["Entrez_Gene_Id"], errors="ignore")
        )
        tcga = raw[[c for c in raw if str(c)[13:15] == "01"]].T
        tcga.index = tcga.index.str[:12]
        discarded["duplicate_primary_patient_samples"] = int(
            tcga.index.duplicated().sum()
        )
        tcga = tcga.loc[~tcga.index.duplicated(keep="first")]
        clinical = read(
            "brca_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
            sep="\t",
            comment="#",
        ).set_index("PATIENT_ID")
        labels = clinical["SUBTYPE"].dropna().str.replace("BRCA_", "", regex=False)
        ids = tcga.index.intersection(labels.index, sort=False)
        tcga = np.log2(tcga.loc[ids].apply(pd.to_numeric, errors="raise") + 1)
        other = read(
            "brca_cptac/HS_CPTAC_BRCA_2018_RNA_GENE.cct", sep="\t", index_col=0
        )
        other = other.loc[~other.index.duplicated(keep="first")].T
        oc = read("brca_cptac/HS_CPTAC_BRCA_2018_CLI.tsi", sep="\t", index_col=0).iloc[
            1:
        ]
        genes = tcga.columns.intersection(other.columns, sort=False)
        tcga = tcga[genes].copy()
        tcga.insert(0, "sample_id", tcga.index)
        tcga["PAM50"] = labels.loc[tcga.index].values
        ids = other.index.intersection(oc.index[oc.PAM50.notna()], sort=False)
        other = other.loc[ids, genes].copy()
        other.insert(0, "sample_id", other.index)
        other["PAM50"] = oc.loc[ids, "PAM50"].replace({"Normal-like": "Normal"}).values
        save("tcga_pam50.csv", tcga)
        save("cptac_pam50.csv", other)
    elif study == "S4":
        base = "hnscc_mlca/data/Case-Study-HNSCC-ML-Copilot-Agent/processed_data/"
        genes = None
        for cohort, expected in [("tcga", 415), ("cptac", 102)]:
            expression = read(base + cohort + ".survival.fcptac.csv")
            survival = read(base + "survival_" + cohort + ".csv")
            if genes is not None and list(expression.columns[1:]) != genes:
                raise ValueError("Cohort gene columns differ")
            genes = list(expression.columns[1:])
            frame = expression.merge(
                survival, on="PatientID", how="inner", validate="one_to_one", sort=False
            )
            if len(frame) != expected:
                raise ValueError("Unexpected joined cohort count: " + cohort)
            save(cohort + "_hnscc.csv", frame)
    else:
        raise ValueError("Unknown study")
    manifest = {
        "study": study,
        "discarded": discarded,
        "versions": {"pandas": pd.__version__, "numpy": np.__version__},
        "inputs": inputs,
        "outputs": outputs,
        "method": "Fixed assembly only; no fitted preprocessing. Duplicate TCGA gene symbols and primary-tumour patient samples retain first occurrence; no outcome-based selection.",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    return manifest
