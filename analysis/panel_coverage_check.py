# -*- coding: utf-8 -*-
"""Check gene coverage across MSK-IMPACT panels (341 / 410 / 468)."""
import io
import os

import numpy as np
import pandas as pd

CB = r"E:\01.WorkSpace\01.Breast_cancer_gene_panel\resistance_signature\data\06_metastatic\cbioportal"
D = os.path.join(CB, "breast_ink4_msk_2021")
OUT = []


def log(*a):
    OUT.append(" ".join(str(x) for x in a))


gp = pd.read_csv(os.path.join(D, "data_gene_panel_matrix.txt"), sep="\t",
                 comment="#", low_memory=False)
log("gene_panel_matrix shape:", gp.shape)
log("columns:", list(gp.columns)[:8])
log("head:")
log(gp.head(3).to_string())
log("")
log("panel value counts:")
for c in gp.columns[1:]:
    vc = gp[c].value_counts(dropna=False)
    log(f"  {c}: {vc.to_dict()}")

sp = pd.read_csv(os.path.join(D, "data_clinical_sample.txt"), sep="\t",
                 comment="#", low_memory=False)
log("")
log("GENE_PANEL in sample table:", sp["GENE_PANEL"].value_counts().to_dict())

cna = pd.read_csv(os.path.join(D, "data_cna.txt"), sep="\t", comment="#", low_memory=False)
cna = cna.set_index("Hugo_Symbol")
log("")
log("CNA matrix:", cna.shape)
log("total NaN cells:", int(cna.isna().sum().sum()),
    f"({100*cna.isna().sum().sum()/cna.size:.2f}%)")
log("")
log("per-gene NaN (not-covered) count for key genes:")
for g in ["ESR1", "RB1", "CCND1", "CDK4", "CDK6", "CCNE1", "FGFR1", "ERBB2",
          "MDM2", "PTEN", "NF1", "CDKN2A", "CDKN2B", "MYC", "ARID1A", "ATM",
          "BRCA2", "IGF1R", "AKT2", "MAP2K4"]:
    if g in cna.index:
        row = pd.to_numeric(cna.loc[g], errors="coerce")
        n_nan = int(row.isna().sum())
        # observed frequency among COVERED samples only
        cov = row.dropna()
        freq_all = float((row.fillna(0) > 0).mean()) if g not in ("RB1", "PTEN", "CDKN2A") else float((row.fillna(0) < 0).mean())
        direction = "del" if g in ("RB1", "PTEN", "CDKN2A", "CDKN2B", "NF1", "ARID1A", "ATM", "BRCA2", "MAP2K4") else "amp"
        f_cov = float(((cov < 0) if direction == "del" else (cov > 0)).mean()) if len(cov) else np.nan
        log(f"  {g:<8} covered={1365-n_nan:>4}/1365  not_covered={n_nan:>4}  "
            f"freq_all={freq_all*100:5.2f}%  freq_covered={f_cov*100:5.2f}%")
    else:
        log(f"  {g:<8} ABSENT from CNA matrix")

# mutation side: which genes appear only in the larger panels?
m = pd.read_csv(os.path.join(D, "data_mutations.txt"), sep="\t", comment="#", low_memory=False)
NONCODING = {"Silent", "Intron", "IGR", "RNA", "3'UTR", "5'UTR", "3'Flank", "5'Flank"}
dmg = m[~m["Variant_Classification"].isin(NONCODING)]
log("")
log("mutation coverage for key genes across panels:")
sp2 = sp.set_index("SAMPLE_ID")
for g in ["ESR1", "RB1", "CDKN2A", "TP53", "PIK3CA", "NF1", "PTEN", "AKT1", "ERBB2"]:
    s = dmg[dmg["Hugo_Symbol"] == g]["Tumor_Sample_Barcode"].unique()
    sub = sp2.reindex(s)
    log(f"  {g:<8} n_samples={len(s):>4}  panels={sub['GENE_PANEL'].value_counts().to_dict()}")

with io.open(r"C:\Users\mao02\AppData\Local\Temp\panel_cov.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(OUT))
print("ok")
