"""Write results/rep52_trim/REP52_REP30_report.md from the computed tables."""
import os

import numpy as np
import pandas as pd

ROOT = r"E:/01.WorkSpace/01.Breast_cancer_gene_panel/resistance_signature"
OUT = os.path.join(ROOT, "results", "rep52_trim")
d = pd.read_csv(os.path.join(OUT, "REP52_percohort_performance.csv"))
s = pd.read_csv(os.path.join(OUT, "REP30_panel_table.csv"))

L = []
A = L.append

A("# REP52 panel: per-cohort performance, and the trimmed REP30 panel\n")
A("Source: `results/rep_genes.csv` (52-gene overall-survival panel) re-evaluated gene by gene "
  "in TCGA-BRCA and METABRIC. Script: `analysis/rep52_percohort_performance.py`; "
  "env = managed venv (lifelines 0.30.3).\n")

A("## Cohorts\n")
A("| cohort | n | events | expression |")
A("|---|---|---|---|")
A("| METABRIC | 1979 | 1143 | Illumina microarray |")
A("| TCGA-BRCA | 1069 | 151 | RNA-seq V2 RSEM |")
A("| POOLED | 3048 | 1294 | per-cohort z-scores, cohort as covariate |\n")
A("**The two cohorts are not equally powered at gene level.** METABRIC carries 1143 OS events, "
  "TCGA-BRCA only 151, so a gene-level effect that is overwhelming in METABRIC is usually "
  "unresolvable in TCGA-BRCA. This is why the panel was validated as a composite score, not gene by gene.\n")

A("## How the 52 genes behave in each cohort\n")
A("| | METABRIC | TCGA-BRCA | POOLED |")
A("|---|---|---|---|")
for lab, key in [("nominal p < 0.05", "p"), ("BH q < 0.05", "q")]:
    row = []
    for c in ["METABRIC", "TCGA", "POOLED"]:
        row.append("%d / 52" % int((d["%s_%s" % (c, key)] < 0.05).sum()))
    A("| %s | %s |" % (lab, " | ".join(row)))
A("")
A("Direction of the univariate effect agrees between METABRIC and TCGA-BRCA for "
  "**49 / 52** genes; the three discordant genes are DIO2, KRT14 and PDGFRA "
  "(all protective in METABRIC, mildly risk-increasing in TCGA-BRCA; KRT14 and PDGFRA were dropped).\n")

A("## Full 52-gene ranking with per-cohort performance\n")
cols = ["rank", "gene", "panel_coef", "panel_HR",
        "METABRIC_HR_perSD", "METABRIC_CI95", "METABRIC_q",
        "TCGA_HR_perSD", "TCGA_CI95", "TCGA_q",
        "POOLED_HR_perSD", "POOLED_q",
        "sign_vs_METABRIC_univariate", "tier", "in_REP30"]
hdr = ["rank", "gene", "Cox coef", "HR",
       "METABRIC HR/SD", "95% CI", "q",
       "TCGA HR/SD", "95% CI", "q",
       "POOLED HR/SD", "q",
       "coef vs univariate", "tier", "in REP30"]
A("| " + " | ".join(hdr) + " |")
A("|" + "|".join(["---"] * len(hdr)) + "|")
for _, r in d.iterrows():
    A("| " + " | ".join([
        str(int(r["rank"])), r["gene"], "%.4f" % r["panel_coef"], "%.4f" % r["panel_HR"],
        "%.3f" % r["METABRIC_HR_perSD"], r["METABRIC_CI95"], "%.2g" % r["METABRIC_q"],
        "%.3f" % r["TCGA_HR_perSD"], r["TCGA_CI95"], "%.2g" % r["TCGA_q"],
        "%.3f" % r["POOLED_HR_perSD"], "%.2g" % r["POOLED_q"],
        r["sign_vs_METABRIC_univariate"], r["tier"], r["in_REP30"],
    ]) + " |")
A("")
A("`panel_coef` / `panel_HR` is the weight the panel was built with (`rep_genes.csv`). "
  "`*_HR_perSD` is a fresh univariate Cox per gene, per cohort, on z-scored expression "
  "(HR per 1 SD). `*_multivar_coef` in the CSV is a penalised multivariate Cox over the 52 genes "
  "refitted within that cohort.\n")

A("## Coefficient sign versus univariate direction\n")
A("For **12 / 52** genes the multivariate coefficient points the opposite way from that gene's own "
  "univariate association in METABRIC. That is expected collinearity behaviour in a penalised "
  "multivariate Cox (these genes carry information conditional on the other 51, not marginally), "
  "but it means the weight must not be read as that gene's standalone prognostic direction. "
  "Three of the 12 are in REP30 (ANAPC16, PFDN5, RPL5 - all A_core, retained on single-cell "
  "replication rather than on their marginal effect) and one is TPX2.\n")
f = d[d["sign_vs_METABRIC_univariate"] == "FLIP"]
A("| gene | coef | METABRIC HR/SD | TCGA HR/SD | in REP30 |")
A("|---|---|---|---|---|")
for _, r in f.iterrows():
    A("| %s | %.4f | %.3f | %.3f | %s |" % (r["gene"], r["panel_coef"],
                                            r["METABRIC_HR_perSD"], r["TCGA_HR_perSD"], r["in_REP30"]))
A("")

A("## Trimming: 52 -> 30\n")
A("Three independent lines of evidence; a gene is retained if any one holds.\n")
A("| line | meaning |")
A("|---|---|")
A("| top-tertile contribution | |coef| in the top third of the 52 |")
A("| HR+ rebuild | also present in the HR-positive 108-gene reconstruction |")
A("| single-cell replication | top-50 DE hit in high-scoring cells in **both** GSE161529 and GSE249361 |\n")
A("| tier | n | genes |")
A("|---|---|---|")
for t in ["A_core", "B_keep", "C_drop"]:
    g = d[d["tier"] == t]["gene"].tolist()
    A("| %s | %d | %s |" % (t, len(g), ", ".join(g)))
A("")
A("**Retained (A+B) = 30; dropped (C) = 22.**\n")

A("## REP30 - the 30-gene panel\n")
A("Cox coefficient as used in the score, direction, per-cohort HR per SD, and coverage "
  "across the nine data sources.\n")
cov = ["TCGA", "METABRIC", "AURORA", "MBCproject", "GSE264205",
       "GSE249361", "GSE161529", "GSE45255", "GSE26459"]
hdr = ["gene", "Cox coef", "direction", "METABRIC HR/SD", "q", "TCGA HR/SD", "q"] + \
      [c + " cov" for c in cov] + ["n sources"]
A("| " + " | ".join(hdr) + " |")
A("|" + "|".join(["---"] * len(hdr)) + "|")
for _, r in s.iterrows():
    A("| " + " | ".join([
        r["gene"], "%.4f" % r["cox_coefficient"],
        "risk-inc" if r["cox_coefficient"] > 0 else "risk-dec",
        "%.3f" % r["METABRIC_HR_perSD"], "%.2g" % r["METABRIC_q"],
        "%.3f" % r["TCGA_HR_perSD"], "%.2g" % r["TCGA_q"],
    ] + [("Y" if r[c] == "present" else "-") for c in cov] +
        [str(int(r["n_cohorts_present"]))]) + " |")
A("")
A("Coverage: Y = gene measured in that source, - = absent. "
  "All 30 are present in TCGA-BRCA and METABRIC; the minimum coverage is 6 of 9 sources.\n")

# --- pooled-vs-METABRIC coefficient comparison
A("## Note on where the weights were estimated\n")
for a, b, la, lb in [("panel_coef", "POOLED_multivar_coef", "panel (as published)", "refit pooled"),
                     ("panel_coef", "TCGA_multivar_coef", "panel (as published)", "refit TCGA"),
                     ("panel_coef", "METABRIC_multivar_coef", "panel (as published)", "refit METABRIC")]:
    r = np.corrcoef(d[a], d[b])[0, 1]
    same = int(((d[a] > 0) == (d[b] > 0)).sum())
    A("- %s vs %s: Pearson r = %.3f, sign agreement %d/52" % (la, lb, r, same))
A("")
A("The published weight vector is almost identical to a **METABRIC-only** refit and clearly "
  "distinct from a TCGA-only refit, i.e. `rep_build.py` fitted the penalised multivariate Cox on "
  "METABRIC (train) and used TCGA-BRCA as the held-out test set. The manuscript text and the "
  "description of the construction both say the weights were estimated in the pooled cohorts. "
  "These two statements need reconciling before submission - either refit genuinely pooled, or "
  "change the wording to name METABRIC as the fitting cohort.\n")

A("## Discrepancy to fix in the manuscript\n")
A("Materials and Methods currently reads: *\"25 were retained on at least one line and 22 without "
  "support were dropped.\"* The retained set is A_core (5) + B_keep (25) = **30**, which is what "
  "REP30 is named for; 25 is the B_keep count alone. 25 + 22 = 47, not 52, so the sentence is "
  "internally inconsistent. It should read **30 retained, 22 dropped**.\n")

p = os.path.join(OUT, "REP52_REP30_report.md")
open(p, "w", encoding="utf-8").write("\n".join(L))
print("written:", p, len("\n".join(L)), "chars")
