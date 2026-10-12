"""Post-process REP52_percohort_performance.csv: add multivariate-vs-univariate
direction-consistency flags and a coverage note, then rebuild the Excel bundle."""
import os

import numpy as np
import pandas as pd

ROOT = r"E:/01.WorkSpace/01.Breast_cancer_gene_panel/resistance_signature"
OUT = os.path.join(ROOT, "results", "rep52_trim")

d = pd.read_csv(os.path.join(OUT, "REP52_percohort_performance.csv"))

d["sign_vs_METABRIC_univariate"] = [
    "concordant" if (c > 0) == (h > 1) else "FLIP"
    for c, h in zip(d["panel_coef"], d["METABRIC_HR_perSD"])
]
d["sign_vs_TCGA_univariate"] = [
    "concordant" if (c > 0) == (h > 1) else "FLIP"
    for c, h in zip(d["panel_coef"], d["TCGA_HR_perSD"])
]
d["sign_vs_POOLED_univariate"] = [
    "concordant" if (c > 0) == (h > 1) else "FLIP"
    for c, h in zip(d["panel_coef"], d["POOLED_HR_perSD"])
]

# order columns for readability
lead = ["rank", "gene", "panel_coef", "panel_HR"]
blocks = []
for c in ["METABRIC", "TCGA", "POOLED"]:
    blocks += ["%s_HR_perSD" % c, "%s_CI95" % c, "%s_p" % c, "%s_q" % c,
               "%s_multivar_coef" % c]
tail = ["direction_concordant", "sign_vs_METABRIC_univariate", "sign_vs_TCGA_univariate",
        "sign_vs_POOLED_univariate", "top_tertile_contribution",
        "confirmed_in_HRplus_rebuild", "sc_hit_GSE161529", "sc_hit_GSE249361",
        "support_lines", "tier", "in_REP30"]
d = d[lead + blocks + tail]
d.to_csv(os.path.join(OUT, "REP52_percohort_performance.csv"), index=False)

s7 = pd.read_csv(os.path.join(OUT, "REP30_panel_table.csv"))
with pd.ExcelWriter(os.path.join(OUT, "REP52_and_REP30_tables.xlsx"), engine="openpyxl") as w:
    d.to_excel(w, sheet_name="REP52_percohort", index=False)
    s7.to_excel(w, sheet_name="REP30_panel", index=False)

print("=== sign concordance of the panel coefficient vs univariate direction ===")
for c in ["METABRIC", "TCGA", "POOLED"]:
    vc = d["sign_vs_%s_univariate" % c].value_counts().to_dict()
    print("  %-9s concordant=%d  FLIP=%d" % (c, vc.get("concordant", 0), vc.get("FLIP", 0)))

print("\n=== genes whose multivariate coefficient FLIPS vs METABRIC univariate ===")
f = d[d["sign_vs_METABRIC_univariate"] == "FLIP"]
print(f[["rank", "gene", "panel_coef", "METABRIC_HR_perSD", "TCGA_HR_perSD",
         "tier", "in_REP30"]].to_string(index=False))

print("\n=== retained (REP30, n=%d) vs dropped (n=%d) ===" %
      ((d["in_REP30"] == "yes").sum(), (d["in_REP30"] == "no").sum()))
print("  A_core :", ", ".join(d[d["tier"] == "A_core"]["gene"]))
print("  B_keep :", ", ".join(d[d["tier"] == "B_keep"]["gene"]))
print("  C_drop :", ", ".join(d[d["tier"] == "C_drop"]["gene"]))
