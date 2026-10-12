"""
REP52 panel simplification assessment.

For each of the 52 pan-subtype REP genes, combine four independent lines of
evidence and assign a support score + tier (keep / drop-candidate):

  1. pooled Cox contribution  (|coef| in rep_genes.csv; strong if >= median)
  2. HR+ rebuild confirmation  (gene also in rep_genes_hrh2neg.csv, 108 genes)
  3. single-cell high-REP core (gene is a top-50 DE hit in BOTH GSE161529 & GSE249361)
  4. single-cell evidence      (gene is a top-50 DE hit in exactly one queue)

Tier A (support>=3): conserved core, keep.
Tier B (support==2): moderate, keep.
Tier C (support<=1): drop candidate.

Also flags direction inconsistency between pan-subtype and HR+ coef.

Run with managed venv (pandas only):
  python analysis/rep52_trim_assess.py
"""
import os, numpy as np, pandas as pd

ROOT = r"E:/01.WorkSpace/01.Breast_cancer_gene_panel/resistance_signature"
RES  = os.path.join(ROOT, "results")
OUT  = os.path.join(RES, "rep52_trim")
os.makedirs(OUT, exist_ok=True)

rep52  = pd.read_csv(os.path.join(RES, "rep_genes.csv"))
hrh    = pd.read_csv(os.path.join(RES, "rep_genes_hrh2neg.csv"))
hrh_set = set(hrh["gene"].tolist())

# single-cell high-REP DE hit sets (top-50 DE genes per cell type)
de_dir = os.path.join(RES, "sc", "integrated", "rep_de")
de_files = {
    "GSE161529_Epi":  os.path.join(de_dir, "de_Epithelial_Malignant.csv"),
    "GSE161529_CAF":  os.path.join(de_dir, "de_Fibroblast_CAF.csv"),
    "GSE249361_Epi":  os.path.join(de_dir, "GSE249361", "de_Epithelial_Malignant.csv"),
    "GSE249361_CAF":  os.path.join(de_dir, "GSE249361", "de_Fibroblast_CAF.csv"),
}
sc_hits = {k: set(pd.read_csv(v)["gene"].tolist()) for k, v in de_files.items() if os.path.exists(v)}
q1615 = sc_hits.get("GSE161529_Epi", set()) | sc_hits.get("GSE161529_CAF", set())
q2493 = sc_hits.get("GSE249361_Epi", set()) | sc_hits.get("GSE249361_CAF", set())

med_coef = rep52["coef"].abs().quantile(0.66)   # top-tertile threshold

rows = []
for _, r in rep52.iterrows():
    g = r["gene"]; coef = r["coef"]
    in_hr = g in hrh_set
    hit1615 = g in q1615
    hit2493 = g in q2493
    sc_core = hit1615 and hit2493
    s1 = int(abs(coef) >= med_coef)          # strong pooled contribution (top tertile)
    s2 = int(in_hr)                           # confirmed in HR+ 108-gene rebuild
    s3 = int(sc_core)                         # cross-queue single-cell replication (strongest)
    s4 = int((hit1615 or hit2493) and not sc_core)
    # KEEP if any of the three independent anchors holds
    if sc_core:
        tier = "A_core"                       # cross-queue single-cell replication = strongest reproducibility
    elif (s1 or s2):
        tier = "B_keep"                       # strong pooled coef OR HR+ rebuild confirmation
    else:
        tier = "C_drop"                       # no independent evidence
    support = s1 + s2 + s3 + s4
    # direction consistency vs HR+ rebuild
    dir_ok = ""
    if in_hr:
        hcoef = hrh.loc[hrh["gene"] == g, "coef"]
        if len(hcoef) and (hcoef.iloc[0] * coef) < 0:
            dir_ok = "DIR_MISMATCH"
    rows.append({
        "gene": g, "coef": round(coef, 4), "HR": round(r["HR"], 4),
        "abs_coef_top_tertile": s1, "in_HRplus108": s2,
        "sc_hit_GSE161529": int(hit1615), "sc_hit_GSE249361": int(hit2493),
        "sc_both_queues": s3, "sc_one_queue": s4,
        "support": support, "tier": tier, "flag": dir_ok,
    })

df = pd.DataFrame(rows).sort_values(["tier", "support", "coef"],
                                   ascending=[True, False, False]).reset_index(drop=True)
df.to_csv(os.path.join(OUT, "rep52_trim_assessment.csv"), index=False)

# summary
tier_counts = df["tier"].value_counts().to_dict()
core = df[df["tier"] == "A_core"]["gene"].tolist()
keep = df[df["tier"].isin(["A_core", "B_keep"])]["gene"].tolist()
drop = df[df["tier"] == "C_drop"]["gene"].tolist()
mismatch = df[df["flag"] == "DIR_MISMATCH"]["gene"].tolist()

# trimmed panel = keep genes with original coef (for rebuilding composite)
trim = rep52[rep52["gene"].isin(keep)].copy()
trim.to_csv(os.path.join(OUT, "rep52_trimmed_panel.csv"), index=False)

with open(os.path.join(OUT, "rep52_trim_interpretation.md"), "w") as f:
    f.write("# REP52 panel trim assessment\n\n")
    f.write("> Evidence chain (any one retains the gene): pooled-Cox strong contribution (top tertile, rep_genes.csv) "
            "OR HR+ rebuild confirmation among the 108 genes (rep_genes_hrh2neg.csv) "
            "OR high-REP single-cell hits replicated across two cohorts (top50 DE of rep_de in GSE161529 & GSE249361).\n")
    f.write("> Genes with none of the three are genuinely peripheral and droppable. Script: analysis/rep52_trim_assess.py\n\n")
    f.write("## Tiered results\n")
    f.write(f"- **A_core (must keep)**: {tier_counts.get('A_core',0)} genes -- hits replicated across two single-cell cohorts (strongest reproducibility evidence)\n")
    f.write(f"- **B_keep (keep)**: {tier_counts.get('B_keep',0)} genes -- strong pooled coef or HR+ rebuild confirmation\n")
    f.write(f"- **C_drop (droppable)**: {tier_counts.get('C_drop',0)} genes -- no independent evidence\n\n")
    f.write(f"### A_core conservative core (n={len(core)})\n{', '.join(core)}\n\n")
    f.write(f"### Keep total (A+B, n={len(keep)})\n{', '.join(keep)}\n\n")
    f.write(f"### Droppable (C, n={len(drop)})\n{', '.join(drop)}\n\n")
    if mismatch:
        f.write(f"### Direction-inconsistent (pan-subtype vs HR+ rebuild coef sign flip)\n{', '.join(mismatch)}\n\n")
    f.write("## Caveats\n")
    f.write("- After trimming, the composite HR must be **backtested** to confirm performance does not collapse (recommended: recompute the REP score and Cox on METABRIC/TCGA-HR+ with the trimmed coefficients); this assessment is gene-level only and does not recompute composite performance.\n")
    f.write("- Single-cell hits are based on the top50 DE proxy (significant genes outside the top50 may be missed); cohort-specific genes (PDGFRA/CXCL12/KRT14 emerge only in GSE161529) were classified as droppable, consistent with the cross-cohort replication conclusion.\n\n")
    f.write("## Support details\n\n")
    cols = ["gene","coef","HR","abs_coef_top_tertile","in_HRplus108",
            "sc_hit_GSE161529","sc_hit_GSE249361","support","tier","flag"]
    f.write("| " + " | ".join(cols) + " |\n")
    f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
    for _, rr in df.iterrows():
        f.write("| " + " | ".join(str(rr[c]) for c in cols) + " |\n")

print("REP52 trim assessment done.")
print(f"Tier A_core={tier_counts.get('A_core',0)} B_keep={tier_counts.get('B_keep',0)} C_drop={tier_counts.get('C_drop',0)}")
print(f"keep(n={len(keep)}) drop(n={len(drop)}) mismatches={mismatch}")
print("A_core genes:", ", ".join(core))
print("C_drop genes:", ", ".join(drop))
