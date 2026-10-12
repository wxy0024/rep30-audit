"""
Build ONE annotated master table for the REP panel construction step:

   52-gene overall-survival panel  ->  three lines of evidence  ->  30-gene REP30
                                     (22 genes dropped, with the exact criteria they failed)

Columns
-------
identity      : rank_by_abs_coef, gene
weights       : panel_coef, panel_HR                       (weights carried into the score)
METABRIC Cox  : HR per SD, 95% CI, p, q   (univariate, METABRIC z-scored)
TCGA-BRCA Cox : HR per SD, 95% CI, p, q   (univariate, TCGA z-scored)
POOLED Cox    : HR per SD, 95% CI, p, q   (per-cohort z, cohort as covariate)
criteria      : E1 top-tertile contribution / E2 HR+ rebuild confirmation /
                E3 single-cell cross-queue replication (both / one / none)
outcome       : n_criteria_passed, decision (Retained / Dropped), failure_reason

Inputs  (all already computed, no refitting here):
  results/rep52_trim/REP52_percohort_performance.csv
  results/rep52_trim/rep52_trim_assessment.csv      (independent cross-check of E1-E3)
Outputs:
  results/rep52_trim/REP52_to_REP30_master.csv
  results/rep52_trim/REP52_to_REP30_master.xlsx     (4 sheets)
Run: managed venv python
"""
import os, numpy as np, pandas as pd

ROOT = r"E:/01.WorkSpace/01.Breast_cancer_gene_panel/resistance_signature"
OUT  = os.path.join(ROOT, "results", "rep52_trim")

d = pd.read_csv(os.path.join(OUT, "REP52_percohort_performance.csv"))
chk = pd.read_csv(os.path.join(OUT, "rep52_trim_assessment.csv"))

# ---- top-tertile threshold (66th percentile of |coef| over the 52 genes) ----
abscoef = d["panel_coef"].abs()
THR = float(abscoef.quantile(0.66))

# ---- cross-check the criteria columns against the independent assessment file ----
m = d.merge(chk[["gene", "abs_coef_top_tertile", "in_HRplus108",
                 "sc_both_queues", "sc_one_queue", "tier"]],
            on="gene", suffixes=("", "_chk"))
bad = m[(m["top_tertile_contribution"] != m["abs_coef_top_tertile"]) |
        (m["confirmed_in_HRplus_rebuild"] != m["in_HRplus108"]) |
        (m["tier"] != m["tier_chk"])]
assert bad.empty, f"criteria mismatch:\n{bad[['gene']]}"
print(f"[check] criteria columns agree with rep52_trim_assessment.csv for all 52 genes")
print(f"[check] top-tertile threshold |coef| >= {THR:.4f}")

# ---- tidy table ----
t = d.copy()
t["rank_by_abs_coef"] = abscoef.rank(ascending=False, method="first").astype(int)
t["decision"] = np.where(t["in_REP30"] == "yes", "Retained (REP30)",
                         np.where(t["in_REP30"] == True, "Retained (REP30)", "Dropped"))

def sc_state(r):
    if r["sc_hit_GSE161529"] == 1 and r["sc_hit_GSE249361"] == 1:
        return "both queues"
    if r["sc_hit_GSE161529"] == 1 or r["sc_hit_GSE249361"] == 1:
        return "one queue only"
    return "none"

def sc_queues(r):
    q = []
    if r["sc_hit_GSE161529"] == 1: q.append("GSE161529")
    if r["sc_hit_GSE249361"] == 1: q.append("GSE249361")
    return "+".join(q) if q else ""

t["E3_sc_replication"] = t.apply(sc_state, axis=1)
t["E3_sc_which_queues"] = t.apply(sc_queues, axis=1)

t["E1_top_tertile_contribution"] = np.where(t["top_tertile_contribution"] == 1, "yes", "no")
t["E2_HRplus_rebuild_confirmed"] = np.where(t["confirmed_in_HRplus_rebuild"] == 1, "yes", "no")

# keep rule actually used by the pipeline:
#   keep if (E3 both queues) OR E1 OR E2 ; otherwise drop
anchor = (t["E3_sc_replication"].eq("both queues")) | \
         (t["top_tertile_contribution"] == 1) | \
         (t["confirmed_in_HRplus_rebuild"] == 1)
t["decision_rule_recheck"] = np.where(anchor, "Retained (REP30)", "Dropped")
mism = t[t["decision_rule_recheck"] != t["decision"]]
print(f"[check] keep rule reproduces in_REP30 for {52 - len(mism)}/52 genes"
      + ("" if mism.empty else f"  MISMATCH: {list(mism['gene'])}"))

def reason(r):
    if r["decision"].startswith("Retained"):
        got = []
        if r["E3_sc_replication"] == "both queues":
            got.append("E3 single-cell replication in both queues")
        if r["top_tertile_contribution"] == 1:
            got.append("E1 top-tertile pooled contribution")
        if r["confirmed_in_HRplus_rebuild"] == 1:
            got.append("E2 HR+ rebuild confirmation")
        return "kept on " + "; ".join(got) if got else "kept (tier)"
    fails = []
    if r["top_tertile_contribution"] != 1:
        fails.append(f"E1 fails (|coef|={abs(r['panel_coef']):.4f} < {THR:.4f} tertile cut)")
    if r["confirmed_in_HRplus_rebuild"] != 1:
        fails.append("E2 fails (absent from HR+ 108-gene rebuild)")
    if r["E3_sc_replication"] == "one queue only":
        fails.append(f"E3 partial ({r['E3_sc_which_queues']} only, not replicated across queues)")
    elif r["E3_sc_replication"] == "none":
        fails.append("E3 fails (no top-50 DE hit in either single-cell queue)")
    return "DROPPED - all three anchors fail: " + "; ".join(fails)

t["failure_reason"] = t.apply(reason, axis=1)

keep = ["rank_by_abs_coef", "gene", "decision", "panel_coef", "panel_HR",
        "METABRIC_HR_perSD", "METABRIC_CI95", "METABRIC_p", "METABRIC_q",
        "TCGA_HR_perSD", "TCGA_CI95", "TCGA_p", "TCGA_q",
        "POOLED_HR_perSD", "POOLED_CI95", "POOLED_p", "POOLED_q",
        "E1_top_tertile_contribution", "E2_HRplus_rebuild_confirmed",
        "E3_sc_replication", "E3_sc_which_queues", "support_lines",
        "tier", "failure_reason"]
out = t[keep].sort_values("rank_by_abs_coef").reset_index(drop=True)

# numeric rounding for readability
for c in ["panel_coef", "METABRIC_HR_perSD", "TCGA_HR_perSD", "POOLED_HR_perSD", "panel_HR"]:
    out[c] = out[c].round(4)
out.to_csv(os.path.join(OUT, "REP52_to_REP30_master.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- Excel bundle
def fmt(df):
    x = df.copy()
    for c in ["METABRIC_p", "METABRIC_q", "TCGA_p", "TCGA_q", "POOLED_p", "POOLED_q"]:
        x[c] = x[c].map(lambda v: f"{v:.3g}")
    return x

retained = out[out["decision"].str.startswith("Retained")].copy()
dropped  = out[out["decision"].eq("Dropped")].copy()

criteria = pd.DataFrame([
    ["E1", "top-tertile contribution to the pooled Cox model",
     f"|Cox coefficient| >= {THR:.4f}",
     "66th percentile of |coef| across the 52 genes (top third)"],
    ["E2", "confirmation in an HR-positive reconstruction of the model",
     "gene also selected in the HR+/HER2- rebuild",
     "rep_genes_hrh2neg.csv, 108 genes"],
    ["E3", "replication of differential expression in high-scoring cells",
     "top-50 DE hit in high-REP cells",
     "strong = hit in BOTH GSE161529 and GSE249361; partial = exactly ONE queue"],
    ["rule", "keep rule",
     "keep if E3(strong) OR E1 OR E2",
     "E3 strong -> tier A_core; E1 or E2 -> tier B_keep; none -> tier C_drop (dropped)"],
], columns=["criterion", "definition", "threshold", "source / note"])

with pd.ExcelWriter(os.path.join(OUT, "REP52_to_REP30_master.xlsx"),
                    engine="openpyxl") as w:
    fmt(out).to_excel(w, sheet_name="Master_52genes", index=False)
    fmt(retained).to_excel(w, sheet_name="Retained_REP30", index=False)
    fmt(dropped).to_excel(w, sheet_name="Dropped_22", index=False)
    criteria.to_excel(w, sheet_name="Criteria_definitions", index=False)

# ---------------------------------------------------------------- console summary
print("\n=== COUNTS ===")
print("  52-gene panel ->", out["decision"].value_counts().to_dict())
print("  tiers:", out["tier"].value_counts().to_dict())
print(f"  criteria passed per dropped gene: "
      f"{dropped['support_lines'].value_counts().sort_index().to_dict()}")

print("\n=== 22 DROPPED GENES: which criteria they failed ===")
for _, r in dropped.iterrows():
    print(f"  {r['gene']:9} E1={r['E1_top_tertile_contribution']:3} "
          f"E2={r['E2_HRplus_rebuild_confirmed']:3} E3={r['E3_sc_replication']:14} "
          f"| {r['failure_reason'].split(': ',1)[1]}")

print("\n=== per-cohort Cox of dropped genes (are they 'bad genes' or just redundant?) ===")
print(f"  dropped genes significant in METABRIC (q<0.05): "
      f"{int((dropped['METABRIC_q']<0.05).sum())}/22")
print(f"  dropped genes significant in TCGA     (q<0.05): "
      f"{int((dropped['TCGA_q']<0.05).sum())}/22")
print(f"  dropped genes with |METABRIC HR per SD - 1| > 0.05: "
      f"{int((dropped['METABRIC_HR_perSD'].sub(1).abs()>0.05).sum())}/22")
print(f"  mean |panel_coef| retained vs dropped: "
      f"{abs(retained['panel_coef']).mean():.4f} vs {abs(dropped['panel_coef']).mean():.4f}")

print("\nwrote:", os.path.join(OUT, "REP52_to_REP30_master.csv"))
print("wrote:", os.path.join(OUT, "REP52_to_REP30_master.xlsx"))
