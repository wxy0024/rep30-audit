# -*- coding: utf-8 -*-
"""Interrogate the only positive signals in the MBCproject cohort.

Two findings need deconstruction before they can be reported:
  (1) REP30 is HIGHER in ESR1-mutant samples (delta +0.33, p=0.08; patients +0.55,
      p=0.02; HR+ patients +0.88, p=0.006) -- the first *positive* resistance
      direction seen in this project, and therefore the one most likely to be
      over-claimed. What are ESR1-mutant samples actually like?
  (2) REP30 is higher in ERBB2-amplified samples -- but ERBB2 amp and HER2+
      subtype overlap, so is it a subtype effect?

Also attaches proper uncertainty (bootstrap CI + permutation) and runs the
method-sensitivity sweep on both contrasts.
"""
import os, sys, io
import numpy as np, pandas as pd
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = r"E:/01.WorkSpace/01.Breast_cancer_gene_panel/resistance_signature"
OUT = os.path.join(ROOT, "results/metastatic/MBCproject")
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import stats_toolkit as stk  # noqa: E402

_log = []
def log(s=""):
    print(s); _log.append(str(s))

df = pd.read_csv(os.path.join(OUT, "mbcproject_linked_samples.csv"))
pl = pd.read_csv(os.path.join(OUT, "mbcproject_linked_patients.csv"))
for c in ["tx_days_any", "tx_days_endo", "tx_days_chemo", "tx_days_cdk46",
          "BX_TIME_DAYS", "BX_KI67", "TMB_NONSYNONYMOUS"]:
    for d in (df, pl):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
log(f"linked: {df.shape} samples / {pl.shape} patients")


def describe_group(d, flag, cols):
    f = d[flag].fillna(False).astype(bool)
    log(f"\n  --- {flag}: {int(f.sum())} carriers vs {int((~f).sum())} wild-type")
    for c in cols:
        if c not in d:
            continue
        a = pd.to_numeric(d.loc[f, c], errors="coerce")
        b = pd.to_numeric(d.loc[~f, c], errors="coerce")
        if a.notna().sum() < 2 or b.notna().sum() < 2:
            continue
        _, p = stats.mannwhitneyu(a.dropna(), b.dropna(), alternative="two-sided")
        log(f"      {c:26s} carrier med={a.median():>9.3f}  WT med={b.median():>9.3f}  "
            f"d={stk.cliffs_delta(a.dropna(), b.dropna()):+.3f}  p={p:.3g}")


log("=" * 78)
log("PART 1 -- who are the ESR1-mutant samples? (confound profile)")
log("=" * 78)
prof = ["rep30", "prolif", "er_activity", "ESR1", "MKI67", "TMB_NONSYNONYMOUS",
        "BX_TIME_DAYS", "tx_days_any", "tx_days_endo", "tx_days_chemo", "met_lines"]
for fl in ["ESR1_mut_any", "ESR1_mut_hot", "ERBB2_amp", "RB1_loss"]:
    describe_group(df, fl, prof)

log("\n  categorical profile of ESR1-mutant samples (all-157):")
f = df["ESR1_mut_any"].fillna(False).astype(bool)
for c in ["BX_LOCATION", "CALC_MET_SETTING", "CALC_TREATMENT_NAIVE", "BX_ER",
          "BX_HER2OVERALL", "SAMPLE_TIMEPOINT"]:
    if c in df:
        tabs = pd.crosstab(f, df[c], normalize="index").round(3)
        log(f"    {c}: carriers -> {df.loc[f, c].value_counts().head(4).to_dict()}")

log("\n  biopsy-site / metastatic-setting composition of ESR1 carriers vs WT:")
log("    carriers BX_LOCATION: " + str(df.loc[f, "BX_LOCATION"].value_counts().to_dict()))
log("    carriers CALC_MET_SETTING: " + str(df.loc[f, "CALC_MET_SETTING"].value_counts().to_dict()))
log("    carriers BX_TIME_DAYS: " + str(np.round(df.loc[f, "BX_TIME_DAYS"].dropna().values, 1)))

log("\n" + "=" * 78)
log("PART 2 -- method sweep + uncertainty on the two positive contrasts")
log("=" * 78)

rows = []
for scope_name, d in [("all-157", df), ("patients-120", pl),
                      ("HR+ samples", df[df["hr_pos"]]),
                      ("post-treatment", df[df["tx_days_any"].fillna(0) > 0]),
                      ("treatment-naive", df[df["tx_days_any"].fillna(0) == 0])]:
    for flag, nm in [("ESR1_mut_any", "ESR1 mut (any)"), ("ESR1_mut_hot", "ESR1 hotspot"),
                     ("ERBB2_amp", "ERBB2 amp"), ("RB1_loss", "RB1 loss")]:
        if flag not in d:
            continue
        g = d[flag].fillna(False).astype(bool)
        a = d.loc[g, "rep30"].values
        b = d.loc[~g, "rep30"].values
        if len(a) < 3 or len(b) < 3:
            continue
        res = stk.compare_continuous(a, b, nperm=20000, seed=7)
        ci = stk.bootstrap_diff_ci(a, b, "mean", nboot=20000, seed=11)
        for r in res:
            rows.append(dict(scope=scope_name, contrast=nm, family=r["family"],
                             method=r["method"], p=r["p"],
                             effect_name=r["effect_name"], effect=r["effect"]))
        rows.append(dict(scope=scope_name, contrast=nm, family="uncertainty",
                         method="bootstrap mean-diff 95% CI", p=np.nan,
                         effect_name="mean diff", effect=ci["stat"]))
        rows.append(dict(scope=scope_name, contrast=nm, family="uncertainty",
                         method=f"CI bounds [{ci['lo']:.3f}, {ci['hi']:.3f}]", p=np.nan,
                         effect_name="mean diff", effect=np.nan))
        log(f"\n{scope_name} | {nm}: n={len(a)}/{len(b)}  "
            f"median {np.median(a):.3f} vs {np.median(b):.3f}")
        log(f"   Cliff's delta = {stk.cliffs_delta(a, b):+.3f}   "
            f"Cohen's d = {stk.cohens_d(a, b):+.3f}   "
            f"bootstrap mean-diff 95% CI = [{ci['lo']:.3f}, {ci['hi']:.3f}]")
        ps = [r["p"] for r in res if r["p"] == r["p"]]
        if ps:
            log(f"   p across {len(ps)} methods: min={min(ps):.4g} max={max(ps):.4g} "
                f"significant(<0.05) in {sum(1 for p in ps if p<0.05)}/{len(ps)}")

ms = pd.DataFrame(rows)
ms.to_csv(os.path.join(OUT, "positive_signal_method_sweep.csv"), index=False)

log("\n" + "=" * 78)
log("PART 3 -- is REP30 in ESR1 mutants explained by proliferation?")
log("=" * 78)
def resid_on(y, C):
    C = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in C])
    ok = np.isfinite(C).all(axis=1) & np.isfinite(y)
    b, *_ = np.linalg.lstsq(C[ok], y[ok], rcond=None)
    out = np.full(len(y), np.nan)
    out[ok] = y[ok] - C[ok] @ b
    return out

for scope_name, d in [("all-157", df), ("patients-120", pl)]:
    g = d["ESR1_mut_any"].fillna(False).astype(bool).values
    y = d["rep30"].values
    log(f"\n{scope_name}: ESR1 mut {int(g.sum())} vs WT {int((~g).sum())}")
    for covs, lab in [([d["prolif"].values], "prolif"),
                      ([d["prolif"].values, d["er_activity"].values], "prolif+ERact"),
                      ([d["prolif"].values, pd.to_numeric(d["tx_days_endo"], errors="coerce").fillna(0).values],
                       "prolif+endo_days")]:
        r = resid_on(y, covs)
        ok = np.isfinite(r)
        a, b = r[ok & g], r[ok & ~g]
        if len(a) >= 3 and len(b) >= 3:
            _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            log(f"    adjusted for {lab:20s}: delta={stk.cliffs_delta(a, b):+.3f} p={p:.4g}")

with open(os.path.join(OUT, "positive_signal_deconstruction.txt"), "w",
          encoding="utf-8") as fh:
    fh.write("\n".join(_log))
log("\nwritten.")
