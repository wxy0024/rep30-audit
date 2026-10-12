#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
audit_prolif_control.py -- answers two open questions from the adversarial review

(1) The "unfair yardstick" objection to the proliferation control
   PROLIF_cox forward (TCGA-trained -> METABRIC) HR = 0.928, p = 0.036 -- **nominally significant but protective in direction**.
   A genuine proliferation score cannot be protective for OS -> a reviewer would object that this control fails to transfer even its own known direction.
   Two explanations must be distinguished:
     (a) cross-cohort transfer itself destroys the proliferation signal (the control is unfair; the whole section needs rewriting)
     (b) merely "sign instability of the re-estimated weights" (the control is still usable; a footnote suffices)
   Decisive tests:
     - Unweighted PROLIF mean (**zero fitting**, no coefficients to transfer) in METABRIC HR+/HER2-: HR / C
       -- if HR>1 persists, proliferation biology is intact and the problem lies in the re-estimated weights -> supports (b)
     - Correlation between the PROLIF weight vectors fitted separately in TCGA and METABRIC
       -- if weak/negative, this directly demonstrates "sign instability" -> supports (b)
     - Control: correlation of the REP30 weight vector across the two cohorts (if clearly higher than PROLIF, the 30 genes are more stable)

(2) Same-n confirmation of "C 0.582 < Age 0.642 / NPI 0.617"
   NPI = size + nodal stage + grade; if METABRIC has missing values, its C may not be computed on n=1398.
   Report, for each comparator, the **valid n, event count, missingness, and whether it enters the model as a continuous variable**.

Outputs: prolif_control_results.csv, comparator_n.csv, log prolif_control_log.txt
"""
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "manuscript_support")
LOGPATH = os.path.join(OUT, "prolif_control_log.txt")

from rep52_backtest import load_panel, rep_score, to_event
from audit_venet_null import (cox_psd, cox_fit_beta, score_from_beta, read_cbio,
                              load_z_subset, MB_DIR, TC_DIR)
from audit_venet_null4 import load_metabric

PROLIF = ["MKI67", "PCNA", "TOP2A", "CCNB1", "BUB1", "AURKA", "PLK1", "MCM2",
          "RRM2", "TYMS", "CDC20", "BIRC5"]
LOG = []


def log(s=""):
    print(s)
    LOG.append(str(s))
    with open(LOGPATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG) + "\n")


def load_tcga(coef):
    z_path = os.path.join(TC_DIR, "data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt")
    mat = load_z_subset(z_path, set(coef) | set(PROLIF))
    S = pd.DataFrame({"REP30": rep_score(mat, coef)})
    S.index = S.index.astype(str)
    S.index.name = "SAMPLE_ID"
    S = S.reset_index()
    S["patient"] = S["SAMPLE_ID"].str.replace(r"-01[ABC]?$", "", regex=True).str[:12]
    cl = read_cbio(os.path.join(TC_DIR, "data_clinical_patient.txt"))
    cl.columns = [c.strip() for c in cl.columns]
    C = pd.DataFrame({"patient": cl["PATIENT_ID"].astype(str),
                      "SUBTYPE": cl["SUBTYPE"].astype(str),
                      "os_months": pd.to_numeric(cl["OS_MONTHS"], errors="coerce"),
                      "os_status": cl["OS_STATUS"].astype(str).str.split(":").str[-1].str.strip()})
    C["event"] = to_event(C["os_status"])
    M = S.merge(C, on="patient", how="inner").rename(columns={"os_months": "t", "event": "e"})
    M["HRposHER2neg"] = M["SUBTYPE"].isin(["BRCA_LumA", "BRCA_LumB"]).astype(int)
    return mat, M


def main():
    coef = load_panel(os.path.join(ROOT, "results/rep52_trim/rep52_trimmed_panel.csv"))
    genes30 = list(coef)
    log("=" * 78)
    log("[1] proliferation control: unweighted vs re-estimated weights -- separating 'transfer destroys the signal' from 'weight sign instability'")

    mat_mb, mb = load_metabric(coef)
    # requires AGE / NPI
    cp = read_cbio(os.path.join(MB_DIR, "data_clinical_patient.txt"))
    cp.columns = [c.strip() for c in cp.columns]
    extra = pd.DataFrame({"patient": cp["PATIENT_ID"].astype(str),
                          "AGE": pd.to_numeric(cp["AGE_AT_DIAGNOSIS"], errors="coerce"),
                          "NPI": pd.to_numeric(cp.get("NPI"), errors="coerce"),
                          "GRADE": pd.to_numeric(cp.get("GRADE"), errors="coerce"),
                          "SIZE": pd.to_numeric(cp.get("TUMOR_SIZE"), errors="coerce"),
                          "NODES": pd.to_numeric(cp.get("LYMPH_NODES_EXAMINED_POS"), errors="coerce")})
    mb = mb.merge(extra, on="patient", how="left")

    sub = mb[mb["HRposHER2neg"] == 1].dropna(subset=["REP30", "t", "e"])
    sub = sub[sub["t"] > 0].copy()
    idx = sub["SAMPLE_ID"].astype(str).tolist()
    t = sub["t"].values.astype(float)
    e = sub["e"].values.astype(float)
    log(f"   METABRIC HR+/HER2- n={len(sub)} events={int(e.sum())}")

    Zm = mat_mb[idx]
    g12 = [g for g in PROLIF if g in Zm.index]
    g30 = [g for g in genes30 if g in Zm.index]
    log(f"   PROLIF {len(g12)}/12, REP30 panel {len(g30)}/30")

    # ---------- (2) valid n for each comparator ----------
    log(f"\n[2] valid n / events / missingness per comparator (baseline n={len(sub)}, events={int(e.sum())})")
    sub["PROLIF_mean"] = Zm.loc[g12].mean(axis=0).reindex(idx).values
    sub["MKI67"] = Zm.loc["MKI67"].reindex(idx).values if "MKI67" in Zm.index else np.nan
    rows_n = []
    for nm, col in [("REP30", "REP30"), ("Age", "AGE"), ("NPI", "NPI"),
                    ("PROLIF (12-gene mean)", "PROLIF_mean"), ("MKI67", "MKI67"),
                    ("GRADE", "GRADE"), ("TUMOR_SIZE", "SIZE"),
                    ("NODES_POSITIVE", "NODES")]:
        x = pd.to_numeric(sub[col], errors="coerce").values.astype(float)
        ok = np.isfinite(x)
        rows_n.append(dict(predictor=nm, n_valid=int(ok.sum()), n_missing=int((~ok).sum()),
                           events=int(e[ok].sum()), continuous=True,
                           imputed=False))
        log(f"   {nm:22s}: valid n={int(ok.sum()):5d}  missing={int((~ok).sum()):4d}  "
            f"events={int(e[ok].sum()):4d}")
    pd.DataFrame(rows_n).to_csv(os.path.join(OUT, "comparator_n.csv"),
                                index=False, encoding="utf-8")
    log("   -> if NPI/AGE/PROLIF all have valid n=1398 and events=812, they are same-n comparable with REP30cf")

    # ---------- (1) unweighted PROLIF ----------
    log(f"\n[1a] unweighted PROLIF mean (zero fitting, no coefficients to transfer)")
    r = cox_psd(sub["PROLIF_mean"].values.astype(float), t, e)
    log(f"   METABRIC HR+/HER2-: HR={r['HR']:.3f} [{r['lo']:.3f},{r['hi']:.3f}] "
        f"p={r['p']:.3g} C={r['cindex']:.4f}  n={r['n']}")

    # same on the TCGA side
    mat_tc, tc = load_tcga(coef)
    tl = tc[tc["HRposHER2neg"] == 1].dropna(subset=["REP30", "t", "e"])
    tl = tl[tl["t"] > 0]
    tids = tl["SAMPLE_ID"].astype(str).tolist()
    t_tc, e_tc = tl["t"].values.astype(float), tl["e"].values.astype(float)
    Zt = mat_tc[tids]
    g12t = [g for g in PROLIF if g in Zt.index]
    prolif_tc = Zt.loc[g12t].mean(axis=0).reindex(tids).values
    r2 = cox_psd(prolif_tc, t_tc, e_tc)
    log(f"   TCGA LumA/LumB    : HR={r2['HR']:.3f} [{r2['lo']:.3f},{r2['hi']:.3f}] "
        f"p={r2['p']:.3g} C={r2['cindex']:.4f}  n={r2['n']}")

    # ---------- ① PROLIF_cox in-sample / cross-fit ----------
    Z12m = Zm.loc[g12, idx]
    Z12t = Zt.loc[g12t, tids]
    Z30m = Zm.loc[g30, idx]
    Z30t = Zt.loc[[g for g in genes30 if g in Zt.index], tids]

    b12_mb, _ = cox_fit_beta(Z12m, t, e)          # fitted within METABRIC
    b12_tc, _ = cox_fit_beta(Z12t, t_tc, e_tc)    # fitted within TCGA
    b30_mb, _ = cox_fit_beta(Z30m, t, e)
    b30_tc, _ = cox_fit_beta(Z30t, t_tc, e_tc)

    log(f"\n[1b] three sources of PROLIF_cox (all evaluated on the same METABRIC HR+/HER2-)")
    out = [dict(score="PROLIF_mean_unweighted", fit="none (zero fitting)",
                HR=r["HR"], lo=r["lo"], hi=r["hi"], p=r["p"], cindex=r["cindex"])]
    sc = score_from_beta(Z12m, b12_mb).reindex(idx).values.astype(float)
    rr = cox_psd(sc, t, e)
    log(f"   PROLIF_cox fitted within METABRIC : HR={rr['HR']:.3f} [{rr['lo']:.3f},{rr['hi']:.3f}] "
        f"p={rr['p']:.3g} C={rr['cindex']:.4f}")
    out.append(dict(score="PROLIF_cox", fit="METABRIC in-sample", HR=rr["HR"], lo=rr["lo"],
                    hi=rr["hi"], p=rr["p"], cindex=rr["cindex"]))
    sc = score_from_beta(Z12m, b12_tc).reindex(idx).values.astype(float)
    rr = cox_psd(sc, t, e)
    log(f"   PROLIF_cox TCGA-trained (transfer) : HR={rr['HR']:.3f} [{rr['lo']:.3f},{rr['hi']:.3f}] "
        f"p={rr['p']:.3g} C={rr['cindex']:.4f}   <- the previously reported 0.928")
    out.append(dict(score="PROLIF_cox", fit="TCGA-trained (cross-fit)", HR=rr["HR"], lo=rr["lo"],
                    hi=rr["hi"], p=rr["p"], cindex=rr["cindex"]))

    # ---------- cross-cohort consistency of the weight vectors ----------
    log(f"\n[1c] weight vectors: TCGA-fitted vs METABRIC-fitted (explains the 'sign instability')")
    from scipy import stats as st
    res = []
    for nm, a, b, gn in [("PROLIF(12)", b12_tc, b12_mb, g12),
                         ("REP30(30)", b30_tc, b30_mb, g30)]:
        pr = st.pearsonr(a, b)
        sr = st.spearmanr(a, b)
        sgn = float(np.mean(np.sign(a) == np.sign(b)))
        log(f"   {nm}: Pearson r={pr[0]:+.3f} (p={pr[1]:.3g}) | "
            f"Spearman rho={sr[0]:+.3f} (p={sr[1]:.3g}) | sign agreement={sgn*100:.0f}%")
        res.append(dict(panel=nm, n_genes=len(gn), pearson_r=float(pr[0]),
                        pearson_p=float(pr[1]), spearman_rho=float(sr[0]),
                        spearman_p=float(sr[1]), same_sign_frac=sgn))
    pd.DataFrame(res).to_csv(os.path.join(OUT, "weight_vector_agreement.csv"),
                             index=False, encoding="utf-8")

    # REP30 cross-fit reproduction (control)
    sc = score_from_beta(Z30m, b30_tc).reindex(idx).values.astype(float)
    rr = cox_psd(sc, t, e)
    log(f"\n   [control] REP30_cox TCGA-trained : HR={rr['HR']:.3f} [{rr['lo']:.3f},{rr['hi']:.3f}] "
        f"p={rr['p']:.3g} C={rr['cindex']:.4f}")
    out.append(dict(score="REP30_cox", fit="TCGA-trained (cross-fit)", HR=rr["HR"], lo=rr["lo"],
                    hi=rr["hi"], p=rr["p"], cindex=rr["cindex"]))
    pd.DataFrame(out).to_csv(os.path.join(OUT, "prolif_control_results.csv"),
                             index=False, encoding="utf-8")
    log("\nDONE")


if __name__ == "__main__":
    sys.excepthook = lambda a, b, tb: (
        LOG.append("EXCEPTION:\n" + "".join(traceback.format_exception(a, b, tb))),
        open(LOGPATH, "w", encoding="utf-8").write("\n".join(LOG) + "\n"))
    main()
