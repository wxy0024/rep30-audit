#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
audit_transfer_stability.py -- the correct "transfer stability" diagnostic

Why this is done
--------
The adversarial review notes that per-gene weight-sign consistency **cannot** serve as positive evidence
(REP30 itself has only rho=+0.180 with 63% sign agreement, only 13 points above the random 50%,
and would be dismissed with "both sides are noise"). The correct diagnostic it proposes:
    **the correlation, on the test set, between the risk scores (linear predictors) fitted separately in the two cohorts**
    individual coefficients are unstable under collinearity; what is stable is the linear predictor.

This script computes it:
    on METABRIC HR+/HER2-,
    the correlation (Pearson / Spearman) between score_TCGAfit and score_MBfit,
    separately for REP30(30) and PROLIF(12).
    Their respective C-index values are reported as context.

Purpose: to explain "under the same transfer obstacle, why REP30 retains its signal while PROLIF does not".
Not intended to prove superiority -- both sides are reported, avoiding selective citation.

Outputs: transfer_stability.csv, log transfer_stability_log.txt
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
LOGPATH = os.path.join(OUT, "transfer_stability_log.txt")

from rep52_backtest import load_panel, rep_score, to_event
from audit_venet_null import read_cbio, load_z_subset, MB_DIR, TC_DIR
from audit_venet_null4 import load_metabric
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy import stats as st

PROLIF = ["MKI67", "PCNA", "TOP2A", "CCNB1", "BUB1", "AURKA", "PLK1", "MCM2",
          "RRM2", "TYMS", "CDC20", "BIRC5"]
LOG = []


def log(s=""):
    print(s)
    LOG.append(str(s))
    with open(LOGPATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG) + "\n")


def C_of(t, x, e):
    m = np.isfinite(x) & np.isfinite(t) & np.isfinite(e)
    return float(concordance_index(t[m], -x[m], e[m]))


def fit_beta_std(Ztr, t_tr, e_tr):
    X = Ztr.T.copy()
    X["t"] = t_tr
    X["e"] = e_tr
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    X = X[X["t"] > 0]
    cols = list(Ztr.index)
    mu, sd = {}, {}
    for c in cols:
        s = X[c].astype(float)
        mu[c] = float(s.mean())
        sd[c] = float(s.std()) if s.std() > 0 else 1.0
        X[c] = (s - mu[c]) / sd[c]
    cph = CoxPHFitter().fit(X[cols + ["t", "e"]], "t", "e")
    beta = np.array([float(cph.params_[c]) for c in cols])
    return beta, np.array([mu[c] for c in cols]), np.array([sd[c] for c in cols])


def score_apply(Zte, beta, mu, sd):
    Zv = np.asarray(Zte.values, dtype=float)
    Zv = np.where(np.isfinite(Zv), Zv, 0.0)
    return ((Zv - mu[:, None]) / sd[:, None]).T @ beta


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
    log("[transfer stability] consistency of the risk scores (linear predictors) across fitting sources")

    mat_mb, mb = load_metabric(coef)
    sub = mb[mb["HRposHER2neg"] == 1].dropna(subset=["REP30", "t", "e"])
    sub = sub[sub["t"] > 0]
    idx = sub["SAMPLE_ID"].astype(str).tolist()
    t, e = sub["t"].values.astype(float), sub["e"].values.astype(float)
    log(f"   METABRIC HR+/HER2- n={len(sub)} events={int(e.sum())}")

    mat_tc, tc = load_tcga(coef)
    tl = tc[tc["HRposHER2neg"] == 1].dropna(subset=["REP30", "t", "e"])
    tl = tl[tl["t"] > 0]
    tids = tl["SAMPLE_ID"].astype(str).tolist()
    t_tc, e_tc = tl["t"].values.astype(float), tl["e"].values.astype(float)
    log(f"   TCGA LumA/LumB     n={len(tl)} events={int(e_tc.sum())}")

    Zm, Zt = mat_mb[idx], mat_tc[tids]
    g30m = [g for g in genes30 if g in Zm.index]
    g12m = [g for g in PROLIF if g in Zm.index]
    g30t = [g for g in genes30 if g in Zt.index]
    g12t = [g for g in PROLIF if g in Zt.index]

    rows = []
    for nm, gm, gt in [("REP30(30)", g30m, g30t), ("PROLIF(12)", g12m, g12t)]:
        A_mb, A_tc = Zm.loc[gm, idx], Zt.loc[gt, tids]
        b_mb, mu_mb, sd_mb = fit_beta_std(A_mb, t, e)
        b_tc, mu_tc, sd_tc = fit_beta_std(A_tc, t_tc, e_tc)
        # both scores computed on METABRIC HR+/HER2-
        s_mb = score_apply(A_mb, b_mb, mu_mb, sd_mb)   # METABRIC-fitted (same cohort)
        s_tc = score_apply(A_mb, b_tc, mu_tc, sd_tc)   # TCGA-fitted (transfer)
        pr = st.pearsonr(s_tc, s_mb)
        sr = st.spearmanr(s_tc, s_mb)
        c_mb, c_tc = C_of(t, s_mb, e), C_of(t, s_tc, e)
        log(f"\n   {nm}:")
        log(f"      risk-score correlation (TCGA-fit vs METABRIC-fit, same test set):")
        log(f"        Pearson  r    = {pr[0]:+.3f}  (p={pr[1]:.3g})")
        log(f"        Spearman rho  = {sr[0]:+.3f}  (p={sr[1]:.3g})")
        log(f"      C-index: METABRIC-fit={c_mb:.4f}   TCGA-fit (transfer)={c_tc:.4f}")
        rows.append(dict(panel=nm, n_genes=len(gm),
                         pearson_r=float(pr[0]), pearson_p=float(pr[1]),
                         spearman_rho=float(sr[0]), spearman_p=float(sr[1]),
                         c_metabric_fit=float(c_mb), c_tcga_fit=float(c_tc)))

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "transfer_stability.csv"),
                              index=False, encoding="utf-8")
    log("\nDONE")


if __name__ == "__main__":
    sys.excepthook = lambda a, b, tb: (
        LOG.append("EXCEPTION:\n" + "".join(traceback.format_exception(a, b, tb))),
        open(LOGPATH, "w", encoding="utf-8").write("\n".join(LOG) + "\n"))
    main()
