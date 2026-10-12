#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
audit_cindex_cost2.py -- fills a gap: CI for the delta-C of the CV out-of-fold comparison

Background
----
The bootstrap CI for the delta-C in CINDEX_COST.md covers only the two **fixed-score** versions:
  - REP30cf (TCGA-trained, fixed) - NPI / - Age
  - Published REP30 (fixed) - NPI / - Age
but **not** the CI for the CV out-of-fold (0.671) vs NPI / Age -- yet 0.671 is exactly
the pivot of the "within-cohort advantage is not overfitting" argument. In the academic-writer draft,
"CV 0.671" and "delta-C +0.059 [+0.034,+0.085]" appear side by side, which could be mistaken for the CI of the CV version
(it is the CI of the **published-value** version). Hence this recomputation.

Approach
----
Run one 10-fold CV (fixed seed) to obtain the out-of-fold score vector; treat it as a fixed score and
run a paired bootstrap (B=2000, resampling patients) for the CI of dC(oof - NPI) and dC(oof - Age).
NOTE: this CI reflects **patient-sampling** variability only; it does **not** include fold-splitting / refitting variability;
  the latter is covered separately by the very narrow range across 20 repeated partitions (0.6685-0.6733).

Outputs: cindex_cv_delta_bootstrap.csv; log cindex_cv_log.txt
"""
import os
import sys
import io
import time
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "manuscript_support")
LOGPATH = os.path.join(OUT, "cindex_cv_log.txt")

MB_DIR = os.path.join(ROOT, "data/01_core/METABRIC/brca_metabric")
PANEL = os.path.join(ROOT, "results/rep52_trim/rep52_trimmed_panel.csv")

from rep52_backtest import load_panel, rep_score, to_event
from audit_venet_null import (cox_psd, read_cbio, load_z_all, cox_fit_beta,
                              score_from_beta)
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold

SEED = 20260924
B_BOOT = 2000
LOG = []


def log(s=""):
    print(s)
    LOG.append(str(s))
    with open(LOGPATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG) + "\n")


def C_of(t, x, e):
    """Harrell C for a risk score (lifelines requires negation, see CINDEX_COST.md section 0)."""
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
    from lifelines import CoxPHFitter
    cph = CoxPHFitter().fit(X[cols + ["t", "e"]], "t", "e")
    beta = np.array([float(cph.params_[c]) for c in cols])
    return beta, np.array([mu[c] for c in cols]), np.array([sd[c] for c in cols])


def load_metabric(coef):
    z_path = os.path.join(MB_DIR, "data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt")
    log("  loading METABRIC full z matrix ...")
    mat = load_z_all(z_path)
    S = pd.DataFrame({"REP30": rep_score(mat, coef)})
    S.index = S.index.astype(str)
    S.index.name = "SAMPLE_ID"
    S = S.reset_index()
    S["patient"] = S["SAMPLE_ID"]
    cp = read_cbio(os.path.join(MB_DIR, "data_clinical_patient.txt"))
    cp.columns = [c.strip() for c in cp.columns]
    P = pd.DataFrame({"patient": cp["PATIENT_ID"].astype(str),
                      "os_months": pd.to_numeric(cp["OS_MONTHS"], errors="coerce"),
                      "os_status": cp["OS_STATUS"].astype(str).str.split(":").str[-1].str.strip(),
                      "AGE": pd.to_numeric(cp["AGE_AT_DIAGNOSIS"], errors="coerce"),
                      "NPI": pd.to_numeric(cp.get("NPI"), errors="coerce")})
    P["event"] = to_event(P["os_status"])
    cs = read_cbio(os.path.join(MB_DIR, "data_clinical_sample.txt"))
    cs.columns = [c.strip() for c in cs.columns]
    Sm = pd.DataFrame({"SAMPLE_ID": cs["SAMPLE_ID"].astype(str),
                       "ER_STATUS": cs["ER_STATUS"].astype(str).str.strip(),
                       "HER2_STATUS": cs["HER2_STATUS"].astype(str).str.strip()})
    M = S.merge(Sm, on="SAMPLE_ID", how="inner").merge(P, on="patient", how="inner")
    M = M.rename(columns={"os_months": "t", "event": "e"})
    M["HRposHER2neg"] = ((M["ER_STATUS"].str.upper().str.startswith("POS")) &
                         (M["HER2_STATUS"].str.upper().str.startswith("NEG"))).astype(int)
    return mat, M


def main():
    coef = load_panel(PANEL)
    genes30 = list(coef)
    mat, mb = load_metabric(coef)
    sub = mb[mb["HRposHER2neg"] == 1].dropna(subset=["REP30", "t", "e"])
    sub = sub[sub["t"] > 0].copy()
    idx = sub["SAMPLE_ID"].astype(str).tolist()
    t = sub["t"].values.astype(float)
    e = sub["e"].values.astype(float)
    age = pd.to_numeric(sub["AGE"], errors="coerce").values.astype(float)
    npi = pd.to_numeric(sub["NPI"], errors="coerce").values.astype(float)
    rep30_pub = sub["REP30"].values.astype(float)
    log(f"n={len(sub)} events={int(e.sum())}")

    Z30 = mat.loc[[g for g in genes30 if g in mat.index], idx]
    Zv = np.asarray(Z30.values, dtype=float)
    Zv = np.where(np.isfinite(Zv), Zv, 0.0)
    Z30c = pd.DataFrame(Zv, index=Z30.index, columns=idx)

    # ---- one 10-fold CV -> out-of-fold scores ----
    log(f"\n[CV] 10-fold, seed={SEED}")
    t0 = time.time()
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    oof = np.full(len(t), np.nan)
    names = list(Z30c.columns)
    for tr, te in skf.split(np.arange(len(t)), e):
        Ztr = pd.DataFrame(Zv[:, tr], index=Z30c.index, columns=[names[i] for i in tr])
        beta, mu, sd = fit_beta_std(Ztr, t[tr], e[tr])
        oof[te] = ((Zv[:, te] - mu[:, None]) / sd[:, None]).T @ beta
    c_oof = C_of(t, oof, e)
    log(f"   REP30 out-of-fold C = {c_oof:.4f}   ({time.time()-t0:.0f}s)")
    np.save(os.path.join(OUT, "cv_oof_rep30.npy"), oof)

    # ---- paired bootstrap ----
    log(f"\n[BOOT] B={B_BOOT}")
    rng = np.random.default_rng(SEED)
    n = len(t)
    res = {"oof_vs_npi": [], "oof_vs_age": [], "pub_vs_npi": [], "pub_vs_age": []}
    for b in range(B_BOOT):
        s = rng.integers(0, n, n)
        tb, eb = t[s], e[s]
        if eb.sum() < 5:
            continue
        c_o, c_n, c_a, c_p = (C_of(tb, oof[s], eb), C_of(tb, npi[s], eb),
                              C_of(tb, age[s], eb), C_of(tb, rep30_pub[s], eb))
        res["oof_vs_npi"].append(c_o - c_n)
        res["oof_vs_age"].append(c_o - c_a)
        res["pub_vs_npi"].append(c_p - c_n)
        res["pub_vs_age"].append(c_p - c_a)
    rows = []
    for k, v in res.items():
        v = np.array(v)
        lo, hi = np.percentile(v, [2.5, 97.5])
        log(f"   ΔC {k:12s}: mean={v.mean():+.4f} 95% CI [{lo:+.4f}, {hi:+.4f}] "
            f"P(Δ>0)={float((v>0).mean()):.3f}")
        rows.append(dict(contrast=k, delta=float(v.mean()), lo=float(lo), hi=float(hi),
                         p_delta_gt0=float((v > 0).mean()), n_boot=len(v)))
    rows.append(dict(contrast="REP30_CV_out_of_fold_C", delta=float(c_oof),
                     lo=np.nan, hi=np.nan, p_delta_gt0=np.nan, n_boot=0))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "cindex_cv_delta_bootstrap.csv"),
                              index=False, encoding="utf-8")
    log("\nDONE")


if __name__ == "__main__":
    sys.excepthook = lambda a, b, tb: (
        LOG.append("EXCEPTION:\n" + "".join(traceback.format_exception(a, b, tb))),
        open(LOGPATH, "w", encoding="utf-8").write("\n".join(LOG) + "\n"))
    main()
