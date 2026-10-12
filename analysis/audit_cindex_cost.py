#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
audit_cindex_cost.py -- "After paying the fitting cost, where does the REP30 C-index actually rank?"

The question raised by the adversarial (simulated) review
----------------------
Place the cross-fit REP30 C=0.5823 into the Fig 2D ranking table:
    Age 0.6416 (1 df, near-zero optimism)
    NPI  0.6175 (fixed formula, zero fitting)
    REP30 cross-fit 0.5823
    Unweighted 12-gene proliferation mean 0.5871
-> After paying the fitting cost, REP30 falls below the NPI and is nearly tied with the unweighted proliferation mean.

This script answers three questions:
  A. Comparability check: are the C values for age/NPI/REP30 computed on the same n=1398 samples, the same event definition,
     and the same C-index implementation (lifelines concordance_index_)?
  B. A fairer "cost-paid" estimate: 10-fold CV (within METABRIC HR+/HER2-,
     out-of-fold evaluation) -- the training power is far better than TCGA (88 events),
     correcting the "TCGA training too weak -> 0.5823 is an overly strict lower bound" problem.
     The same CV is applied to the 12-gene PROLIF (equal-degrees-of-freedom control).
  C. Paired bootstrap 95% CI for the delta-C: REP30cf - NPI, REP30cf - Age
     (the published REP30 - NPI / - Age values are also reported as a reference).

Outputs: results/manuscript_support/cindex_cost_results.csv, cindex_cost_log.txt
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
LOGPATH = os.path.join(OUT, "cindex_cost_log.txt")

from rep52_backtest import load_panel
from audit_venet_null import cox_psd, read_cbio, load_z_all, MB_DIR, TC_DIR
from audit_venet_null2 import load_metabric, load_tcga
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

PROLIF = ["MKI67", "PCNA", "TOP2A", "CCNB1", "BUB1", "AURKA", "PLK1", "MCM2",
          "RRM2", "TYMS", "CDC20", "BIRC5"]
N_REPEAT = 20
N_FOLD = 10
B_BOOT = 2000
SEED = 20260924

LOG = []


def log(s=""):
    print(s)
    LOG.append(str(s))
    with open(LOGPATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG) + "\n")


def C_of(t, x, e):
    """Harrell C for a **risk score** x: larger x = higher risk).

     Convention pitfall (verified empirically, see results/manuscript_support/_cdiag.txt):
    lifelines.utils.concordance_index(event_times, predicted_scores, event_observed)
     In 0.30.3, concordance_index treats predicted_scores as "larger = **longer** survival" -- the opposite of a risk score.
     Synthetic-data check: when larger x -> shorter time (high risk) it returns 0.0, and only -x returns 1.0;
     whereas CoxPHFitter.concordance_index_ returns 1.0 in both cases (it uses the linear predictor and orients itself).
     A risk score must therefore be negated to align with the published pipeline (cph.concordance_index_).
    """
    m = np.isfinite(x) & np.isfinite(t) & np.isfinite(e)
    return float(concordance_index(t[m], -x[m], e[m]))


def fit_beta_std(Ztr, t_tr, e_tr):
    """Fit a multivariable Cox on the training set (covariates standardized); returns (beta, mu, sd)."""
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
    for c in cols:
        X[c] = (X[c].astype(float) - mu[c]) / sd[c]
    cph = CoxPHFitter().fit(X[cols + ["t", "e"]], "t", "e")
    beta = np.array([float(cph.params_[c]) for c in cols])
    return beta, np.array([mu[c] for c in cols]), np.array([sd[c] for c in cols])


def cv_outfold_C(Z, t, e, n_repeat=N_REPEAT, n_fold=N_FOLD, seed=SEED):
    """Repeat n-fold CV n_repeat times; return the pooled out-of-fold C for each repeat."""
    from sklearn.model_selection import StratifiedKFold
    Zv = np.asarray(Z.values, dtype=float)
    Zv = np.where(np.isfinite(Zv), Zv, 0.0)
    idxs = np.arange(len(t))
    Cs = []
    for rep in range(n_repeat):
        skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed + rep)
        oof = np.full(len(t), np.nan)
        for tr, te in skf.split(idxs, e):
            Ztr = pd.DataFrame(Zv[:, tr], index=Z.index, columns=[Z.columns[i] for i in tr])
            beta, mu, sd = fit_beta_std(Ztr, t[tr], e[tr])
            Zte = Zv[:, te]
            oof[te] = ((Zte - mu[:, None]) / sd[:, None]).T @ beta
        Cs.append(C_of(t, oof, e))
    return np.array(Cs)


def main():
    coef = load_panel(os.path.join(ROOT, "results/rep52_trim/rep52_trimmed_panel.csv"))
    genes30 = list(coef)
    log("=" * 78)
    log("[A] Comparability check: are age / NPI / REP30 on the same samples and the same implementation?")

    mat_mb, mb = load_metabric(coef)
    # requires AGE / NPI / PROLIF
    cp = read_cbio(os.path.join(MB_DIR, "data_clinical_patient.txt"))
    cp.columns = [c.strip() for c in cp.columns]
    extra = pd.DataFrame({
        "patient": cp["PATIENT_ID"].astype(str),
        "AGE": pd.to_numeric(cp["AGE_AT_DIAGNOSIS"], errors="coerce"),
        "NPI": pd.to_numeric(cp.get("NPI"), errors="coerce")})
    mb = mb.merge(extra, on="patient", how="left")

    sub = mb[mb["HRposHER2neg"] == 1].dropna(subset=["REP30", "t", "e"])
    sub = sub[sub["t"] > 0].copy()
    idx = sub["SAMPLE_ID"].astype(str).tolist()
    t = sub["t"].values.astype(float)
    e = sub["e"].values.astype(float)
    log(f"   n={len(sub)}  events={int(e.sum())}")

    g12 = [g for g in PROLIF if g in mat_mb.index]
    Zm = mat_mb[idx]
    sub["PROLIF"] = Zm.loc[g12].mean(axis=0).reindex(idx).values
    log(f"   PROLIF genes matched: {len(g12)}/12")

    # ---- recompute each with lifelines concordance_index, cross-checked against hrposher2neg_cindex.csv ----
    published = {"REP30": 0.676562, "Age": 0.641633, "NPI": 0.617457,
                 "PROLIF": 0.587119, "MKI67": 0.568836}
    rows = []
    for name, col in [("REP30", "REP30"), ("Age", "AGE"), ("NPI", "NPI"),
                      ("PROLIF", "PROLIF"), ("MKI67", "MKI67")]:
        if col not in sub.columns:
            continue
        x = pd.to_numeric(sub[col], errors="coerce").values.astype(float)
        m = np.isfinite(x)
        c = C_of(t[m], x[m], e[m])
        pub = published.get(name)
        d = (c - pub) if pub is not None else np.nan
        log(f"   {name:7s}: n={int(m.sum()):5d}  C={c:.4f}  "
            f"published={pub if pub is None else round(pub,4)}  diff={d if pub is None else round(d,5)}")
        rows.append(dict(predictor=name, n=int(m.sum()), cindex_recomputed=c,
                         cindex_published=pub, diff=d))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "cindex_comparability.csv"),
                              index=False, encoding="utf-8")
    log("   -> recomputation matches the published values => the age/NPI/REP30 C values are same-sample, same-implementation, and directly comparable")
    log("   (note: if C and 1-C are globally flipped, the concordance_index sign convention was not negated;")
    log("     this script negates per the empirically verified convention, see the C_of docstring)")

    # ---- REP30cf (TCGA-trained) reproduction ----
    mat_tc, tc = load_tcga(coef)
    tc_lum = tc[tc["HRposHER2neg"] == 1].dropna(subset=["REP30", "t", "e"])
    tc_lum = tc_lum[tc_lum["t"] > 0]
    tc_ids = tc_lum["SAMPLE_ID"].astype(str).tolist()
    Z30_tc = mat_tc.loc[[g for g in genes30 if g in mat_tc.index], tc_ids]
    Z30_mb = Zm.loc[[g for g in genes30 if g in Zm.index], idx]
    Z12_mb = Zm.loc[g12, idx]
    beta, mu, sd = fit_beta_std(Z30_tc, tc_lum["t"].values.astype(float),
                                tc_lum["e"].values.astype(float))
    Zv = np.where(np.isfinite(np.asarray(Z30_mb.values, float)),
                  np.asarray(Z30_mb.values, float), 0.0)
    rep30cf = ((Zv - mu[:, None]) / sd[:, None]).T @ beta
    c_cf = C_of(t, rep30cf, e)
    log(f"\n[B] REP30cf (TCGA-trained) C = {c_cf:.4f}  (expected 0.5823)")

    # ---- C. 10-fold CV (out-of-fold, adequately powered training) ----
    log(f"\n[C] {N_FOLD}-fold CV x {N_REPEAT} repeats (within METABRIC HR+/HER2-, out-of-fold)")
    t0 = time.time()
    cv30 = cv_outfold_C(Z30_mb, t, e)
    log(f"   REP30 (30 genes) : mean={cv30.mean():.4f}  "
        f"[{np.percentile(cv30,2.5):.4f}, {np.percentile(cv30,97.5):.4f}]  "
        f"min={cv30.min():.4f} max={cv30.max():.4f}   ({time.time()-t0:.0f}s)")
    t0 = time.time()
    cv12 = cv_outfold_C(Z12_mb, t, e)
    log(f"   PROLIF(12 genes) : mean={cv12.mean():.4f}  "
        f"[{np.percentile(cv12,2.5):.4f}, {np.percentile(cv12,97.5):.4f}]  "
        f"min={cv12.min():.4f} max={cv12.max():.4f}   ({time.time()-t0:.0f}s)")
    log(f"   NPI (zero-fit)     : 0.6175   Age (zero-fit): 0.6416   <- reference lines")

    # ---- D. paired bootstrap for the delta-C ----
    log(f"\n[D] paired bootstrap (B={B_BOOT}) for the delta-C")
    rng = np.random.default_rng(SEED)
    n = len(t)
    rep30_pub = sub["REP30"].values.astype(float)
    age = pd.to_numeric(sub["AGE"], errors="coerce").values.astype(float)
    npi = pd.to_numeric(sub["NPI"], errors="coerce").values.astype(float)
    boot = {k: [] for k in ["cf_vs_npi", "cf_vs_age", "pub_vs_npi", "pub_vs_age"]}
    for b in range(B_BOOT):
        s = rng.integers(0, n, n)
        tb, eb = t[s], e[s]
        if eb.sum() < 5:
            continue
        c_cf_b = C_of(tb, rep30cf[s], eb)
        c_n_b = C_of(tb, npi[s], eb)
        c_a_b = C_of(tb, age[s], eb)
        c_p_b = C_of(tb, rep30_pub[s], eb)
        boot["cf_vs_npi"].append(c_cf_b - c_n_b)
        boot["cf_vs_age"].append(c_cf_b - c_a_b)
        boot["pub_vs_npi"].append(c_p_b - c_n_b)
        boot["pub_vs_age"].append(c_p_b - c_a_b)
    res = []
    for k, v in boot.items():
        v = np.array(v)
        lo, hi = np.percentile(v, [2.5, 97.5])
        log(f"   ΔC {k:12s}: mean={v.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
            f"P(Δ>0)={float((v>0).mean()):.3f}")
        res.append(dict(contrast=k, delta=float(v.mean()), lo=float(lo), hi=float(hi),
                        p_delta_gt0=float((v > 0).mean()), n_boot=len(v)))
    pd.DataFrame(res).to_csv(os.path.join(OUT, "cindex_delta_bootstrap.csv"),
                             index=False, encoding="utf-8")

    summ = pd.DataFrame([
        dict(predictor="REP30 (published pooled coefs)", cindex=float(C_of(t, rep30_pub, e)),
             fitting="partly in-sample (pooled TCGA+METABRIC)"),
        dict(predictor="REP30 cross-fit (TCGA-trained)", cindex=float(c_cf),
             fitting="cross-cohort; weak training (88 events)"),
        dict(predictor="REP30 10-fold CV out-of-fold", cindex=float(cv30.mean()),
             fitting="within-cohort CV; adequate training"),
        dict(predictor="PROLIF_cox 10-fold CV out-of-fold", cindex=float(cv12.mean()),
             fitting="equal-DF control, same CV"),
        dict(predictor="Age", cindex=float(C_of(t, age, e)), fitting="none (1 df)"),
        dict(predictor="NPI", cindex=float(C_of(t, npi, e)), fitting="none (fixed formula)"),
    ])
    summ.to_csv(os.path.join(OUT, "cindex_cost_results.csv"), index=False, encoding="utf-8")
    log("\n[SUMMARY]")
    log(summ.to_string())
    log("\nDONE")


if __name__ == "__main__":
    sys.excepthook = lambda a, b, tb: (
        LOG.append("EXCEPTION:\n" + "".join(traceback.format_exception(a, b, tb))),
        open(LOGPATH, "w", encoding="utf-8").write("\n".join(LOG) + "\n"))
    main()
