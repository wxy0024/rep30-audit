#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
audit_venet_null.py -- two "must-compute-new" analyses raised by the adversarial review

Background
----
The simulated review (deaigc-reviewer) flagged two fatal weaknesses in the positive results of section 2.4; both
**cannot be fixed by rewording, only by new data**:

 (1) The Venet-2011 random-signature null hypothesis
     The manuscript cites Venet 2011 as its "starting point" but never runs it. The 52 parent genes of REP30
     were selected on OS, and the 30 retained genes carry pooled TCGA+METABRIC Cox weights,
     so HR 1.93 / C 0.677 is exactly the magnitude Venet predicted "random gene sets can reach".
     Needed: size-matched random 30-gene sets -> per-SD Cox under exactly the same protocol
            -> the median and 2.5-97.5 percentiles of the random-set C-index, REP30's percentile,
               and the fraction of random sets exceeding REP30.
     Two null hypotheses:
       Null A (unweighted): random 30-gene mean-z          -- the classic Venet null
       Null B (Cox-weighted on the same data): random 30 genes scored with Cox weights fitted within the same cohort
              -- stricter, exactly matching "REP30 selected genes on OS and fitted weights"

 (2) Within-cohort back-substitution + equal-degrees-of-freedom control
     (i) cross-fitting: fit the coefficients on TCGA only and apply them unchanged to METABRIC HR+/HER2-, and vice versa
     (ii) equal-df control: fit a Cox-weighted proliferation score (PROLIF_cox) from the same 12 proliferation genes in the same way,
         and put it in the same model as REP30, separating "gene content" from "fitting/weighting degrees of freedom"

Protocol (identical to analysis/subtype_rep30.py, for digit-by-digit reproducibility)
--------------------------------------------------------
  REP30 = sum coef*z      (coef from results/rep52_trim/rep52_trimmed_panel.csv)
  HR+/HER2-: METABRIC uses sample-level ER_STATUS/HER2_STATUS (IHC)
             TCGA has no IHC -> PAM50 LumA/LumB surrogate
  Cox: lifelines CoxPHFitter, covariates standardized as (x-mean)/sd -> HR per 1 SD
  C   : lifelines concordance_index_

Outputs
----
  results/manuscript_support/venet_null_draws.csv      per-draw results
  results/manuscript_support/venet_null_summary.csv    null-hypothesis summary
  results/manuscript_support/crossfit_results.csv      cross-fit / equal-df control
  results/manuscript_support/venet_run_log.txt         full log (incl. reproduction check)
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
os.makedirs(OUT, exist_ok=True)
LOGPATH = os.path.join(OUT, "venet_run_log.txt")

MB_DIR = os.path.join(ROOT, "data/01_core/METABRIC/brca_metabric")
TC_DIR = os.path.join(ROOT, "data/01_core/TCGA-BRCA/brca_tcga_pan_can_atlas_2018")
PANEL = os.path.join(ROOT, "results/rep52_trim/rep52_trimmed_panel.csv")

PROLIF = ["MKI67", "PCNA", "TOP2A", "CCNB1", "BUB1", "AURKA", "PLK1", "MCM2",
          "RRM2", "TYMS", "CDC20", "BIRC5"]

N_DRAW = 1000          # draws per null hypothesis
SEED = 20260921
NAN_MAX = 0.10         # max missing fraction for a gene to enter the sampling universe

from rep52_backtest import load_panel, load_z_subset, rep_score, to_event
from lifelines import CoxPHFitter

LOG = []


def log(s=""):
    print(s)
    LOG.append(str(s))
    with open(LOGPATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG) + "\n")


def _hook(t, v, tb):
    LOG.append("EXCEPTION:\n" + "".join(traceback.format_exception(t, v, tb)))
    with open(LOGPATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG) + "\n")


sys.excepthook = _hook


# ------------------------------------------------------------------ helpers
def read_cbio(path):
    with open(path, "rt", encoding="utf-8-sig", errors="replace") as fh:
        lines = fh.read().splitlines()
    data = [l for l in lines if l and not l.startswith("#")]
    return pd.read_csv(io.StringIO("\n".join(data)), sep="\t", low_memory=False)


def load_z_all(matrix_path, dtype=np.float32):
    """Read the full gene x sample z-score matrix (float32 to save memory)."""
    genes, rows = [], []
    with open(matrix_path, "rt", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        samples = header[2:]
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue
            g = parts[0]
            if g in genes:
                continue
            vals = [np.nan if x in ("", "NA", "NaN") else float(x) for x in parts[2:]]
            rows.append(np.array(vals, dtype=dtype))
            genes.append(g)
    mat = pd.DataFrame(np.vstack(rows), index=genes, columns=samples)
    return mat


def cox_psd(x, t, e):
    """Univariable Cox (covariate standardized -> per 1 SD). Returns a dict."""
    m = np.isfinite(x) & np.isfinite(t) & np.isfinite(e)
    x, t, e = x[m], t[m], e[m]
    if len(x) < 20 or e.sum() < 5:
        return None
    xs = (x - x.mean()) / x.std()
    dd = pd.DataFrame({"t": t, "e": e, "x": xs})
    try:
        cph = CoxPHFitter().fit(dd, "t", "e")
        return dict(n=len(x), events=int(e.sum()),
                    HR=float(np.exp(cph.params_["x"])),
                    lo=float(np.exp(cph.summary.loc["x", "coef lower 95%"])),
                    hi=float(np.exp(cph.summary.loc["x", "coef upper 95%"])),
                    p=float(cph.summary.loc["x", "p"]),
                    cindex=float(cph.concordance_index_))
    except Exception as ex:
        return dict(n=len(x), events=int(e.sum()), note=str(ex)[:80])


def cox_fit_beta(Z, t, e):
    """Multivariable Cox; returns (beta array, fit info). Z: genes x samples (DataFrame)."""
    X = Z.T.copy()
    X["t"] = t
    X["e"] = e
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    X = X[X["t"] > 0]
    if len(X) < 50 or X["e"].sum() < 20:
        return None, None
    cols = list(Z.index)
    dd = X[cols + ["t", "e"]].copy()
    for c in cols:
        s = dd[c].astype(float)
        dd[c] = (s - s.mean()) / s.std()
    try:
        cph = CoxPHFitter().fit(dd, "t", "e")
        beta = np.array([float(cph.params_[c]) for c in cols])
        return beta, dict(n=len(dd), events=int(dd["e"].sum()))
    except Exception as ex:
        return None, dict(n=len(dd), events=int(X["e"].sum()), note=str(ex)[:80])


def score_from_beta(Z, beta):
    """score = sum beta_i * z_i (column-aligned)."""
    return pd.Series(np.asarray(Z.values).T @ beta, index=Z.columns)


# ------------------------------------------------------------------ cohorts
def load_metabric(coef):
    z_path = os.path.join(MB_DIR, "data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt")
    log("  METABRIC: loading FULL z matrix (for random-gene universe) ...")
    t0 = time.time()
    mat = load_z_all(z_path)
    log(f"    full matrix {mat.shape[0]} genes x {mat.shape[1]} samples, "
        f"{time.time()-t0:.1f}s")

    rep = rep_score(mat, coef)
    S = pd.DataFrame({"REP30": rep})
    S.index = S.index.astype(str)
    S.index.name = "SAMPLE_ID"
    S = S.reset_index()                      # SAMPLE_ID becomes a column so the merge does not drop it
    S["patient"] = S["SAMPLE_ID"]            # METABRIC: SAMPLE_ID is PATIENT_ID

    cp = read_cbio(os.path.join(MB_DIR, "data_clinical_patient.txt"))
    cp.columns = [c.strip() for c in cp.columns]
    P = pd.DataFrame({
        "patient": cp["PATIENT_ID"].astype(str),
        "os_months": pd.to_numeric(cp["OS_MONTHS"], errors="coerce"),
        "os_status": cp["OS_STATUS"].astype(str).str.split(":").str[-1].str.strip(),
    })
    P["event"] = to_event(P["os_status"])

    cs = read_cbio(os.path.join(MB_DIR, "data_clinical_sample.txt"))
    cs.columns = [c.strip() for c in cs.columns]
    Sm = pd.DataFrame({
        "SAMPLE_ID": cs["SAMPLE_ID"].astype(str),
        "ER_STATUS": cs["ER_STATUS"].astype(str).str.strip(),
        "HER2_STATUS": cs["HER2_STATUS"].astype(str).str.strip(),
    })
    M = S.merge(Sm, on="SAMPLE_ID", how="inner").merge(P, on="patient", how="inner")
    M = M.rename(columns={"os_months": "t", "event": "e"})
    hrp = (M["ER_STATUS"].str.upper().str.startswith("POS")) & \
          (M["HER2_STATUS"].str.upper().str.startswith("NEG"))
    M["HRposHER2neg"] = hrp.astype(int)
    log(f"  METABRIC merged {M.shape[0]}; HR+/HER2- n={int(M['HRposHER2neg'].sum())}")
    return mat, M


def load_tcga(coef):
    z_path = os.path.join(TC_DIR, "data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt")
    wanted = set(coef) | set(PROLIF)
    mat = load_z_subset(z_path, wanted)
    rep = rep_score(mat, coef)
    S = pd.DataFrame({"REP30": rep})
    S.index = S.index.astype(str)
    S.index.name = "SAMPLE_ID"
    S = S.reset_index()                      # SAMPLE_ID becomes a column so the merge does not drop it
    S["patient"] = S["SAMPLE_ID"].str.replace(r"-01[ABC]?$", "", regex=True).str[:12]

    cl = read_cbio(os.path.join(TC_DIR, "data_clinical_patient.txt"))
    cl.columns = [c.strip() for c in cl.columns]
    C = pd.DataFrame({
        "patient": cl["PATIENT_ID"].astype(str),
        "SUBTYPE": cl["SUBTYPE"].astype(str),
        "os_months": pd.to_numeric(cl["OS_MONTHS"], errors="coerce"),
        "os_status": cl["OS_STATUS"].astype(str).str.split(":").str[-1].str.strip(),
    })
    C["event"] = to_event(C["os_status"])
    M = S.merge(C, on="patient", how="inner").rename(columns={"os_months": "t", "event": "e"})
    lum = M["SUBTYPE"].isin(["BRCA_LumA", "BRCA_LumB"])
    M["HRposHER2neg"] = lum.astype(int)
    log(f"  TCGA merged {M.shape[0]}; LumA/LumB (HR+/HER2- proxy) n={int(M['HRposHER2neg'].sum())}")
    return mat, M


# ------------------------------------------------------------------ main
def main():
    rng = np.random.default_rng(SEED)
    coef = load_panel(PANEL)
    genes30 = [g for g in coef]
    log(f"REP30 panel: {len(genes30)} genes")
    log("=" * 78)

    mat_mb, mb = load_metabric(coef)
    mat_tc, tc = load_tcga(coef)
    log("=" * 78)

    # ---------------- reproduction check (must match first, otherwise stop) ----------------
    sub = mb[mb["HRposHER2neg"] == 1].copy()
    sub = sub.dropna(subset=["REP30", "t", "e"])
    sub = sub[sub["t"] > 0]
    t_mb = sub["t"].values.astype(float)
    e_mb = sub["e"].values.astype(float)
    ref = cox_psd(sub["REP30"].values.astype(float), t_mb, e_mb)
    log("\n[REPRODUCE] METABRIC HR+/HER2- REP30 per-SD Cox:")
    log(f"   n={ref['n']}  events={ref['events']}  HR={ref['HR']:.4f} "
        f"[{ref['lo']:.3f},{ref['hi']:.3f}]  p={ref['p']:.3g}  C={ref['cindex']:.4f}")
    log("   expected (subtype_cox.csv): n=1398 events=812 HR=1.9289 C=0.6766")
    ok = (ref["n"] == 1398 and ref["events"] == 812
          and abs(ref["HR"] - 1.9289) < 0.01 and abs(ref["cindex"] - 0.6766) < 0.005)
    log(f"   REPRODUCTION {'OK' if ok else 'FAILED -- STOPPING'}")
    if not ok:
        log("!! reproduction failed; downstream results are untrustworthy; aborting.")
        return
    REP_C = ref["cindex"]
    REP_HR = ref["HR"]

    # ---------------- sampling universe ----------------
    if "SAMPLE_ID" not in sub.columns:
        log("!! merged frame is missing the SAMPLE_ID column")
        return
    cols = sub["SAMPLE_ID"].astype(str).tolist()
    miss = [c for c in cols if c not in set(mat_mb.columns)]
    if miss:
        log(f"!! {len(miss)}/{len(cols)} SAMPLE_IDs not found among the expression-matrix columns (e.g. {miss[:3]})")
        return
    Zall = mat_mb[cols]
    nanfrac = Zall.isna().mean(axis=1)
    sd = Zall.std(axis=1)
    universe = Zall.index[(nanfrac <= NAN_MAX) & (sd > 0)].tolist()
    log(f"\n[UNIVERSE] total genes={Zall.shape[0]}; "
        f"kept (NaN<={NAN_MAX:.0%} & sd>0)={len(universe)}")
    log(f"   REP30 panel within universe: {len([g for g in genes30 if g in set(universe)])}/{len(genes30)}")
    # marginal-match check: the z matrix is already per-gene z
    ug = Zall.loc[[g for g in universe[:5000]]]
    log(f"   marginal check (first 5000 genes): mean of gene-means={ug.mean(axis=1).mean():+.4f}, "
        f"SD of gene-SDs={ug.std(axis=1).std():.4f}  -> the z matrix is already marginally standardized; sampling matches margins automatically")

    idx = cols
    Zu = Zall.loc[universe, idx]
    Zuf = Zu.values.astype(float)
    Zuf = np.where(np.isfinite(Zuf), Zuf, 0.0)   # NaN -> 0 (= the gene mean, in z space)

    # ---------------- Null A: unweighted mean-z ----------------
    log(f"\n[NULL A] random 30-gene mean-z, {N_DRAW} draws ...")
    t0 = time.time()
    rowsA = []
    for i in range(N_DRAW):
        k = rng.choice(len(universe), size=30, replace=False)
        sc = np.nanmean(Zuf[k, :], axis=0)
        r = cox_psd(sc, t_mb, e_mb)
        if r and "cindex" in r:
            rowsA.append(dict(null="A_unweighted", draw=i, cindex=r["cindex"],
                              HR=r["HR"], p=r["p"], n_overlap_rep30=len(set(np.array(universe)[k]) & set(genes30))))
        if (i + 1) % 200 == 0:
            log(f"    {i+1}/{N_DRAW}  ({time.time()-t0:.0f}s)")
    A = pd.DataFrame(rowsA)
    log(f"    done {len(A)} draws in {time.time()-t0:.0f}s")

    # ---------------- Null B: Cox-weighted on the same data ----------------
    log(f"\n[NULL B] random 30 genes + Cox weights fitted within the same cohort, {N_DRAW} draws ...")
    t0 = time.time()
    rowsB = []
    for i in range(N_DRAW):
        k = rng.choice(len(universe), size=30, replace=False)
        gnames = [universe[j] for j in k]
        Zk = pd.DataFrame(Zuf[k, :], index=gnames, columns=idx)
        beta, info = cox_fit_beta(Zk, t_mb, e_mb)
        if beta is None:
            continue
        sc = (Zk.values.T @ beta)
        r = cox_psd(sc, t_mb, e_mb)
        if r and "cindex" in r:
            rowsB.append(dict(null="B_in_sample_cox_weighted", draw=i, cindex=r["cindex"],
                              HR=r["HR"], p=r["p"], n_overlap_rep30=len(set(gnames) & set(genes30))))
        if (i + 1) % 200 == 0:
            log(f"    {i+1}/{N_DRAW}  ({time.time()-t0:.0f}s)")
    B = pd.DataFrame(rowsB)
    log(f"    done {len(B)} draws in {time.time()-t0:.0f}s")

    draws = pd.concat([A, B], ignore_index=True)
    draws.to_csv(os.path.join(OUT, "venet_null_draws.csv"), index=False, encoding="utf-8")

    # ---------------- summary ----------------
    summ = []
    for nm, D in [("A_unweighted", A), ("B_in_sample_cox_weighted", B)]:
        c = D["cindex"].values
        h = D["HR"].values
        pct = float((c < REP_C).mean() * 100)
        exc = float((c >= REP_C).mean())
        log(f"\n[{nm}] n={len(c)}")
        log(f"   C-index : median={np.median(c):.4f}  "
            f"2.5th={np.percentile(c,2.5):.4f}  97.5th={np.percentile(c,97.5):.4f}  "
            f"max={c.max():.4f}")
        log(f"   HR/SD   : median={np.median(h):.3f}  "
            f"2.5th={np.percentile(h,2.5):.3f}  97.5th={np.percentile(h,97.5):.3f}")
        log(f"   REP30 C={REP_C:.4f} -> percentile {pct:.1f}th ; fraction of random sets >= REP30 = {exc:.4f} ({int(exc*len(c))}/{len(c)})")
        summ.append(dict(null=nm, n_draws=len(c),
                         c_median=float(np.median(c)),
                         c_p2p5=float(np.percentile(c, 2.5)),
                         c_p97p5=float(np.percentile(c, 97.5)),
                         c_max=float(c.max()),
                         hr_median=float(np.median(h)),
                         hr_p2p5=float(np.percentile(h, 2.5)),
                         hr_p97p5=float(np.percentile(h, 97.5)),
                         rep30_cindex=REP_C, rep30_HR=REP_HR,
                         rep30_percentile=pct,
                         prop_random_ge_rep30=exc))
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(OUT, "venet_null_summary.csv"), index=False, encoding="utf-8")

    # ---------------- (2) cross-fit ----------------
    log("\n" + "=" * 78)
    log("[CROSS-FIT] fit the coefficients on the training cohort only, apply them unchanged to the test cohort")

    tc_lum = tc[tc["HRposHER2neg"] == 1].dropna(subset=["REP30", "t", "e"])
    tc_lum = tc_lum[tc_lum["t"] > 0]
    g30_tc = [g for g in genes30 if g in mat_tc.index]
    g12_tc = [g for g in PROLIF if g in mat_tc.index]
    g30_mb = [g for g in genes30 if g in Zall.index]
    g12_mb = [g for g in PROLIF if g in Zall.index]
    log(f"   TCGA LumA/LumB n={len(tc_lum)} events={int(tc_lum['e'].sum())}; "
        f"panel {len(g30_tc)}/30, PROLIF {len(g12_tc)}/12")
    log(f"   METABRIC HR+/HER2- n={len(sub)} events={int(e_mb.sum())}; "
        f"panel {len(g30_mb)}/30, PROLIF {len(g12_mb)}/12")

    tc_ids = tc_lum["SAMPLE_ID"].astype(str).tolist()
    Z30_tc = mat_tc.loc[g30_tc, tc_ids]
    Z12_tc = mat_tc.loc[g12_tc, tc_ids]
    Z30_mb = Zall.loc[g30_mb, idx]
    Z12_mb = Zall.loc[g12_mb, idx]

    t_tc = tc_lum["t"].values.astype(float)
    e_tc = tc_lum["e"].values.astype(float)

    out = []

    def add(analysis, train, test, score_col, r, extra=""):
        out.append(dict(analysis=analysis, train=train, test=test, score=score_col,
                        n=r.get("n"), events=r.get("events"),
                        HR=r.get("HR"), lo=r.get("lo"), hi=r.get("hi"),
                        p=r.get("p"), cindex=r.get("cindex"), note=extra))

    # baseline: original REP30 (pooled panel coefficients)
    r = cox_psd(sub["REP30"].values.astype(float), t_mb, e_mb)
    add("REP30_original_pooled_coefs", "pooled TCGA+METABRIC (as published)",
        "METABRIC HR+/HER2-", "REP30", r, "in-sample w.r.t. METABRIC")
    r = cox_psd(tc_lum["REP30"].values.astype(float), t_tc, e_tc)
    add("REP30_original_pooled_coefs", "pooled TCGA+METABRIC (as published)",
        "TCGA LumA/LumB", "REP30", r, "in-sample w.r.t. TCGA")

    # (2a) cross-fit REP30: fit coefs on TCGA -> apply METABRIC
    beta30_t, info_t = cox_fit_beta(Z30_tc, t_tc, e_tc)
    log(f"   fit 30-gene Cox on TCGA LumA/LumB: {info_t}")
    if beta30_t is not None:
        sc = score_from_beta(Z30_mb, beta30_t)
        r = cox_psd(sc.reindex(idx).values.astype(float), t_mb, e_mb)
        add("REP30_crossfit", "TCGA LumA/LumB", "METABRIC HR+/HER2-", "REP30_TCGAcoefs", r,
            "coefs fitted on TCGA only, applied unchanged")
    # reverse: fit on METABRIC -> apply TCGA
    beta30_m, info_m = cox_fit_beta(Z30_mb, t_mb, e_mb)
    log(f"   fit 30-gene Cox on METABRIC HR+/HER2-: {info_m}")
    if beta30_m is not None:
        sc = score_from_beta(Z30_tc, beta30_m)
        r = cox_psd(sc.reindex(tc_ids).values.astype(float), t_tc, e_tc)
        add("REP30_crossfit", "METABRIC HR+/HER2-", "TCGA LumA/LumB", "REP30_MBcoefs", r,
            "coefs fitted on METABRIC only, applied unchanged")
        # in-sample reference (optimistic)
        sc_in = score_from_beta(Z30_mb, beta30_m)
        r_in = cox_psd(sc_in.reindex(idx).values.astype(float), t_mb, e_mb)
        add("REP30_insample_ref", "METABRIC HR+/HER2-", "METABRIC HR+/HER2-",
            "REP30_MBcoefs", r_in, "OPTIMISTIC in-sample reference")

    # (2b) equal-DF control: Cox-weighted 12-gene proliferation score
    gamma12_t, info_p = cox_fit_beta(Z12_tc, t_tc, e_tc)
    log(f"   fit 12-gene Cox on TCGA LumA/LumB: {info_p}")
    if gamma12_t is not None and beta30_t is not None:
        sc_p = score_from_beta(Z12_mb, gamma12_t)
        r = cox_psd(sc_p.reindex(idx).values.astype(float), t_mb, e_mb)
        add("PROLIFcox_equalDF", "TCGA LumA/LumB", "METABRIC HR+/HER2-",
            "PROLIF_cox(12 genes, Cox-weighted)", r,
            "SAME fitting procedure as REP30_crossfit; differs only in gene content (12 vs 30)")

        # joint model on METABRIC: REP30_TCGAcoefs + PROLIF_cox_TCGAcoefs
        a = score_from_beta(Z30_mb, beta30_t).reindex(idx).values.astype(float)
        b = sc_p.reindex(idx).values.astype(float)
        J = pd.DataFrame({"t": t_mb, "e": e_mb, "REP30cf": a, "PROLIFcf": b})
        J = J[J["t"] > 0].dropna()
        for c in ["REP30cf", "PROLIFcf"]:
            s = J[c].astype(float)
            J[c] = (s - s.mean()) / s.std()
        try:
            cph = CoxPHFitter().fit(J, "t", "e")
            log(f"\n   JOINT model on METABRIC HR+/HER2- (both TCGA-fitted), n={len(J)} ev={int(J['e'].sum())}:")
            for c in ["REP30cf", "PROLIFcf"]:
                log(f"      {c}: HR/SD={np.exp(cph.params_[c]):.3f} "
                    f"[{np.exp(cph.summary.loc[c,'coef lower 95%']):.3f},"
                    f"{np.exp(cph.summary.loc[c,'coef upper 95%']):.3f}] "
                    f"p={cph.summary.loc[c,'p']:.3g}")
            log(f"      model C={cph.concordance_index_:.4f}")
            out.append(dict(analysis="JOINT_REP30cf+PROLIFcf", train="TCGA LumA/LumB",
                            test="METABRIC HR+/HER2-", score="REP30cf", n=len(J),
                            events=int(J["e"].sum()),
                            HR=float(np.exp(cph.params_["REP30cf"])),
                            lo=float(np.exp(cph.summary.loc["REP30cf", "coef lower 95%"])),
                            hi=float(np.exp(cph.summary.loc["REP30cf", "coef upper 95%"])),
                            p=float(cph.summary.loc["REP30cf", "p"]),
                            cindex=float(cph.concordance_index_),
                            note="joint model; PROLIFcf HR="
                                 f"{np.exp(cph.params_['PROLIFcf']):.3f} p={cph.summary.loc['PROLIFcf','p']:.3g}"))
            out.append(dict(analysis="JOINT_REP30cf+PROLIFcf", train="TCGA LumA/LumB",
                            test="METABRIC HR+/HER2-", score="PROLIFcf", n=len(J),
                            events=int(J["e"].sum()),
                            HR=float(np.exp(cph.params_["PROLIFcf"])),
                            lo=float(np.exp(cph.summary.loc["PROLIFcf", "coef lower 95%"])),
                            hi=float(np.exp(cph.summary.loc["PROLIFcf", "coef upper 95%"])),
                            p=float(cph.summary.loc["PROLIFcf", "p"]),
                            cindex=float(cph.concordance_index_),
                            note="joint model; REP30cf HR="
                                 f"{np.exp(cph.params_['REP30cf']):.3f} p={cph.summary.loc['REP30cf','p']:.3g}"))
        except Exception as ex:
            log(f"   joint model failed: {str(ex)[:120]}")

    CF = pd.DataFrame(out)
    CF.to_csv(os.path.join(OUT, "crossfit_results.csv"), index=False, encoding="utf-8")
    log("\n[CROSS-FIT] summary table:")
    log(CF.to_string())
    log("\nDONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _hook(*sys.exc_info())
        raise
