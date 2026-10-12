#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
audit_venet_null4.py -- Null D: a random-signature null **with gene screening**

Why this is needed
----------
Nulls B/C let random sets fit Cox weights but involve **no gene screening**; yet the 52 parent genes of REP30
were themselves selected on overall survival. This is the most faithful layer of the Venet 2011 critique and
currently the only remaining point of attack (report section 5.1 concedes this).

Null D: rank all genes within METABRIC HR+/HER2- by the univariable Cox score-test statistic,
take the top-300 as the "already screened by outcome" pool; draw 30 genes from it each time, fit Cox weights,
and evaluate the in-sample C-index / HR -- i.e. all three layers (random genes + screening + fitting) are allowed.

(the full version would re-screen 17k genes univariably at every draw, which is prohibitively expensive; the top-300
pool is a one-shot pre-screened approximation, equivalent to "randomly pick 30 genes from those already outcome-associated".)

Outputs: venet_null_draws_D.csv, venet_null_summary_D.csv, log venet4_run_log.txt
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
LOGPATH = os.path.join(OUT, "venet4_run_log.txt")

MB_DIR = os.path.join(ROOT, "data/01_core/METABRIC/brca_metabric")
PANEL = os.path.join(ROOT, "results/rep52_trim/rep52_trimmed_panel.csv")

N_DRAW = 1000
SEED = 20260925
POOL = 300

from rep52_backtest import load_panel, rep_score, to_event
from audit_venet_null import (cox_fit_beta, score_from_beta, cox_psd, read_cbio,
                              load_z_all)

LOG = []


def log(s=""):
    print(s)
    LOG.append(str(s))
    with open(LOGPATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG) + "\n")


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
                      "os_status": cp["OS_STATUS"].astype(str).str.split(":").str[-1].str.strip()})
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
    log(f"  merged {M.shape[0]}; HR+/HER2- n={int(M['HRposHER2neg'].sum())}")
    return mat, M


def univariate_cox_z(X, t, e, chunk=2000):
    """Vectorized univariable Cox score-test statistic z (Breslow risk sets, >= t definition).

    X: genes x samples (np.float32, zeros-filled)
    Returns z = U / sqrt(V) for each gene.
    """
    n = len(t)
    order = np.argsort(t, kind="mergesort")      # ascending
    ts, es = t[order], e[order]
    ef = (es == 1)
    g = X.shape[0]
    zs = np.full(g, np.nan, dtype=np.float64)
    for s in range(0, g, chunk):
        Xe = np.asarray(X[s:s + chunk, order], dtype=np.float64)
        Xe = np.where(np.isfinite(Xe), Xe, 0.0)
        # suffix statistics (accumulated from the back): the risk set at position i = {j >= i}
        rev = Xe[:, ::-1]
        c1 = np.cumsum(rev, axis=1)[:, ::-1]           # sum_{j>=i} x
        c2 = np.cumsum(rev * rev, axis=1)[:, ::-1]     # sum_{j>=i} x^2
        cnt = np.arange(n, 0, -1, dtype=np.float64)    # n - i
        mean = c1 / cnt
        mean2 = c2 / cnt
        U = ((Xe - mean) * ef).sum(axis=1)
        V = ((mean2 - mean * mean) * ef).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            zs[s:s + chunk] = np.where(V > 0, U / np.sqrt(V), 0.0)
    return zs


def main():
    coef = load_panel(PANEL)
    genes30 = list(coef)
    log("=" * 78)
    log("[NULL D] random 30 genes drawn from the top-%d outcome-associated pool (screening + fitting both allowed), in-sample" % POOL)

    mat, mb = load_metabric(coef)
    sub = mb[mb["HRposHER2neg"] == 1].dropna(subset=["REP30", "t", "e"])
    sub = sub[sub["t"] > 0]
    idx = sub["SAMPLE_ID"].astype(str).tolist()
    t = sub["t"].values.astype(float)
    e = sub["e"].values.astype(float)
    log(f"   n={len(sub)} events={int(e.sum())}")

    Zall = mat[idx]
    X = np.asarray(Zall.values, dtype=np.float32)
    X = np.where(np.isfinite(X), X, 0.0)
    gnames = np.array(Zall.index)

    log("   computing univariate Cox score-test z for all %d genes ..." % len(gnames))
    t0 = time.time()
    zs = univariate_cox_z(X, t, e)
    log(f"      done in {time.time()-t0:.0f}s")

    ordr = np.argsort(-np.abs(zs))
    pool_idx = ordr[:POOL]
    pool_genes = list(gnames[pool_idx])
    pset = set(pool_genes)
    n_rep_in_pool = len([g for g in genes30 if g in pset])
    log(f"   pool = top {POOL} by |z|; |z| range in pool: "
        f"{np.abs(zs[pool_idx]).min():.2f} - {np.abs(zs[pool_idx]).max():.2f}")
    log(f"   REP30 panel inside the pool: {n_rep_in_pool}/30")
    rep_z = np.abs(zs[[int(np.where(gnames == g)[0][0]) for g in genes30 if g in set(gnames)]])
    log(f"   REP30 gene |z|: median={np.median(rep_z):.2f}, "
        f"min={rep_z.min():.2f}, max={rep_z.max():.2f}")

    rng = np.random.default_rng(SEED)
    rows = []
    log(f"\n[DRAWS] {N_DRAW} draws: sample 30 from pool -> fit Cox in-sample -> C, HR")
    t0 = time.time()
    for i in range(N_DRAW):
        k = rng.choice(POOL, size=30, replace=False)
        gsel = [pool_genes[j] for j in k]
        Zk = pd.DataFrame(X[pool_idx[k], :], index=gsel, columns=idx)
        beta, _ = cox_fit_beta(Zk, t, e)
        if beta is None:
            continue
        sc = score_from_beta(Zk, beta).reindex(idx).values.astype(float)
        r = cox_psd(sc, t, e)
        if r and "cindex" in r:
            rows.append(dict(null="D_selected_in_sample_cox_weighted", draw=i,
                             cindex=r["cindex"], HR=r["HR"], p=r["p"],
                             n_overlap_rep30=len(set(gsel) & set(genes30))))
        if (i + 1) % 200 == 0:
            log(f"    {i+1}/{N_DRAW}  ({time.time()-t0:.0f}s)")
    D = pd.DataFrame(rows)
    log(f"    done {len(D)} draws in {time.time()-t0:.0f}s")
    D.to_csv(os.path.join(OUT, "venet_null_draws_D.csv"), index=False, encoding="utf-8")

    c = D["cindex"].values
    h = D["HR"].values
    REP_C, REP_HR = 0.676562, 1.928457
    pct = float((c < REP_C).mean() * 100)
    log(f"\n[NULL D] n={len(c)}")
    log(f"   C-index : median={np.median(c):.4f}  2.5th={np.percentile(c,2.5):.4f}  "
        f"97.5th={np.percentile(c,97.5):.4f}  max={c.max():.4f}")
    log(f"   HR/SD   : median={np.median(h):.3f}  2.5th={np.percentile(h,2.5):.3f}  "
        f"97.5th={np.percentile(h,97.5):.3f}")
    log(f"   REP30 in-sample C={REP_C:.4f} -> Null D {pct:.1f}th pct ; "
        f"random >= REP30 = {(c>=REP_C).mean():.4f} ({int((c>=REP_C).sum())}/{len(c)})")
    log(f"   reference Null B (no screening, same-cohort fit) median C = 0.6004 ; this Null D median = {np.median(c):.4f}")

    pd.DataFrame([dict(null="D_selected_in_sample_cox_weighted", n_draws=len(c),
                       pool_top=POOL, c_median=float(np.median(c)),
                       c_p2p5=float(np.percentile(c, 2.5)), c_p97p5=float(np.percentile(c, 97.5)),
                       c_max=float(c.max()), hr_median=float(np.median(h)),
                       rep30_cindex=REP_C, rep30_HR=REP_HR,
                       rep30_percentile=pct,
                       prop_random_ge_rep30=float((c >= REP_C).mean()))]).to_csv(
        os.path.join(OUT, "venet_null_summary_D.csv"), index=False, encoding="utf-8")
    log("\nDONE")


if __name__ == "__main__":
    sys.excepthook = lambda a, b, tb: (
        LOG.append("EXCEPTION:\n" + "".join(traceback.format_exception(a, b, tb))),
        open(LOGPATH, "w", encoding="utf-8").write("\n".join(LOG) + "\n"))
    main()
