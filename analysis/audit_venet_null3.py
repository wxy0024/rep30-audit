#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
audit_venet_null3.py -- Null C: a cross-fit random-signature null

Why this is needed
----------
Null B is "random 30 genes + Cox weights fitted **within the same cohort**", an in-sample null,
and can only fairly be compared against the **in-sample** C=0.6766 of REP30.

But the honest (cross-cohort) estimate in section 2.4 is the cross-fit version:
    TCGA LumA/LumB fitted coefficients -> evaluated on METABRIC HR+/HER2-: HR=1.339, C=0.5823
Comparing that against the in-sample Null B (0.5823 sits at the 7.2nd percentile of Null B) is an
**asymmetric comparison** that a reviewer could justifiably push back on.

Null C: random 30 genes whose weights are fitted **only on TCGA LumA/LumB** and applied unchanged to
METABRIC HR+/HER2- -- exactly the same pipeline, training samples and evaluation samples as REP30_crossfit.
This is the fair null for the cross-fit REP30.

Sampling universe = genes shared by METABRIC and TCGA (otherwise cross-fitting is impossible).

(ships its own loader and log; does not overwrite the audit_venet_null / null2 logs)
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
LOGPATH = os.path.join(OUT, "venet3_run_log.txt")

MB_DIR = os.path.join(ROOT, "data/01_core/METABRIC/brca_metabric")
TC_DIR = os.path.join(ROOT, "data/01_core/TCGA-BRCA/brca_tcga_pan_can_atlas_2018")
PANEL = os.path.join(ROOT, "results/rep52_trim/rep52_trimmed_panel.csv")

PROLIF = ["MKI67", "PCNA", "TOP2A", "CCNB1", "BUB1", "AURKA", "PLK1", "MCM2",
          "RRM2", "TYMS", "CDC20", "BIRC5"]
N_DRAW = 1000
SEED = 20260923
NAN_MAX = 0.10

from rep52_backtest import load_panel, rep_score, to_event
from audit_venet_null import cox_psd, cox_fit_beta, score_from_beta, read_cbio, load_z_all

LOG = []


def log(s=""):
    print(s)
    LOG.append(str(s))
    with open(LOGPATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG) + "\n")


def load_metabric(coef):
    z_path = os.path.join(MB_DIR, "data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt")
    log("  METABRIC: loading FULL z matrix ...")
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
    log(f"  METABRIC merged {M.shape[0]}; HR+/HER2- n={int(M['HRposHER2neg'].sum())}")
    return mat, M


def load_tcga(coef):
    z_path = os.path.join(TC_DIR, "data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt")
    log("  TCGA: loading FULL z matrix ...")
    mat = load_z_all(z_path)
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
    log(f"  TCGA merged {M.shape[0]}; LumA/LumB n={int(M['HRposHER2neg'].sum())}")
    return mat, M


def main():
    rng = np.random.default_rng(SEED)
    coef = load_panel(PANEL)
    genes30 = list(coef)
    log("=" * 78)
    log("[NULL C] random 30 genes, weights fitted only on TCGA LumA/LumB -> applied to METABRIC HR+/HER2-")

    mat_mb, mb = load_metabric(coef)
    mat_tc, tc = load_tcga(coef)

    sub = mb[mb["HRposHER2neg"] == 1].dropna(subset=["REP30", "t", "e"])
    sub = sub[sub["t"] > 0]
    idx = sub["SAMPLE_ID"].astype(str).tolist()
    t_mb, e_mb = sub["t"].values.astype(float), sub["e"].values.astype(float)

    tc_lum = tc[tc["HRposHER2neg"] == 1].dropna(subset=["REP30", "t", "e"])
    tc_lum = tc_lum[tc_lum["t"] > 0]
    tc_ids = tc_lum["SAMPLE_ID"].astype(str).tolist()
    t_tc, e_tc = tc_lum["t"].values.astype(float), tc_lum["e"].values.astype(float)
    log(f"   train TCGA n={len(tc_lum)} ev={int(e_tc.sum())} | "
        f"test METABRIC n={len(sub)} ev={int(e_mb.sum())}")

    # ---- sampling universe: genes shared by both cohorts ----
    Zm = mat_mb[idx]
    Zt = mat_tc[tc_ids]
    ok_m = set(Zm.index[(Zm.isna().mean(axis=1) <= NAN_MAX) & (Zm.std(axis=1) > 0)])
    ok_t = set(Zt.index[(Zt.isna().mean(axis=1) <= NAN_MAX) & (Zt.std(axis=1) > 0)])
    universe = sorted(ok_m & ok_t)
    log(f"   METABRIC genes kept={len(ok_m)}; TCGA genes kept={len(ok_t)}; "
        f"intersection universe={len(universe)}")
    log(f"   REP30 panel within universe: {len([g for g in genes30 if g in set(universe)])}/{len(genes30)}")

    Zm_u = Zm.loc[universe].values.astype(float)
    Zm_u = np.where(np.isfinite(Zm_u), Zm_u, 0.0)
    Zt_u = Zt.loc[universe].values.astype(float)
    Zt_u = np.where(np.isfinite(Zt_u), Zt_u, 0.0)
    uset = set(universe)

    rows = []
    t0 = time.time()
    for i in range(N_DRAW):
        k = rng.choice(len(universe), size=30, replace=False)
        gnames = [universe[j] for j in k]
        Zk_t = pd.DataFrame(Zt_u[k, :], index=gnames, columns=tc_ids)
        beta, _ = cox_fit_beta(Zk_t, t_tc, e_tc)
        if beta is None:
            continue
        sc = Zm_u[k, :].T @ beta
        r = cox_psd(sc, t_mb, e_mb)
        if r and "cindex" in r:
            rows.append(dict(null="C_crossfit_cox_weighted", draw=i, cindex=r["cindex"],
                             HR=r["HR"], p=r["p"],
                             n_overlap_rep30=len(set(gnames) & set(genes30))))
        if (i + 1) % 200 == 0:
            log(f"    {i+1}/{N_DRAW}  ({time.time()-t0:.0f}s)")
    C = pd.DataFrame(rows)
    log(f"    done {len(C)} draws in {time.time()-t0:.0f}s")
    C.to_csv(os.path.join(OUT, "venet_null_draws_C.csv"), index=False, encoding="utf-8")

    c = C["cindex"].values
    h = C["HR"].values
    log(f"\n[NULL C] n={len(c)}")
    log(f"   C-index : median={np.median(c):.4f}  2.5th={np.percentile(c,2.5):.4f}  "
        f"97.5th={np.percentile(c,97.5):.4f}  max={c.max():.4f}")
    log(f"   HR/SD   : median={np.median(h):.3f}  2.5th={np.percentile(h,2.5):.3f}  "
        f"97.5th={np.percentile(h,97.5):.3f}")

    REP_C = 0.582285
    REP_HR = 1.338725
    pct = float((c < REP_C).mean() * 100)
    log(f"\n[POSITION of REP30_crossfit] C={REP_C:.4f} HR={REP_HR:.3f}")
    log(f"   Null C percentile = {pct:.1f}th ; random >= REP30cf = {(c>=REP_C).mean():.4f} "
        f"({int((c>=REP_C).sum())}/{len(c)})")
    log(f"   HR: random >= REP30cf HR = {(h>=REP_HR).mean():.4f} "
        f"({int((h>=REP_HR).sum())}/{len(h)})")

    summ = pd.DataFrame([dict(null="C_crossfit_cox_weighted", n_draws=len(c),
                              c_median=float(np.median(c)),
                              c_p2p5=float(np.percentile(c, 2.5)),
                              c_p97p5=float(np.percentile(c, 97.5)),
                              c_max=float(c.max()),
                              hr_median=float(np.median(h)),
                              hr_p2p5=float(np.percentile(h, 2.5)),
                              hr_p97p5=float(np.percentile(h, 97.5)),
                              rep30cf_cindex=REP_C, rep30cf_HR=REP_HR,
                              rep30cf_percentile=pct,
                              prop_random_ge_rep30cf=float((c >= REP_C).mean()))])
    summ.to_csv(os.path.join(OUT, "venet_null_summary_C.csv"), index=False, encoding="utf-8")
    log("\nDONE")


if __name__ == "__main__":
    sys.excepthook = lambda t, v, tb: (
        LOG.append("EXCEPTION:\n" + "".join(traceback.format_exception(t, v, tb))),
        open(LOGPATH, "w", encoding="utf-8").write("\n".join(LOG) + "\n"))
    main()
