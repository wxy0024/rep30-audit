#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
audit_venet_null2.py -- supplement: the reverse (METABRIC-trained -> TCGA-tested) equal-degrees-of-freedom control

Why this is needed
----------
The equal-df control in audit_venet_null.py was fitted on TCGA LumA/LumB, where the TCGA side
has only 88 events -- fitting 12-30 covariates there is high-dimensional noise fitting that depresses the transfer
performance of both REP30 and PROLIF_cox, leaving "too few training events" as an alternative explanation.
Here we run a **well-powered** reverse control: fit on METABRIC HR+/HER2- (812 events),
evaluate on TCGA LumA/LumB (88 events), with REP30(30 genes) and PROLIF_cox(12 genes)
going through exactly the same fit-evaluate pipeline.

Also: place the two C-index values of REP30 (in-sample 0.6766 / cross-fit 0.5823)
into the Null A / Null B distributions and report their percentiles.

(this script ships its own loaders and does not reuse the audit_venet_null log, to avoid overwriting the main log)
"""
import os
import sys
import io
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "manuscript_support")
os.makedirs(OUT, exist_ok=True)
LOGPATH = os.path.join(OUT, "venet2_run_log.txt")

MB_DIR = os.path.join(ROOT, "data/01_core/METABRIC/brca_metabric")
TC_DIR = os.path.join(ROOT, "data/01_core/TCGA-BRCA/brca_tcga_pan_can_atlas_2018")
PANEL = os.path.join(ROOT, "results/rep52_trim/rep52_trimmed_panel.csv")

PROLIF = ["MKI67", "PCNA", "TOP2A", "CCNB1", "BUB1", "AURKA", "PLK1", "MCM2",
          "RRM2", "TYMS", "CDC20", "BIRC5"]

from rep52_backtest import load_panel, load_z_subset, rep_score, to_event
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
    P = pd.DataFrame({
        "patient": cp["PATIENT_ID"].astype(str),
        "os_months": pd.to_numeric(cp["OS_MONTHS"], errors="coerce"),
        "os_status": cp["OS_STATUS"].astype(str).str.split(":").str[-1].str.strip()})
    P["event"] = to_event(P["os_status"])
    cs = read_cbio(os.path.join(MB_DIR, "data_clinical_sample.txt"))
    cs.columns = [c.strip() for c in cs.columns]
    Sm = pd.DataFrame({
        "SAMPLE_ID": cs["SAMPLE_ID"].astype(str),
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
    mat = load_z_subset(z_path, set(coef) | set(PROLIF))
    S = pd.DataFrame({"REP30": rep_score(mat, coef)})
    S.index = S.index.astype(str)
    S.index.name = "SAMPLE_ID"
    S = S.reset_index()
    S["patient"] = S["SAMPLE_ID"].str.replace(r"-01[ABC]?$", "", regex=True).str[:12]
    cl = read_cbio(os.path.join(TC_DIR, "data_clinical_patient.txt"))
    cl.columns = [c.strip() for c in cl.columns]
    C = pd.DataFrame({
        "patient": cl["PATIENT_ID"].astype(str),
        "SUBTYPE": cl["SUBTYPE"].astype(str),
        "os_months": pd.to_numeric(cl["OS_MONTHS"], errors="coerce"),
        "os_status": cl["OS_STATUS"].astype(str).str.split(":").str[-1].str.strip()})
    C["event"] = to_event(C["os_status"])
    M = S.merge(C, on="patient", how="inner").rename(columns={"os_months": "t", "event": "e"})
    M["HRposHER2neg"] = M["SUBTYPE"].isin(["BRCA_LumA", "BRCA_LumB"]).astype(int)
    log(f"  TCGA merged {M.shape[0]}; LumA/LumB n={int(M['HRposHER2neg'].sum())}")
    return mat, M


def main():
    coef = load_panel(PANEL)
    genes30 = list(coef)
    log("=" * 78)
    log("[EQUAL-DF, reverse] train METABRIC HR+/HER2- -> test TCGA LumA/LumB")

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
    log(f"   train METABRIC n={len(sub)} ev={int(e_mb.sum())} | "
        f"test TCGA n={len(tc_lum)} ev={int(e_tc.sum())}")

    g30_mb = [g for g in genes30 if g in mat_mb.index]
    g12_mb = [g for g in PROLIF if g in mat_mb.index]
    g30_tc = [g for g in genes30 if g in mat_tc.index]
    g12_tc = [g for g in PROLIF if g in mat_tc.index]
    Z30_mb, Z12_mb = mat_mb.loc[g30_mb, idx], mat_mb.loc[g12_mb, idx]
    Z30_tc, Z12_tc = mat_tc.loc[g30_tc, tc_ids], mat_tc.loc[g12_tc, tc_ids]

    out = []
    b30, i30 = cox_fit_beta(Z30_mb, t_mb, e_mb)
    b12, i12 = cox_fit_beta(Z12_mb, t_mb, e_mb)
    log(f"   fit REP30(30) on METABRIC: {i30}")
    log(f"   fit PROLIF(12) on METABRIC: {i12}")

    from lifelines import CoxPHFitter
    if b30 is not None:
        sc = score_from_beta(Z30_tc, b30).reindex(tc_ids)
        r = cox_psd(sc.values.astype(float), t_tc, e_tc)
        log(f"   REP30_MBcoefs   -> TCGA: HR={r['HR']:.3f} [{r['lo']:.3f},{r['hi']:.3f}] "
            f"p={r['p']:.3g} C={r['cindex']:.4f}")
        out.append(dict(analysis="REP30_crossfit_rev", train="METABRIC HR+/HER2-",
                        test="TCGA LumA/LumB", score="REP30_MBcoefs", n=r["n"],
                        events=r["events"], HR=r["HR"], lo=r["lo"], hi=r["hi"],
                        p=r["p"], cindex=r["cindex"], note="30 genes"))
    if b12 is not None:
        sc = score_from_beta(Z12_tc, b12).reindex(tc_ids)
        r = cox_psd(sc.values.astype(float), t_tc, e_tc)
        log(f"   PROLIFcox_MBcoefs-> TCGA: HR={r['HR']:.3f} [{r['lo']:.3f},{r['hi']:.3f}] "
            f"p={r['p']:.3g} C={r['cindex']:.4f}")
        out.append(dict(analysis="PROLIFcox_equalDF_rev", train="METABRIC HR+/HER2-",
                        test="TCGA LumA/LumB", score="PROLIF_cox_MBcoefs", n=r["n"],
                        events=r["events"], HR=r["HR"], lo=r["lo"], hi=r["hi"],
                        p=r["p"], cindex=r["cindex"],
                        note="12 genes, SAME fitting procedure as REP30"))

    if b30 is not None and b12 is not None:
        a = score_from_beta(Z30_tc, b30).reindex(tc_ids).values.astype(float)
        b = score_from_beta(Z12_tc, b12).reindex(tc_ids).values.astype(float)
        J = pd.DataFrame({"t": t_tc, "e": e_tc, "REP30cf": a, "PROLIFcf": b})
        J = J[J["t"] > 0].dropna()
        for c in ["REP30cf", "PROLIFcf"]:
            s = J[c].astype(float)
            J[c] = (s - s.mean()) / s.std()
        cph = CoxPHFitter().fit(J, "t", "e")
        log(f"\n   JOINT (rev) n={len(J)} ev={int(J['e'].sum())} C={cph.concordance_index_:.4f}")
        for c in ["REP30cf", "PROLIFcf"]:
            log(f"      {c}: HR/SD={np.exp(cph.params_[c]):.3f} "
                f"[{np.exp(cph.summary.loc[c,'coef lower 95%']):.3f},"
                f"{np.exp(cph.summary.loc[c,'coef upper 95%']):.3f}] p={cph.summary.loc[c,'p']:.3g}")
            out.append(dict(analysis="JOINT_rev", train="METABRIC HR+/HER2-",
                            test="TCGA LumA/LumB", score=c, n=len(J),
                            events=int(J["e"].sum()), HR=float(np.exp(cph.params_[c])),
                            lo=float(np.exp(cph.summary.loc[c, "coef lower 95%"])),
                            hi=float(np.exp(cph.summary.loc[c, "coef upper 95%"])),
                            p=float(cph.summary.loc[c, "p"]),
                            cindex=float(cph.concordance_index_), note="joint model (rev)"))

    # ---- where the two C values of REP30 sit in the random distributions ----
    try:
        D = pd.read_csv(os.path.join(OUT, "venet_null_draws.csv"))
        A = D[D["null"] == "A_unweighted"]["cindex"].values
        B = D[D["null"] == "B_in_sample_cox_weighted"]["cindex"].values
        log(f"\n[POSITIONING] Null A n={len(A)}, Null B n={len(B)}")
        for nm, v in [("REP30 in-sample C", 0.676562),
                      ("REP30 cross-fit C (TCGA coefs)", 0.582285)]:
            log(f"   {nm:34s} = {v:.4f}")
            for lab, arr in [("Null A", A), ("Null B", B)]:
                log(f"      {lab}: {float((arr < v).mean()*100):5.1f}th pct ; "
                    f"random >= REP30 = {(arr >= v).mean():.4f} ({int((arr>=v).sum())}/{len(arr)})")
    except Exception as ex:
        log(f"   positioning skipped: {str(ex)[:120]}")

    pd.DataFrame(out).to_csv(os.path.join(OUT, "crossfit_results_rev.csv"),
                             index=False, encoding="utf-8")
    log("\nDONE")


if __name__ == "__main__":
    sys.excepthook = lambda t, v, tb: (
        LOG.append("EXCEPTION:\n" + "".join(traceback.format_exception(t, v, tb))),
        open(LOGPATH, "w", encoding="utf-8").write("\n".join(LOG) + "\n"))
    main()
