#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
subtype_rep30.py  --  does REP30 hold within subtypes? Focus: HR+/HER2- (luminal)

Background
----
The pooled analysis (8 independent data sources) established: REP30 = a pan-proliferation/prognosis programme, **not a resistance marker**.
This script answers the follow-up: **if it cannot serve as a resistance marker globally, does it act within subtypes (especially HR+/HER2-)?**

Four-tier progressive design
-----------
 L-A subtype-stratified prognosis   TCGA (PAM50 SUBTYPE) / METABRIC (ER_STATUS+HER2_STATUS, CLAUDIN_SUBTYPE)
      Cox per subtype (per 1 SD) -> is the prognostic value of REP30 subtype-dependent? With interaction tests (LR test).
 L-B is it just proliferation   within HR+/HER2-
      - pairwise correlations of REP30 / PROLIF (12 genes) / ER_TARGETS (20 genes) + a univariable Cox for each
      - multivariable Cox: REP30 + PROLIF (+ age/stage) -> is REP30 independent of proliferation
      - **decisive test**: within the "low proliferation" tertile, can REP30 still separate survival?
        (if it fails in the low-proliferation group -> within this subtype REP30 is also just a proliferation surrogate)
      - C-index comparison: PROLIF alone / REP30 alone / both
 L-C resistance association       within HR+/HER2-: REP30 vs ESR1/RB1 CNA-mutations + endocrine-treatment exposure
 L-D external validation       repeat the L-B decisive test within 4 ER+ endocrine-treated GEO cohorts

Scoring protocol (consistent across the whole project)
----------------------
  REP30  = sum coef*z         (coef from results/rep52_trim/rep52_trimmed_panel.csv)
  PROLIF = mean z(12 proliferation genes)
  ER_act = mean z(20 direct ER target genes)     <- deliberately excludes ESR1/FOXA1/GATA3, reading "pathway output" rather than lineage
  TCGA/METABRIC: cBioPortal precomputed per-gene z matrices used directly
  GEO: within-cohort per-gene z

Outputs: results/subtype_rep30/{subtype_cox.csv, interaction_tests.csv, hrposher2neg_*.csv,
        score_correlations.csv, summary.md, fig_subtype_rep30.png, run_log.txt}
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
OUT = os.path.join(ROOT, "results", "subtype_rep30")
os.makedirs(OUT, exist_ok=True)

MB_DIR = os.path.join(ROOT, "data/01_core/METABRIC/brca_metabric")
TC_DIR = os.path.join(ROOT, "data/01_core/TCGA-BRCA/brca_tcga_pan_can_atlas_2018")
PANEL = os.path.join(ROOT, "results/rep52_trim/rep52_trimmed_panel.csv")

from rep52_backtest import load_panel, load_z_subset, rep_score, to_event

try:
    from lifelines import CoxPHFitter
    from lifelines.statistics import logrank_test
    HAVE_LIFELINES = True
except Exception as _e:
    HAVE_LIFELINES = False

# ------------------------------------------------------------------ logging
LOG = []


def log(s=""):
    print(s)
    LOG.append(str(s))


def flush():
    with open(os.path.join(OUT, "run_log.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG))


def _hook(t, v, tb):
    LOG.append("EXCEPTION:\n" + "".join(traceback.format_exception(t, v, tb)))
    flush()


sys.excepthook = _hook

# ------------------------------------------------------------------ gene sets
PROLIF = ["MKI67", "PCNA", "TOP2A", "CCNB1", "BUB1", "AURKA", "PLK1", "MCM2",
          "RRM2", "TYMS", "CDC20", "BIRC5"]
ER_TARGETS = ["TFF1", "TFF3", "GREB1", "PGR", "PRLR", "PDZK1", "CA12", "SLC39A6",
              "NPY1R", "MYB", "XBP1", "SERPINA3", "GPR160", "HMGCS2", "IL6ST",
              "MYBL1", "HSPB8", "KCNK5", "SCUBE2", "CTSD"]


def read_cbio(path):
    with open(path, "rt", encoding="utf-8-sig", errors="replace") as fh:
        lines = fh.read().splitlines()
    data = [l for l in lines if l and not l.startswith("#")]
    return pd.read_csv(io.StringIO("\n".join(data)), sep="\t", low_memory=False)


def zmean(mat, genes):
    g = [x for x in genes if x in mat.index]
    if not g:
        return None, []
    return mat.loc[g].mean(axis=0), g


# ------------------------------------------------------------------ Cox utils
def prep(df, cols, tcol="t", ecol="e"):
    d = df[[c for c in cols] + [tcol, ecol]].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    d = d[d[tcol] > 0]
    return d


def uni_cox(df, col, tcol="t", ecol="e", per_sd=True):
    """Univariable Cox. per_sd=True -> HR per 1 SD. Returns a dict."""
    d = prep(df, [col], tcol, ecol)
    n, ev = len(d), int(d[ecol].sum()) if len(d) else 0
    if n < 20 or ev < 5:
        return dict(n=n, events=ev, note="too few")
    x = d[col].astype(float)
    x = (x - x.mean()) / x.std() if per_sd else x
    dd = pd.DataFrame({"t": d[tcol].values, "e": d[ecol].values, "x": x.values})
    try:
        cph = CoxPHFitter().fit(dd, "t", "e")
        return dict(n=n, events=ev,
                    HR=float(np.exp(cph.params_["x"])),
                    lo=float(np.exp(cph.summary.loc["x", "coef lower 95%"])),
                    hi=float(np.exp(cph.summary.loc["x", "coef upper 95%"])),
                    p=float(cph.summary.loc["x", "p"]),
                    cindex=float(cph.concordance_index_))
    except Exception as ex:
        return dict(n=n, events=ev, note=str(ex)[:60])


def multi_cox(df, cols, tcol="t", ecol="e"):
    d = prep(df, cols, tcol, ecol)
    n, ev = len(d), int(d[ecol].sum()) if len(d) else 0
    if n < 30 or ev < 10:
        return None
    dd = pd.DataFrame({"t": d[tcol].values, "e": d[ecol].values})
    for c in cols:
        x = d[c].astype(float)
        dd[c] = ((x - x.mean()) / x.std()).values
    try:
        cph = CoxPHFitter().fit(dd, "t", "e")
        r = {c: dict(HR=float(np.exp(cph.params_[c])),
                     lo=float(np.exp(cph.summary.loc[c, "coef lower 95%"])),
                     hi=float(np.exp(cph.summary.loc[c, "coef upper 95%"])),
                     p=float(cph.summary.loc[c, "p"])) for c in cols}
        r["_cindex"] = float(cph.concordance_index_)
        r["_n"], r["_events"] = n, ev
        return r
    except Exception:
        return None


def interaction_lrt(df, score_col, grp_col, tcol="t", ecol="e"):
    """LR test for the score x binary-group interaction. Returns (p_int, HR_int)."""
    d = prep(df, [score_col, grp_col], tcol, ecol)
    if len(d) < 40 or d[ecol].sum() < 15:
        return None, None
    dd = pd.DataFrame({"t": d[tcol].values, "e": d[ecol].values})
    for c in (score_col, grp_col):
        x = d[c].astype(float)
        dd[c] = ((x - x.mean()) / x.std()).values
    dd["score"] = dd[score_col]
    dd["grp"] = dd[grp_col]
    dd["sc_x_grp"] = dd["score"] * dd["grp"]
    try:
        full = CoxPHFitter().fit(dd[["t", "e", "score", "grp", "sc_x_grp"]], "t", "e")
        red = CoxPHFitter().fit(dd[["t", "e", "score", "grp"]], "t", "e")
        from scipy import stats as _st
        lr = 2.0 * (full.log_likelihood_ - red.log_likelihood_)
        p = float(_st.chi2.sf(max(lr, 0.0), 1))
        return p, float(np.exp(full.params_["sc_x_grp"]))
    except Exception:
        return None, None


# ============================================================== cohort loading
def load_tcga(coef):
    """TCGA-BRCA: z matrix + patient clinical (SUBTYPE/OS). Returns (score_df, clin_df, subtype_series)."""
    z_path = os.path.join(TC_DIR, "data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt")
    wanted = set(coef) | set(PROLIF) | set(ER_TARGETS)
    log("  TCGA: loading z-subset ...")
    mat = load_z_subset(z_path, wanted)
    rep = rep_score(mat, coef)
    prolif, pg = zmean(mat, PROLIF)
    eract, eg = zmean(mat, ER_TARGETS)
    S = pd.DataFrame({"REP30": rep, "PROLIF": prolif, "ER_act": eract})
    S["MKI67"] = mat.loc["MKI67"] if "MKI67" in mat.index else np.nan
    S.index = S.index.astype(str)
    S.index.name = "SAMPLE_ID"

    cl = read_cbio(os.path.join(TC_DIR, "data_clinical_patient.txt"))
    cl.columns = [c.strip() for c in cl.columns]
    C = pd.DataFrame({
        "patient": cl["PATIENT_ID"].astype(str),
        "SUBTYPE": cl["SUBTYPE"].astype(str),
        "os_months": pd.to_numeric(cl["OS_MONTHS"], errors="coerce"),
        "os_status": cl["OS_STATUS"].astype(str).str.split(":").str[-1].str.strip(),
        "AGE": pd.to_numeric(cl.get("AGE"), errors="coerce"),
        "STAGE": cl.get("AJCC_PATHOLOGIC_TUMOR_STAGE", pd.Series(index=cl.index)).astype(str),
        "GRADE": pd.Series([""] * len(cl)),
    })
    C["event"] = to_event(C["os_status"])

    S2 = S.copy()
    S2["patient"] = S2.index.str.replace(r"-01[ABC]?$", "", regex=True).str[:12]
    M = S2.merge(C, on="patient", how="inner")
    M = M.rename(columns={"os_months": "t", "event": "e"})
    # TCGA has no IHC -> PAM50 LumA/LumB as the HR+/HER2- surrogate
    lum = M["SUBTYPE"].isin(["BRCA_LumA", "BRCA_LumB"])
    M["HRposHER2neg"] = lum.astype(int)
    M["STAGE_NUM"] = M["STAGE"].map(stage_num)
    log(f"  TCGA: scored {S.shape[0]} samples; merged {M.shape[0]}; "
        f"REP30 panel {len([g for g in coef if g in mat.index])}/{len(coef)}; "
        f"PROLIF {len(pg)}; ER {len(eg)}")
    log(f"  TCGA SUBTYPE counts: {dict(M['SUBTYPE'].value_counts())}")
    return M


def load_metabric(coef):
    """METABRIC: z matrix + patient (OS/CLAUDIN_SUBTYPE/ER_IHC) + sample (ER/HER2_STATUS)."""
    z_path = os.path.join(MB_DIR, "data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt")
    wanted = set(coef) | set(PROLIF) | set(ER_TARGETS)
    log("  METABRIC: loading z-subset ...")
    mat = load_z_subset(z_path, wanted)
    rep = rep_score(mat, coef)
    prolif, pg = zmean(mat, PROLIF)
    eract, eg = zmean(mat, ER_TARGETS)
    S = pd.DataFrame({"REP30": rep, "PROLIF": prolif, "ER_act": eract})
    S["MKI67"] = mat.loc["MKI67"] if "MKI67" in mat.index else np.nan
    S.index = S.index.astype(str)
    S.index.name = "SAMPLE_ID"
    S["patient"] = S.index

    cp = read_cbio(os.path.join(MB_DIR, "data_clinical_patient.txt"))
    cp.columns = [c.strip() for c in cp.columns]
    P = pd.DataFrame({
        "patient": cp["PATIENT_ID"].astype(str),
        "CLAUDIN_SUBTYPE": cp["CLAUDIN_SUBTYPE"].astype(str),
        "ER_IHC": cp["ER_IHC"].astype(str),
        "HER2_SNP6": cp["HER2_SNP6"].astype(str),
        "os_months": pd.to_numeric(cp["OS_MONTHS"], errors="coerce"),
        "os_status": cp["OS_STATUS"].astype(str).str.split(":").str[-1].str.strip(),
        "AGE": pd.to_numeric(cp["AGE_AT_DIAGNOSIS"], errors="coerce"),
        "NPI": pd.to_numeric(cp.get("NPI"), errors="coerce"),
        "HT": cp.get("HORMONE_THERAPY", pd.Series(index=cp.index)).astype(str),
        "CT": cp.get("CHEMOTHERAPY", pd.Series(index=cp.index)).astype(str),
    })
    P["event"] = to_event(P["os_status"])

    cs = read_cbio(os.path.join(MB_DIR, "data_clinical_sample.txt"))
    cs.columns = [c.strip() for c in cs.columns]
    Sm = pd.DataFrame({
        "SAMPLE_ID": cs["SAMPLE_ID"].astype(str),
        "ER_STATUS": cs["ER_STATUS"].astype(str).str.strip(),
        "HER2_STATUS": cs["HER2_STATUS"].astype(str).str.strip(),
        "PR_STATUS": cs["PR_STATUS"].astype(str).str.strip(),
        "GRADE": cs["GRADE"].astype(str),
    })

    M = S.merge(Sm, on="SAMPLE_ID", how="inner").merge(P, on="patient", how="inner")
    M = M.rename(columns={"os_months": "t", "event": "e"})
    hrp = (M["ER_STATUS"].str.upper().str.startswith("POS")) & \
          (M["HER2_STATUS"].str.upper().str.startswith("NEG"))
    lum = M["CLAUDIN_SUBTYPE"].isin(["LumA", "LumB"])
    M["HRposHER2neg"] = hrp.astype(int)
    M["luminal_pam50"] = lum.astype(int)
    log(f"  METABRIC: scored {S.shape[0]}; merged {M.shape[0]}; "
        f"REP30 panel {len([g for g in coef if g in mat.index])}/{len(coef)}; "
        f"PROLIF {len(pg)}; ER {len(eg)}")
    log(f"  METABRIC ER_STATUS: {dict(M['ER_STATUS'].value_counts())}")
    log(f"  METABRIC HER2_STATUS: {dict(M['HER2_STATUS'].value_counts())}")
    log(f"  METABRIC CLAUDIN_SUBTYPE: {dict(M['CLAUDIN_SUBTYPE'].value_counts())}")
    log(f"  METABRIC HR+/HER2- n={int(M['HRposHER2neg'].sum())}")
    return M


# ============================================================== analyses
def subtype_strata(coh_name, M, subtype_col, values, label_map=None):
    """Univariable REP30 Cox within each subtype."""
    rows = []
    for v in values:
        sub = M[M[subtype_col] == v]
        r = uni_cox(sub, "REP30")
        if r:
            nm = (label_map or {}).get(v, v)
            r = dict(cohort=coh_name, subtype_col=subtype_col, subtype=nm, **r)
            rows.append(r)
    return rows


def resistance_labels_hrpos(M, coh_name):
    """REP30 vs resistance/exposure labels within HR+/HER2-."""
    rows = []
    sub = M[M["HRposHER2neg"] == 1].copy()
    if len(sub) < 30:
        return rows
    # treatment exposure (METABRIC has HORMONE_THERAPY/CHEMOTHERAPY)
    for col, lab in [("HT", "hormone_therapy_exposed")]:
        if col in sub.columns:
            v = sub[col].astype(str).str.upper()
            if v.nunique() > 1 and v.isin(["YES", "NO", "TRUE", "FALSE"]).any():
                g = v.isin(["YES", "TRUE"]).astype(int)
                a = sub.loc[g == 1, "REP30"].values
                b = sub.loc[g == 0, "REP30"].values
                if len(a) >= 5 and len(b) >= 5:
                    from scipy import stats as _st
                    u = _st.mannwhitneyu(a, b, alternative="two-sided")
                    rows.append(dict(cohort=coh_name, label=lab, n_label=len(a), n_ctrl=len(b),
                                     median_label=float(np.median(a)), median_ctrl=float(np.median(b)),
                                     delta=float(np.median(a) - np.median(b)), p=float(u.pvalue)))
    return rows


def tertile_test(M, coh_name, tag, which):
    """After stratifying by PROLIF tertiles, can REP30 still separate survival (decisive test)?"""
    sub = M.dropna(subset=["REP30", "PROLIF", "t", "e"]).copy()
    if len(sub) < 60:
        return None
    q1, q2 = sub["PROLIF"].quantile([1 / 3.0, 2 / 3.0])
    if which == "low":
        s, lab = sub[sub["PROLIF"] <= q1], "PROLIF-low tertile"
    elif which == "high":
        s, lab = sub[sub["PROLIF"] >= q2], "PROLIF-high tertile"
    else:
        s, lab = sub[(sub["PROLIF"] > q1) & (sub["PROLIF"] < q2)], "PROLIF-mid tertile"
    r = uni_cox(s, "REP30")
    if not r:
        return None
    return dict(cohort=coh_name, stratum=f"{tag} · {lab}", **r)


def stage_num(x):
    s = str(x).upper()
    for k, v in [("IV", 4), ("III", 3), ("II", 2), ("I", 1)]:
        if k in s.replace("STAGE", "").strip():
            return v
    return np.nan


def main():
    if not HAVE_LIFELINES:
        log("FATAL: lifelines not available")
        return
    coef = load_panel(PANEL)
    log(f"REP30 panel: {len(coef)} genes")
    log("=" * 78)

    tcga = load_tcga(coef)
    mb = load_metabric(coef)
    log("=" * 78)

    # ---------------- L-A subtype-stratified prognosis ----------------
    rows = []
    rows += subtype_strata("TCGA", tcga, "SUBTYPE",
                           ["BRCA_LumA", "BRCA_LumB", "BRCA_Her2", "BRCA_Basal", "BRCA_Normal"],
                           {"BRCA_LumA": "LumA", "BRCA_LumB": "LumB", "BRCA_Her2": "HER2-enriched",
                            "BRCA_Basal": "Basal", "BRCA_Normal": "Normal-like"})
    rows += subtype_strata("METABRIC", mb, "CLAUDIN_SUBTYPE",
                           ["LumA", "LumB", "Her2", "Basal", "claudin-low", "Normal"])
    rows += subtype_strata("METABRIC", mb, "HRposHER2neg", [1, 0],
                           {1: "HR+/HER2- (IHC)", 0: "other (all subtypes)"})
    SX = pd.DataFrame(rows)
    SX.to_csv(os.path.join(OUT, "subtype_cox.csv"), index=False, encoding="utf-8")
    log("\n[L-A] subtype-stratified REP30 univariable Cox (per 1 SD)")
    for _, r in SX.iterrows():
        if "HR" in r and np.isfinite(r.get("HR", np.nan)):
            log(f"   {r['cohort']:9s} {str(r['subtype']):22s} n={int(r['n']):5d} ev={int(r['events']):4d} "
                f"HR={r['HR']:.2f} [{r['lo']:.2f},{r['hi']:.2f}] p={r['p']:.2g} C={r['cindex']:.3f}")
        else:
            log(f"   {r['cohort']:9s} {str(r['subtype']):22s} n={int(r.get('n', 0)):5d} "
                f"ev={int(r.get('events', 0)):4d}  -> {r.get('note', 'n/a')}")

    # interaction tests: TCGA PAM50 luminal vs non-luminal; METABRIC ER/HER2
    ia = []
    for coh, M, sc, gp, lab in [
            ("TCGA", tcga, "REP30", "HRposHER2neg", "LumA/LumB vs rest"),
            ("METABRIC", mb, "REP30", "HRposHER2neg", "ER+/HER2- vs rest"),
            ("METABRIC", mb, "REP30", "luminal_pam50", "PAM50 LumA/B vs rest")]:
        p, hr = interaction_lrt(M, sc, gp)
        ia.append(dict(cohort=coh, score=sc, grouping=lab, p_interaction=p, HR_interaction=hr))
        log(f"[L-A] interaction {coh} {sc} x {lab}: p_int={p if p is None else round(p,3)} HR_int={None if hr is None else round(hr,2)}")
    pd.DataFrame(ia).to_csv(os.path.join(OUT, "interaction_tests.csv"), index=False, encoding="utf-8")

    # ---------------- L-B is it just proliferation ----------------
    corr_rows, mv_rows, lp_rows = [], [], []
    CLIN = {"METABRIC": ["AGE", "NPI"], "METABRIC_lumA_B": ["AGE", "NPI"],
            "TCGA": ["AGE", "STAGE_NUM"]}
    for coh, M, mask_col in [("TCGA", tcga, "HRposHER2neg"), ("METABRIC", mb, "HRposHER2neg"),
                             ("METABRIC_lumA_B", mb, "luminal_pam50")]:
        sub = M[M[mask_col] == 1].copy()
        tag = f"{coh}·HR+/HER2-" if mask_col == "HRposHER2neg" else f"{coh}·PAM50 luminal"
        if len(sub) < 30:
            continue
        from scipy import stats as _st
        for a, b in [("REP30", "PROLIF"), ("REP30", "ER_act"), ("REP30", "MKI67"),
                     ("PROLIF", "ER_act")]:
            m = sub[[a, b]].dropna()
            r, p = _st.spearmanr(m[a], m[b])
            corr_rows.append(dict(stratum=tag, a=a, b=b, spearman=float(r), p=float(p), n=len(m)))
        log(f"\n[L-B] {tag}  (n={len(sub)}, events={int(sub['e'].sum())})")
        for rr in corr_rows[-4:]:
            log(f"   Spearman {rr['a']}~{rr['b']} = {rr['spearman']:+.3f} (p={rr['p']:.2g}, n={rr['n']})")
        for sc in ["REP30", "PROLIF", "MKI67", "ER_act"]:
            r = uni_cox(sub, sc)
            if r and "HR" in r:
                log(f"   univariable {sc:7s} HR/SD={r['HR']:.2f} [{r['lo']:.2f},{r['hi']:.2f}] "
                    f"p={r['p']:.2g} C={r['cindex']:.3f}")
            elif r:
                log(f"   univariable {sc:7s} -> {r.get('note')}")
        # multivariable: REP30 + proliferation comparator
        for partner in ["PROLIF", "MKI67"]:
            mv = multi_cox(sub, ["REP30", partner])
            if mv:
                log(f"   multivariable REP30+{partner}: REP30 HR={mv['REP30']['HR']:.2f} p={mv['REP30']['p']:.2g} | "
                    f"{partner} HR={mv[partner]['HR']:.2f} p={mv[partner]['p']:.2g} | C={mv['_cindex']:.3f}")
                mv_rows.append(dict(stratum=tag, model=f"REP30+{partner}", n=mv["_n"],
                                    events=mv["_events"],
                                    REP30_HR=mv["REP30"]["HR"], REP30_lo=mv["REP30"]["lo"],
                                    REP30_hi=mv["REP30"]["hi"], REP30_p=mv["REP30"]["p"],
                                    partner=partner, partner_HR=mv[partner]["HR"],
                                    partner_p=mv[partner]["p"], cindex=mv["_cindex"]))
        # multivariable: REP30 + clinical (age + NPI/stage)
        cc = [c for c in CLIN.get(coh, []) if c in sub.columns]
        if cc and sub[cc].notna().all(axis=None) is not None:
            mv2 = multi_cox(sub, ["REP30"] + cc)
            if mv2:
                parts = " | ".join(f"{c} HR={mv2[c]['HR']:.2f} p={mv2[c]['p']:.2g}" for c in cc)
                log(f"   multivariable REP30+clinical({'+'.join(cc)}): REP30 HR={mv2['REP30']['HR']:.2f} "
                    f"p={mv2['REP30']['p']:.2g} | {parts} | C={mv2['_cindex']:.3f}")
                mv_rows.append(dict(stratum=tag, model="REP30+clinical", n=mv2["_n"],
                                    events=mv2["_events"],
                                    REP30_HR=mv2["REP30"]["HR"], REP30_lo=mv2["REP30"]["lo"],
                                    REP30_hi=mv2["REP30"]["hi"], REP30_p=mv2["REP30"]["p"],
                                    partner="+".join(cc),
                                    partner_HR=np.nan, partner_p=np.nan,
                                    cindex=mv2["_cindex"]))
        # proliferation-tertile test (low/mid/high all reported; the key is whether the low-proliferation group remains effective)
        for which in ["low", "mid", "high"]:
            lp = tertile_test(sub, coh, tag, which)
            if lp:
                lp_rows.append(lp)
                log(f"   [{lp['stratum'].split('· ')[-1]}] REP30: n={lp['n']} ev={lp['events']} "
                    f"HR={lp.get('HR', float('nan')):.2f} p={lp.get('p', float('nan')):.2g} "
                    f"C={lp.get('cindex', float('nan')):.3f}")
    pd.DataFrame(corr_rows).to_csv(os.path.join(OUT, "score_correlations.csv"), index=False, encoding="utf-8")
    pd.DataFrame(mv_rows).to_csv(os.path.join(OUT, "hrposher2neg_multivariable.csv"), index=False, encoding="utf-8")
    pd.DataFrame(lp_rows).to_csv(os.path.join(OUT, "hrposher2neg_prolif_tertiles.csv"), index=False, encoding="utf-8")

    # ---------------- L-C resistance association ----------------
    res_rows = []
    for coh, M in [("METABRIC", mb)]:
        res_rows += resistance_labels_hrpos(M, coh)
    if res_rows:
        pd.DataFrame(res_rows).to_csv(os.path.join(OUT, "resistance_labels_hrposHER2neg.csv"),
                                      index=False, encoding="utf-8")
        log("\n[L-C] treatment exposure vs REP30 within HR+/HER2-")
        for r in res_rows:
            log(f"   {r['cohort']} {r['label']}: n={r['n_label']}/{r['n_ctrl']} "
                f"Δmedian={r['delta']:+.3f} p={r['p']:.3g}")

    # ---------------- L-B2 incremental C-index value ----------------
    log("\n[L-B2] C-index of each predictor within HR+/HER2- (which is stronger)")
    cin_rows = []
    for coh, M, mask_col in [("METABRIC", mb, "HRposHER2neg")]:
        sub = M[M[mask_col] == 1].copy()
        for nm, col, ps in [("NPI (clinical)", "NPI", True), ("Age", "AGE", True),
                            ("PROLIF (12-gene mean)", "PROLIF", True), ("MKI67 alone", "MKI67", True),
                            ("REP30", "REP30", True)]:
            if col in sub.columns:
                rr = uni_cox(sub, col, per_sd=ps)
                if rr and "cindex" in rr:
                    cin_rows.append(dict(stratum="METABRIC·HR+/HER2-", predictor=nm,
                                         n=rr["n"], cindex=rr["cindex"], HR=rr["HR"], p=rr["p"]))
                    log(f"   {nm:24s} C={rr['cindex']:.3f}  HR/SD={rr['HR']:.2f} p={rr['p']:.2g}")
        # PAM50 LumA/LumB as the 'clinical genomic stratification' comparator
        if "luminal_pam50" in sub.columns:
            rr = uni_cox(sub, "luminal_pam50", per_sd=False)
            if rr and "cindex" in rr:
                cin_rows.append(dict(stratum="METABRIC·HR+/HER2-", predictor="PAM50 LumA vs LumB",
                                     n=rr["n"], cindex=rr["cindex"], HR=rr["HR"], p=rr["p"]))
                log(f"   {'PAM50 LumA vs LumB':24s} C={rr['cindex']:.3f}  HR={rr['HR']:.2f} p={rr['p']:.2g}")
    if cin_rows:
        pd.DataFrame(cin_rows).to_csv(os.path.join(OUT, "hrposher2neg_cindex.csv"),
                                      index=False, encoding="utf-8")

    # ---------------- L-D external validation in ER+ GEO cohorts ----------------
    log("\n[L-D] external validation in ER+ GEO cohorts (ER+/endocrine-treated context)")
    geo_rows, geo_tert = [], []
    try:
        import rep_geo_validate as G
        gpl = G.load_gpl_annots()
        wanted = set(coef) | set(PROLIF) | set(ER_TARGETS)
        for gse, (fname, key) in G.COHORTS.items():
            try:
                gexpr, samples = G.parse_series(gse, fname, gpl)
            except Exception as ex:
                log(f"   {gse}: parse fail {str(ex)[:80]}")
                continue
            t, e, unit = G.extract_outcome(gse, key, samples)
            # restrict to ER+ if an ER status field can be identified
            er_key, er_vals = None, set()
            for d in samples:
                for k in d:
                    kk = k.lower()
                    if "her" in kk:
                        continue
                    if "er" == kk or kk.startswith("er ") or kk.startswith("er_") or "estrogen" in kk:
                        er_key = k
                        er_vals = {str(dd.get(k, "")).strip().lower() for dd in samples}
                        break
                if er_key:
                    break
            vals = [str(d.get(er_key, "")).strip().lower() if er_key else "" for d in samples]
            if er_key and ({"positive", "pos", "1"} & er_vals):
                er_pos = np.isin(vals, ["positive", "pos", "1"])
                note = f"{er_key}=pos subset"
            else:
                er_pos = np.ones(len(samples), bool)
                note = "all (cohort is ER+ by design)" if er_key is None else f"{er_key} unclear -> all"
            genes = [g for g in wanted if g in gexpr.index]
            sub = gexpr.loc[genes]
            z = (sub - sub.mean(axis=1).values.reshape(-1, 1)) / sub.std(axis=1).values.reshape(-1, 1)
            z = z.fillna(0.0)
            rep = sum(coef[g] * z.loc[g].values for g in coef if g in z.index)
            pg = [g for g in PROLIF if g in z.index]
            prolif = z.loc[pg].mean(axis=0).values if pg else np.full(z.shape[1], np.nan)
            mk = z.loc["MKI67"].values if "MKI67" in z.index else np.full(z.shape[1], np.nan)
            D = pd.DataFrame({"REP30": rep, "PROLIF": prolif, "MKI67": mk,
                              "t": np.asarray(t, float), "e": np.asarray(e, float),
                              "er_pos": er_pos})
            D = D.replace([np.inf, -np.inf], np.nan).dropna(subset=["t", "e"])
            sub2 = D[D["er_pos"] & (D["t"] > 0)]
            if len(sub2) < 30:
                log(f"   {gse}: n={len(sub2)} too few ({note})")
                continue
            r_rep = uni_cox(sub2, "REP30")
            r_pro = uni_cox(sub2, "PROLIF")
            r_mk = uni_cox(sub2, "MKI67")
            log(f"   {gse} [{note}] n={r_rep['n']} ev={r_rep['events']} ({unit}) | "
                f"REP30 HR={r_rep['HR']:.2f} p={r_rep['p']:.2g} C={r_rep['cindex']:.3f} | "
                f"PROLIF HR={r_pro['HR']:.2f} p={r_pro['p']:.2g} | MKI67 HR={r_mk['HR']:.2f} p={r_mk['p']:.2g}")
            geo_rows.append(dict(cohort=gse, note=note, unit=unit, n=r_rep["n"], events=r_rep["events"],
                                 REP30_HR=r_rep["HR"], REP30_lo=r_rep["lo"], REP30_hi=r_rep["hi"],
                                 REP30_p=r_rep["p"], REP30_cindex=r_rep["cindex"],
                                 PROLIF_HR=r_pro["HR"], PROLIF_p=r_pro["p"],
                                 MKI67_HR=r_mk["HR"], MKI67_p=r_mk["p"]))
            for which in ["low", "mid", "high"]:
                lp = tertile_test(sub2.assign(t=sub2["t"], e=sub2["e"]), gse, gse, which)
                if lp:
                    geo_tert.append(lp)
                    log(f"        [{lp['stratum'].split('· ')[-1]}] REP30 n={lp['n']} ev={lp['events']} "
                        f"HR={lp.get('HR', float('nan')):.2f} p={lp.get('p', float('nan')):.2g} "
                        f"C={lp.get('cindex', float('nan')):.3f}")
    except Exception as ex:
        log(f"   L-D skipped: {str(ex)[:120]}")
    if geo_rows:
        pd.DataFrame(geo_rows).to_csv(os.path.join(OUT, "geo_hrpos_external.csv"),
                                      index=False, encoding="utf-8")
    if geo_tert:
        pd.DataFrame(geo_tert).to_csv(os.path.join(OUT, "geo_hrpos_prolif_tertiles.csv"),
                                      index=False, encoding="utf-8")

    log("\nDONE")
    flush()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _hook(*sys.exc_info())
        raise
