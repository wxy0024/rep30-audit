#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prognostic_metrics_hrpos.py
===========================
Discrimination / calibration / decision-curve metrics for REP30 in
METABRIC HR+/HER2-  (n=1398, 812 events), plus Table S1 (REP30 gene panel).

IMPORTANT (read before interpreting)
------------------------------------
Every quantity produced here is an **apparent / partly in-sample** assessment.
The REP30 weights (results/rep52_trim/rep52_trimmed_panel.csv) were derived from
a multivariable Cox fitted on pooled TCGA-BRCA + METABRIC, so METABRIC is partly a
DERIVATION sample. The bootstrap optimism correction below only accounts for the
within-stratum univariable refit; it CANNOT remove the optimism inherited from
the original panel derivation. All net benefits, calibration statistics and
time-dependent AUCs below carry that caveat.

Conventions are copied (not reinvented) from:
  analysis/rep52_backtest.py   load_panel / load_z_subset / rep_score / to_event
  analysis/subtype_rep30.py    load_metabric / PROLIF / ER_TARGETS / uni_cox

Usage
-----
  ENABLE_PLOTS=0 python analysis/prognostic_metrics_hrpos.py
  python analysis/prognostic_metrics_hrpos.py --B 1000 --seed 20260922

Outputs -> results/venet_null/
  discrimination_calibration.csv   A1 (optimism-corrected C) + A3 (slope/intercept)
  calibration_deciles.csv          A3 decile calibration table
  delta_cindex.csv                 A2
  td_auc_ipcw.csv                  A4
  dca_net_benefit.csv              A5
  TableS1_rep30_genes.xlsx/.csv    B
  run_log.txt
"""
import os, sys, io, time, argparse, platform, traceback
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
os.chdir(ROOT)

ENABLE_PLOTS = os.environ.get("ENABLE_PLOTS", "0") not in ("0", "", "no", "False")

OUT = os.path.join(ROOT, "results", "venet_null")
os.makedirs(OUT, exist_ok=True)

MB_DIR = os.path.join(ROOT, "data/01_core/METABRIC/brca_metabric")
PANEL_CSV = os.path.join(ROOT, "results/rep52_trim/rep52_trimmed_panel.csv")
MB_Z = os.path.join(MB_DIR, "data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt")

TAU = 120.0                      # 10-year horizon, months
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

PREDICTORS = ["REP30", "AGE", "NPI", "PROLIF", "MKI67"]
PRED_LABEL = {"REP30": "REP30", "AGE": "Age", "NPI": "NPI",
              "PROLIF": "PROLIF (12-gene mean)", "MKI67": "MKI67"}

from rep52_backtest import load_panel, load_z_subset, rep_score, to_event
from subtype_rep30 import PROLIF, ER_TARGETS, read_cbio

from lifelines import CoxPHFitter, KaplanMeierFitter
from scipy.optimize import minimize_scalar
from scipy.special import log_expit
from scipy import stats as sps

# ------------------------------------------------------------------ logging
LOG = []
def log(s=""):
    s = str(s)
    print(s, flush=True)
    LOG.append(s)
def flush_log():
    with open(os.path.join(OUT, "run_log.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG))


# ==================================================================
# 1. data: METABRIC HR+/HER2-  (exact copy of subtype_rep30.load_metabric)
# ==================================================================
def load_hrpos():
    coef = load_panel(PANEL_CSV)
    wanted = set(coef) | set(PROLIF) | set(ER_TARGETS)
    log("reading METABRIC z-subset ...")
    mat = load_z_subset(MB_Z, wanted)
    log(f"  z matrix gene x sample = {mat.shape}")
    log(f"  REP30 panel coverage METABRIC = {len([g for g in coef if g in mat.index])}/{len(coef)}")

    rep = rep_score(mat, coef)
    prolif = mat.loc[[g for g in PROLIF if g in mat.index]].mean(axis=0)
    S = pd.DataFrame({"REP30": rep, "PROLIF": prolif})
    S["MKI67"] = mat.loc["MKI67"] if "MKI67" in mat.index else np.nan
    S.index = S.index.astype(str)
    S.index.name = "SAMPLE_ID"
    S["patient"] = S.index

    cp = read_cbio(os.path.join(MB_DIR, "data_clinical_patient.txt"))
    cp.columns = [c.strip() for c in cp.columns]
    P = pd.DataFrame({
        "patient": cp["PATIENT_ID"].astype(str),
        "os_months": pd.to_numeric(cp["OS_MONTHS"], errors="coerce"),
        "os_status": cp["OS_STATUS"].astype(str).str.split(":").str[-1].str.strip(),
        "AGE": pd.to_numeric(cp["AGE_AT_DIAGNOSIS"], errors="coerce"),
        "NPI": pd.to_numeric(cp.get("NPI"), errors="coerce"),
    })
    P["event"] = to_event(P["os_status"])

    cs = read_cbio(os.path.join(MB_DIR, "data_clinical_sample.txt"))
    cs.columns = [c.strip() for c in cs.columns]
    Sm = pd.DataFrame({"SAMPLE_ID": cs["SAMPLE_ID"].astype(str),
                       "ER_STATUS": cs["ER_STATUS"].astype(str).str.strip(),
                       "HER2_STATUS": cs["HER2_STATUS"].astype(str).str.strip()})

    M = S.merge(Sm, on="SAMPLE_ID", how="inner").merge(P, on="patient", how="inner")
    M = M.rename(columns={"os_months": "t", "event": "e"})
    hrp = (M["ER_STATUS"].str.upper().str.startswith("POS")) & \
          (M["HER2_STATUS"].str.upper().str.startswith("NEG"))
    M["HRposHER2neg"] = hrp.astype(int)
    sub = M[M["HRposHER2neg"] == 1].copy()
    log(f"  merged n={len(M)};  HR+/HER2- n={len(sub)}  events={int(sub['e'].sum())}")

    # common complete-case + t>0  (matches subtype_rep30.prep)
    cc = sub.dropna(subset=PREDICTORS + ["t", "e"]).replace([np.inf, -np.inf], np.nan).dropna()
    cc = cc[cc["t"] > 0].reset_index(drop=True)
    log(f"  common complete-case set n={len(cc)} events={int(cc['e'].sum())}")
    return coef, mat, cc


# ==================================================================
# 2. concordance
# ==================================================================
def fast_cindex(t, e, x):
    """Harrell's C for a single risk score. Ties in the predictor count 0.5;
    only pairs with t_j > t_i are comparable. Invariant to increasing transforms."""
    t = np.asarray(t, float); e = np.asarray(e, int); x = np.asarray(x, float)
    later = t[None, :] > t[:, None]
    d = x[:, None] - x[None, :]
    ev = e == 1
    pos = float(((d > 0) & later)[ev].sum())
    neg = float(((d < 0) & later)[ev].sum())
    tie = float(((d == 0) & later)[ev].sum())
    den = pos + neg + tie
    return (pos + 0.5 * tie) / den if den > 0 else np.nan


def fit_uni(cc, col, per_sd=True):
    """Univariable Cox. Returns (cph, beta, hr_per_unit_scale, C_lifelines)."""
    x = cc[col].astype(float)
    xs = ((x - x.mean()) / x.std()) if per_sd else x
    dd = pd.DataFrame({"t": cc["t"].values, "e": cc["e"].values, "x": xs.values})
    cph = CoxPHFitter().fit(dd, "t", "e")
    return cph, float(cph.params_["x"]), float(cph.concordance_index_)


def apparent_table(cc):
    rows = []
    for c in PREDICTORS:
        cph, b, cli = fit_uni(cc, c, per_sd=True)
        # per-unit (per SD) HR
        hr = float(np.exp(b))
        lo = float(np.exp(cph.summary.loc["x", "coef lower 95%"]))
        hi = float(np.exp(cph.summary.loc["x", "coef upper 95%"]))
        p = float(cph.summary.loc["x", "p"])
        rows.append(dict(predictor=PRED_LABEL[c], variable=c, n=len(cc),
                         events=int(cc["e"].sum()), HR_per_SD=hr,
                         HR_lo=lo, HR_hi=hi, p=p,
                         cindex_apparent_lifelines=cli,
                         cindex_apparent_fast=fast_cindex(cc["t"].values, cc["e"].values, cc[c].values),
                         beta_per_SD=b))
    return pd.DataFrame(rows)


# ==================================================================
# 3. IPCW machinery
# ==================================================================
def ipcw_weights(t, e, tau):
    """IPCW case / non-case weights for the binary event-by-tau outcome.
    case_w[i]  = weight that subject i is a case  (event by tau)
    ctrl_w[i]  = weight that subject i is event-free at tau
    Subjects with follow-up past tau have known status -> weight 1."""
    t = np.asarray(t, float); e = np.asarray(e, int)
    km_c = KaplanMeierFitter().fit(t, event_observed=1 - e)     # censoring survival G
    km_s = KaplanMeierFitter().fit(t, event_observed=e)         # outcome survival S
    G_t = np.asarray(km_c.predict(t).values, float)
    S_t = np.asarray(km_s.predict(t).values, float)
    S_tau = float(km_s.predict(tau))
    G_tau = float(km_c.predict(tau))
    G_t = np.clip(G_t, 1e-12, None)
    S_t = np.clip(S_t, 1e-12, None)

    n = len(t)
    case_w = np.zeros(n); ctrl_w = np.zeros(n)
    early = t <= tau
    late = ~early
    ctrl_w[late] = 1.0                                  # status known, event-free at tau
    ie = early & (e == 1)
    case_w[ie] = 1.0 / G_t[ie]                          # observed event before tau
    ic = early & (e == 0)                               # censored before tau -> redistribute
    if ic.any():
        w = 1.0 / G_t[ic]
        q = 1.0 - S_tau / S_t[ic]                       # P(event by tau | alive at t_i)
        q = np.clip(q, 0.0, 1.0)
        case_w[ic] = q * w
        ctrl_w[ic] = (1.0 - q) * w
    return case_w, ctrl_w, S_tau, G_tau


def uno_ipcw_auc(t, e, marker, tau):
    """Uno IPCW cumulative/dynamic AUC at horizon tau.
    cases   : event observed at or before tau, weight 1/G(T_i)
    controls: subjects still under observation after tau, weight 1/G(tau)"""
    t = np.asarray(t, float); e = np.asarray(e, int); m = np.asarray(marker, float)
    km_c = KaplanMeierFitter().fit(t, event_observed=1 - e)
    G_tau = float(km_c.predict(tau))
    G_at = np.asarray(km_c.predict(np.minimum(t, tau)).values, float)
    G_at = np.clip(G_at, 1e-12, None)
    case = (t <= tau) & (e == 1)
    ctrl = t > tau
    if not case.any() or not ctrl.any():
        return np.nan, 0, 0
    wc_case = 1.0 / G_at[case]
    wc_ctrl = np.full(int(ctrl.sum()), 1.0 / max(G_tau, 1e-12))
    mi = m[case][:, None]; mj = m[ctrl][None, :]
    W = wc_case[:, None] * wc_ctrl[None, :]
    psi = 1.0 * (mi > mj) + 0.5 * (mi == mj)
    num = float((W * psi).sum())
    den = float(W.sum())
    return num / den if den > 0 else np.nan, int(case.sum()), int(ctrl.sum())


# ==================================================================
# 4. DCA
# ==================================================================
def net_benefit(case_w, ctrl_w, risk, pt):
    flagged = np.asarray(risk, float) >= pt
    N = float(case_w.sum() + ctrl_w.sum())
    if N <= 0:
        return np.nan
    tp = float(case_w[flagged].sum())
    fp = float(ctrl_w[flagged].sum())
    return (tp - fp * (pt / (1.0 - pt))) / N


def dca_curves(cc, risks, case_w, ctrl_w):
    rows = []
    N = float(case_w.sum() + ctrl_w.sum())
    prevalence = float(case_w.sum()) / N
    for pt in THRESHOLDS:
        row = dict(threshold_pt=pt, N_weighted=N, weighted_event_prevalence_at_tau=prevalence,
                   net_benefit_treat_none=0.0)
        row["net_benefit_treat_all"] = net_benefit(case_w, ctrl_w, np.ones(len(cc)) + 1, pt)
        row["net_benefit_treat_all"] = (float(case_w.sum())
                                        - float(ctrl_w.sum()) * (pt / (1.0 - pt))) / N
        for name, r in risks.items():
            row[f"net_benefit_{name}"] = net_benefit(case_w, ctrl_w, r, pt)
        rows.append(row)
    return pd.DataFrame(rows)


# ==================================================================
# 5. calibration
# ==================================================================
def calibration_intercept_ipcw(p_risk, case_w, ctrl_w):
    """Calibration-in-the-large at horizon tau: fit logit(q) = g + logit(p)
    (coefficient on logit(p) fixed at 1) by IPCW-weighted Bernoulli ML.
    NOTE: a true intercept is not identifiable from the Cox partial likelihood
    (adding a constant to the linear predictor leaves Breslow ratios unchanged),
    so the horizon-specific weighted logistic form is used instead."""
    p = np.clip(np.asarray(p_risk, float), 1e-6, 1 - 1e-6)
    lpoff = np.log(p / (1 - p))          # offset, fixed coefficient 1
    cw = np.asarray(case_w, float); nw = np.asarray(ctrl_w, float)

    def nll(g):
        eta = g + lpoff
        ll = cw * log_expit(eta) + nw * log_expit(-eta)
        return -float(ll.sum())

    eta0 = lpoff
    ll0 = cw * log_expit(eta0) + nw * log_expit(-eta0)    # gamma = 0  -> perfect calibration
    res = minimize_scalar(nll, bounds=(-8, 8), method="bounded",
                          options={"xatol": 1e-8})
    return float(res.x), float(-ll0.sum())


def decile_calibration(cc, risk, tau):
    df = pd.DataFrame({"risk": np.asarray(risk, float),
                       "t": cc["t"].values, "e": cc["e"].values})
    df["decile"] = pd.qcut(df["risk"], 10, labels=False, duplicates="drop") + 1
    rows = []
    for d, g in df.groupby("decile"):
        kmf = KaplanMeierFitter().fit(g["t"], g["e"])
        obs = 1.0 - float(kmf.predict(tau))
        rows.append(dict(decile=int(d), n=len(g), events=int(g["e"].sum()),
                         mean_predicted_10y_risk=float(g["risk"].mean()),
                         min_predicted=float(g["risk"].min()),
                         max_predicted=float(g["risk"].max()),
                         observed_10y_risk_KM=obs,
                         difference_observed_minus_predicted=obs - float(g["risk"].mean())))
    return pd.DataFrame(rows)


# ==================================================================
# PART B -- Table S1
# ==================================================================
def file_gene_set(path):
    gs = set()
    with open(path, "rt", encoding="utf-8", errors="replace") as fh:
        fh.readline()
        for line in fh:
            g = line.split("\t", 1)[0].strip()
            if g:
                gs.add(g)
    return gs


def geo_gene_set(gse, data_sub="data/02_acquired_resistance"):
    import rep_geo_validate as G
    gpl = G.load_gpl_annots()
    dpath = os.path.join(ROOT, data_sub, gse)
    fn = [f for f in os.listdir(dpath) if "series_matrix" in f]
    if not fn:
        return None, None, f"no series_matrix in {dpath}"
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        gexpr, samples = G.parse_series(gse, fn[0], gpl)
    finally:
        sys.stdout = old
    return set(gexpr.index), buf.getvalue(), fn[0]


def build_table_s7(coef):
    genes = list(coef)
    rows = []
    # ---- per-cohort gene sets
    sets = {}
    note = {}
    # cBioPortal-style matrices
    cands = {
        "TCGA": "data/01_core/TCGA-BRCA/brca_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt",
        "METABRIC": "data/01_core/METABRIC/brca_metabric/data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt",
        "AURORA": "data/06_metastatic/cbioportal/brca_aurora_2023/data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt",
        "MBCproject": "data/06_metastatic/cbioportal/brca_mbcproject_2022/data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt",
    }
    for k, v in cands.items():
        note[k] = v
        try:
            sets[k] = file_gene_set(v)
            note[k] = f"{v}  ({len(sets[k])} genes in matrix)"
        except Exception as ex:
            sets[k] = set(); note[k] = f"ERROR {str(ex)[:80]} ({v})"
    # single cell
    try:
        import anndata
        for k, v in {"GSE264205": "results/sc/GSE264205/reload/GSE264205_fixed.h5ad",
                     "GSE249361": "results/sc/GSE249361/GSE249361.h5ad",
                     "GSE161529": "results/sc/GSE161529/GSE161529.h5ad"}.items():
            try:
                ad = anndata.read_h5ad(v, backed="r")
                sets[k] = set(str(g) for g in ad.var.index)
                note[k] = f"{v}; {tuple(ad.shape)} (cells x features); gene symbols taken from var_names"
                del ad
            except Exception as ex:
                sets[k] = set(); note[k] = f"ERROR {str(ex)[:80]}"
    except Exception as ex:
        log("  anndata unavailable: " + str(ex)[:80])
    # GEO arrays
    for k in ["GSE45255", "GSE26459"]:
        try:
            gs, msg, fn = geo_gene_set(k)
            sets[k] = gs if gs is not None else set()
            parse_lines = [l for l in (msg or "").splitlines() if l.strip().startswith("[parse]")]
            note[k] = f"data/02_acquired_resistance/{k}/{fn}; " + (
                parse_lines[-1] if parse_lines else "platform not reported")
        except Exception as ex:
            sets[k] = set(); note[k] = f"ERROR {str(ex)[:80]}"

    order = ["TCGA", "METABRIC", "AURORA", "MBCproject",
             "GSE264205", "GSE249361", "GSE161529", "GSE45255", "GSE26459"]
    for g in genes:
        r = {"gene": g, "cox_coefficient": coef[g],
             "direction": "risk-increasing (higher expression -> shorter OS)" if coef[g] > 0
                          else "risk-decreasing (higher expression -> longer OS)"}
        n = 0
        for c in order:
            present = g in sets.get(c, set())
            r[c] = "present" if present else "absent"
            n += int(present)
        r["n_cohorts_present"] = n
        rows.append(r)
    panel = pd.DataFrame(rows)

    # ---- metadata block
    counts = [f"{c} {sum((panel[c] == 'present').astype(int))}/30" for c in order]
    missing = {c: ", ".join(panel.loc[panel[c] == "absent", "gene"]) for c in order}
    prolif_overlap = sorted(set(genes) & set(PROLIF))
    er_overlap = sorted(set(genes) & set(ER_TARGETS))

    meta = [
        ("S7b: n genes in panel", "30"),
        ("S7b: score formula", "REP30 = sum_g coef_g * z_g ; z_g = per-gene z-score from the "
                               "cBioPortal reference matrix (diploid-samples reference) for METABRIC/TCGA/"
                               "AURORA/MBCproject, and per-gene z computed within cohort for GEO cohorts"),
        ("S7b: coefficient direction", "coef>0 => risk-increasing; coef<0 => risk-decreasing"),
        ("S7b: n risk-increasing / risk-decreasing",
         f"{int((panel['cox_coefficient']>0).sum())} / {int((panel['cox_coefficient']<0).sum())}"),
        ("S7b: per-cohort coverage", "; ".join(counts)),
        ("S7b: missing genes per cohort", "; ".join(f"{c}: [{v}]" for c, v in missing.items() if v)),
        ("S7b: probe collapsing rule (GEO Affymetrix)",
         "UNWEIGHTED ARITHMETIC MEAN of all probes annotated to the same gene symbol "
         "(analysis/rep_geo_validate.py parse_series: gene_rows.setdefault(g,[]).append(row) then "
         "pd.concat(v,axis=1).mean(axis=1)). NOT max-mean / NOT jetset / NOT first-probe."),
        ("S7b: probe->gene mapping",
         "GPL annotation tables in data/05_gpl/*.annot; multi-symbol entries split on '///','//','/ ' "
         "and the FIRST symbol is taken. Platform is auto-selected by maximising the fraction of the "
         "series' own probes that the platform annotates."),
        ("S7b: missing-gene rule (scoring)",
         "WEIGHTED SUM over available genes only - the score is NOT rescaled by the number of genes "
         "available, and a gene absent from the platform contributes 0 (rep_score / score_cohort: "
         "sum over [g for g in coef if g in matrix]; z matrices use per-gene standardisation and "
         ".fillna(0.0)). NOTE: this differs from 'mean of available genes' - no division occurs."),
        ("S7b: minimum gene requirement", "rep_geo_validate.score_cohort returns None if <5 panel genes present"),
        ("S7b: overlap with PROLIF (12-gene module)",
         (", ".join(prolif_overlap) if prolif_overlap else "NONE (0 genes)")),
        ("S7b: overlap with ER_TARGETS (20-gene module)",
         (", ".join(er_overlap) if er_overlap else "NONE (0 genes)")),
        ("S7b: PROLIF gene list", ", ".join(PROLIF)),
        ("S7b: ER_TARGETS gene list", ", ".join(ER_TARGETS)),
    ]
    for c in order:
        meta.append((f"S7b: source used for {c}", (note.get(c, "") or os.path.basename(c))))
    meta_df = pd.DataFrame(meta, columns=["item", "value"])
    return panel, meta_df, order, counts, prolif_overlap, er_overlap


def write_xlsx(panel, meta, order):
    """Minimal openpyxl writer (no matplotlib): bold header, grey fill, autofit."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
    HF = PatternFill("solid", fgColor="F2F2F2")
    wb = Workbook()
    for sheet_name, df in (("S7a_panel", panel), ("S7b_metadata", meta)):
        ws = wb.active if sheet_name == "S7a_panel" else wb.create_sheet()
        ws.title = sheet_name
        ws.sheet_view.showGridLines = False
        for j, c in enumerate(df.columns, 1):
            cell = ws.cell(row=1, column=j, value=str(c))
            cell.font = Font(bold=True, size=10); cell.fill = HF
            cell.alignment = Alignment(horizontal="left", vertical="center")
        for i, (_, r) in enumerate(df.iterrows(), start=2):
            for j, c in enumerate(df.columns, 1):
                v = r[c]
                if isinstance(v, (np.integer,)):
                    v = int(v)
                elif isinstance(v, (np.floating,)):
                    v = float(v)
                elif isinstance(v, bool):
                    v = str(v)
                ws.cell(row=i, column=j, value=v)
            ws.row_dimensions[i].height = 15
        widths = {}
        for row in ws.iter_rows():
            for cell in row:
                if cell.value is None:
                    continue
                w = sum(2 if ord(ch) > 0x2E80 else 1 for ch in str(cell.value))
                widths[cell.column] = max(widths.get(cell.column, 0), w)
        for col, w in widths.items():
            ws.column_dimensions[get_column_letter(col)].width = min(max(w + 2, 9), 62)
        ws.freeze_panes = "A2"
    wb.save(os.path.join(OUT, "TableS1_rep30_genes.xlsx"))


# ==================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--tau", type=float, default=TAU)
    args = ap.parse_args()
    tau = args.tau

    log("=" * 78)
    log("REP30 discrimination / calibration / decision curve - METABRIC HR+/HER2-")
    log("=" * 78)
    log(f"python      : {platform.python_version()}")
    import numpy as _n, pandas as _p, lifelines as _lf, scipy as _sp
    log(f"numpy={_n.__version__} pandas={_p.__version__} lifelines={_lf.__version__} scipy={_sp.__version__}")
    log(f"seed        : {args.seed}   B={args.B}   tau={tau} months (10 years)")
    log(f"ENABLE_PLOTS: {ENABLE_PLOTS}")
    log(f"output dir  : {OUT}")
    log("")

    coef, mat, cc = load_hrpos()
    n, ev = len(cc), int(cc["e"].sum())
    if not (1390 <= n <= 1410) or not (805 <= ev <= 820):
        log(f"!! STOP: cohort size n={n} events={ev} does not reproduce n~1398 / 812.")
        flush_log(); return
    log(f"OK: reproduced n={n} events={ev}")
    log(f"follow-up months: min={cc['t'].min():.1f} median={cc['t'].median():.1f} max={cc['t'].max():.1f}")
    log("")

    # ---------------- apparent performance ----------------
    log("=" * 78)
    log("[0] apparent univariable performance (reproduces results/subtype_rep30/)")
    app = apparent_table(cc)
    app.to_csv(os.path.join(OUT, "_apparent_check.csv"), index=False)
    maxdiff = float((app["cindex_apparent_lifelines"] - app["cindex_apparent_fast"]).abs().max())
    log(app.to_string(index=False))
    log(f"max |lifelines C - fast C| = {maxdiff:.2e}  (fast C validated; used inside the bootstrap)")
    C_full = {c: float(app.loc[app["variable"] == c, "cindex_apparent_lifelines"].iloc[0]) for c in PREDICTORS}
    log("")

    # ---------------- A1 + A2 bootstrap (shared resamples) ----------------
    log("=" * 78)
    log(f"[A1/A2] bootstrap B={args.B}, seed={args.seed} (shared resamples per iteration)")
    rng = np.random.RandomState(args.seed)
    t_arr = cc["t"].values; e_arr = cc["e"].values
    cols = {c: cc[c].values.astype(float) for c in PREDICTORS}

    bootC = {c: np.empty(args.B) for c in PREDICTORS}
    app_rep30 = np.empty(args.B)      # apparent C from the RE-FITTED REP30 model (bootstrap data)
    test_rep30 = np.empty(args.B)     # that model evaluated on the original cohort
    t0 = time.time()
    for b in range(args.B):
        idx = rng.randint(0, n, n)
        tb, eb = t_arr[idx], e_arr[idx]
        # refit REP30 univariable Cox on the bootstrap sample
        xb = cols["REP30"][idx]
        xb = (xb - xb.mean()) / max(xb.std(), 1e-12)
        dd = pd.DataFrame({"t": tb, "e": eb, "x": xb})
        try:
            cph = CoxPHFitter().fit(dd, "t", "e")
            beta = float(cph.params_["x"])
            app_rep30[b] = float(cph.concordance_index_)
            # apply to the original cohort using the bootstrap-fitted coefficient
            risk_full = beta * ((cols["REP30"] - xb.mean()) / max(xb.std(), 1e-12))
            test_rep30[b] = fast_cindex(t_arr, e_arr, risk_full)
        except Exception as ex:
            app_rep30[b] = np.nan; test_rep30[b] = np.nan
        for c in PREDICTORS:
            bootC[c][b] = fast_cindex(tb, eb, cols[c][idx])
        if (b + 1) % 100 == 0:
            log(f"   {b+1}/{args.B}  ({time.time()-t0:.0f}s)")
    log(f"   bootstrap loop done in {time.time()-t0:.0f}s")

    opt = app_rep30 - test_rep30
    opt_mean = float(np.nanmean(opt))
    C_app_full = C_full["REP30"]
    C_opt = C_app_full - opt_mean
    se_C = float(np.nanstd(bootC["REP30"], ddof=1))
    rows = []
    opt_all = {}
    for c in PREDICTORS:
        oc = opt_all[c] = float(np.nanmean(bootC[c]) - C_full[c])   # optimism for this predictor
        pclo, pchi = np.percentile(bootC[c], [2.5, 97.5])
        rows.append(dict(predictor=PRED_LABEL[c], variable=c, n=n, events=ev,
                         cindex_apparent=C_full[c],
                         optimism=oc,
                         cindex_optimism_corrected=C_full[c] - oc,
                         cindex_se_bootstrap=float(np.nanstd(bootC[c], ddof=1)),
                         cindex_ci95_lo=C_full[c] - oc - 1.959963985 * float(np.nanstd(bootC[c], ddof=1)),
                         cindex_ci95_hi=C_full[c] - oc + 1.959963985 * float(np.nanstd(bootC[c], ddof=1)),
                         cindex_percentile_ci95_lo=float(pclo),
                         cindex_percentile_ci95_hi=float(pchi)))
    dc = pd.DataFrame(rows)
    dc.to_csv(os.path.join(OUT, "optimism_corrected_cindex.csv"), index=False)
    log("")
    log("[A1] optimism-corrected C-index")
    log(dc.to_string(index=False))
    log(f"REP30 apparent C={C_app_full:.6f}  mean optimism (refit-based)={opt_mean:.6f}  "
        f"optimism-corrected={C_opt:.6f}")
    log("")

    # detect OOB rows? not requested; skip.

    # ---------------- A2 delta C-index ----------------
    log("=" * 78)
    log("[A2] paired bootstrap delta C-index  (REP30 - comparator, same resample)")
    drows = []
    for c in PREDICTORS:
        if c == "REP30":
            continue
        dstar = bootC["REP30"] - bootC[c]
        d_pt = C_full["REP30"] - C_full[c]
        d_opt = (C_full["REP30"] - opt_all["REP30"]) - (C_full[c] - opt_all[c])
        lo, hi = np.percentile(dstar, [2.5, 97.5])
        p = 2.0 * min(float((dstar <= 0).mean()), float((dstar >= 0).mean()))
        drows.append(dict(comparator=PRED_LABEL[c], variable=c, n=n, events=ev,
                          cindex_REP30=C_full["REP30"], cindex_comparator=C_full[c],
                          delta_apparent=d_pt,
                          delta_optimism_corrected=d_opt,
                          delta_boot_mean=float(dstar.mean()),
                          delta_ci95_lo=float(lo), delta_ci95_hi=float(hi),
                          delta_se=float(dstar.std(ddof=1)),
                          p_bootstrap_two_sided=p))
        log(f"   REP30 - {PRED_LABEL[c]:22s}: delta={d_pt:+.4f} 95%CI [{lo:+.4f},{hi:+.4f}] "
            f"p={p:.3g}   (optimism-corrected {d_opt:+.4f})")
    dd = pd.DataFrame(drows)
    dd.to_csv(os.path.join(OUT, "delta_cindex.csv"), index=False)
    log("")

    # ---------------- A3 calibration ----------------
    log("=" * 78)
    log("[A3] calibration (in-sample, see caveat at top of run_log)")
    # predicted 10y risk from each univariable Cox
    risks = {}
    for c in PREDICTORS:
        x = cc[c].astype(float)
        xs = ((x - x.mean()) / x.std())
        X = pd.DataFrame({"x": xs.values})
        df = pd.DataFrame({"t": cc["t"].values, "e": cc["e"].values, "x": xs.values})
        cph = CoxPHFitter().fit(df, "t", "e")
        sf = cph.predict_survival_function(X, times=[tau])
        risks[c] = 1.0 - np.asarray(sf.iloc[0].values, float)

    # calibration slope: refit Cox on the RAW composite score (implicit development coefficient = 1)
    raw = cc["REP30"].astype(float).values
    df_slope = pd.DataFrame({"t": cc["t"].values, "e": cc["e"].values, "raw": raw})
    cph_s = CoxPHFitter().fit(df_slope, "t", "e")
    slope = float(cph_s.params_["raw"])
    slope_lo = float(cph_s.summary.loc["raw", "coef lower 95%"])
    slope_hi = float(cph_s.summary.loc["raw", "coef upper 95%"])
    slope_p = float(cph_s.summary.loc["raw", "p"])
    sd_rep30 = float(raw.std())

    case_w, ctrl_w, S_tau, G_tau = ipcw_weights(cc["t"].values, cc["e"].values, tau)
    intercept, ll0 = calibration_intercept_ipcw(risks["REP30"], case_w, ctrl_w)
    kmf_all = KaplanMeierFitter().fit(cc["t"], cc["e"])
    obs10 = 1.0 - float(kmf_all.predict(tau))
    mean_pred10 = float(risks["REP30"].mean())

    cal_rows = [
        dict(metric="apparent C-index (REP30)", value=C_app_full, ci_lo=np.nan, ci_hi=np.nan, p=np.nan,
             note="Harrell C from univariable Cox, per-SD; reproduces 0.677"),
        dict(metric="bootstrap optimism (REP30)", value=opt_mean, ci_lo=np.nan, ci_hi=np.nan, p=np.nan,
             note=f"B={args.B}, mean(apparent - test) with refit inside each resample"),
        dict(metric="optimism-corrected C-index (REP30)", value=C_opt,
             ci_lo=C_opt - 1.959963985 * se_C, ci_hi=C_opt + 1.959963985 * se_C, p=np.nan,
             note="bootstrap SE CI on the optimism-corrected value"),
        dict(metric="calibration slope (univariable, raw REP30 score)", value=slope,
             ci_lo=slope_lo, ci_hi=slope_hi, p=slope_p,
             note="Cox refit with raw composite score; development LP implies 1"),
        dict(metric="calibration slope implied per SD", value=app.loc[app['variable'] == 'REP30', 'HR_per_SD'].iloc[0],
             ci_lo=np.nan, ci_hi=np.nan, p=np.nan, note="HR per 1 SD, same refit"),
        dict(metric="SD of raw REP30 score", value=sd_rep30, ci_lo=np.nan, ci_hi=np.nan, p=np.nan,
             note="converts per-unit slope to per-SD scale"),
        dict(metric="calibration intercept at 10y (IPCW logistic, logit p fixed at 1)",
             value=intercept, ci_lo=np.nan, ci_hi=np.nan, p=np.nan,
             note="0 = perfect calibration-in-the-large"),
        dict(metric="weighted log-likelihood at gamma=0 (reference)",
             value=ll0, ci_lo=np.nan, ci_hi=np.nan, p=np.nan,
             note="log-lik of the perfectly-calibrated constrained model, for LR-type comparison"),
        dict(metric="observed 10y risk (KM)", value=obs10, ci_lo=np.nan, ci_hi=np.nan, p=np.nan, note="1 - S(120)"),
        dict(metric="mean predicted 10y risk (REP30 Cox)", value=mean_pred10, ci_lo=np.nan, ci_hi=np.nan, p=np.nan,
             note="averaged over the 1398 patients"),
    ]
    # ---- harmonise schema and add a few diagnostics, then combine A1 + A3
    cal_rows = [dict(block="A3_calibration", predictor="REP30",
                     **{k: v for k, v in r.items() if k != "p"}) for r in cal_rows]
    extra = [
        dict(block="A3_calibration", predictor="REP30", metric="bootstrap SE of C-index",
             value=se_C, ci_lo=np.nan, ci_hi=np.nan, note="SD of C* over resamples"),
        dict(block="A3_calibration", predictor="REP30", metric="IPCW weighted cohort size N",
             value=float(case_w.sum() + ctrl_w.sum()), ci_lo=np.nan, ci_hi=np.nan,
             note="sum of case+control IPCW weights; net benefit uses this denominator for all models"),
        dict(block="A3_calibration", predictor="REP30", metric="IPCW weighted event prevalence at 10y",
             value=float(case_w.sum()) / float(case_w.sum() + ctrl_w.sum()),
             ci_lo=np.nan, ci_hi=np.nan, note=f"compare with the KM estimate {obs10:.4f}"),
    ]
    for c in PREDICTORS:
        extra.append(dict(block="A3_calibration", predictor=PRED_LABEL[c],
                          metric="predicted 10y risk range (in-sample Cox)",
                          value=float(risks[c].min()), ci_lo=float(np.median(risks[c])),
                          ci_hi=float(risks[c].max()),
                          note="value=min, ci_lo=median, ci_hi=max of predicted risk"))

    def _r(block, predictor, metric, value, lo=np.nan, hi=np.nan, note=""):
        return dict(block=block, predictor=predictor, metric=metric, value=value,
                    ci_lo=lo, ci_hi=hi, note=note)

    comb = [_r("A1_discrimination", PRED_LABEL[r["variable"]], "apparent C-index", r["cindex_apparent"],
               np.nan, np.nan, "Harrell C, univariable Cox per 1 SD") for r in rows]
    comb += [_r("A1_discrimination", PRED_LABEL[r["variable"]], "bootstrap optimism", r["optimism"],
                np.nan, np.nan, f"mean(apparent-test) over B={args.B} resamples") for r in rows]
    comb += [_r("A1_discrimination", PRED_LABEL[r["variable"]], "optimism-corrected C-index",
                r["cindex_optimism_corrected"], r["cindex_ci95_lo"], r["cindex_ci95_hi"],
                "apparent minus optimism; CI = corrected +/- 1.96 bootstrap SE") for r in rows]
    comb += [_r("A1_discrimination", PRED_LABEL[r["variable"]], "optimism-corrected C-index percentile CI low",
                r["cindex_percentile_ci95_lo"], np.nan, np.nan, "2.5th percentile of bootstrap C*") for r in rows]
    comb += [_r("A1_discrimination", PRED_LABEL[r["variable"]], "optimism-corrected C-index percentile CI high",
                r["cindex_percentile_ci95_hi"], np.nan, np.nan, "97.5th percentile of bootstrap C*") for r in rows]
    comb += cal_rows + extra
    pd.DataFrame(comb).to_csv(os.path.join(OUT, "discrimination_calibration.csv"), index=False)

    log(f"   calibration slope = {slope:.4f} [{slope_lo:.4f},{slope_hi:.4f}] (SD(REP30)={sd_rep30:.4f})")
    log(f"   calibration intercept at 10y = {intercept:.4f}")
    log(f"   observed 10y risk {obs10:.4f} vs mean predicted {mean_pred10:.4f}")
    log("   predicted 10y risk ranges (in-sample Cox):")
    for c in PREDICTORS:
        log(f"      {PRED_LABEL[c]:22s} min={risks[c].min():.4f} median={np.median(risks[c]):.4f} "
            f"max={risks[c].max():.4f}")
    log("   NOTE: several models never predict a risk below some of the low DCA thresholds;")
    log("         for those they flag everyone and their net benefit equals 'treat all'.")

    dec = decile_calibration(cc, risks["REP30"], tau)
    dec.to_csv(os.path.join(OUT, "calibration_deciles.csv"), index=False)
    log("   decile table:")
    log(dec.to_string(index=False))
    log("")

    # ---------------- A4 time-dependent AUC ----------------
    log("=" * 78)
    log("[A4] Uno IPCW cumulative/dynamic AUC at 10 years")
    arows = []
    for c in PREDICTORS:
        a, ncase, nctrl = uno_ipcw_auc(cc["t"].values, cc["e"].values, cc[c].values, tau)
        arows.append(dict(marker=PRED_LABEL[c], variable=c, horizon_months=tau,
                          uno_ipcw_auc=a, n_cases_weighted=ncase, n_controls_at_risk=nctrl,
                          method="Uno IPCW cumulative/dynamic AUC; cases = events <= tau weighted 1/G(T), "
                                 "controls = subjects with T > tau weighted 1/G(tau)"))
        log(f"   {PRED_LABEL[c]:22s} AUC({tau:.0f}mo) = {a:.4f}  (cases={ncase}, controls={nctrl})")
    pd.DataFrame(arows).to_csv(os.path.join(OUT, "td_auc_ipcw.csv"), index=False)
    log("")

    # ---------------- A5 DCA ----------------
    log("=" * 78)
    log("[A5] decision curve analysis (IPCW-weighted net benefit at 10 years)")
    dca = dca_curves(cc, risks, case_w, ctrl_w)
    dca.to_csv(os.path.join(OUT, "dca_net_benefit.csv"), index=False)
    log(dca.to_string(index=False))
    for pt in (0.10, 0.20):
        row = dca[dca["threshold_pt"] == pt].iloc[0]
        log(f"   > {int(pt*100)}% threshold: " + ", ".join(
            f"{k.replace('net_benefit_','')}={row[k]:.4f}" for k in row.index if k.startswith("net_benefit_")))
    log("")

    # ---------------- PART B ----------------
    log("=" * 78)
    log("[B] Table S1 - REP30 gene panel + per-cohort coverage")
    panel, meta, order, counts, prolif_ov, er_ov = build_table_s7(coef)
    write_xlsx(panel, meta, order)
    panel.to_csv(os.path.join(OUT, "TableS1_rep30_genes.csv"), index=False)
    meta.to_csv(os.path.join(OUT, "TableS1_metadata.csv"), index=False)
    log("   coverage: " + "; ".join(counts))
    log(f"   overlap with PROLIF      : {prolif_ov if prolif_ov else 'NONE (0 genes)'}")
    log(f"   overlap with ER_TARGETS  : {er_ov if er_ov else 'NONE (0 genes)'}")
    log(panel.to_string(index=False))
    log("")
    log("DONE")
    flush_log()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOG.append("EXCEPTION:\n" + traceback.format_exc())
        print(traceback.format_exc())
        flush_log()
        raise
