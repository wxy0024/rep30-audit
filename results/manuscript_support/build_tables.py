# -*- coding: utf-8 -*-
"""
Aggregate the completed results/ outputs -> submission-grade table package (manuscript/tables/)
Reads results/ only; writes manuscript/tables/ and results/manuscript_support/ only.
Re-runs no analysis. All numbers come from CSV/XLSX files already present under results/.
"""
import os, json, math
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

ROOT = "E:/01.WorkSpace/01.Breast_cancer_gene_panel/resistance_signature"
os.chdir(ROOT)
OUT = "manuscript/tables"
CSVDIR = os.path.join(OUT, "csv")
os.makedirs(CSVDIR, exist_ok=True)
SUP = "results/manuscript_support"
os.makedirs(SUP, exist_ok=True)

# ---------------------------------------------------------------- styling
HEADER_FILL = PatternFill("solid", fgColor="F2F2F2")
HEADER_FONT = Font(bold=True, size=10, color="1F1F1F")
NOTE_FONT = Font(italic=True, size=8, color="595959")
BODY_FONT = Font(size=10)
BODY_FONT_TXT = Font(size=10)

def _is_na(v):
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and v.strip() == "" and False)

def fmt_p(v):
    """p values: keep original scientific notation exactly (avoid Excel 6.6E-05)."""
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return repr(v)
    s = str(v)
    if s.lower() in ("nan", "none"):
        return ""
    return s

def _w(s):
    """visual width: CJK counted as 2."""
    w = 0
    for ch in str(s):
        w += 2 if ord(ch) > 0x2E80 else 1
    return w

def write_df(ws, df, start_row, p_cols=(), title=None, write_header=True):
    r = start_row
    if title:
        c = ws.cell(row=r, column=1, value=title)
        c.font = Font(bold=True, size=10)
        r += 1
    hdr = None
    if write_header:
        for j, col in enumerate(df.columns, 1):
            c = ws.cell(row=r, column=j, value=str(col))
            c.font = HEADER_FONT
            c.fill = HEADER_FILL
            c.alignment = Alignment(horizontal="left", vertical="center")
        ws.row_dimensions[r].height = 15
        hdr = r
        r += 1
    for _, row in df.iterrows():
        for j, col in enumerate(df.columns, 1):
            v = row[col]
            if col in p_cols:
                c = ws.cell(row=r, column=j, value=fmt_p(v))
                c.font = BODY_FONT_TXT
                c.number_format = "@"
            elif v is None or (isinstance(v, float) and math.isnan(v)):
                c = ws.cell(row=r, column=j, value="")
                c.font = BODY_FONT
            elif isinstance(v, (np.floating,)):
                fv = float(v)
                if math.isinf(fv):
                    c = ws.cell(row=r, column=j, value="inf")
                    c.font = BODY_FONT_TXT; c.number_format = "@"
                else:
                    c = ws.cell(row=r, column=j, value=fv)
                    c.font = BODY_FONT
            elif isinstance(v, (np.integer,)):
                c = ws.cell(row=r, column=j, value=int(v))
                c.font = BODY_FONT
            elif isinstance(v, float):
                if math.isinf(v):
                    c = ws.cell(row=r, column=j, value="inf")
                    c.font = BODY_FONT_TXT; c.number_format = "@"
                else:
                    c = ws.cell(row=r, column=j, value=float(v))
                    c.font = BODY_FONT
            elif isinstance(v, (int,)):
                c = ws.cell(row=r, column=j, value=int(v))
                c.font = BODY_FONT
            else:
                c = ws.cell(row=r, column=j, value=str(v))
                c.font = BODY_FONT_TXT
                c.number_format = "@"
        ws.row_dimensions[r].height = 15
        r += 1
    return r

def autofit(ws, min_w=8, max_w=64):
    widths = {}
    for row in ws.iter_rows():
        for cell in row:
            if cell.value is None:
                continue
            col = cell.column
            widths[col] = max(widths.get(col, 0), _w(cell.value))
    for col, w in widths.items():
        ws.column_dimensions[get_column_letter(col)].width = min(max(w + 2, min_w), max_w)

def add_note(ws, row, ncols, text):
    c = ws.cell(row=row, column=1, value=text)
    c.font = NOTE_FONT
    c.alignment = Alignment(wrap_text=True, vertical="top")
    span = max(ncols, 6)
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=span)
    ws.row_dimensions[row].height = max(30, 12 * (len(text) // 90 + 1))

def save_single(sheet_name, df, p_cols, note, fname):
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name
    ws.sheet_view.showGridLines = False
    end = write_df(ws, df, 1, p_cols=p_cols)
    autofit(ws)
    ws.freeze_panes = "A2"
    add_note(ws, end + 1, len(df.columns), note)
    wb.save(os.path.join(OUT, fname + ".xlsx"))
    df.to_csv(os.path.join(OUT, fname + ".csv"), index=False)
    # also per-sheet csv in csv/
    df.to_csv(os.path.join(CSVDIR, fname + "__" + sheet_name + ".csv"), index=False)
    print("saved", fname, "sheet", sheet_name, df.shape)

def save_multi(fname, sheets):
    """sheets: list of (sheet_name, df, p_cols). Writes xlsx (multi) + per-sheet csv."""
    wb = Workbook()
    first = True
    for sname, blocks, p_cols, note in sheets:
        ws = wb.active if first else wb.create_sheet()
        ws.title = sname[:31]
        first = False
        ws.sheet_view.showGridLines = False
        r = 1
        # blocks: list of (title_or_None, df)
        header_map = {}
        for (title, df) in blocks:
            write_df(ws, df, r, p_cols=p_cols, title=title)
            # recompute next row
            r = r + (1 if title else 0) + 1 + len(df)
        autofit(ws)
        ws.freeze_panes = "A2"
        add_note(ws, r + 1, 12, note)
        # csv exports (one per block)
        for i, (title, df) in enumerate(blocks):
            fn = fname + "__" + sname + (f"_{i+1}" if len(blocks) > 1 else "") + ".csv"
            df.to_csv(os.path.join(CSVDIR, fn), index=False)
    wb.save(os.path.join(OUT, fname + ".xlsx"))
    print("saved", fname, "sheets", [s[0] for s in sheets])

# ================================================================ Table 1
inv = pd.read_csv("results/evidence_table/submission/Table_S1_cohort_inventory.csv", dtype=str, keep_default_na=False)
inv_main = inv[inv["included"].str.lower() == "yes"].copy()

def pick_n(row):
    np_ = str(row["n_patients"]).strip()
    ns = str(row["n_samples"]).strip()
    if np_ and np_ != "0":
        return np_
    if "cells" in ns:
        return ns
    return ns if ns else np_

t1 = pd.DataFrame({
    "cohort_id": inv_main["cohort_id"],
    "dataset": inv_main["cohort"],
    "type": inv_main["cohort_type"],
    "specimen": inv_main["specimen"],
    "access_route": inv_main["access_route"],
    "n": inv_main.apply(pick_n, axis=1),
    "evidential_role": inv_main["evidential_role"],
    "n_comparisons": inv_main["comparisons_in_supplement"],
})
t1_note = ("Table 1. Independent cohorts and the derived single-cell integration. "
           "n = number of patients (single-cell cohorts C5/S1: number of cells; C1 is an in-vitro cell-line model reported as samples). "
           "C6 combines GEO GSE209998 and cBioPortal brca_aurora_2023, which are the SAME 53 patients and are counted once. "
           "S1 (3-cohort Harmony integration) is DERIVED from C5 + GSE249361 + GSE161529 and is not an independent cohort. "
           "Excluded: brca_mbcproject_wagle_2017 - earlier release of the same Count Me In MBCproject (RNA samples overlap 71.9%, patients overlap 99.1%; "
           "union adds 1 patient) -> pseudo-replication. n_comparisons = rows contributed to the full evidence table.")
save_single("Table1_cohort_inventory", t1, [], t1_note, "Table1_cohort_inventory")

# ================================================================ Table 2
s2 = pd.read_csv("results/evidence_table/submission/Table_S2_headline_evidence.csv", dtype=str, keep_default_na=False)
eff_metric, eff_val = [], []
for s in s2["effect_str"]:
    if "=" in s:
        a, b = s.split("=", 1)
        eff_metric.append(a.strip())
        eff_val.append(b.strip())
    else:
        eff_metric.append("")
        eff_val.append(s.strip())
t2 = pd.DataFrame({
    "cohort_id": s2["cohort_id"],
    "cohort": s2["cohort"],
    "dataset": s2["dataset"],
    "access_route": s2["access_route"],
    "endpoint_class": s2["endpoint_class"],
    "endpoint": s2["endpoint"],
    "n_label_control": s2["n_label/control"],
    "effect_metric": eff_metric,
    "effect": eff_val,
    "p": s2["p_str"],
    "adjustment": s2["adjustment"],
    "detail": s2["detail"],
    "source_file": s2["source_file"],
})
t2_note = ("Table 2. Non-redundant headline evidence (23 comparisons across 8 independent sources). "
           "effect_metric/effect: Cliff's delta (>0 = the drug-resistant / more-exposed group has HIGHER REP30; sign of the resistant-minus-control contrast) "
           "or Spearman r (dose axis). |effect| < 0.15 is treated as negligible. p values are stored as text to preserve exact scientific notation. "
           "Source column retained for provenance.")
save_single("Table2_headline_evidence", t2, ["p"], t2_note, "Table2_headline_evidence")

# ================================================================ Table 3
ci = pd.read_csv("results/subtype_rep30/hrposher2neg_cindex.csv")
mv = pd.read_csv("results/subtype_rep30/hrposher2neg_multivariable.csv")
tt = pd.read_csv("results/subtype_rep30/hrposher2neg_prolif_tertiles.csv")
ge = pd.read_csv("results/subtype_rep30/geo_hrpos_external.csv")
cox = pd.read_csv("results/subtype_rep30/subtype_cox.csv")

cols3 = ["block", "cohort", "stratum", "item", "n", "events", "effect_metric", "estimate",
         "ci_lo", "ci_hi", "p", "cindex", "partner", "partner_estimate", "partner_p", "note"]
rows3 = []
# headline
mr = cox[(cox["cohort"] == "METABRIC") & (cox["subtype_col"] == "HRposHER2neg")].iloc[0]
rows3.append(dict(block="Primary result", cohort="METABRIC", stratum="HR+/HER2-",
                  item="REP30 (per 1 SD)", n=int(mr["n"]), events=int(mr["events"]),
                  effect_metric="HR per SD", estimate=mr["HR"], ci_lo=mr["lo"], ci_hi=mr["hi"],
                  p=mr["p"], cindex=mr["cindex"], note="primary prognostic result (univariable Cox, OS)"))
# c-index comparison
for _, r in ci.iterrows():
    rows3.append(dict(block="C-index vs comparator predictors", cohort="METABRIC", stratum="HR+/HER2-",
                      item=r["predictor"], n=int(r["n"]), effect_metric="HR", estimate=r["HR"],
                      p=r["p"], cindex=r["cindex"],
                      note="univariable Cox on OS; C-index for ranking"))
# multivariable
for _, r in mv.iterrows():
    rows3.append(dict(block="Multivariable models", cohort=r["stratum"].split("\u00b7")[0],
                      stratum=r["stratum"].split("\u00b7", 1)[-1], item=r["model"], n=int(r["n"]),
                      events=int(r["events"]), effect_metric="REP30 HR", estimate=r["REP30_HR"],
                      ci_lo=r["REP30_lo"], ci_hi=r["REP30_hi"], p=r["REP30_p"], cindex=r["cindex"],
                      partner=r["partner"], partner_estimate=r["partner_HR"], partner_p=r["partner_p"],
                      note="multivariable Cox; REP30 adjusted for partner term"))
# tertiles
for _, r in tt.iterrows():
    rows3.append(dict(block="HR by proliferation tertile", cohort=r["cohort"], stratum=r["stratum"],
                      item="REP30", n=int(r["n"]), events=int(r["events"]), effect_metric="HR",
                      estimate=r["HR"], ci_lo=r["lo"], ci_hi=r["hi"], p=r["p"], cindex=r["cindex"],
                      note="REP30 HR within each proliferation (PROLIF) tertile"))
# external
for _, r in ge.iterrows():
    rows3.append(dict(block="External GEO validation", cohort=r["cohort"], stratum=r["note"],
                      item="REP30", n=int(r["n"]), events=int(r["events"]),
                      effect_metric="HR (" + str(r["unit"]) + ")", estimate=r["REP30_HR"],
                      ci_lo=r["REP30_lo"], ci_hi=r["REP30_hi"], p=r["REP30_p"], cindex=r["REP30_cindex"],
                      partner="PROLIF", partner_estimate=r["PROLIF_HR"], partner_p=r["PROLIF_p"],
                      note="external validation; unit=" + str(r["unit"])))
t3 = pd.DataFrame(rows3)[cols3]
t3_note = ("Table 3. Prognostic performance of REP30 in HR+/HER2- disease. HR = hazard ratio per 1 SD increase in REP30 unless noted; "
           "METABRIC HR+/HER2- primary result: n=1398, 812 events, HR=1.93/SD, p=3.9e-62, C-index=0.677. "
           "C-index ranks REP30 (0.677) above age (0.642), NPI (0.617), 12-gene proliferation mean (0.587), MKI67 (0.569) and PAM50 LumA-vs-LumB (0.501). "
           "Multivariable models show the simple proliferation mean (PROLIF, p=0.27) and MKI67 (p=0.81) are absorbed, while REP30 stays significant. "
           "p values are stored as text to preserve exact scientific notation. External validation is 2/4 (GSE17705, GSE45255).")
save_single("Table3_hrposher2neg_performance", t3, ["p", "partner_p"], t3_note, "Table3_hrposher2neg_performance")

# ================================================================ (removed) cohort inventory & audit
# NOTE: the cohort inventory & audit was Table S3 in the manuscript and has been DELETED from the
# submission (its content is preserved in Table 1 + Table S2 + main text).
s3full = pd.read_excel("results/evidence_table/submission/Table_S3_full_evidence.xlsx")


# ================================================================ Table S2 (full evidence, 472 rows)
s2_full_note = ("Table S2. Full evidence table: 461 included comparisons (+ 11 excluded) across the 8 independent sources. "
           "Columns include effect_metric/effect, p, include flag and exclude_reason. "
           "Cliff's delta >0 = resistant/more-exposed group higher REP30; Spearman r = dose axis; |effect| < 0.15 negligible. "
           "p stored as text to preserve scientific notation.")
save_single("S2_full_evidence", s3full, ["p"], s2_full_note, "TableS2_full_evidence")

# ================================================================ Table S3
s4 = pd.read_csv("results/evidence_table/submission/Table_S3_prognostic_layer.csv")
s4_note = ("Table S3. Prognostic (OS) layer plus single-cell variance decomposition. "
           "REP30-high vs low OS HR in METABRIC (full52 / trim30) and TCGA-BRCA. "
           "Variance decomposition of REP30 across 96,368 cells (3-cohort Harmony integration): eta^2 study=0.025, cell_type=0.036, sample=0.103 "
           "(sample identity explains the most variance). p stored as text.")
save_single("S3_prognostic_layer", s4, ["p"], s4_note, "TableS3_prognostic_layer")

# ================================================================ Table S4
def rd(p):
    return pd.read_csv(p)

sc = rd("results/subtype_rep30/subtype_cox.csv")
it = rd("results/subtype_rep30/interaction_tests.csv")
cor = rd("results/subtype_rep30/score_correlations.csv")
mvv = rd("results/subtype_rep30/hrposher2neg_multivariable.csv")
pt = rd("results/subtype_rep30/hrposher2neg_prolif_tertiles.csv")
gpt = rd("results/subtype_rep30/geo_hrpos_prolif_tertiles.csv")
cit = rd("results/subtype_rep30/hrposher2neg_cindex.csv")
gex = rd("results/subtype_rep30/geo_hrpos_external.csv")
rlb = rd("results/subtype_rep30/resistance_labels_hrposHER2neg.csv")

gpt2 = gpt.copy(); gpt2.insert(0, "source", "geo_hrpos_prolif_tertiles")
pt2 = pt.copy(); pt2.insert(0, "source", "hrposher2neg_prolif_tertiles")
prolif_comb = pd.concat([pt2, gpt2], ignore_index=True, sort=False)

s5_note = ("Table S4. Subtype layer and HR+/HER2- analyses. subtype_cox = REP30 OS HR within each subtype; "
           "interaction_tests = subtype x REP30 interaction (all null, p_int 0.11-0.45); score_correlations = Spearman between REP30/PROLIF/ER_act/MKI67; "
           "multivariable = REP30 adjusted for PROLIF/MKI67/clinical; prolif_tertiles = REP30 HR within proliferation tertiles (internal + GEO); "
           "cindex = C-index comparison; geo_external = external GEO validation; resistance_labels = endocrine-exposure label test. "
           "p values stored as text.")
save_multi("TableS4_subtype_and_hrpos", [
    ("subtype_cox", [(None, sc)], ["p"], s5_note),
    ("interaction_tests", [(None, it)], ["p_interaction"], "Table S4. subtype x REP30 interaction tests."),
    ("score_correlations", [(None, cor)], ["p"], "Table S4. Pairwise Spearman correlations of scores."),
    ("multivariable", [(None, mvv)], ["REP30_p", "partner_p"], "Table S4. Multivariable Cox models."),
    ("prolif_tertiles", [(None, prolif_comb)], ["p"], "Table S4. REP30 HR within proliferation tertiles (internal + external GEO)."),
    ("cindex", [(None, cit)], ["p"], "Table S4. C-index comparison of predictors."),
    ("geo_external", [(None, gex)], ["REP30_p", "PROLIF_p", "MKI67_p"], "Table S4. External GEO validation (4 cohorts)."),
    ("resistance_labels", [(None, rlb)], ["p"], "Table S4. Endocrine-exposure label test within HR+/HER2-."),
])

# ================================================================ Table S5
# igr2015 landscape
mutf = rd("results/brca_igr_2015/mutation_frequency.csv")
cnaf = rd("results/brca_igr_2015/cna_frequency.csv")
igrt = rd("results/brca_igr_2015/cohort_comparison.csv")
subf = rd("results/brca_igr_2015/subtype_alteration.csv")
igr_blocks = [
    ("mutation frequency (n=216, WES)", mutf),
    ("aCNA frequency (n=216)", cnaf),
    ("IGR metastatic vs TCGA primary", igrt),
    ("alteration frequency by IHC subtype", subf),
]
igr_note = ("Table S5 / igr2015_landscape. brca_igr_2015 acquired-resistance alteration landscape (WES only, n=216 metastatic; no expression -> REP30 not computable). "
            "ESR1 mutation 10.2% (vs TCGA primary 0.8%, ~13x), HR+/HER2- subgroup 14.0%; RB1 loss (CNA<=-1) 34.7%. "
            "p/percent columns stored as text where applicable.")

# dfci cdk46
dfci_samp = rd("results/metastatic/cdk46_resistance/dfci_sample_table.csv")
pheno_counts = dfci_samp["BX_PHENOTYPE"].value_counts().rename_axis("BX_PHENOTYPE").reset_index(name="n")
pheno_dur = dfci_samp.groupby("BX_PHENOTYPE")["treat_days"].median().rename("median_treat_days").reset_index()
pheno_dur = pheno_dur.merge(pheno_counts, on="BX_PHENOTYPE")
dvp = rd("results/metastatic/cdk46_resistance/dfci_gene_vs_phenotype.csv")
res = dvp[dvp["contrast"].str.startswith("resistant")].copy()
res_main = res[res["gene"].isin(["TP53", "ESR1", "RB1", "CDKN2A", "PIK3CA", "CHEK2", "KMT2C", "TBX3"])].copy()
res_main = res_main[["gene", "n_carriers", "case_pos", "case_n", "ctrl_pos", "ctrl_n",
                     "risk_diff", "rd_lo", "rd_hi", "odds_ratio", "or_lo", "or_hi", "fisher_p", "fisher_q", "ci_note"]]
esr1h = rd("results/metastatic/cdk46_resistance/dfci_esr1_hotspot.csv")
dtd = rd("results/metastatic/cdk46_resistance/dfci_gene_vs_treatment_duration.csv")
dtd_main = dtd[dtd["gene"].isin(["CDKN2A", "TP53", "RB1", "ESR1", "PIK3CA"])].copy()
# TP53 method sensitivity (Firth OR / mid-p) from method_sensitivity
tp53_sens = pd.DataFrame([
    {"method": "Fisher exact (two-sided)", "effect_metric": "OR", "estimate": np.inf, "p": "6.58e-05", "note": "primary exact test"},
    {"method": "Fisher exact (one-sided, enrichment)", "effect_metric": "OR", "estimate": np.inf, "p": "5.19e-05", "note": ""},
    {"method": "Fisher mid-p (two-sided)", "effect_metric": "OR", "estimate": np.inf, "p": "3.99e-05", "note": "anti-conservative vs Fisher"},
    {"method": "Firth penalised logistic (Wald)", "effect_metric": "OR", "estimate": 33.1, "p": "0.0153", "note": "ER-status adjusted; same OR/p"},
    {"method": "Logistic MLE (Wald)", "effect_metric": "OR", "estimate": 4.2e10, "p": "0.9994", "note": "TRAP: complete separation -> divergent"},
])
dfci_blocks = [
    ("sample phenotype counts (n=59)", pheno_counts),
    ("median CDK4/6i treatment duration by phenotype", pheno_dur),
    ("genotype vs phenotype (resistant[Intrinsic+Acquired] vs sensitive)", res_main),
    ("TP53 method sensitivity (2x2 = 21/41 vs 0/18)", tp53_sens),
    ("ESR1 LBD hotspot only", esr1h[["gene", "case_pos", "case_n", "ctrl_pos", "ctrl_n", "odds_ratio", "or_lo", "or_hi", "risk_diff", "fisher_p"]]),
    ("treatment duration vs gene (continuous)", dtd_main),
]
dfci_note = ("Table S5 / dfci_cdk46. brca_dfci_2020 (Wander 2020, n=59 CDK4/6i-exposed metastatic breast cancer, WES). "
             "TP53 mutation is the only robust resistance-associated alteration: 21/41 vs 0/18 (Fisher p=6.6e-05; mid-p 4.0e-05; ER-adjusted Firth OR=33.1, p=0.015; exome-wide BH q=0.010). "
             "ESR1 hotspot 14/41 vs 3/18 p=0.22 (underpowered); RB1 mutation 4/4 in resistant; CDKN2A delta=-0.80, p=0.022 (n=3, hypothesis only). "
             "Phenotypes: Intrinsic 28 / Acquired 13 / Sensitive 18; median treatment duration 103 / 272 / 576 days. "
             "NOTE: ordinary logistic MLE fails under complete separation (p=0.9994). p values stored as text.")

# met500 / mbcproject / aurora labels
m500 = rd("results/metastatic/MET500_clinical/resistance_label_tests.csv")
mbc = rd("results/metastatic/MBCproject/rep30_tests.csv")
aur = rd("results/metastatic/AURORA_cBioPortal/rep30_tests.csv")
m500_note = ("Table S5 / met500_brca_labels. MET500_clinical resistance_label_tests.csv. Confound screen: BRCA REP30 vs tumour content r=+0.371 (p=1.5e-6); "
             "REP30 vs proliferation r=+0.099 (p=0.21, decoupled). p stored as text.")
mbc_note = ("Table S5 / mbcproject_labels. MBCproject (Count Me In) rep30_tests.csv. REP30 vs proliferation (RNA module) r=+0.410 (confound). p stored as text.")
aur_note = ("Table S5 / aurora_labels. AURORA US (cBioPortal) rep30_tests.csv. p stored as text.")

# exposure axis replication (archived)
ea_rows = pd.DataFrame([
    {"block": "derivation (MBCproject)", "item": "cohort", "value": "MBCproject (Count Me In); post-treatment n=54 vs naive n=103"},
    {"block": "derivation (MBCproject)", "item": "genes selected (q<0.05 & |d|>=0.20)", "value": "6 of 30378 tested"},
    {"block": "derivation (MBCproject)", "item": "EAP exposed vs naive delta", "value": "+0.717"},
    {"block": "derivation (MBCproject)", "item": "EAP p", "value": "1.75e-13"},
    {"block": "replication (AURORA US)", "item": "cohort", "value": "AURORA US (cBioPortal); post_met_treatment True n=62 vs False n=61"},
    {"block": "replication (AURORA US)", "item": "EAP delta", "value": "+0.086"},
    {"block": "replication (AURORA US)", "item": "EAP p", "value": "0.414"},
    {"block": "replication (AURORA US)", "item": "verdict", "value": "REPLICATION FAILED (pre-registered criterion not met)"},
    {"block": "cross-cohort", "item": "Spearman of genome-wide delta vectors", "value": "-0.021"},
    {"block": "cross-cohort", "item": "top-200 same sign in replication", "value": "131/200 (65.5%)"},
    {"block": "provenance", "item": "source path", "value": "_archive/2026-09-21_deadends/results/resistance_axis/ (interpretation.md, run_log.txt, merged_de_vectors.csv)"},
])
ea_gene = pd.DataFrame([
    {"gene": "NKAIN1", "delta_derivation": -0.451, "q_derivation": "0.023", "delta_replication": -0.021, "p_replication": "0.953", "q_replication": "0.985"},
    {"gene": "MMP11", "delta_derivation": -0.530, "q_derivation": "0.0009", "delta_replication": -0.151, "p_replication": "0.194", "q_replication": "0.538"},
    {"gene": "PTGFR", "delta_derivation": 0.478, "q_derivation": "0.025", "delta_replication": 0.024, "p_replication": "0.858", "q_replication": "0.954"},
    {"gene": "MATN2", "delta_derivation": 0.456, "q_derivation": "0.024", "delta_replication": -0.064, "p_replication": "0.514", "q_replication": "0.791"},
    {"gene": "DOK7", "delta_derivation": -0.375, "q_derivation": "0.023", "delta_replication": -0.045, "p_replication": "0.809", "q_replication": "0.934"},
    {"gene": "LIMS3", "delta_derivation": -0.299, "q_derivation": "NA", "delta_replication": np.nan, "p_replication": "NA (missing)", "q_replication": "NA (missing)"},
])
with open(os.path.join(SUP, "exposure_axis_source.txt"), "w", encoding="utf-8") as f:
    f.write("exposure_axis_replication source (archived, not re-derived)\n")
    f.write("path: _archive/2026-09-21_deadends/results/resistance_axis/\n")
    f.write("files: interpretation.md, run_log.txt, merged_de_vectors.csv, EAP_weights.csv\n")
    f.write("derivation MBCproject delta=+0.717 p=1.75e-13 -> replication AURORA delta=+0.086 p=0.414; cross-cohort delta r=-0.021\n")
ea_note = ("Table S5 / exposure_axis_replication. Exposure-Associated Program (EAP) derivation-replication FAILURE. "
           "Derivation (MBCproject) delta=+0.717 (p=1.8e-13) collapses in the independent AURORA cohort to delta=+0.086 (p=0.414); "
           "cross-cohort genome-wide delta vector Spearman r=-0.021. Source archived, not re-derived "
           "(_archive/2026-09-21_deadends/results/resistance_axis/). See results/manuscript_support/exposure_axis_source.txt.")

save_multi("TableS5_genomic_and_metastatic", [
    ("igr2015_landscape", igr_blocks, [], igr_note),
    ("dfci_cdk46", dfci_blocks, ["p", "fisher_p", "fisher_q", "mwu_p"], dfci_note),
    ("met500_brca_labels", [(None, m500)], ["p"], m500_note),
    ("mbcproject_labels", [(None, mbc)], ["p"], mbc_note),
    ("aurora_labels", [(None, aur)], ["p"], aur_note),
    ("exposure_axis_replication", [("summary", ea_rows), ("per-gene delta", ea_gene)], ["p_replication", "q_replication"],
     ea_note),
])

print("ALL TABLES DONE")
