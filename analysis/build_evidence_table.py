#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_evidence_table.py   (v2 -- submission build)
==================================================
Assemble the REP30 "does it capture the resistant state?" evidence base into a
publication-grade main figure plus supplementary tables, under an explicit
curation policy:

    POLICY  keep only data that carries independent evidential weight for the
            claim; anything redundant or non-informative is dropped, and the
            drop is documented.

Two consequences of that policy are encoded here:

  * `brca_mbcproject_wagle_2017` is EXCLUDED. It is not an independent cohort:
    it is an earlier release of the same Count Me In MBCproject, sharing
    71.9% of RNA samples and 99.1% of RNA patients with `brca_mbcproject_2022`.
    Widening to the union adds 1 patient -> pseudo-replication, not power.
  * The two AURORA US access routes (GEO GSE209998 and cBioPortal
    brca_aurora_2023) are ONE cohort: the same 53 patients (Jaccard = 1.000).
    They are merged into a single cohort entry and never counted twice.

So the "nine cohorts" of the working draft become EIGHT independent cohorts,
plus one derived single-cell synthesis.

OUTPUTS (results/evidence_table/ and results/evidence_table/submission/)
    evidence_table_full.csv              every comparison ingested (with an
                                         include/exclude decision per row)
    evidence_table_headline.csv          curated rows drawn in the main figure
    evidence_table.md                    human-readable headline tables
    submission/Fig1_evidence_main.png    300 dpi raster
    submission/Fig1_evidence_main.pdf    vector, for typesetting
    submission/Table_S1_cohort_inventory.xlsx   cohort list + redundancy audit
    submission/Table_S2_headline_evidence.xlsx  headline (main-figure) table
    submission/Table_S3_full_evidence.xlsx      all 470+ comparisons
    submission/Table_S4_prognostic_layer.xlsx   the positive, different question
    submission/audit_overlap.json        machine-readable overlap numbers
    run_log.txt

SIGN CONVENTION (uniform everywhere)
    effect > 0  ->  REP30 is HIGHER in the resistance / more-exposed group
    effect < 0  ->  REP30 is LOWER  in the resistance / more-exposed group
Effect metric is Cliff's delta for two-group contrasts and Spearman r for dose
axes; both live on a comparable [-1, 1] magnitude scale.

Run with the managed venv (has matplotlib + openpyxl):
  C:/Users/mao02/.workbuddy/binaries/python/envs/default/Scripts/python.exe \
      analysis/build_evidence_table.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

R = r"E:\01.WorkSpace\01.Breast_cancer_gene_panel\resistance_signature"
RES = os.path.join(R, "results")
OUT = os.path.join(RES, "evidence_table")
SUB = os.path.join(OUT, "submission")
os.makedirs(OUT, exist_ok=True)
os.makedirs(SUB, exist_ok=True)

LOG = []


def log(s=""):
    print(s)
    LOG.append(str(s))


def _rd(p):
    return pd.read_csv(p, low_memory=False)


def cliffs_delta(a, b):
    """P(a>b) - P(a<b). Positive -> a tends to exceed b."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return np.nan
    a = a[:, None]
    b = b[None, :]
    return float(((a > b).sum() - (a < b).sum()) / (na * nb))


def split_n(s):
    """'6/117' -> (6, 117); 157 -> (nan, 157)."""
    s = str(s)
    if "/" in s:
        try:
            return int(s.split("/")[0]), int(s.split("/")[1])
        except Exception:
            return np.nan, np.nan
    try:
        return np.nan, int(float(s))
    except Exception:
        return np.nan, np.nan


def fnum(v, nd=3):
    if v is None:
        return "-"
    try:
        v = float(v)
    except Exception:
        return str(v)
    if not np.isfinite(v):
        return "-"
    return f"{v:.{nd}g}"


# =====================================================================
# COHORT REGISTRY  (the curation layer)
# =====================================================================
# cohort_id -> canonical description.  `datasets` lists every result-file
# `dataset` string that belongs to that cohort.
COHORT_REG = {
    "C1": dict(cohort="GSE26459", label="GSE26459 (MCF-7, 4-OHT acquired)",
               ctype="bulk in-vitro model", tumour="cell line",
               access="GEO GSE26459", role="acquired-resistance model"),
    "C2": dict(cohort="GSE45255", label="GSE45255 (tumour, tamoxifen-treated)",
               ctype="bulk primary", tumour="primary",
               access="GEO GSE45255", role="subtype surrogate (ER-)"),
    "C3": dict(cohort="TCGA-BRCA", label="TCGA-BRCA",
               ctype="bulk primary", tumour="primary",
               access="GDC / public processed matrices", role="labels + prognosis"),
    "C4": dict(cohort="METABRIC", label="METABRIC",
               ctype="bulk primary", tumour="primary",
               access="cBioPortal brca_metabric", role="labels + prognosis"),
    "C5": dict(cohort="GSE264205", label="GSE264205 (longitudinal pre/post/metastatic scRNA)",
               ctype="single-cell", tumour="primary + metastasis",
               access="GEO GSE264205", role="tumour-state trajectory"),
    "C6": dict(cohort="AURORA US", label="AURORA US (metastatic biopsy programme)",
               ctype="bulk metastatic", tumour="primary + metastasis",
               access="GEO GSE209998 + cBioPortal brca_aurora_2023 (same 53 patients)",
               role="labels, tumour state, treatment dose"),
    "C7": dict(cohort="MET500", label="MET500 (Michigan metastatic biopsy)",
               ctype="bulk metastatic", tumour="metastasis",
               access="MET500 public tables", role="labels, pan-cancer vs breast"),
    "C8": dict(cohort="MBCproject", label="MBCproject / Count Me In (2022 release)",
               ctype="bulk metastatic", tumour="mostly primary specimen",
               access="cBioPortal brca_mbcproject_2022", role="labels + treatment exposure"),
    "S1": dict(cohort="3-cohort Harmony", label="3-cohort Harmony integration (derived)",
               ctype="single-cell (derived)", tumour="mixed",
               access="derived from C5 + GSE249361 + GSE161529",
               role="variance decomposition (supporting)"),
}

# dataset-string (as written by the ingest below) -> cohort_id
DS2COH = {
    "GSE26459": "C1",
    "GSE45255": "C2",
    "TCGA": "C3",
    "METABRIC": "C4",
    "GSE264205 (scRNA)": "C5",
    "AURORA US (GSE209998)": "C6",
    "AURORA US (cBioPortal)": "C6",
    "MET500": "C7",
    "MBCproject (Count Me In)": "C8",
    "3-cohort Harmony (scRNA)": "S1",
}

# curated exclusion registry -- what was dropped and exactly why
EXCLUDED = [
    dict(dataset="brca_mbcproject_wagle_2017",
         reason="not an independent cohort -- earlier release of the same Count Me In "
                "MBCproject as brca_mbcproject_2022",
         overlap="71.9% of RNA samples and 99.1% of RNA patients shared; union adds 1 patient",
         decision="excluded (pseudo-replication would inflate n and shrink p)",
         detail_path="results/metastatic/wagle2017/interpretation.md"),
]

# =====================================================================
# INGEST
# =====================================================================
rows = []


def add(**kw):
    rows.append(kw)


# ---- 1. bulk cell line / tumour GEO resistance series --------------------
for gse, tag in [("GSE26459", "bulk-cell-line, MCF-7 4-OHT acquired"),
                 ("GSE45255", "bulk-tumour, Tamoxifen ER- surrogate")]:
    f = os.path.join(RES, "gse_resistance", gse, "rep_score_by_group.csv")
    d = _rd(f)
    col = "REPtrim_score" if "REPtrim_score" in d.columns else "REP52_score"
    res = d.loc[d["group"] == "resistant", col].values
    sen = d.loc[d["group"] == "sensitive", col].values
    add(dataset=gse, cohort_type="bulk", layer="expression",
        endpoint_class="molecular-resistance-genotype" if gse == "GSE26459" else "subtype-surrogate",
        endpoint=("4-OHT acquired resistance (resistant vs sensitive)" if gse == "GSE26459"
                  else "ER- (intrinsic tamoxifen resistance surrogate) vs ER+"),
        n_carrier=len(res), n_control=len(sen), n_total=len(d),
        effect_metric="cliffs_delta", effect=cliffs_delta(res, sen),
        p=stats.mannwhitneyu(res, sen, alternative="two-sided")[1],
        adjustment="none",
        detail=f"mean {res.mean():.3f} vs {sen.mean():.3f}; {tag}",
        source_file=os.path.relpath(f, R))

# ---- 2. TCGA / METABRIC reported resistance labels -----------------------
for sub, tag in [("label_comparison.csv", "all"),
                 (os.path.join("within_er", "label_comparison.csv"), "ER-restricted")]:
    f = os.path.join(RES, "rep30_resistance_validation", sub)
    d = _rd(f)
    for _, r in d.iterrows():
        lab = str(r["label"])
        is_tx = any(t in lab for t in ["treated", "therapy", "chemotherapy", "immunotherapy"])
        add(dataset=r["cohort"], cohort_type="bulk-primary", layer="expression",
            endpoint_class=("treatment-exposure" if is_tx else "molecular-resistance-genotype"),
            endpoint=lab,
            n_carrier=int(r["n_label"]), n_control=int(r["n_ctrl"]),
            n_total=int(r["n_label"]) + int(r["n_ctrl"]),
            effect_metric="cliffs_delta", effect=float(r["cliffs_delta"]), p=float(r["p"]),
            adjustment="none" if tag == "all" else "ER-restricted",
            detail=f"median {r['median_label']:.3f} vs {r['median_ctrl']:.3f}",
            source_file=os.path.relpath(f, R))

# ---- 3. single-cell GSE264205 -------------------------------------------
f = os.path.join(RES, "sc", "GSE264205", "reload", "rep30_comparisons.csv")
d = _rd(f)
# group_A = earlier/less-resistant state, group_B = later/more-resistant state.
# Cliff's delta here is P(A>B)-P(A<B) => flip sign into the resistance convention.
for _, r in d.iterrows():
    add(dataset="GSE264205 (scRNA)", cohort_type="single-cell", layer="expression",
        endpoint_class="tumour-state",
        endpoint=f"{r['compartment']}: {r['group_A']} vs {r['group_B']}",
        n_carrier=int(r["n_B"]), n_control=int(r["n_A"]),
        n_total=int(r["n_A"]) + int(r["n_B"]),
        effect_metric="cliffs_delta", effect=-float(r["cliffs_delta"]),
        p=float(r["mwu_p"]),
        adjustment="none (sample offset not removed)",
        detail=f"per-cell score, {int(r['n_genes_used'])}/30 panel genes present",
        source_file=os.path.relpath(f, R))

# ---- 4. AURORA US, GEO access route --------------------------------------
f = os.path.join(RES, "metastatic", "AURORA_US_rep30", "rep30_comparisons.csv")
d = _rd(f)
for _, r in d.iterrows():
    if r["score"] != "REP30":
        continue
    con = str(r["contrast"])
    if "FFPE vs fresh frozen" in con:
        cls = "confound"                       # preservation / assay artefact
    elif "vs Normal" in con:
        cls = "other"                          # tumour vs normal tissue, not resistance
    elif con.startswith("Metastatic vs") or con.startswith("Met site"):
        cls = "tumour-state"                   # metastasis, or one site vs another
    elif "Post vs Pre" in con or "Post-treatment vs Pre-treatment" in con:
        cls = "treatment-exposure"
    else:
        cls = "other"
    add(dataset="AURORA US (GSE209998)", cohort_type="bulk-metastatic", layer="expression",
        endpoint_class=cls, endpoint=con,
        n_carrier=int(r["n_A"]), n_control=int(r["n_B"]),
        n_total=int(r["n_A"]) + int(r["n_B"]),
        effect_metric="cliffs_delta", effect=float(r["cliffs_delta"]), p=float(r["p_mwu"]),
        adjustment="FFPE/fresh-frozen shown separately",
        detail=f"median {r['median_A']:.3f} vs {r['median_B']:.3f}",
        source_file=os.path.relpath(f, R))

# ---- 5. AURORA US, cBioPortal access route -------------------------------
AUR_CB = os.path.join(RES, "metastatic", "AURORA_cBioPortal", "rep30_tests.csv")
d = _rd(AUR_CB)
for _, r in d.iterrows():
    t = str(r["test"])
    if t.startswith("REP30 vs") and ("mutation" in t or "loss" in t or "amplification" in t):
        cls = "molecular-resistance-genotype"
    elif t.startswith("REP30") and ("days" in t or "lines" in t or "start dates" in t):
        cls = "treatment-exposure"
    elif ("proliferation" in t or "MKI67" in t or "tumour cells" in t or "stromal" in t):
        cls = "confound"
    elif "post-treatment vs" in t or "post- vs pre-metastatic" in t:
        cls = "treatment-exposure"
    elif "metastatic vs primary" in t:
        cls = "tumour-state"
    else:
        cls = "other"
    nc, nt = split_n(r["n"])
    adj = "none"
    if "[adjusted" in t or "[prolif" in t:
        adj = "prolif+TC+stroma"
    if t.startswith("ER-target score"):
        cls = "other"
    add(dataset="AURORA US (cBioPortal)", cohort_type="bulk-metastatic", layer="expression",
        endpoint_class=cls, endpoint=f"{r['scope']} | {t}",
        n_carrier=nc, n_control=nt, n_total=nt,
        effect_metric=("spearman_r" if t.endswith("(Spearman)") or "partial Spearman" in str(r["detail"])
                       else "cliffs_delta"),
        effect=float(r["statistic"]) if pd.notna(r["statistic"]) else np.nan,
        p=float(r["p"]) if pd.notna(r["p"]) else np.nan,
        adjustment=adj, detail=str(r["detail"]),
        source_file=os.path.relpath(AUR_CB, R))

# ---- 6. MET500 -----------------------------------------------------------
MET = os.path.join(RES, "metastatic", "MET500_clinical", "resistance_label_tests.csv")
d = _rd(MET)
for _, r in d.iterrows():
    t = str(r["test"])
    if t.startswith("REP30 vs") and ("mutation" in t or "loss" in t or "amplification" in t
                                     or "alteration" in t):
        cls = "molecular-resistance-genotype"
    elif ("proliferation" in t or "MKI67" in t or "tc (" in t or "age" in t
          or "interval" in t):
        cls = "confound"
    elif "biopsy sites" in t:
        cls = "other"                          # Kruskal-Wallis statistic, not on [-1,1]
    elif "prior therapy" in t:
        cls = "treatment-exposure"
    else:
        cls = "other"
    nc, nt = split_n(r["n"])
    add(dataset="MET500", cohort_type="bulk-metastatic", layer="expression",
        endpoint_class=cls, endpoint=f"{r['scope']} | {t}",
        n_carrier=nc, n_control=nt, n_total=nt,
        effect_metric=("spearman_r" if ("Spearman" in t or "partial spearman" in str(r["detail"]))
                       else "cliffs_delta"),
        effect=float(r["statistic"]) if pd.notna(r["statistic"]) else np.nan,
        p=float(r["p"]) if pd.notna(r["p"]) else np.nan,
        adjustment=("prolif+TC" if "[prolif+tc" in t else "prolif" if "[prolif-adjusted]" in t else "none"),
        detail=str(r["detail"]), source_file=os.path.relpath(MET, R))

# ---- 7. MBCproject -------------------------------------------------------
MBC = os.path.join(RES, "metastatic", "MBCproject", "rep30_tests.csv")
d = _rd(MBC)
for _, r in d.iterrows():
    t = str(r["test"])
    sc = str(r["scope"])
    if t.startswith("REP30 vs") and ("mutation" in t or "loss" in t or "amplification" in t):
        cls = "molecular-resistance-genotype"
    elif t.startswith("REP30 vs") and ("days" in t or "lines" in t or "start dates" in t):
        cls = "treatment-exposure"
    elif t.startswith("REP30") and ("proliferation" in t or "MKI67" in t or "TMB" in t
                                    or "Ki67" in t or "days from reference" in t
                                    or "time to metastatic" in t):
        cls = "confound"
    elif t.startswith("REP30") and ("HR+" in t or "TNBC" in t or "HER2+" in t or "ER+ biopsy" in t):
        cls = "subtype"
    elif t.startswith("REP30") and ("post-treatment vs" in t):
        cls = "treatment-exposure"
    elif t.startswith("dREP30") or t.startswith("REP30 change"):
        cls = "treatment-exposure"
    elif t.startswith("ER-target score") or t.startswith("REP30 vs ER-target"):
        cls = "other"
    elif "by " in t and "KW" in t.upper():
        cls = "confound"
    else:
        cls = "other"
    nc, nt = split_n(r["n"])
    adj = "none"
    if "[adj" in t:
        adj = "prolif" + ("+ER-status" if "er_pos" in str(r["detail"]) or "ER-status" in t else "")
    if "| ER-status adjusted" in t:
        adj = "ER-status"
    add(dataset="MBCproject (Count Me In)", cohort_type="bulk-metastatic", layer="expression",
        endpoint_class=cls, endpoint=f"{sc} | {t}",
        n_carrier=nc, n_control=nt, n_total=nt,
        effect_metric=("spearman_r" if ("Spearman" in t or "partial Spearman" in str(r["detail"]))
                       else "cliffs_delta"),
        effect=float(r["statistic"]) if pd.notna(r["statistic"]) else np.nan,
        p=float(r["p"]) if pd.notna(r["p"]) else np.nan,
        adjustment=adj, detail=str(r["detail"]), source_file=os.path.relpath(MBC, R))

# ---- 8. prognostic layer (POSITIVE evidence, different question) ---------
f = os.path.join(RES, "rep52_trim", "backtest_results.csv")
d = _rd(f)
for _, r in d.iterrows():
    if r["model"] != "high_vs_low_median":
        continue
    coh, scope, panel = str(r["prefix"]).split("|")
    add(dataset=(coh if scope == "all" else f"{coh} ({scope})"),
        cohort_type="bulk-primary", layer="expression",
        endpoint_class="prognostic-OS",
        endpoint=f"OS, REP30-high vs low ({panel})",
        n_carrier=np.nan, n_control=np.nan, n_total=int(r["n"]),
        effect_metric="HR", effect=float(r["HR"]), p=float(r["p"]),
        adjustment="univariable",
        detail=f"HR 95% CI {r['lower']:.2f}-{r['upper']:.2f}",
        source_file=os.path.relpath(f, R))

# ---- 9. single-cell variance decomposition -------------------------------
f = os.path.join(RES, "sc", "integrated", "report", "integrated_rep_tests.csv")
d = _rd(f)
for _, r in d.iterrows():
    t = str(r["test"])
    if not t.startswith("eta^2"):
        continue
    add(dataset="3-cohort Harmony (scRNA)", cohort_type="single-cell", layer="expression",
        endpoint_class="variance-decomposition",
        endpoint=t.replace("eta^2 REP_score ~ ", "eta^2 REP ~ "),
        n_carrier=np.nan, n_control=np.nan, n_total=int(r["n"]),
        effect_metric="eta_squared", effect=float(r["statistic"]), p=np.nan,
        adjustment="none", detail="variance explained by each factor",
        source_file=os.path.relpath(f, R))

full = pd.DataFrame(rows)

# =====================================================================
# attach the curation columns
# =====================================================================
full["cohort_id"] = full["dataset"].map(DS2COH)
missing = sorted(set(full.loc[full["cohort_id"].isna(), "dataset"]))
if missing:
    log(f"!! datasets with no cohort_id mapping: {missing}")
full["cohort"] = full["cohort_id"].map({k: v["cohort"] for k, v in COHORT_REG.items()})
full["access_route"] = full["dataset"]
full["include"] = "yes"
full["exclude_reason"] = ""

# Column order that reads well in a spreadsheet
COLS = ["cohort_id", "cohort", "dataset", "access_route", "cohort_type", "layer",
        "endpoint_class", "endpoint", "n_carrier", "n_control", "n_total",
        "effect_metric", "effect", "p", "adjustment", "detail", "include",
        "exclude_reason", "source_file"]
full = full[COLS]
# cosmetic: a missing source "detail" cell became the string "nan" during ingest
full["detail"] = ["-" if str(x).strip().lower() in ("nan", "none", "") else str(x)
                  for x in full["detail"]]

# ---- de-duplicate statistics that the source files report twice ----------
# Several source tables emit the SAME test under two scope labels (e.g. AURORA
# "Metastatic | ..." and "treatment-exposure/met-only | ...").  Identical
# effect AND p means one test, so the second copy is flagged, not deleted --
# the row stays in the supplement for auditability but is excluded from every
# count reported in the figure and in the text.
DUPE_KEY = ["cohort_id", "endpoint_class", "effect_metric", "effect", "p"]
_is_dupe = full.duplicated(subset=DUPE_KEY, keep="first") & full["effect"].notna()
full.loc[_is_dupe, "include"] = "no"
full.loc[_is_dupe, "exclude_reason"] = ("identical statistic to an earlier row in the same cohort "
                                        "(the same contrast reported twice, or a collinear locus / "
                                        "nested subset tested on the same samples)")
log(f"duplicate statistics flagged (excluded from all counts): {int(_is_dupe.sum())}")
if _is_dupe.any():
    for _, r in full[_is_dupe].iterrows():
        log(f"    {r['cohort']} | {r['endpoint']} | {r['effect_metric']}={r['effect']:+.3f} "
            f"p={r['p']:.3g}")
INC = full[full["include"] == "yes"].copy()          # the analysis set
log(f"analysis set: {len(INC)} of {len(full)} ingested rows")
full.to_csv(os.path.join(OUT, "evidence_table_full.csv"), index=False, encoding="utf-8")
log(f"full evidence table rows: {len(full)}")
log("rows by cohort:\n" + full.groupby(["cohort_id", "cohort"]).size().to_string())

# =====================================================================
# REDUNDANCY AUDIT -- recomputed from the sample tables, not asserted
# =====================================================================
audit = {}


def _audit_pair(name, a_ids, a_lab, b_ids, b_lab):
    A, B = set(a_ids), set(b_ids)
    inter = A & B
    rec = dict(a=a_lab, b=b_lab,
               n_a=len(A), n_b=len(B), n_shared=len(inter),
               n_a_only=len(A - B), n_b_only=len(B - A),
               pct_of_a=100.0 * len(inter) / max(len(A), 1),
               jaccard=len(inter) / max(len(A | B), 1))
    audit[name] = rec
    log(f"  {name}: {a_lab} n={rec['n_a']} vs {b_lab} n={rec['n_b']} -> "
        f"shared {rec['n_shared']} ({rec['pct_of_a']:.1f}% of first, Jaccard {rec['jaccard']:.3f})")
    return rec


log("\n-- redundancy audit --")
# (a) wagle_2017 vs MBCproject-2022
try:
    wg = _rd(os.path.join(RES, "metastatic", "wagle2017", "wagle2017_linked_samples.csv"))
    m22 = _rd(os.path.join(RES, "metastatic", "MBCproject", "mbcproject_linked_samples.csv"))
    _audit_pair("wagle2017_vs_MBCproject2022_samples",
                wg["SAMPLE_ID"], "wagle2017 RNA samples",
                m22["Unnamed: 0"], "MBCproject-2022 RNA samples")
    _audit_pair("wagle2017_vs_MBCproject2022_patients",
                wg["PATIENT_ID"].dropna(), "wagle2017 patients",
                m22["PATIENT_ID"].dropna(), "MBCproject-2022 patients")
except Exception as e:
    log(f"  wagle/MBCproject audit failed: {type(e).__name__}: {e}")

# (b) AURORA GEO vs AURORA cBioPortal
try:
    geo = _rd(os.path.join(RES, "metastatic", "AURORA_US_rep30", "rep30_per_sample.csv"))
    cbp = _rd(os.path.join(RES, "metastatic", "AURORA_cBioPortal", "aurora_linked.csv"))

    def _strip(s):
        return str(s).upper().replace("AUR-", "").replace("AUR_", "")

    _audit_pair("AURORA_geo_vs_cbioportal_patients",
                [_strip(x) for x in geo["patient"].dropna()], "AURORA GEO patients",
                [_strip(x) for x in cbp["PATIENT_ID"].dropna()], "AURORA cBioPortal patients")
except Exception as e:
    log(f"  AURORA audit failed: {type(e).__name__}: {e}")

with open(os.path.join(SUB, "audit_overlap.json"), "w", encoding="utf-8") as fh:
    json.dump({"overlaps": audit, "excluded": EXCLUDED}, fh, indent=2, ensure_ascii=False)

# =====================================================================
# TABLE S1 -- cohort inventory + audit
# =====================================================================
n_by_coh = INC.groupby("cohort_id").size().to_dict()
inv_rows = []
for cid, meta in COHORT_REG.items():
    n_pat = ""
    n_smp = ""
    if cid == "C6":
        n_pat, n_smp = 53, "129 (GEO) / 123 (cBioPortal)"
    elif cid == "C8":
        n_pat, n_smp = 120, "157 with RNA"
    elif cid == "C5":
        n_pat, n_smp = "", "16,368 cells; 4 timepoints (Pre/Post1/Post2/Meta)"
    elif cid == "C3":
        n_pat, n_smp = 1069, 1069
    elif cid == "C4":
        n_pat, n_smp = 1979, 1979
    elif cid == "C7":
        n_pat, n_smp = 496, "868 (834 with complete labels)"
    elif cid == "C1":
        n_pat, n_smp = 0, 24
    elif cid == "C2":
        n_pat, n_smp = 137, 137
    elif cid == "S1":
        n_pat, n_smp = "", "96,368 cells"
    inv_rows.append(dict(cohort_id=cid, cohort=meta["label"], cohort_type=meta["ctype"],
                         specimen=meta["tumour"], access_route=meta["access"],
                         n_patients=n_pat, n_samples=n_smp,
                         evidential_role=meta["role"],
                         comparisons_in_supplement=n_by_coh.get(cid, 0),
                         included="yes"))
for ex in EXCLUDED:
    inv_rows.append(dict(cohort_id="--", cohort=ex["dataset"], cohort_type="bulk metastatic",
                         specimen="mix", access_route="cBioPortal " + ex["dataset"],
                         n_patients=113, n_samples="146 with RNA",
                         evidential_role="none (redundant release)",
                         comparisons_in_supplement=0, included="NO",
                         ))
inv = pd.DataFrame(inv_rows)

audit_rows = []
for k, v in audit.items():
    audit_rows.append(dict(check=k, set_a=v["a"], n_a=v["n_a"], set_b=v["b"], n_b=v["n_b"],
                           n_shared=v["n_shared"], pct_of_a=round(v["pct_of_a"], 2),
                           jaccard=round(v["jaccard"], 4),
                           verdict=("SAME cohort -- must not be counted twice"
                                    if v["jaccard"] > 0.5 else "partly overlapping")))
audf = pd.DataFrame(audit_rows)
exdf = pd.DataFrame(EXCLUDED)
dropdf = full[full["include"] == "no"][
    ["cohort_id", "cohort", "dataset", "endpoint_class", "endpoint",
     "effect_metric", "effect", "p", "exclude_reason"]]

with pd.ExcelWriter(os.path.join(SUB, "Table_S1_cohort_inventory.xlsx"),
                    engine="openpyxl") as xw:
    inv.to_excel(xw, sheet_name="S1a_cohort_inventory", index=False)
    audf.to_excel(xw, sheet_name="S1b_redundancy_audit", index=False)
    exdf.to_excel(xw, sheet_name="S1c_excluded_datasets", index=False)
    dropdf.to_excel(xw, sheet_name="S1d_excluded_rows", index=False)
inv.to_csv(os.path.join(SUB, "Table_S1_cohort_inventory.csv"), index=False, encoding="utf-8")
audf.to_csv(os.path.join(SUB, "Table_S1b_redundancy_audit.csv"), index=False, encoding="utf-8")
log(f"\nTable S1 written: {len(inv)} cohort rows, {len(audf)} audit rows, "
    f"{len(dropdf)} excluded-row entries")

# =====================================================================
# HEADLINE rows for the main figure
# =====================================================================
HEAD = [
    # --- true / genomic resistance genotypes ------------------------------
    ("GSE26459", "4-OHT acquired resistance", "molecular-resistance-genotype",
     "GSE26459 · 4-OHT acquired resistance"),
    ("TCGA", "ESR1_mutation", "molecular-resistance-genotype",
     "TCGA · ESR1 mutation"),
    ("AURORA US (cBioPortal)", "all-samples | REP30 vs ESR1 mutation",
     "molecular-resistance-genotype", "AURORA · ESR1 mutation"),
    ("MET500", "BRCA | REP30 vs ESR1 mutation", "molecular-resistance-genotype",
     "MET500 · ESR1 mutation (breast)"),
    ("MBCproject (Count Me In)", "all-157 | REP30 vs ESR1 mutation (any nonsyn)",
     "molecular-resistance-genotype", "MBCproject · ESR1 mutation"),
    ("TCGA", "RB1_any_loss", "molecular-resistance-genotype", "TCGA · RB1 loss"),
    ("METABRIC", "RB1_any_loss", "molecular-resistance-genotype", "METABRIC · RB1 loss"),
    ("AURORA US (cBioPortal)", "all-samples | REP30 vs RB1 loss",
     "molecular-resistance-genotype", "AURORA · RB1 loss"),
    ("MET500", "BRCA | REP30 vs RB1 loss", "molecular-resistance-genotype",
     "MET500 · RB1 loss (breast)"),
    ("MBCproject (Count Me In)", "all-157 | REP30 vs RB1 loss",
     "molecular-resistance-genotype", "MBCproject · RB1 loss"),
    # --- subtype surrogate -------------------------------------------------
    ("GSE45255", "ER- (intrinsic tamoxifen resistance surrogate)", "subtype-surrogate",
     "GSE45255 · ER- vs ER+ (subtype)"),
    # --- tumour state ------------------------------------------------------
    ("AURORA US (cBioPortal)", "primary-vs-met | REP30 metastatic vs primary",
     "tumour-state", "AURORA · metastatic vs primary"),
    ("AURORA US (GSE209998)", "Metastatic vs Primary", "tumour-state",
     "AURORA · met vs primary (paired)"),
    ("GSE264205 (scRNA)", "Epithelial/Malignant: Pre vs Meta", "tumour-state",
     "GSE264205 · pre vs met (malignant)"),
    # --- treatment dose ----------------------------------------------------
    ("AURORA US (cBioPortal)", "Metastatic | REP30 vs days on metastatic therapy",
     "treatment-exposure", "AURORA · days on metastatic therapy"),
    ("AURORA US (cBioPortal)", "Metastatic | REP30 vs metastatic lines before biopsy",
     "treatment-exposure", "AURORA · lines before biopsy"),
    ("MBCproject (Count Me In)",
     "serial-biopsy | dREP30 vs change in cumulative therapy days",
     "treatment-exposure", "MBCproject · ΔREP30 vs Δ therapy days"),
    ("MET500", "all-cases | REP30 vs prior therapy", "treatment-exposure",
     "MET500 · prior therapy (Y/N)"),
    # --- confounds (the explanation) ---------------------------------------
    ("MBCproject (Count Me In)", "confound-screen | REP30 vs proliferation", "confound",
     "MBCproject · proliferation"),
    ("AURORA US (cBioPortal)", "all-samples | REP30 vs proliferation", "confound",
     "AURORA · proliferation"),
    ("MET500", "all-cases | REP30 vs proliferation", "confound", "MET500 · proliferation"),
    ("AURORA US (cBioPortal)", "all-samples | REP30 vs percent tumour cells", "confound",
     "AURORA · tumour cells (%)"),
    ("MET500", "BRCA | REP30 vs tc", "confound", "MET500 · tumour content (breast)"),
]

head_rows = []
for ds, key, grp, lab in HEAD:
    sub = full[(full["dataset"] == ds) & (full["endpoint"].str.contains(key, regex=False))]
    if sub.empty:
        log(f"  !! HEADLINE MISS: {ds} | {key}")
        continue
    r = sub.iloc[0].copy()
    r["grp"] = grp
    r["label"] = lab
    head_rows.append(r)
head = pd.DataFrame(head_rows)
head.to_csv(os.path.join(OUT, "evidence_table_headline.csv"), index=False, encoding="utf-8")
log(f"headline rows: {len(head)}")


def _lookup(ds, key):
    s = full[(full["dataset"] == ds) & (full["endpoint"].str.contains(key, regex=False))]
    if s.empty:
        log(f"  !! lookup miss: {ds} | {key}")
        return None
    return s.iloc[0]


# ---- Table S2 : headline -------------------------------------------------
def pretty(df):
    o = df.copy()
    o["n_label/control"] = [
        (f"{fnum(a,4)}/{fnum(b,4)}" if fnum(a, 4) != "-" and fnum(b, 4) != "-" else "-")
        for a, b in zip(o["n_carrier"], o["n_control"])]
    o["effect_str"] = [f"{r['effect_metric']}={r['effect']:+.3f}"
                       if np.isfinite(r["effect"]) else "-" for _, r in o.iterrows()]
    o["p_str"] = [f"{r['p']:.3g}" if np.isfinite(r["p"]) else "-" for _, r in o.iterrows()]
    o["detail"] = ["-" if str(x).strip().lower() in ("nan", "none", "") else str(x)
                   for x in o["detail"]]
    return o[["cohort_id", "cohort", "dataset", "access_route", "endpoint_class", "endpoint",
              "n_label/control", "effect_str", "p_str", "adjustment", "detail", "source_file"]]


pretty(head).to_csv(os.path.join(SUB, "Table_S2_headline_evidence.csv"),
                    index=False, encoding="utf-8")
with pd.ExcelWriter(os.path.join(SUB, "Table_S2_headline_evidence.xlsx"),
                    engine="openpyxl") as xw:
    pretty(head).to_excel(xw, sheet_name="S2_headline", index=False)

# ---- Table S3 : full -----------------------------------------------------
with pd.ExcelWriter(os.path.join(SUB, "Table_S3_full_evidence.xlsx"),
                    engine="openpyxl") as xw:
    full.to_excel(xw, sheet_name="S3_full_evidence", index=False)

# ---- Table S4 : prognostic + variance ------------------------------------
prog = full[full["endpoint_class"].isin(["prognostic-OS", "variance-decomposition"])]
with pd.ExcelWriter(os.path.join(SUB, "Table_S4_prognostic_layer.xlsx"),
                    engine="openpyxl") as xw:
    prog.to_excel(xw, sheet_name="S4_prognostic_variance", index=False)
prog.to_csv(os.path.join(SUB, "Table_S4_prognostic_layer.csv"), index=False, encoding="utf-8")
log(f"Tables S2-S4 written (headline {len(head)}, full {len(full)}, prognostic/variance {len(prog)})")

# =====================================================================
# MARKDOWN supplement
# =====================================================================
CLS_ZH = {
    "molecular-resistance-genotype": "resistance genotype",
    "subtype-surrogate": "subtype surrogate",
    "tumour-state": "tumour state",
    "treatment-exposure": "treatment exposure",
    "confound": "confounder (control)",
    "prognostic-OS": "prognosis",
    "variance-decomposition": "variance decomposition",
    "subtype": "subtype",
    "other": "other",
}
lines = ["# REP30 resistance-state evidence table (submission version: 8 independent cohorts)", "",
         "> Sign convention: effect > 0 = the more resistant/exposed group has a **higher** REP30; effect < 0 = **lower**.",
         "> Effect sizes: Cliff's delta for two-group comparisons, Spearman r for dose axes; both lie in [-1,1].",
         "> After de-redundancy the two AURORA access routes are merged into 1 cohort (same 53 patients, Jaccard = 1.000);",
         "> `brca_mbcproject_wagle_2017` excluded due to 99.1% patient overlap with MBCproject-2022.", "",
         "## Cohort inventory", "",
         "| cohort_id | cohort | type | access route | n(patients) | n(samples) | comparisons in the evidence table |",
         "|---|---|---|---|---|---|---|"]
for _, r in inv.iterrows():
    lines.append(f"| {r['cohort_id']} | {r['cohort']} | {r['cohort_type']} | {r['access_route']} "
                 f"| {r['n_patients']} | {r['n_samples']} | {r['comparisons_in_supplement']} |")
lines += ["", "## Redundancy audit (recomputed from sample tables, not asserted by hand)", "",
          "| check | set A | n | set B | n | intersection | % of A | Jaccard | verdict |",
          "|---|---|---|---|---|---|---|---|---|"]
for _, r in audf.iterrows():
    lines.append(f"| {r['check']} | {r['set_a']} | {r['n_a']} | {r['set_b']} | {r['n_b']} "
                 f"| {r['n_shared']} | {r['pct_of_a']}% | {r['jaccard']} | {r['verdict']} |")
lines += ["", "## Exclusions", "", "| item | reason | overlap fact | decision |", "|---|---|---|---|"]
for ex in EXCLUDED:
    lines.append(f"| {ex['dataset']} | {ex['reason']} | {ex['overlap']} | {ex['decision']} |")
lines += ["", "## Dropped rows (the same statistic reported by source files under different scopes)", "",
          f"{len(dropdf)} rows flagged `include = no` (kept in the supplementary table for auditing, excluded from all statistics):", "",
          "| cohort | endpoint class | endpoint | effect size | p | exclusion reason |", "|---|---|---|---|---|---|"]
for _, r in dropdf.iterrows():
    lines.append(f"| {r['cohort']} | {CLS_ZH.get(r['endpoint_class'], r['endpoint_class'])} "
                 f"| {str(r['endpoint']).replace('|', '/')} | {r['effect']:+.3f} | {r['p']:.3g} "
                 f"| {r['exclude_reason']} |")
lines += ["", f"> Analysis set = {len(INC)} rows (ingested {len(full)} rows).", "",
          "## Compact table used in the main figure (headline)", "",
          "| cohort | endpoint class | endpoint | n(label/control) | effect size | p | correction | interpretation |",
          "|---|---|---|---|---|---|---|---|"]
for _, r in head.iterrows():
    nlab, nctl = fnum(r["n_carrier"], 4), fnum(r["n_control"], 4)
    nstr = f"{nlab}/{nctl}" if nlab != "-" and nctl != "-" else "-"
    eff = (f"δ={r['effect']:+.3f}" if r["effect_metric"] == "cliffs_delta"
           else f"r={r['effect']:+.3f}" if r["effect_metric"] == "spearman_r"
           else f"{r['effect_metric']}={r['effect']:.3f}")
    pstr = "-" if not np.isfinite(r["p"]) else f"{r['p']:.3g}" + (" *" if r["p"] < 0.05 else "")
    det = str(r["detail"])
    det = "-" if det.strip().lower() in ("nan", "none", "") else det
    lines.append(f"| {r['cohort']} | {CLS_ZH.get(r['endpoint_class'], r['endpoint_class'])} "
                 f"| {str(r['endpoint']).replace('|', '/')} | {nstr} | {eff} | {pstr} "
                 f"| {r['adjustment']} | {det[:66]} |")
lines += ["", "## Prognostic layer (a different question: is REP30 a prognostic programme)", "",
          "| cohort | endpoint | n | HR (95% CI) | p |", "|---|---|---|---|---|"]
for _, r in full[full["endpoint_class"] == "prognostic-OS"].iterrows():
    ci = str(r["detail"]).replace("HR 95% CI ", "")
    lines.append(f"| {r['cohort']} | {r['endpoint']} | {int(r['n_total'])} | {ci} | {fnum(r['p'])} |")
lines += ["", "## Variance decomposition (three-cohort Harmony integration, 96,368 cells)", "", "| factor | eta-squared |", "|---|---|"]
for _, r in full[full["endpoint_class"] == "variance-decomposition"].iterrows():
    lines.append(f"| {r['endpoint']} | {r['effect']:.4f} |")
lines += ["", "> sample (batch) eta-squared is ~2.8x that of cell type -> the variance of REP_score is driven mainly by 'which sample / which library prep'.", ""]
with open(os.path.join(OUT, "evidence_table.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
log("wrote evidence_table.md")

# =====================================================================
# MAIN FIGURE  --  180 mm page width, explicit axes placement
# =====================================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 6.6,
    "axes.titlesize": 7.0,
    "axes.labelsize": 6.4,
    "xtick.labelsize": 5.8,
    "ytick.labelsize": 5.8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "legend.frameon": False,
})
RED = "#c0392b"      # REP30 higher in the resistant / more-exposed group
BLUE = "#2471a3"     # REP30 lower
GREY = "#9aa0a6"
BAND = "#eef0f2"

CM = 1 / 2.54
fig = plt.figure(figsize=(18.0 * CM, 24.0 * CM))

# ---------------------------------------------------------------------
# Panel A : the evidence grid
# ---------------------------------------------------------------------
axA = fig.add_axes([0.055, 0.545, 0.935, 0.385])

GRP_ORDER = ["molecular-resistance-genotype", "subtype-surrogate", "tumour-state",
             "treatment-exposure", "confound"]
GRP_LAB = {"molecular-resistance-genotype": "RESISTANCE GENOTYPE",
           "subtype-surrogate": "SUBTYPE SURROGATE",
           "tumour-state": "TUMOUR STATE  (met vs primary)",
           "treatment-exposure": "TREATMENT DOSE",
           "confound": "CONFOUND  (positive control)"}

seq = []
for g in GRP_ORDER:
    sub = head[head["grp"] == g]
    if sub.empty:
        continue
    seq.append(("H", g))
    for _, rr in sub.iterrows():
        seq.append(("R", rr))
n = len(seq)
pos = {i: n - 1 - i for i in range(n)}

LAY = -1.01      # single x for the row labels  (inside the axes = no clipping)
PC = 0.66        # fixed column for the p value  (right of the data)
axA.axvspan(-0.15, 0.15, color=BAND, zorder=0, lw=0)
axA.axvline(0, color="black", lw=0.7, zorder=1)
axA.axvline(PC - 0.075, color="#e6e6e6", lw=0.6, zorder=1)

for i, (kind, payload) in enumerate(seq):
    yv = pos[i]
    if kind == "H":
        axA.axhline(yv + 0.55, color="#b5b5b5", lw=0.7, zorder=1)
        axA.text(LAY, yv, GRP_LAB[payload], va="center", ha="left", fontsize=6.1,
                 fontweight="bold", color="#111111")
        continue
    r = payload
    e = float(r["effect"])
    sig = np.isfinite(r["p"]) and r["p"] < 0.05
    col = (RED if e > 0 else BLUE) if sig else GREY
    axA.plot([0, e], [yv, yv], color=col, lw=1.8 if sig else 1.0,
             alpha=1.0 if sig else 0.85, zorder=2, solid_capstyle="butt")
    axA.scatter([e], [yv], s=24 if sig else 15, color=col, zorder=3,
                edgecolor="white", linewidth=0.5,
                marker="o" if r["effect_metric"] == "cliffs_delta" else "D")
    pv = float(r["p"]) if np.isfinite(r["p"]) else np.nan
    lab = "" if not np.isfinite(pv) else f"{pv:.2g}" + (" *" if pv < 0.05 else "")
    axA.text(PC, yv, lab, va="center", ha="left", fontsize=5.4,
             color="#111111" if sig else "#666666")
    axA.text(LAY, yv, str(r["label"]), va="center", ha="left", fontsize=5.5,
             color="#111111" if sig else "#444444")

axA.text(PC, n - 0.55, "p", fontsize=5.6, color="#666666", ha="left", va="center")
axA.set_yticks([])
axA.spines["left"].set_visible(False)
axA.set_ylim(-0.9, n - 0.12)
axA.set_xlim(-1.05, 0.95)
axA.spines["bottom"].set_bounds(-0.60, 0.60)
axA.set_xticks([-0.6, -0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6])
axA.set_xticklabels(["-0.60", "-0.45", "-0.30", "-0.15", "0", "0.15", "0.30", "0.45", "0.60"])
axA.set_xlabel("effect size on REP30   (Cliff's δ  ○   |   Spearman r  ◆)      "
               ">0 = REP30 higher in the resistant / more-exposed group", labelpad=3)
axA.set_title("A   REP30 versus the resistant state, in every dataset where it can be tested\n"
              "      grey band = |effect| < 0.15 (negligible);   filled marker + * = p < 0.05;   "
              "grey = non-significant", loc="left", pad=3.5)

# ---------------------------------------------------------------------
# Panel B : the few signals decay under stratification
# ---------------------------------------------------------------------
axB = fig.add_axes([0.125, 0.283, 0.390, 0.205])
DECAY = [
    ("MET500", "all-cases | REP30 vs ESR1 mutation", "MET500 · ESR1"),
    ("MET500", "all-cases | REP30 vs RB1 loss", "MET500 · RB1"),
    ("AURORA US (cBioPortal)", "all-samples | REP30 vs ESR1 mutation", "AURORA · ESR1"),
    ("AURORA US (cBioPortal)", "all-samples | REP30 vs RB1 loss", "AURORA · RB1"),
    ("MBCproject (Count Me In)", "all-157 | REP30 vs ESR1 mutation (any nonsyn)", "MBCproject · ESR1"),
    ("MBCproject (Count Me In)", "all-157 | REP30 vs ERBB2 amplification", "MBCproject · ERBB2"),
]
DECAY_CTRL = {
    "MET500 · ESR1": "BRCA | REP30 vs ESR1 mutation",
    "MET500 · RB1": "BRCA | REP30 vs RB1 loss",
    "AURORA · ESR1": "all-samples | REP30 vs ESR1 mutation [adjusted",
    "AURORA · RB1": "all-samples | REP30 vs RB1 loss [adjusted",
    "MBCproject · ESR1": "post-treatment | REP30 vs ESR1 mutation (any nonsyn)",
    "MBCproject · ERBB2": "ERBB2-stratified | REP30 vs ERBB2 amp within HER2- only",
}
axB.axvspan(-0.15, 0.15, color=BAND, zorder=0, lw=0)
axB.axvline(0, color="black", lw=0.7, zorder=1)
C1, C2 = 0.66, 1.06          # text columns (effect, p)
axB.axvline(C1 - 0.045, color="#e6e6e6", lw=0.6, zorder=1)
axB.axvline(C2 - 0.045, color="#e6e6e6", lw=0.6, zorder=1)
yB = []
for k, (ds, k0, lab) in enumerate(DECAY):
    r0, r1 = _lookup(ds, k0), _lookup(ds, DECAY_CTRL[lab])
    if r0 is None or r1 is None:
        continue
    e0, e1 = float(r0["effect"]), float(r1["effect"])
    yv = len(DECAY) - 1 - k
    axB.annotate("", xy=(e1, yv), xytext=(e0, yv),
                 arrowprops=dict(arrowstyle="-|>", color="#777777", lw=0.8,
                                 shrinkA=3, shrinkB=3), zorder=1)
    axB.scatter([e0], [yv], s=24, color=RED, zorder=3, edgecolor="white", linewidth=0.5)
    axB.scatter([e1], [yv], s=24, zorder=3, edgecolor="black", linewidth=0.55,
                marker="s", color=(RED if e1 > 0 else BLUE))
    axB.text(C1, yv, f"{e0:+.2f} → {e1:+.2f}", fontsize=5.2, va="center", ha="left",
             color="#333333")
    axB.text(C2, yv, f"{r0['p']:.1g} → {r1['p']:.1g}", fontsize=5.2, va="center",
             ha="left", color="#333333")
    yB.append((yv, lab))
axB.text(C1, len(DECAY) + 0.05, "effect  raw → ctrl", fontsize=5.1, ha="left",
         va="center", color="#777777")
axB.text(C2, len(DECAY) + 0.05, "p  raw → ctrl", fontsize=5.1, ha="left",
         va="center", color="#777777")
axB.set_yticks([p[0] for p in yB])
axB.set_yticklabels([p[1] for p in yB], fontsize=5.5)
axB.set_ylim(-0.85, len(DECAY) + 0.42)
axB.set_xlim(-0.34, 1.32)
axB.spines["bottom"].set_bounds(-0.30, 0.40)
axB.set_xticks([-0.3, -0.15, 0, 0.15, 0.3])
axB.set_xticklabels(["-0.30", "-0.15", "0", "0.15", "0.30"], fontsize=5.0)
axB.set_xlabel("effect size on REP30      ● raw (all cases)     ■ after controlling", labelpad=3)
axB.set_title("B   The few 'signals' vanish once the relevant\n"
              "      subgroup or the confound is controlled", loc="left", pad=3.5)

# ---------------------------------------------------------------------
# Panel C : the confound is larger than the signal
# ---------------------------------------------------------------------
axC = fig.add_axes([0.655, 0.283, 0.330, 0.205])
CONF = [
    ("MBCproject (Count Me In)", "confound-screen | REP30 vs proliferation", "proliferation",
     "MBCproject · proliferation"),
    ("AURORA US (cBioPortal)", "all-samples | REP30 vs proliferation", "proliferation",
     "AURORA · proliferation"),
    ("MET500", "all-cases | REP30 vs proliferation", "proliferation", "MET500 · proliferation"),
    ("AURORA US (cBioPortal)", "all-samples | REP30 vs percent tumour cells", "tumour content",
     "AURORA · tumour cells (%)"),
    ("MET500", "BRCA | REP30 vs tc", "tumour content", "MET500 · tumour content"),
    ("AURORA US (cBioPortal)", "all-samples | REP30 vs percent stromal cells", "tumour content",
     "AURORA · stromal cells (%)"),
]
RESID = [
    ("GSE26459", "4-OHT acquired resistance (resistant vs sensitive)", "GSE26459 · 4-OHT resistance"),
    ("AURORA US (cBioPortal)", "all-samples | REP30 vs ESR1 mutation", "AURORA · ESR1 mutation"),
    ("AURORA US (cBioPortal)", "all-samples | REP30 vs RB1 loss", "AURORA · RB1 loss"),
    ("MET500", "BRCA | REP30 vs ESR1 mutation", "MET500 · ESR1 mutation"),
    ("MBCproject (Count Me In)", "all-157 | REP30 vs RB1 loss", "MBCproject · RB1 loss"),
    ("MBCproject (Count Me In)", "all-157 | REP30 vs ESR1 mutation (any nonsyn)",
     "MBCproject · ESR1 mutation"),
]
CAT_COL = {"proliferation": "#d98880", "tumour content": "#a9cce3", "resistance label": "#7dcea0"}
rowsC = []
for ds, key, cat, lab in CONF:
    r = _lookup(ds, key)
    if r is not None and np.isfinite(r["effect"]):
        rowsC.append((lab, cat, abs(float(r["effect"]))))
for ds, key, lab in RESID:
    r = _lookup(ds, key)
    if r is not None and np.isfinite(r["effect"]):
        rowsC.append((lab, "resistance label", abs(float(r["effect"]))))
rowsC = rowsC[::-1]
yy = np.arange(len(rowsC))
axC.barh(yy, [z[2] for z in rowsC], height=0.58,
         color=[CAT_COL[z[1]] for z in rowsC], zorder=2)
for i, (lab, cat, v) in enumerate(rowsC):
    axC.text(v + 0.008, yy[i], f"{v:.2f}", va="center", fontsize=5.4)
axC.set_yticks(yy)
axC.set_yticklabels([z[0] for z in rowsC], fontsize=5.4)
res_only = [z[2] for z in rowsC if z[1] == "resistance label"]
axC.axvline(np.mean(res_only), color="#2c3e50", ls="--", lw=0.9, zorder=3)
axC.set_xlim(0, max([z[2] for z in rowsC]) * 2.10)
axC.set_ylim(-0.75, len(rowsC) - 0.25)
axC.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
axC.set_xlabel(f"|effect size|      (dashed line = mean |effect| of the resistance labels "
               f"= {np.mean(res_only):.2f})", labelpad=3)
axC.legend(handles=[Patch(color=CAT_COL["proliferation"], label="vs proliferation"),
                    Patch(color=CAT_COL["tumour content"], label="vs tumour / stromal content"),
                    Patch(color=CAT_COL["resistance label"], label="vs a resistance label")],
           fontsize=5.2, loc="center right", bbox_to_anchor=(1.0, 0.52),
           handlelength=1.1, handleheight=0.8, labelspacing=0.35)
axC.set_title("C   The confound is larger than the signal", loc="left", pad=3.5)

# ---------------------------------------------------------------------
# Panel D : where the signature does work
# ---------------------------------------------------------------------
axD = fig.add_axes([0.135, 0.040, 0.850, 0.152])
PR = full[full["endpoint_class"] == "prognostic-OS"].copy()
PR["panel"] = PR["endpoint"].str.extract(r"\((full52|trim30)\)")
PR = PR.sort_values(["cohort", "panel"], ascending=[True, False]).reset_index(drop=True)
PR["short"] = PR["cohort"].str.replace("TCGA-BRCA", "TCGA").str.replace("METABRIC", "METABRIC")
PR["short"] = PR["short"] + " · " + PR["panel"].str.replace("full52", "REP52 panel") \
    .str.replace("trim30", "REP30 panel")
ylabD = []
for i, (_, r) in enumerate(PR.iterrows()):
    yv = len(PR) - 1 - i
    lo, hi = [float(x) for x in str(r["detail"]).replace("HR 95% CI ", "").split("-")]
    col = RED if r["effect"] > 1 else GREY
    axD.plot([lo, hi], [yv, yv], color=col, lw=1.3, solid_capstyle="butt", zorder=2)
    axD.scatter([r["effect"]], [yv], s=26, color=col, zorder=3,
                edgecolor="white", linewidth=0.5)
    axD.text(hi * 1.035, yv, f"{r['effect']:.2f}   (p = {r['p']:.1g},  n = {int(r['n_total'])})",
             va="center", fontsize=5.3, color="#333333")
    ylabD.append(r["short"])
axD.axvline(1.0, color="black", lw=0.7, ls=":")
axD.set_yticks(range(len(PR)))
axD.set_yticklabels(ylabD[::-1], fontsize=5.6)
axD.set_xscale("log")
axD.set_xlim(1.15, 4.6)
axD.set_xticks([1.5, 2, 3, 4])
axD.set_xticklabels(["1.5", "2", "3", "4"])
axD.set_ylim(-0.65, len(PR) - 0.35)
axD.set_xlabel("hazard ratio for overall survival, REP30-high versus low  (log scale)", labelpad=3)
axD.set_title("D   What the signature actually reads out: prognosis and proliferation, not resistance\n"
              "      REP30 tracks proliferation  r = +0.32 to +0.42 (p < 1e-6, 3 cohorts)   |   "
              "REP30 tracks the ER programme  r = -0.02 to +0.12 (n.s.)",
              loc="left", pad=3.5)

fig.suptitle("REP30 in eight independent breast-cancer datasets is a proliferation / prognosis read-out, "
             "not a resistance-state marker", fontsize=8.0, x=0.055, y=0.976, ha="left")
fig.savefig(os.path.join(SUB, "Fig1_evidence_main.png"), dpi=300)
fig.savefig(os.path.join(SUB, "Fig1_evidence_main.pdf"))
fig.savefig(os.path.join(OUT, "fig1_evidence_main.png"), dpi=300)
plt.close(fig)
log("wrote Fig1 (png 300 dpi + pdf)")

# =====================================================================
# numeric summary
# =====================================================================
sig = INC[(INC["endpoint_class"].isin(["molecular-resistance-genotype", "tumour-state",
                                       "treatment-exposure"])) & INC["p"].notna()]
log(f"\nresistance-state tests with p<0.05: {int((sig['p'] < 0.05).sum())}/{len(sig)}")
log("positive-direction & significant:")
for _, r in sig[(sig["p"] < 0.05) & (sig["effect"] > 0)].iterrows():
    log(f"   {r['cohort']} | {r['endpoint']} | {r['effect_metric']}={r['effect']:+.3f} p={r['p']:.3g}")
conf = INC[(INC["endpoint_class"] == "confound") & INC["p"].notna()].copy()
conf = conf[conf["effect"].abs() <= 1.0]        # drop chi-square/KW statistics, keep delta/r
log(f"\nconfound tests on a [-1,1] scale: {len(conf)}; mean |effect| = {conf['effect'].abs().mean():.3f}; "
    f"max = {conf['effect'].abs().max():.3f}")
log("  confound rows with |effect| > 0.40:")
for _, r in conf[conf["effect"].abs() > 0.40].iterrows():
    log(f"    {r['cohort']} | {r['endpoint']} | {r['effect']:+.3f}")
log("\nprognostic layer (different question):")
for _, r in INC[INC["endpoint_class"] == "prognostic-OS"].iterrows():
    log(f"   {r['cohort']} | {r['endpoint']} | HR={r['effect']:.2f} ({r['detail']}) p={r['p']:.3g}")

with open(os.path.join(OUT, "run_log.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(LOG))
print("\nDONE")
