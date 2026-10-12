# -*- coding: utf-8 -*-
"""
The Metastatic Breast Cancer Project (brca_mbcproject_2022, cBioPortal) -- the
first public cohort in this project that lets us test "does REP30 capture a
drug-resistant state?" against BOTH

  (a) a *dose* axis  -- days on endocrine therapy / any metastatic therapy /
      CDK4/6 inhibitor before the biopsy was taken, and
  (b) a *genotype* axis -- ESR1 LBD hotspot mutation, RB1 loss, ERBB2 amp ...

with receptor status (BX_ER / BX_PR / BX_HER2OVERALL), an independent
proliferation readout (BX_KI67) and per-patient serial biopsies (SAMPLE_TIMEPOINT
T1..T5) all present.

Design decisions that matter for interpretation
-----------------------------------------------
* 379 samples / 301 patients; only 157 samples / 120 patients have RNA, and of
  those **125/157 are BREAST-site (mostly primary)** -> this cohort is a
  *primary-biopsy-heavy* metastatic-disease cohort, not a metastasis-RNA cohort.
  Reported explicitly; `CALC_MET_SETTING` gives whether metastatic disease was
  present at biopsy.
* TUMOR_FRACTION is EMPTY for all RNA samples, so no tumour-content adjustment.
  The confounds available are the RNA proliferation module, BX_KI67 (n=26 only),
  TMB, receptor status and biopsy site.
* Sample IDs come in two schemes (`MBC-MBCProject_xxxx-Tumor-SM-yyy` and
  `RP-1156_MBCProject_xxxx_T1_v4_Exome`); both resolve to `SAMPLE_TIMEPOINT`.
* `BX_TIME_DAYS` == the specimen timeline `START_DATE` (verified 153/153) and the
  treatment timeline shares that origin -> exposure can be computed exactly.
* Observations are NOT independent (29 patients contribute 2-3 RNA samples), so
  every test is reported at sample level AND at patient level (latest biopsy).

Outputs: results/metastatic/MBCproject/
"""
import os
import sys
import io
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = r"E:/01.WorkSpace/01.Breast_cancer_gene_panel/resistance_signature"
CD = os.path.join(ROOT, "data/06_metastatic/cbioportal/brca_mbcproject_2022")
PANEL30 = os.path.join(ROOT, "results/rep52_trim/rep52_trimmed_panel.csv")
OUT = os.path.join(ROOT, "results/metastatic/MBCproject")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, "analysis"))
import stats_toolkit as stk  # noqa: E402

# ---------------------------------------------------------------- modules
PROLIF = ["MKI67", "PCNA", "TOP2A", "CCNB1", "BUB1", "AURKA", "PLK1", "MCM2",
          "RRM2", "TYMS", "CDC20", "BIRC5"]

# Canonical oestrogen-receptor transcriptional targets (MSigDB HALLMARK_ESTROGEN_
# RESPONSE_EARLY/LATE core members that are direct ER targets; deliberately
# EXCLUDES ESR1/FOXA1/GATA3 so the score reads pathway OUTPUT, not lineage).
ER_TARGETS = ["TFF1", "TFF3", "GREB1", "PGR", "PRLR", "PDZK1", "CA12", "SLC39A6",
              "NPY1R", "MYB", "XBP1", "SERPINA3", "GPR160", "HMGCS2", "IL6ST",
              "MYBL1", "HSPB8", "KCNK5", "SCUBE2", "CTSD"]

ENDOCRINE = {"TAMOXIFEN", "LETROZOLE", "ANASTROZOLE", "EXEMESTANE", "FULVESTRANT",
             "LEUPROLIDE", "GOSERELIN", "MEGESTROL", "TOREMIFENE", "ELACESTRANT",
             "TAMOXIFEN CITRATE", "FLUOXYMESTERONE", "MEDROXYPROGESTERONE",
             "ETHINYL ESTRADIOL", "ESTRADIOL", "DIETHYLSTILBESTROL"}
CDK46 = {"PALBOCICLIB", "RIBOCICLIB", "ABEMACICLIB"}
CHEMO = {"CYCLOPHOSPHAMIDE", "DOCETAXEL", "DOXORUBICIN", "PACLITAXEL", "CARBOPLATIN",
         "GEMCITABINE", "PACLITAXEL PROTEIN BOUND", "5-FLUOROURACIL", "ERIBULIN",
         "VINORELBINE", "METHOTREXATE", "CISPLATIN", "LIPOSOMAL DOXORUBICIN",
         "CAPECITABINE", "IXABEPILONE", "EPIRUBICIN", "FLUOROURACIL"}
HER2DIR = {"TRASTUZUMAB", "PERTUZUMAB", "ADO-TRASTUZUMAB EMTANSINE", "LAPATINIB",
           "TRASTUZUMAB EMTANSINE", "NERATINIB", "TUCATINIB", "MARGETUXIMAB"}

LBD_LO, LBD_HI = 310, 547          # ER-alpha ligand-binding domain, canonical
ESR1_HOTSPOT = {"Y537S", "Y537N", "Y537C", "D538G", "E380Q", "L536H", "L536R",
                "V534E", "V560M", "S463P"}

TRUNCATING = {"Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
              "Splice_Site", "Translation_Start_Site", "Nonstop_Mutation"}
NONSYN = TRUNCATING | {"Missense_Mutation", "In_Frame_Del", "In_Frame_Ins",
                       "Splice_Region"}

_log = []
def log(s=""):
    print(s)
    _log.append(str(s))


def rd(fn):
    """read a cBioPortal tsv, skipping the '#'-comment block"""
    with open(os.path.join(CD, fn), encoding="utf-8", errors="replace") as f:
        hdr, rows = None, []
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if hdr is None:
                hdr = p
                continue
            rows.append(p + [""] * (len(hdr) - len(p)))
    return pd.DataFrame(rows, columns=hdr)


def mwu(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(u), float(p)


def partial_spearman(x, y, z):
    x, y = np.asarray(x, float), np.asarray(y, float)
    Z = np.column_stack([np.asarray(t, float) for t in z])
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(Z).all(axis=1)
    x, y, Z = x[ok], y[ok], Z[ok]
    if len(x) < 6:
        return np.nan, np.nan
    x, y = stats.rankdata(x), stats.rankdata(y)
    Z = np.column_stack([stats.rankdata(Z[:, j]) for j in range(Z.shape[1])])

    def resid(v, C):
        C = np.column_stack([np.ones(len(v)), C])
        b, *_ = np.linalg.lstsq(C, v, rcond=None)
        return v - C @ b
    return stats.pearsonr(resid(x, Z), resid(y, Z))


def ols_adjust(score, flag, covars):
    """residualise score on covars, then MWU + Cliff's delta on the residual."""
    y = np.asarray(score, float)
    C = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in covars])
    ok = np.isfinite(C).all(axis=1) & np.isfinite(y)
    y2, C2 = y[ok], C[ok]
    if len(y2) < 8:
        return np.nan, np.nan, int(ok.sum())
    beta, *_ = np.linalg.lstsq(C2, y2, rcond=None)
    r = y2 - C2 @ beta
    f = np.asarray(flag, bool)[ok]
    if f.sum() < 3 or (~f).sum() < 3:
        return np.nan, np.nan, int(ok.sum())
    _, p = stats.mannwhitneyu(r[f], r[~f], alternative="two-sided")
    return stk.cliffs_delta(r[f], r[~f]), float(p), int(ok.sum())


# ================================================================== main
def main():
    panel = pd.read_csv(PANEL30)
    panel["gene"] = panel["gene"].astype(str).str.strip()
    coef = dict(zip(panel["gene"], panel["coef"]))
    log(f"REP panel: {len(coef)} genes")

    # ---------------- RNA ----------------
    # The shipped z-score table is NOT usable as-is: one pseudogene (NOS2P2) carries
    # the sentinel value 8.0e15 in every sample, 21,069 gene rows are entirely empty,
    # and the median of the remaining values is -0.62 (i.e. it was standardised
    # against a different reference set), so its moments are meaningless.
    # We therefore rebuild the score from raw RSEM: log2(x+1) then per-gene z within
    # these 157 samples -- transparent, and it removes the sentinel automatically.
    raw = pd.read_csv(os.path.join(CD, "data_mrna_seq_v2_rsem.txt"), sep="\t", low_memory=False)
    log(f"raw RSEM: {raw.shape[0]} rows x {raw.shape[1]} cols")
    sym = raw["Hugo_Symbol"].astype(str).str.strip()
    X = raw.drop(columns=["Hugo_Symbol", "Entrez_Gene_Id"]).apply(pd.to_numeric, errors="coerce")
    X = np.log2(X + 1.0)
    # drop gene rows that are entirely missing, then collapse duplicate symbols
    X = X.loc[~(X.isna().all(axis=1))]
    sym = sym.loc[X.index]
    X.index = sym.values
    X = X[~X.index.duplicated(keep="first")]
    X = X.replace([np.inf, -np.inf], np.nan)
    log(f"RNA matrix (log2 RSEM): {X.shape[0]} genes x {X.shape[1]} samples")
    log(f"  NaN fraction before z: {float(np.isnan(X.values).mean()):.5f}")

    mu = X.mean(axis=1)
    sd = X.std(axis=1).replace(0, np.nan)
    Zmat = X.sub(mu, axis=0).div(sd, axis=0)
    log(f"  z after standardisation: median={np.nanmedian(Zmat.values):.4f} "
        f"sd={np.nanstd(Zmat.values):.4f} max|z|={np.nanmax(np.abs(Zmat.values)):.2f}")
    X = Zmat

    miss_rep = [g for g in coef if g not in X.index]
    miss_pr = [g for g in PROLIF if g not in X.index]
    miss_er = [g for g in ER_TARGETS if g not in X.index]
    log(f"genes present -- REP {len(coef)-len(miss_rep)}/{len(coef)}, "
        f"prolif {len(PROLIF)-len(miss_pr)}/{len(PROLIF)}, "
        f"ER-targets {len(ER_TARGETS)-len(miss_er)}/{len(ER_TARGETS)}")
    for nm, mm in [("REP missing", miss_rep), ("prolif missing", miss_pr), ("ER missing", miss_er)]:
        if mm:
            log(f"  {nm}: {mm}")

    if miss_rep:
        log(f"  NOTE: REP score computed on {len(coef)-len(miss_rep)}/{len(coef)} genes "
            f"(missing {miss_rep}); earlier cohorts also dropped genes, so the score is "
            f"comparable in construction but not numerically identical.")
    X = X.fillna(0.0)

    Z = X.loc[[g for g in coef if g in X.index]].T
    rep30 = sum(coef[g] * Z[g] for g in Z.columns)
    prolif = X.loc[[g for g in PROLIF if g in X.index]].T.mean(axis=1)
    er_act = X.loc[[g for g in ER_TARGETS if g in X.index]].T.mean(axis=1)
    # ER output relative to lineage: ESR1, FOXA1, GATA3 excluded by construction
    df = pd.DataFrame({"rep30": rep30, "prolif": prolif, "er_activity": er_act,
                       "MKI67": X.loc["MKI67"].T, "ESR1": X.loc["ESR1"].T})

    # ---------------- clinical ----------------
    cs = rd("data_clinical_sample.txt")
    cs = cs.set_index("SAMPLE_ID")
    cp = rd("data_clinical_patient.txt").set_index("PATIENT_ID")

    keep = ["PATIENT_ID", "SAMPLE_TIMEPOINT", "BX_TIME_DAYS", "BX_LOCATION", "BX_TYPE",
            "BX_ER", "BX_PERCENT_ER", "BX_PR", "BX_HER2OVERALL", "BX_KI67",
            "CALC_MET_SETTING", "CALC_TREATMENT_NAIVE", "ONCOTREE_CODE",
            "TMB_NONSYNONYMOUS", "TUMOR_FRACTION"]
    df = df.join(cs[[c for c in keep if c in cs.columns]], how="left")
    df = df.join(cp[["AGE_AT_DIAGNOSIS", "CALC_TIME_TO_METS_DX_DAYS", "DX_ER",
                     "DX_HER2OVERALL", "METS_SITES_AT_METS_DX"]], on="PATIENT_ID")
    log(f"\nclinical join: PATIENT_ID present {df['PATIENT_ID'].notna().sum()}/{len(df)}, "
        f"unique patients {df['PATIENT_ID'].nunique()}")

    for c in ["BX_TIME_DAYS", "BX_KI67", "BX_PERCENT_ER", "TMB_NONSYNONYMOUS",
              "CALC_TIME_TO_METS_DX_DAYS"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # receptor status -> HR+ / HER2+ / TNBC (biopsy-level, fall back to diagnosis)
    def pos(s):
        return s.astype(str).str.upper().str.startswith("POS")
    def neg(s):
        return s.astype(str).str.upper().str.startswith("NEG")

    df["hr_pos"] = (pos(df["BX_ER"]) | pos(df["BX_PR"])) & neg(df["BX_HER2OVERALL"])
    df["hr_neg"] = neg(df["BX_ER"]) & neg(df["BX_PR"]) & neg(df["BX_HER2OVERALL"])
    df["her2_pos"] = pos(df["BX_HER2OVERALL"])
    df["er_pos_bx"] = pos(df["BX_ER"])
    df["met_at_biopsy"] = df["CALC_MET_SETTING"].astype(str).str.contains("METASTATIC_DISEASE_PRESENT")
    df["tx_naive_at_biopsy"] = df["CALC_TREATMENT_NAIVE"].astype(str).str.upper().eq("YES")

    log("  BX_ER          : " + str(df["BX_ER"].value_counts().to_dict()))
    log("  BX_HER2OVERALL : " + str(df["BX_HER2OVERALL"].value_counts().to_dict()))
    log(f"  HR+ {int(df['hr_pos'].sum())} | TNBC-like {int(df['hr_neg'].sum())} | "
        f"HER2+ {int(df['her2_pos'].sum())}")
    log("  biopsy site (top): " + str(df["BX_LOCATION"].value_counts().head(6).to_dict()))
    log("  CALC_MET_SETTING: " + str(df["CALC_MET_SETTING"].value_counts().to_dict()))
    log("  CALC_TREATMENT_NAIVE: " + str(df["CALC_TREATMENT_NAIVE"].value_counts().to_dict()))
    log(f"  BX_TIME_DAYS: median={df['BX_TIME_DAYS'].median():.0f} "

        f"({df['BX_TIME_DAYS'].min():.0f} .. {df['BX_TIME_DAYS'].max():.0f})")

    # ---------------- mutations (two ID schemes -> SAMPLE_TIMEPOINT) ----------------
    mut = pd.read_csv(os.path.join(CD, "data_mutations.txt"), sep="\t", low_memory=False,
                      usecols=["Hugo_Symbol", "Variant_Classification", "Tumor_Sample_Barcode",
                               "HGVSp_Short", "Protein_position"])
    mut["Protein_position"] = pd.to_numeric(
        mut["Protein_position"].astype(str).str.split("/").str[0], errors="coerce")
    sid2tp = dict(zip(cs.index, cs["SAMPLE_TIMEPOINT"]))

    def to_tp(b):
        if b in sid2tp:
            return sid2tp[b]
        if b.startswith("RP-"):
            p = b.split("_")
            if "MBCProject" in p:
                i = p.index("MBCProject")
                return f"MBCProject_{p[i+1]}_{p[i+2]}"
        return None

    mut["tp"] = mut["Tumor_Sample_Barcode"].map(to_tp)
    log(f"\nMAF: {mut.shape[0]} variants; barcodes resolved to timepoint "
        f"{mut['tp'].notna().sum()}/{len(mut)}")

    ns = mut[mut["Variant_Classification"].isin(NONSYN)]
    tr = mut[mut["Variant_Classification"].isin(TRUNCATING)]

    def tp_set(gene, table=None):
        t = ns if table is None else table
        return set(t.loc[t["Hugo_Symbol"] == gene, "tp"].dropna())

    df["TP53_mut"] = df["SAMPLE_TIMEPOINT"].isin(tp_set("TP53"))
    df["PIK3CA_mut"] = df["SAMPLE_TIMEPOINT"].isin(tp_set("PIK3CA"))
    df["GATA3_mut"] = df["SAMPLE_TIMEPOINT"].isin(tp_set("GATA3"))
    df["CDH1_mut"] = df["SAMPLE_TIMEPOINT"].isin(tp_set("CDH1"))
    df["ARID1A_mut"] = df["SAMPLE_TIMEPOINT"].isin(tp_set("ARID1A"))
    df["RB1_mut"] = df["SAMPLE_TIMEPOINT"].isin(tp_set("RB1") | tp_set("RB1", tr))

    e = ns[ns["Hugo_Symbol"] == "ESR1"]
    e_any = set(e["tp"].dropna())
    e_hot = set(e.loc[e["HGVSp_Short"].astype(str).str.replace("p.", "", regex=False)
                      .str.upper().isin({h.upper() for h in ESR1_HOTSPOT}), "tp"].dropna())
    e_lbd = set(e.loc[e["Protein_position"].between(LBD_LO, LBD_HI), "tp"].dropna())
    df["ESR1_mut_any"] = df["SAMPLE_TIMEPOINT"].isin(e_any)
    df["ESR1_mut_hot"] = df["SAMPLE_TIMEPOINT"].isin(e_hot | e_lbd)
    df["ESR1_mut_lbd"] = df["SAMPLE_TIMEPOINT"].isin(e_lbd)
    log(f"  ESR1 ns variants: {e.shape[0]}; samples any={len(e_any)} hotspot/LBD={len(e_hot | e_lbd)}")
    log(f"  ESR1 hotspot variants seen: {sorted(e.loc[e['HGVSp_Short'].isin(['p.'+h for h in ESR1_HOTSPOT]), 'HGVSp_Short'].unique())}")

    # ---------------- CNA ----------------
    cna = rd("data_cna.txt").set_index("Hugo_Symbol")
    cna = cna.drop(columns=[c for c in cna.columns if c.lower() in ("cytoband",)], errors="ignore")
    for c in cna.columns:
        cna[c] = pd.to_numeric(cna[c], errors="coerce")

    def cna_row(g):
        if g not in cna.index:
            return None
        return cna.loc[g]
    amp_flags = {"ERBB2_amp": "ERBB2", "CCND1_amp": "CCND1", "CCNE1_amp": "CCNE1",
                 "MDM2_amp": "MDM2", "CDK4_amp": "CDK4", "FGFR1_amp": "FGFR1",
                 "ESR1_amp": "ESR1"}
    for flag, g in amp_flags.items():
        r = cna_row(g)
        if r is None:
            log(f"  {g}: not in CNA matrix"); continue
        v = r.reindex(df.index)
        df[flag] = (v >= 2).fillna(False)      # GISTIC 2 = high-level amplification
    r = cna_row("RB1")
    if r is not None:
        v = r.reindex(df.index)
        df["RB1_cna"] = v
        df["RB1_homdel"] = (v <= -2).fillna(False)
        df["RB1_loss"] = (df["RB1_homdel"] | df["RB1_mut"]).fillna(False)
    log("\nCNA flags (RNA samples): " + str(
        {k: int(df[k].sum()) for k in list(amp_flags) + ["RB1_homdel", "RB1_loss"] if k in df}))

    # ---------------- treatment exposure ----------------
    tt = rd("data_timeline_treatment.txt")
    for c in ["START_DATE", "STOP_DATE"]:
        tt[c] = pd.to_numeric(tt[c], errors="coerce")
    tt["DRUG"] = tt["DRUG"].astype(str).str.upper().str.strip()
    tt["mode"] = tt["PATIENT_TREATMENT_MODE"].astype(str).str.upper()
    log(f"\ntreatment records: {len(tt)}; patients {tt['PATIENT_ID'].nunique()}")

    def exposure(pid, day):
        """everything about therapy exposure strictly before `day` for one patient"""
        t = tt[tt["PATIENT_ID"] == pid]
        started = t[t["START_DATE"].notna() & (t["START_DATE"] <= day) & (t["START_DATE"] > -9000)]
        res = {}
        groups = [("endo", ENDOCRINE), ("chemo", CHEMO), ("cdk46", CDK46), ("her2", HER2DIR)]
        obs = float(np.nansum((np.minimum(started["STOP_DATE"].fillna(day), day)
                               - started["START_DATE"]).clip(lower=0)))
        res["tx_days_any"] = obs
        for gname, gset in groups:
            s = started[started["DRUG"].isin(gset)]
            res[f"tx_days_{gname}"] = float(np.nansum(
                (np.minimum(s["STOP_DATE"].fillna(day), day) - s["START_DATE"]).clip(lower=0)))
            res[f"tx_n_{gname}"] = int(len(s))
        # metastatic lines (a "line" ~ a distinct start date in metastatic mode)
        met = started[started["mode"].str.contains("METASTATIC")]
        res["met_lines"] = int(met["START_DATE"].nunique())
        res["tx_n_any"] = int(started["START_DATE"].nunique())
        prog = met[(met["STOP_DATE"].notna()) & (met["STOP_DATE"] <= day) &
                   (met["TREATMENT_RESPONSE"].astype(str).str.upper()
                    .str.contains("PROGRESSION|NO_RESPONSE"))]
        res["n_progressions"] = int(len(prog))
        return res

    rows = []
    for _, r in df.iterrows():
        if pd.isna(r.get("BX_TIME_DAYS")) or pd.isna(r.get("PATIENT_ID")):
            rows.append({}); continue
        rows.append(exposure(r["PATIENT_ID"], r["BX_TIME_DAYS"]))
    ex = pd.DataFrame(rows, index=df.index)
    df = pd.concat([df, ex], axis=1)

    log("exposure (samples with a biopsy day): " + str(int(df["tx_days_any"].notna().sum())))
    log("  tx_days_any   median=%.0f (%.0f .. %.0f)" % (
        df["tx_days_any"].median(), df["tx_days_any"].min(), df["tx_days_any"].max()))
    for c in ["tx_days_endo", "tx_days_chemo", "tx_days_cdk46", "tx_days_her2",
              "met_lines", "tx_n_any", "n_progressions"]:
        log(f"  {c:16s} nonzero={int((df[c] > 0).sum()):3d}  median={df[c].median():.0f}  "
            f"max={df[c].max():.0f}")

    # patient-level view: the LATEST RNA biopsy per patient (most exposed state)
    # NOTE: expression/score columns come from the latest biopsy, but GENOTYPE
    # flags must be aggregated as "ever positive across that patient's RNA
    # timepoints" -- an ESR1 LBD clone, once acquired, persists, so reading the
    # flag off only the latest specimen would silently drop carriers whose last
    # biopsy happened to be wild-type.
    df["_ord"] = df["BX_TIME_DAYS"].fillna(-1)
    FLAGS = ["TP53_mut", "PIK3CA_mut", "GATA3_mut", "CDH1_mut", "ARID1A_mut", "RB1_mut",
             "ESR1_mut_any", "ESR1_mut_hot", "ESR1_mut_lbd", "ERBB2_amp", "CCND1_amp",
             "CCNE1_amp", "MDM2_amp", "CDK4_amp", "FGFR1_amp", "ESR1_amp",
             "RB1_homdel", "RB1_loss", "hr_pos", "hr_neg", "her2_pos", "er_pos_bx",
             "met_at_biopsy", "tx_naive_at_biopsy"]
    FLAGS = [c for c in FLAGS if c in df.columns]
    ever = df.groupby("PATIENT_ID")[FLAGS].max()
    pl = df.sort_values("_ord").groupby("PATIENT_ID", as_index=False).tail(1).copy()
    pl = pl.set_index("PATIENT_ID")
    pl[FLAGS] = ever.reindex(pl.index)[FLAGS].values
    pl = pl.reset_index()
    log(f"\npatient-level rows (latest biopsy + ever-positive genotype): {len(pl)} "
        f"(from {len(df)} samples / {df['PATIENT_ID'].nunique()} patients)")
    log("  carrier counts flipped on by the ever-positive rule: " + str(
        {c: (int(pl[c].sum()), int(df.sort_values('_ord').groupby('PATIENT_ID').tail(1)[c].sum()))
         for c in ["ESR1_mut_any", "ESR1_mut_hot", "RB1_loss", "ERBB2_amp"] if c in pl}))

    df.to_csv(os.path.join(OUT, "mbcproject_linked_samples.csv"))
    pl.to_csv(os.path.join(OUT, "mbcproject_linked_patients.csv"))

    # ================================================================== tests
    res = []

    def add(scope, test, n, stat, p, detail=""):
        res.append(dict(scope=scope, test=test, n=n,
                        statistic=round(stat, 3) if stat == stat else np.nan,
                        p=p if p == p else np.nan, detail=detail))

    def label_test(scope, sub, flag, name, covars=("prolif",)):
        f = sub[flag].fillna(False).astype(bool)
        a = sub.loc[f, "rep30"].values
        b = sub.loc[~f, "rep30"].values
        if len(a) < 3 or len(b) < 3:
            add(scope, name, f"{len(a)}/{len(b)}", np.nan, np.nan, "too few"); return
        add(scope, name, f"{len(a)}/{len(b)}", stk.cliffs_delta(a, b), mwu(a, b)[1],
            f"median {np.median(a):.3f} vs {np.median(b):.3f}")
        cv = [sub[c].values for c in covars if c in sub]
        if cv:
            d2, p2, n2 = ols_adjust(sub["rep30"].values, f.values, cv)
            add(scope, name + " [adj " + "+".join(covars) + "]", n2, d2, p2,
                "OLS residual -> MWU + Cliff's delta")

    def corr_test(scope, sub, col, label, covars=None):
        v = pd.to_numeric(sub[col], errors="coerce")
        if v.nunique(dropna=True) < 2:
            return
        ok = v.notna() & sub["rep30"].notna()
        if ok.sum() < 12:
            return
        rr, pv = stats.spearmanr(sub.loc[ok, "rep30"], v[ok])
        add(scope, f"REP30 vs {label} (Spearman)", int(ok.sum()), rr, pv)
        if covars:
            cv = [pd.to_numeric(sub[c], errors="coerce").fillna(0).values for c in covars]
            pr, pp = partial_spearman(sub["rep30"].values, v.values, cv)
            add(scope, f"REP30 vs {label} [adj {'+'.join(covars)}]", int(ok.sum()), pr, pp,
                "partial Spearman")

    LBL = [("ESR1_mut_any", "ESR1 mutation (any nonsyn)"),
           ("ESR1_mut_hot", "ESR1 LBD hotspot mutation"),
           ("RB1_loss", "RB1 loss (homdel or truncating)"),
           ("ERBB2_amp", "ERBB2 amplification"),
           ("CCNE1_amp", "CCNE1 amplification"),
           ("MDM2_amp", "MDM2 amplification"),
           ("CCND1_amp", "CCND1 amplification"),
           ("CDK4_amp", "CDK4 amplification")]
    CORRS = [("tx_days_any", "days on ANY therapy before biopsy",
              ["prolif", "TMB_NONSYNONYMOUS"]),
             ("tx_days_endo", "days on ENDOCRINE therapy (dose axis)",
              ["prolif", "er_activity"]),
             ("tx_days_chemo", "days on chemotherapy", ["prolif"]),
             ("tx_days_cdk46", "days on CDK4/6 inhibitor", ["prolif"]),
             ("met_lines", "metastatic treatment lines before biopsy", ["prolif"]),
             ("tx_n_any", "distinct therapy start dates before biopsy", ["prolif"]),
             ("n_progressions", "documented progressions before biopsy", ["prolif"])]

    scopes = {"all-157": df,
              "patients-120": pl,
              "HR+ (ER/PR+, HER2-)": df[df["hr_pos"]],
              "HR+ patients": pl[pl["hr_pos"]],
              "ER+ biopsy (BX_ER pos)": df[df["er_pos_bx"]],
              "met-at-biopsy": df[df["met_at_biopsy"]],
              "treatment-naive": df[df["tx_days_any"].fillna(0) == 0],
              "post-treatment": df[df["tx_days_any"].fillna(0) > 0]}

    for sname, sub in scopes.items():
        if len(sub) < 15:
            log(f"\nskip scope {sname} (n={len(sub)})"); continue
        log("\n" + "=" * 78)
        log(f"scope {sname}: n={len(sub)} samples, {sub['PATIENT_ID'].nunique()} patients")
        for flag, nm in LBL:
            if flag in sub:
                label_test(sname, sub, flag, "REP30 vs " + nm)
        for col, lab, cv in CORRS:
            if col in sub:
                corr_test(sname, sub, col, lab, cv)

    # ---- therapy-exposure TIMING contrast: naive at biopsy vs already treated ----
    log("\n" + "=" * 78)
    nv = df[df["tx_days_any"].fillna(0) == 0]
    pt = df[df["tx_days_any"].fillna(0) > 0]
    add("treatment-timing", "REP30 post-treatment vs therapy-naive at biopsy (MWU)",
        f"{len(pt)}/{len(nv)}", stk.cliffs_delta(pt["rep30"], nv["rep30"]),
        mwu(pt["rep30"], nv["rep30"])[1],
        f"median {pt['rep30'].median():.3f} vs {nv['rep30'].median():.3f}")
    d, p, n = ols_adjust(df["rep30"].values, (df["tx_days_any"].fillna(0) > 0).values,
                         [df["prolif"].values])
    add("treatment-timing", "REP30 post-treatment vs naive [adj prolif]", n, d, p,
        "OLS residual -> MWU + Cliff's delta")
    # restrict to actually exposed patients: dose-response within the exposed set
    for lab, col in [("days ANY therapy", "tx_days_any"), ("days endocrine", "tx_days_endo"),
                     ("days chemotherapy", "tx_days_chemo")]:
        if pt[col].notna().sum() >= 12:
            rr, pv = stats.spearmanr(pt[col], pt["rep30"])
            add("treatment-timing/exposed-only", f"REP30 vs {lab} (Spearman)",
                len(pt), rr, pv, "free of the naive/treated step change")

    # ---- is the ERBB2-amplification signal just the HER2+ subtype? ----
    log("\n" + "=" * 78)
    for nm, sub in [("HER2+ only", df[df["her2_pos"]]), ("HER2- only", df[~df["her2_pos"]])]:
        if len(sub) < 15:
            log(f"  {nm}: n={len(sub)} too few"); continue
        label_test("ERBB2-stratified", sub, "ERBB2_amp", f"REP30 vs ERBB2 amp within {nm}",
                   covars=("prolif",))
    label_test("ERBB2-stratified", df, "ESR1_mut_any", "REP30 vs ESR1 mut | ER-status adjusted",
               covars=("prolif", "er_pos_bx"))
    label_test("ERBB2-stratified", df, "ESR1_mut_hot", "REP30 vs ESR1 hotspot | ER-status adjusted",
               covars=("prolif", "er_pos_bx"))

    # ---- explicit confound screens (all-157) ----
    log("\n" + "=" * 78)
    for c, lab, cv in [("prolif", "proliferation (RNA module)", None),
                       ("MKI67", "MKI67", None),
                       ("er_activity", "ER-target activity score", None),
                       ("ESR1", "ESR1 expression", None),
                       ("BX_KI67", "Ki67 (IHC, n small)", None),
                       ("TMB_NONSYNONYMOUS", "TMB", None),
                       ("BX_TIME_DAYS", "days from reference to biopsy", None),
                       ("CALC_TIME_TO_METS_DX_DAYS", "time to metastatic dx", None)]:
        corr_test("confound-screen", df, c, lab)
    for nm, col, cv in [("REP30 vs ER activity | proliferation-adjusted", "er_activity", ["prolif"]),
                        ("REP30 vs proliferation | ER-activity-adjusted", "prolif", ["er_activity"])]:
        rr, pp = partial_spearman(df["rep30"].values,
                                  pd.to_numeric(df[col], errors="coerce").values,
                                  [df[c].values for c in cv])
        add("confound-screen", nm, len(df), rr, pp, "partial Spearman")

    # group differences by categorical confounds
    for cat, lab in [("BX_LOCATION", "biopsy site"), ("BX_TYPE", "biopsy type"),
                     ("ONCOTREE_CODE", "histology"), ("CALC_MET_SETTING", "metastatic setting"),
                     ("CALC_TREATMENT_NAIVE", "treatment naive at biopsy")]:
        groups = [g["rep30"].dropna().values for _, g in df.groupby(cat) if len(g) >= 5]
        if len(groups) >= 2:
            h, pk = stats.kruskal(*groups)
            add("confound-screen", f"REP30 by {lab} (Kruskal-Wallis)", len(groups),
                h, pk, f"{len(groups)} groups")

    # receptor-status contrasts
    for flag, nm in [("hr_pos", "HR+ vs rest"), ("hr_neg", "TNBC-like vs rest"),
                     ("her2_pos", "HER2+ vs rest"), ("er_pos_bx", "ER+ biopsy vs rest")]:
        label_test("all-157", df, flag, "REP30 " + nm)

    # ---- serial biopsies (within patient, T1 -> latest) ----
    log("\n" + "=" * 78)
    multi = df[df["PATIENT_ID"].isin(
        df["PATIENT_ID"].value_counts()[lambda s: s >= 2].index)].copy()
    multi = multi.sort_values(["PATIENT_ID", "_ord"])
    pairs = []
    for pid, g in multi.groupby("PATIENT_ID"):
        g = g.dropna(subset=["BX_TIME_DAYS"])
        if len(g) < 2:
            continue
        first, last = g.iloc[0], g.iloc[-1]
        pairs.append(dict(PATIENT_ID=pid, n=len(g),
                          d_rep30=last["rep30"] - first["rep30"],
                          d_prolif=last["prolif"] - first["prolif"],
                          d_er=last["er_activity"] - first["er_activity"],
                          dt=last["BX_TIME_DAYS"] - first["BX_TIME_DAYS"],
                          d_txdays=last["tx_days_any"] - first["tx_days_any"],
                          d_endo=last["tx_days_endo"] - first["tx_days_endo"],
                          d_cdk46=last["tx_days_cdk46"] - first["tx_days_cdk46"],
                          gained_esr1=int(last["ESR1_mut_hot"]) - int(first["ESR1_mut_hot"]),
                          first_rep30=first["rep30"], last_rep30=last["rep30"]))
    pr = pd.DataFrame(pairs)
    log(f"serial-biopsy patients with >=2 RNA timepoints: {len(pr)}")
    if len(pr) >= 6:
        w, pw = stats.wilcoxon(pr["d_rep30"], alternative="two-sided")
        add("serial-biopsy", "REP30 change (last - first) Wilcoxon signed-rank", len(pr),
            float(np.median(pr["d_rep30"])), float(pw),
            f"median d={np.median(pr['d_rep30']):.3f}; up in {int((pr['d_rep30']>0).sum())}/{len(pr)}")
        for col, lab in [("d_txdays", "change in cumulative therapy days"),
                         ("d_endo", "change in endocrine therapy days"),
                         ("d_cdk46", "change in CDK4/6i days")]:
            ok = pr[col].notna()
            if ok.sum() >= 8 and pr.loc[ok, col].abs().sum() > 0:
                rr, pv = stats.spearmanr(pr.loc[ok, "d_rep30"], pr.loc[ok, col])
                add("serial-biopsy", f"dREP30 vs {lab} (Spearman)", int(ok.sum()), rr, pv)
        rr, pv = stats.spearmanr(pr["d_rep30"], pr["d_prolif"])
        add("serial-biopsy", "dREP30 vs dProliferation (Spearman)", len(pr), rr, pv,
            "is the longitudinal change just proliferation?")
        rr, pv = stats.spearmanr(pr["d_rep30"], pr["d_er"])
        add("serial-biopsy", "dREP30 vs dER-activity (Spearman)", len(pr), rr, pv)
        pr.to_csv(os.path.join(OUT, "serial_biopsy_pairs.csv"), index=False)

    # ---- ER-activity alternative endpoint (the proposed variable swap) ----
    log("\n" + "=" * 78)
    add("ER-activity-endpoint", "ER-target score vs ESR1 hotspot mutation",
        f"{int(df['ESR1_mut_hot'].sum())}/{int((~df['ESR1_mut_hot']).sum())}",
        stk.cliffs_delta(df.loc[df["ESR1_mut_hot"], "er_activity"],
                         df.loc[~df["ESR1_mut_hot"], "er_activity"]),
        mwu(df.loc[df["ESR1_mut_hot"], "er_activity"],
            df.loc[~df["ESR1_mut_hot"], "er_activity"])[1])
    add("ER-activity-endpoint", "ER-target score vs RB1 loss",
        f"{int(df['RB1_loss'].sum())}/{int((~df['RB1_loss']).sum())}",
        stk.cliffs_delta(df.loc[df["RB1_loss"], "er_activity"],
                         df.loc[~df["RB1_loss"], "er_activity"]),
        mwu(df.loc[df["RB1_loss"], "er_activity"],
            df.loc[~df["RB1_loss"], "er_activity"])[1])
    add("ER-activity-endpoint", "REP30 vs ER-target score (Spearman)", len(df),
        stats.spearmanr(df["rep30"], df["er_activity"])[0],
        stats.spearmanr(df["rep30"], df["er_activity"])[1])
    _v = pd.to_numeric(df["tx_days_endo"], errors="coerce")
    _ok = _v.notna() & df["er_activity"].notna()
    if _ok.sum() >= 12 and _v[_ok].nunique() > 1:
        rr, pv = stats.spearmanr(df.loc[_ok, "er_activity"], _v[_ok])
        add("ER-activity-endpoint", "ER-target score vs days on endocrine therapy",
            int(_ok.sum()), rr, pv)
    _ok2 = _v.notna()
    if _ok2.sum() >= 12 and _v[_ok2].nunique() > 1:
        rr, pv = stats.spearmanr(df.loc[_ok2, "er_activity"], _v[_ok2])
        add("ER-activity-endpoint", "ER-target score vs days on endocrine therapy (all)",
            int(_ok2.sum()), rr, pv, "zeros included = therapy-naive")

    out = pd.DataFrame(res)
    out.to_csv(os.path.join(OUT, "rep30_tests.csv"), index=False)
    log("\n" + out.to_string(index=False))

    # ================================================================== figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))

        ax = axes[0]
        ax.scatter(df["prolif"], df["rep30"], s=16, alpha=0.6)
        rr, pp = stats.spearmanr(df["prolif"], df["rep30"])
        ax.set_xlabel("proliferation module"); ax.set_ylabel("REP30")
        ax.set_title(f"REP30 vs proliferation\nr={rr:.3f} p={pp:.1e}"); ax.grid(alpha=0.3)

        ax = axes[1]
        ax.scatter(df["tx_days_endo"], df["rep30"], s=16, alpha=0.6, color="tab:red")
        rr, pp = stats.spearmanr(df["tx_days_endo"], df["rep30"])
        ax.set_xlabel("days on endocrine therapy before biopsy"); ax.set_ylabel("REP30")
        ax.set_title(f"REP30 vs endocrine exposure\nr={rr:.3f} p={pp:.1e}"); ax.grid(alpha=0.3)

        ax = axes[2]
        labs, data = [], []
        for flag, nm in [("ESR1_mut_hot", "ESR1 hot"), ("RB1_loss", "RB1 loss"),
                         ("ERBB2_amp", "ERBB2 amp"), ("CCNE1_amp", "CCNE1 amp")]:
            if flag in df:
                labs += [nm + "+", nm + "-"]
                data += [df.loc[df[flag].fillna(False), "rep30"].values,
                         df.loc[~df[flag].fillna(False), "rep30"].values]
        bp = ax.boxplot(data, showfliers=False, patch_artist=False, widths=0.6)
        ax.set_xticks(np.arange(1, len(labs) + 1))
        ax.set_xticklabels(labs, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("REP30"); ax.set_title("REP30 by resistance genotype"); ax.grid(alpha=0.3)

        ax = axes[3]
        if len(pr) >= 6:
            ax.scatter(pr["d_txdays"], pr["d_rep30"], s=40, alpha=0.8, color="tab:purple")
            rr, pp = stats.spearmanr(pr["d_txdays"], pr["d_rep30"])
            ax.set_xlabel("change in cumulative therapy days"); ax.set_ylabel("change in REP30")
            ax.set_title(f"serial biopsies (n={len(pr)})\nr={rr:.3f} p={pp:.1e}")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "mbcproject_rep30_overview.png"), dpi=170)
        plt.close()
        log("\nfigure written")
    except Exception as e:
        log(f"figure skipped: {type(e).__name__}: {e}")

    with open(os.path.join(OUT, "run_log.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(_log))


if __name__ == "__main__":
    main()
