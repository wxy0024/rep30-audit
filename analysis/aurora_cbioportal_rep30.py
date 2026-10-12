# -*- coding: utf-8 -*-
"""
AURORA US (cBioPortal release, brca_aurora_2023) -- the definitive public test of
"does REP30 capture a drug-resistant state?", now with the four confounds that
killed the earlier AURORA-US analysis (GSE209998) all controllable:

  1. receptor status   PRIM_ER / PRIM_PR / PRIM_HER2  -> HR+ subgroup possible
                       (GSE209998 had none; MET500 clinical had none)
  2. tissue preservation  PRESERVATION = FF vs FFPE  -> the FFPE artefact that
                       produced the fake "metastasis REP30 up, p=0.0017" in
                       GSE209998 can now be regressed out explicitly
  3. tumour content    PERCENT_TUMOR_CELLS, PERCENT_STROMAL_CELLS
  4. proliferation     NUMBER_PROLIFERATING_CELLS (a histology readout, i.e.
                       INDEPENDENT of the RNA-based MKI67 module)

Plus, uniquely, a real treatment timeline:
  * data_timeline_specimen.txt : each specimen's collection day (from primary dx)
  * data_timeline_dx.txt       : primary (day 0) and metastatic diagnosis days
  * data_timeline_treatment.txt: every therapy line with START/STOP days
-> we can define "this biopsy was taken after N lines of metastatic therapy" and
   "days on metastatic therapy before this biopsy", i.e. an ACTUAL exposure axis.

Inputs are the gzipped cBioPortal study tarball the user downloaded, already
extracted to data/06_metastatic/cbioportal/brca_aurora_2023/.

Outputs: results/metastatic/AURORA_cBioPortal/
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = r"E:/01.WorkSpace/01.Breast_cancer_gene_panel/resistance_signature"
CD = os.path.join(ROOT, "data/06_metastatic/cbioportal/brca_aurora_2023")
PANEL30 = os.path.join(ROOT, "results/rep52_trim/rep52_trimmed_panel.csv")
OUT = os.path.join(ROOT, "results/metastatic/AURORA_cBioPortal")
os.makedirs(OUT, exist_ok=True)

PROLIF = ["MKI67", "PCNA", "TOP2A", "CCNB1", "BUB1", "AURKA", "PLK1", "MCM2",
          "RRM2", "TYMS", "CDC20", "BIRC5"]

TRUNCATING = {"Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
              "Splice_Site", "Translation_Start_Site", "Nonstop_Mutation"}
NONSYN = TRUNCATING | {"Missense_Mutation", "In_Frame_Del", "In_Frame_Ins",
                       "Splice_Region"}

_log = []
def log(s=""):
    print(s)
    _log.append(str(s))


# ------------------------------------------------------------------ helpers
def cliffs_delta(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return np.nan
    A = a[:, None]; B = b[None, :]
    return ((A > B).sum() - (A < B).sum()) / (na * nb)


def mwu(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    return stats.mannwhitneyu(a, b, alternative="two-sided")


def perm_p(a, b, n=4000, seed=0):
    a = np.asarray(a, float); b = np.asarray(b, float)
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b]); na = len(a)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(pool[:na].mean() - pool[na:].mean()) >= obs:
            cnt += 1
    return (cnt + 1) / (n + 1)


def ols_adjust(score, flag, covars):
    """residualise score on covars, then MWU + Cliff's delta on the residual."""
    y = np.asarray(score, float)
    C = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in covars])
    ok = np.isfinite(C).all(axis=1) & np.isfinite(y)
    y2, C2 = y[ok], C[ok]
    beta, *_ = np.linalg.lstsq(C2, y2, rcond=None)
    r = y2 - C2 @ beta
    f = np.asarray(flag, bool)[ok]
    if f.sum() < 3 or (~f).sum() < 3:
        return np.nan, np.nan, np.nan
    u, p = stats.mannwhitneyu(r[f], r[~f], alternative="two-sided")
    return cliffs_delta(r[f], r[~f]), p, int(ok.sum())


def partial_spearman(x, y, z):
    """Spearman of x,y after removing the rank-linear effect of z.
    z may be a 1-D array or a list/tuple of 1-D arrays (one per covariate).
    NOTE: np.asarray([arr1, arr2]) silently makes a (2, n) array, so a list must
    be column-stacked explicitly -- otherwise the covariate matrix is transposed
    and lstsq raises 'Incompatible dimensions'."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if isinstance(z, (list, tuple)):
        Z = np.column_stack([np.asarray(t, float) for t in z])
        ok &= np.isfinite(Z).all(axis=1)
    else:
        Z = np.asarray(z, float)
        if Z.ndim == 1:
            Z = Z[:, None]
        ok &= np.isfinite(Z).all(axis=1)
    x, y, Z = x[ok], y[ok], Z[ok]
    x, y = stats.rankdata(x), stats.rankdata(y)
    Z = np.column_stack([stats.rankdata(Z[:, j]) for j in range(Z.shape[1])])
    def resid(v, c):
        c = np.column_stack([np.ones(len(v)), c])
        b, *_ = np.linalg.lstsq(c, v, rcond=None)
        return v - c @ b
    rx, ry = resid(x, Z), resid(y, Z)
    return stats.pearsonr(rx, ry)


def rd(fn):
    """read a cBioPortal tsv, skipping the '#'-comment header block"""
    with open(os.path.join(CD, fn), encoding="utf-8", errors="replace") as f:
        hdr = None; rows = []
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if hdr is None:
                hdr = p; continue
            rows.append(p)
    return pd.DataFrame(rows, columns=hdr)


# ------------------------------------------------------------------ main
def main():
    panel = pd.read_csv(PANEL30)
    panel["gene"] = panel["gene"].astype(str).str.strip()
    coef = dict(zip(panel["gene"], panel["coef"]))

    # ---- RNA -------------------------------------------------------------
    rna_p = os.path.join(CD, "data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt")
    rna = pd.read_csv(rna_p, sep="\t", low_memory=False)
    sym = rna["Hugo_Symbol"].astype(str).str.strip()
    eid = rna["Entrez_Gene_Id"].astype(str).str.strip()
    ent2sym = dict(zip(eid, sym))
    X = rna.drop(columns=["Hugo_Symbol", "Entrez_Gene_Id"]).astype(float)
    X.index = sym.values                                  # index by SYMBOL
    X = X[~X.index.duplicated(keep="first")]
    log(f"RNA matrix: {X.shape[0]} genes x {X.shape[1]} samples")
    log(f"  values look z-scored: mean={X.values.mean():.4f} sd={X.values.std():.4f}")
    log(f"  index sample: {list(X.index[:5])}")

    genes = sorted(set(coef) | set(PROLIF))
    missing = [g for g in genes if g not in X.index]
    log(f"REP30 genes present: {len([g for g in coef if g in X.index])}/30; "
        f"prolif {len([g for g in PROLIF if g in X.index])}/{len(PROLIF)}")
    if missing:
        log(f"  MISSING: {missing}")

    Z = X.loc[[g for g in coef if g in X.index]].T          # samples x genes
    rep30 = sum(coef[g] * Z[g] for g in Z.columns)
    P = X.loc[[g for g in PROLIF if g in X.index]].T
    prolif = P.mean(axis=1)
    df = pd.DataFrame({"rep30": rep30, "prolif": prolif, "MKI67": X.loc["MKI67"].T})

    # ---- annotation ------------------------------------------------------
    cs = rd("data_clinical_sample.txt").set_index("SAMPLE_ID")
    cp = rd("data_clinical_patient.txt").set_index("PATIENT_ID")
    keep_s = ["PATIENT_ID", "METASTATIC_SITE", "SAMPLE_TYPE", "PRESERVATION",
              "NUMBER_PROLIFERATING_CELLS", "PERCENT_TUMOR_CELLS",
              "PERCENT_STROMAL_CELLS", "PERCENT_NORMAL_CELLS",
              "PERCENT_LYMPHOCYTE_INFILTRATION", "ONCOTREE_CODE",
              "CANCER_TYPE_DETAILED"]
    df = df.join(cs[keep_s], how="left")
    log(f"\nclinical sample joined: {df['PATIENT_ID'].notna().sum()}/{len(df)}")
    log("  SAMPLE_TYPE: " + str(df['SAMPLE_TYPE'].value_counts().to_dict()))
    log("  PRESERVATION: " + str(df['PRESERVATION'].value_counts().to_dict()))

    keep_p = ["PRIM_ER", "PRIM_PR", "PRIM_HER2", "PRIM_HER2_INTERPRETATION",
              "PRIM_STAGE_DX", "AGE", "OS_STATUS", "DFS_MONTHS", "PRIM_ONCOTYPE"]
    df = df.join(cp[keep_p], on="PATIENT_ID")
    log(f"  PRIM_ER: {df['PRIM_ER'].value_counts().to_dict()}")
    log(f"  PRIM_HER2_INTERPRETATION: {df['PRIM_HER2_INTERPRETATION'].value_counts().to_dict()}")

    for c in ["PERCENT_TUMOR_CELLS", "PERCENT_STROMAL_CELLS", "NUMBER_PROLIFERATING_CELLS",
              "PERCENT_NORMAL_CELLS", "PERCENT_LYMPHOCYTE_INFILTRATION", "AGE"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # HR+ : ER+ or PR+ AND HER2 negative
    her2 = df["PRIM_HER2_INTERPRETATION"].astype(str).str.lower()
    er = df["PRIM_ER"].astype(str).str.lower()
    pr = df["PRIM_PR"].astype(str).str.lower()
    df["hr_pos"] = (er.str.startswith("pos") | pr.str.startswith("pos")) & her2.str.startswith("neg")
    df["hr_neg"] = (er.str.startswith("neg") & pr.str.startswith("neg")) & her2.str.startswith("neg")
    log(f"  HR+ (ER/PR+ & HER2-): {int(df['hr_pos'].sum())} samples; "
        f"HR- (TNBC-like): {int(df['hr_neg'].sum())} samples")

    # ---- mutations -------------------------------------------------------
    mut = rd("data_mutations.txt")
    mut["Tumor_Sample_Barcode"] = mut["Tumor_Sample_Barcode"].astype(str)
    ns = mut[mut["Variant_Classification"].isin(NONSYN)]
    tr = mut[mut["Variant_Classification"].isin(TRUNCATING)]

    def mut_set(gene, table):
        return set(table.loc[table["Hugo_Symbol"] == gene, "Tumor_Sample_Barcode"])

    df["ESR1_mut"] = df.index.isin(mut_set("ESR1", ns))
    df["ERBB2_mut"] = df.index.isin(mut_set("ERBB2", ns))
    df["PIK3CA_mut"] = df.index.isin(mut_set("PIK3CA", ns))
    df["TP53_alt"] = df.index.isin(mut_set("TP53", ns))
    rb1mut = mut_set("RB1", tr) | mut_set("RB1", ns)

    # ---- CNA -------------------------------------------------------------
    cna = rd("data_log2_cna.txt").set_index("Entrez_Gene_Id")
    cna.index = cna.index.astype(str)
    g2e = {v: k for k, v in ent2sym.items()}
    def cna_row(gene):
        e = g2e.get(gene)
        if e is None or e not in cna.index:
            return None
        return pd.to_numeric(cna.loc[e], errors="coerce")
    rb1_cna = cna_row("RB1")
    erbb2_cna = cna_row("ERBB2")
    ccnd1_cna = cna_row("CCND1")
    if erbb2_cna is not None:
        df["ERBB2_cna"] = erbb2_cna.reindex(df.index)
        df["ERBB2_amp"] = df["ERBB2_cna"] > 1.0
        log(f"\nERBB2 log2CNA: p50={erbb2_cna.median():.2f} p95={erbb2_cna.quantile(.95):.2f}; "
            f"amp(>1.0) n={int(df['ERBB2_amp'].sum())}")
    if rb1_cna is not None:
        df["RB1_cna"] = rb1_cna.reindex(df.index)
        df["RB1_del"] = df["RB1_cna"] < -0.5
        log(f"RB1 log2CNA: p5={rb1_cna.quantile(.05):.2f} p50={rb1_cna.median():.2f}; "
            f"del(<-0.5) n={int(df['RB1_del'].sum())}")
    if ccnd1_cna is not None:
        df["CCND1_cna"] = ccnd1_cna.reindex(df.index)
    df["RB1_loss"] = df.index.isin(rb1mut) | df.get("RB1_del", False)
    log(f"mutation carriers: ESR1={int(df['ESR1_mut'].sum())} RB1mut={len(rb1mut)} "
        f"PIK3CA={int(df['PIK3CA_mut'].sum())} TP53={int(df['TP53_alt'].sum())}")

    # ---- treatment / sampling timeline -----------------------------------
    sp = rd("data_timeline_specimen.txt")
    sp["START_DATE"] = pd.to_numeric(sp["START_DATE"], errors="coerce")
    sp = sp.rename(columns={"Specimen_ID": "SAMPLE_ID", "START_DATE": "collect_day",
                            "Specimen_TYPE": "specimen_type", "Specimen_SITE": "specimen_site"})
    df = df.join(sp.set_index("SAMPLE_ID")[["collect_day", "specimen_type", "specimen_site"]],
                 how="left")
    log(f"\nspecimen collection day available: {df['collect_day'].notna().sum()}/{len(df)}")

    dx = rd("data_timeline_dx.txt")
    dx["START_DATE"] = pd.to_numeric(dx["START_DATE"], errors="coerce")
    mdx = (dx[dx["DIAGNOSIS_TYPE"].str.contains("Metastatic", na=False)]
           .groupby("PATIENT_ID")["START_DATE"].min().rename("met_dx_day"))
    df = df.join(mdx, on="PATIENT_ID")
    log(f"metastatic diagnosis day available: {df['met_dx_day'].notna().sum()}/{len(df)}")

    tt = rd("data_timeline_treatment.txt")
    for c in ["START_DATE", "STOP_DATE"]:
        tt[c] = pd.to_numeric(tt[c], errors="coerce")
    met_tx = tt[tt["Treatment_TYPE"].str.contains("Metastatic", na=False)].copy()

    # per-sample: how many metastatic lines started BEFORE the biopsy, and days on therapy
    rows = []
    for sid, r in df.iterrows():
        if pd.isna(r.get("collect_day")):
            rows.append((np.nan, np.nan, np.nan)); continue
        t = met_tx[met_tx["PATIENT_ID"] == r["PATIENT_ID"]]
        started = t[(t["START_DATE"].notna()) & (t["START_DATE"] <= r["collect_day"])]
        # days on metastatic therapy prior to biopsy (clipped at 0)
        days = float(np.nansum((np.minimum(started["STOP_DATE"].fillna(r["collect_day"]),
                                           r["collect_day"])
                                - started["START_DATE"]).clip(lower=0)))
        progressed = int(((started["STOP_DATE"].notna()) & (started["STOP_DATE"] <= r["collect_day"]) &
                          (started["RESPONSE"].astype(str).str.contains("Progression", na=False))).sum())
        rows.append((len(started), days, progressed))
    df["met_lines_before_biopsy"] = [r[0] for r in rows]
    df["days_on_met_tx"] = [r[1] for r in rows]
    df["n_progressions_before_biopsy"] = [r[2] for r in rows]
    df["post_met_treatment"] = df["met_lines_before_biopsy"] > 0
    df["days_since_met_dx"] = df["collect_day"] - df["met_dx_day"]
    log("\nmetastatic lines before biopsy (samples): " +
        str(df["met_lines_before_biopsy"].value_counts(dropna=False).sort_index().to_dict()))
    log("post-metastatic-treatment samples: %d / %d" %
        (int(df["post_met_treatment"].sum()), int(df["met_lines_before_biopsy"].notna().sum())))

    df.to_csv(os.path.join(OUT, "aurora_linked.csv"))

    # ---- tests -----------------------------------------------------------
    res = []
    def add(scope, test, n, stat, p, extra=""):
        res.append(dict(scope=scope, test=test, n=n, statistic=stat, p=p, detail=extra))

    def group_test(scope, sub, flag, name, score="rep30",
                   covars=("prolif", "PERCENT_TUMOR_CELLS", "PERCENT_STROMAL_CELLS")):
        f = sub[flag].fillna(False).astype(bool)
        a = sub.loc[f, score].values; b = sub.loc[~f, score].values
        if len(a) < 3 or len(b) < 3:
            add(scope, name, f"{len(a)}/{len(b)}", np.nan, np.nan, "too few"); return
        add(scope, name, f"{len(a)}/{len(b)}", round(cliffs_delta(a, b), 3), mwu(a, b)[1],
            f"med {np.median(a):.3f} vs {np.median(b):.3f}")
        cv = [sub[c].values for c in covars if c in sub]
        d2, p2, n2 = ols_adjust(sub[score].values, f.values, cv)
        add(scope, name + " [adjusted: prolif+TC+stroma]", n2,
            round(d2, 3) if d2 == d2 else np.nan, p2, "OLS residual -> MWU")

    # scope definitions
    log("\n" + "=" * 78)
    scopes = {
        "all-samples": df,
        "Primary": df[df["SAMPLE_TYPE"] == "Primary"],
        "Metastatic": df[df["SAMPLE_TYPE"] == "Metastatic"],
        "Metastatic(HR+)": df[(df["SAMPLE_TYPE"] == "Metastatic") & df["hr_pos"]],
        "FF-only": df[df["PRESERVATION"] == "FF"],
    }
    for sname, sub in scopes.items():
        if len(sub) < 12:
            log(f"skip {sname} (n={len(sub)})"); continue
        log(f"\n--- scope {sname}: n={len(sub)} samples / {sub['PATIENT_ID'].nunique()} patients")
        for flag, nm in [("ESR1_mut", "REP30 vs ESR1 mutation"),
                         ("RB1_loss", "REP30 vs RB1 loss"),
                         ("ERBB2_amp", "REP30 vs ERBB2 amplification"),
                         ("PIK3CA_mut", "REP30 vs PIK3CA mutation"),
                         ("TP53_alt", "REP30 vs TP53 alteration"),
                         ("hr_pos", "REP30 HR+ vs non-HR+")]:
            group_test(sname, sub, flag, nm)

        # proliferation + confound screens on the RNA-based axis
        for c, lab in [("prolif", "proliferation (RNA module)"),
                       ("MKI67", "MKI67"),
                       ("PERCENT_TUMOR_CELLS", "percent tumour cells"),
                       ("PERCENT_STROMAL_CELLS", "percent stromal cells"),
                       ("NUMBER_PROLIFERATING_CELLS", "proliferating cells (histology)"),
                       ("days_on_met_tx", "days on metastatic therapy"),
                       ("met_lines_before_biopsy", "metastatic lines before biopsy"),
                       ("days_since_met_dx", "days since metastatic dx")]:
            if c not in sub:
                continue
            v = pd.to_numeric(sub[c], errors="coerce")
            ok = v.notna()
            if ok.sum() > 12:
                rr, pv = stats.spearmanr(sub.loc[ok, "rep30"], v[ok])
                add(sname, f"REP30 vs {lab} (Spearman)", int(ok.sum()), round(rr, 3), pv)

    # ---- PRIMARY vs METASTATIC (overall + paired within patient) ---------
    log("\n" + "=" * 78)
    prim = df[df["SAMPLE_TYPE"] == "Primary"]["rep30"]
    met = df[df["SAMPLE_TYPE"] == "Metastatic"]["rep30"]
    add("primary-vs-met", "REP30 metastatic vs primary (MWU)", f"{len(met)}/{len(prim)}",
        round(cliffs_delta(met, prim), 3), mwu(met, prim)[1],
        f"med {np.median(met):.3f} vs {np.median(prim):.3f}")
    f = (df["SAMPLE_TYPE"] == "Metastatic").values
    d, p, n = ols_adjust(df["rep30"].values, f,
                         [df["prolif"].values, df["PERCENT_TUMOR_CELLS"].values,
                          df["PERCENT_STROMAL_CELLS"].values])
    add("primary-vs-met", "REP30 met vs primary [adjusted prolif+TC+stroma]", n,
        round(d, 3), p, "OLS residual -> MWU")
    # FFPE-only (the GSE209998 lesson)
    sub = df[df["PRESERVATION"] == "FF"]
    fm = sub["SAMPLE_TYPE"] == "Metastatic"
    add("primary-vs-met(FF only)", "REP30 met vs primary (MWU)",
        f"{int(fm.sum())}/{int((~fm).sum())}",
        round(cliffs_delta(sub.loc[fm, "rep30"], sub.loc[~fm, "rep30"]), 3),
        mwu(sub.loc[fm, "rep30"], sub.loc[~fm, "rep30"])[1],
        "fresh-frozen only -> FFPE artefact removed")

    # paired within patient
    piv = df.dropna(subset=["PATIENT_ID"]).groupby(["PATIENT_ID", "SAMPLE_TYPE"])["rep30"].mean().unstack()
    piv = piv.dropna(subset=["Primary", "Metastatic"])
    log(f"\npaired patients with both primary and metastatic RNA: {len(piv)}")
    if len(piv) >= 6:
        delta = (piv["Metastatic"] - piv["Primary"]).values
        w, pw = stats.wilcoxon(delta, alternative="two-sided")
        add("paired-within-patient", "REP30 delta (met - primary) Wilcoxon signed-rank",
            len(delta), round(np.median(delta), 3), pw,
            f"median delta={np.median(delta):.3f}; n_increase={int((delta>0).sum())}/{len(delta)}")
        log(f"  median delta={np.median(delta):.3f}; increase in {int((delta>0).sum())}/{len(delta)}")

    # ---- treatment-exposure axis (the unique part) ------------------------
    log("\n" + "=" * 78)
    expo = df[df["met_lines_before_biopsy"].notna()].copy()
    add("treatment-exposure", "REP30 post- vs pre-metastatic-treatment (MWU)",
        f"{int(expo['post_met_treatment'].sum())}/{int((~expo['post_met_treatment']).sum())}",
        round(cliffs_delta(expo.loc[expo["post_met_treatment"], "rep30"],
                           expo.loc[~expo["post_met_treatment"], "rep30"]), 3),
        mwu(expo.loc[expo["post_met_treatment"], "rep30"],
            expo.loc[~expo["post_met_treatment"], "rep30"])[1])
    r, p = stats.spearmanr(expo["rep30"], expo["days_on_met_tx"])
    add("treatment-exposure", "REP30 vs days on metastatic therapy (Spearman)",
        len(expo), round(r, 3), p)
    rp, pp = partial_spearman(expo["rep30"].values, expo["days_on_met_tx"].values,
                              [expo["prolif"].values, expo["PERCENT_TUMOR_CELLS"].values])
    add("treatment-exposure", "REP30 vs days on met therapy [prolif+TC adjusted]",
        len(expo), round(rp, 3), pp, "partial Spearman")
    r2, p2 = stats.spearmanr(expo["rep30"], expo["n_progressions_before_biopsy"])
    add("treatment-exposure", "REP30 vs n progressions before biopsy (Spearman)",
        len(expo), round(r2, 3), p2)
    # within metastatic only, and within HR+ metastatic only
    for tag, sub2 in [("met-only", expo[expo["SAMPLE_TYPE"] == "Metastatic"]),
                      ("met-only HR+", expo[(expo["SAMPLE_TYPE"] == "Metastatic") & expo["hr_pos"]])]:
        if len(sub2) < 15:
            continue
        r, p = stats.spearmanr(sub2["rep30"], sub2["days_on_met_tx"])
        add(f"treatment-exposure/{tag}", "REP30 vs days on met therapy (Spearman)",
            len(sub2), round(r, 3), p)

    out = pd.DataFrame(res)
    out.to_csv(os.path.join(OUT, "rep30_tests.csv"), index=False)
    log("\n" + out.to_string(index=False))

    # ---- figures ---------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        ax = axes[0]
        data = [prim.values, met.values]
        try:
            ax.boxplot(data, tick_labels=["Primary", "Metastatic"], showfliers=False)
        except TypeError:
            ax.boxplot(data, labels=["Primary", "Metastatic"], showfliers=False)
        for i, dd in enumerate(data):
            ax.scatter(np.random.normal(i + 1, 0.06, len(dd)), dd, s=10, alpha=0.5)
        ax.set_ylabel("REP30"); ax.set_title(f"REP30 primary vs met\nn={len(prim)}/{len(met)}")
        ax.grid(alpha=0.3)

        ax = axes[1]
        for lab, mk, col in [("FF", "o", "tab:blue"), ("FFPE", "^", "tab:red")]:
            s = df[df["PRESERVATION"] == lab]
            ax.scatter(s["prolif"], s["rep30"], s=16, alpha=0.6, marker=mk, label=lab)
        ax.set_xlabel("proliferation (RNA module)"); ax.set_ylabel("REP30")
        ax.legend(fontsize=8); ax.set_title("REP30 vs proliferation, by preservation")
        ax.grid(alpha=0.3)

        ax = axes[2]
        e = expo
        ax.scatter(e["days_on_met_tx"], e["rep30"], s=16, alpha=0.6)
        rr, pp = stats.spearmanr(e["days_on_met_tx"], e["rep30"])
        ax.set_xlabel("days on metastatic therapy before biopsy")
        ax.set_ylabel("REP30")
        ax.set_title(f"treatment exposure (n={len(e)})\nSpearman r={rr:.3f}, p={pp:.2e}")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "aurora_rep30_overview.png"), dpi=160)
        plt.close()
        log("figure written")
    except Exception as e:
        log(f"figure skipped: {type(e).__name__}: {e}")

    with open(os.path.join(OUT, "run_log.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(_log))


if __name__ == "__main__":
    main()
