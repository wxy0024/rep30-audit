#!/usr/bin/env python
"""
REP30 / REP52 scoring in the AURORA US metastatic breast cancer cohort
(GSE209998, n=129 RNA-seq samples; 44 primary / 79 metastatic / 6 normal).

Question
--------
Does the REP30 expression signature capture the *acquired-resistance /
treatment-exposed* state?

Design (all within one uniformly processed RNA-seq cohort, so no cross-study
batch artefact; sample-type differences are the signal, not a technical artefact):

  A. TREATMENT EXPOSURE (primary tumours only)
       Post-treatment primary (n=17)  vs  Pre-treatment primary (n=26)
     -> the closest available surrogate for "resistant / treated" state.

  B. METASTASIS
       Metastatic (n=79)  vs  Primary (n=44)
     including a PATIENT-PAIRED primary<->metastasis comparison
     (patient code embedded in the AUR sample id, e.g. AUR-AFEA-TTP1 / -TTM4).

  C. NEGATIVE CONTROL
       Normal tissue (n=6) as a reference for "not cancer".

Confounder / specificity controls
---------------------------------
  * FFPE vs fresh-frozen batch composition per group
  * REP30 vs a proliferation score (MKI67-led 10-gene panel) correlation
  * empirical permutation null: 2000 random 30-gene sets, same group contrast
  * per-gene deltas inside the panel (are the panel genes, or just a couple,
    driving the score?)

Scoring
-------
Expression: GSE209998_AUR_129_UQN.txt.gz (upper-quartile normalised counts).
  -> log2(x+1) -> per-gene z-score across all 129 samples -> REP = sum(coef * z)

Outputs -> results/metastatic/AURORA_US_rep30/
"""
from __future__ import annotations

import gzip
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"E:/01.WorkSpace/01.Breast_cancer_gene_panel/resistance_signature"
D = os.path.join(ROOT, "data", "06_metastatic", "AURORA_US")
OUT = os.path.join(ROOT, "results", "metastatic", "AURORA_US_rep30")
PANEL30 = os.path.join(ROOT, "results", "rep52_trim", "rep52_trimmed_panel.csv")
PANEL52 = os.path.join(ROOT, "results", "rep_genes.csv")

PROLIF = ["MKI67", "PCNA", "TOP2A", "CCNB1", "BUB1", "AURKA", "PLK1", "MCM2",
          "RRM2", "TYMS", "CDC20", "BIRC5"]

os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "run.log"), "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


# ----------------------------------------------------------------- metadata
def parse_series_matrix(path):
    rows = defaultdict(list)
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if not line.startswith("!"):
                continue
            parts = [p.strip().strip('"') for p in line.rstrip("\n").split("\t")]
            if len(parts) >= 2:
                rows[parts[0]].append(parts[1:])
    return rows


def build_meta():
    sm = parse_series_matrix(os.path.join(D, "GSE209998_series_matrix.txt.gz"))
    n = len(sm["!Sample_title"][0])
    recs = []
    for i in range(n):
        title = sm["!Sample_title"][0][i]
        m = re.search(r"\[(.+?)\]", title)
        sid = m.group(1) if m else title
        d = {"sample": sid, "gsm": sm["!Sample_geo_accession"][0][i],
             "source": sm["!Sample_source_name_ch1"][0][i]}
        for row in sm["!Sample_characteristics_ch1"]:
            if len(row) != n:
                continue
            kv = row[i]
            if ":" in kv:
                k, v = kv.split(":", 1)
                d[k.strip()] = v.strip()
        descs = [r[i] for r in sm["!Sample_description"] if len(r) == n]
        d["prep"] = next((x for x in descs if x in ("FFPE", "Fresh frozen")), "NA")
        recs.append(d)
    meta = pd.DataFrame(recs).set_index("sample")
    # patient code: AUR-<CODE>-TT...
    meta["patient"] = [re.match(r"AUR-([A-Z0-9]+)-", s).group(1)
                       if re.match(r"AUR-([A-Z0-9]+)-", s) else s.split("-")[1]
                       for s in meta.index]
    meta["group"] = np.where(meta["disease"].str.contains("Metasta"),
                             "Metastatic",
                             np.where(meta["disease"].str.contains("Primary"),
                                      "Primary", "Normal"))
    meta["trt"] = meta["treatment"]
    return meta


# ----------------------------------------------------------------- expression
def load_expr(meta):
    path = os.path.join(D, "GSE209998_AUR_129_UQN.txt.gz")
    log(f"[expr] reading {os.path.basename(path)} ...")
    df = pd.read_csv(path, sep="\t", index_col=0, low_memory=False)
    df = df.apply(pd.to_numeric, errors="coerce")
    log(f"[expr] raw shape = {df.shape}")
    # keep samples with metadata
    cols = [c for c in df.columns if c in meta.index]
    log(f"[expr] matched to metadata: {len(cols)}/{df.shape[1]} columns")
    df = df[cols]
    # drop all-zero / all-nan genes
    df = df.dropna(how="all")
    df = df.loc[df.sum(axis=1) > 0]
    log(f"[expr] after gene filter = {df.shape}")
    X = np.log2(df.values + 1.0)                      # genes x samples
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    return pd.DataFrame(Z, index=df.index, columns=df.columns), df


def score(Z, genes, coef):
    g = [x for x in genes if x in Z.index]
    if not g:
        return None, []
    w = np.array([coef[x] for x in g])
    M = Z.loc[g].values
    return (w[:, None] * M).sum(axis=0), g


# ----------------------------------------------------------------- stats
def cliffs_delta(a, b):
    a0 = np.asarray(a, float)
    b0 = np.asarray(b, float)
    if len(a0) == 0 or len(b0) == 0:
        return np.nan
    gt = (a0[:, None] > b0[None, :]).sum()
    lt = (a0[:, None] < b0[None, :]).sum()
    return float(gt - lt) / (len(a0) * len(b0))


def mwu(a, b):
    """Mann-Whitney U two-sided p, no scipy dependency fallback."""
    try:
        from scipy.stats import mannwhitneyu
        return mannwhitneyu(a, b, alternative="two-sided").pvalue
    except Exception:
        # normal approximation
        a = np.asarray(a, float); b = np.asarray(b, float)
        n1, n2 = len(a), len(b)
        r = np.concatenate([a, b])
        ranks = pd.Series(r).rank().values
        R1 = ranks[:n1].sum()
        U = R1 - n1 * (n1 + 1) / 2
        mu = n1 * n2 / 2
        sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        from math import erf, sqrt
        zz = abs(U - mu) / sd
        return 2 * (1 - 0.5 * (1 + erf(zz / sqrt(2))))


def wilcoxon(a, b):
    try:
        from scipy.stats import wilcoxon as w
        return w(a, b, zero_method="wilcox").pvalue
    except Exception:
        return np.nan


def bonferroni(p, n):
    return min(1.0, p * n)


def ols(X, y):
    """Least-squares fit; return (beta, se, two-sided p) per term."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    dof = max(1, n - k)
    s2 = float((resid ** 2).sum() / dof)
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv) * s2)
    t = beta / se
    try:
        from scipy.stats import t as tdist
        p = 2 * (1 - tdist.cdf(np.abs(t), dof))
    except Exception:
        p = np.full_like(t, np.nan)
    return beta, se, p


# ----------------------------------------------------------------- main
def main():
    meta = build_meta()
    log("[meta] n =", len(meta))
    log("[meta] group x treatment:")
    log(meta.groupby(["group", "trt"]).size().to_string())
    log("[meta] group x prep:")
    log(meta.groupby(["group", "prep"]).size().to_string())

    Z, raw = load_expr(meta)
    Z = Z[[c for c in Z.columns]]

    p30 = pd.read_csv(PANEL30)
    p30 = p30.set_index("gene")["coef"].to_dict()
    p52 = pd.read_csv(PANEL52).set_index("gene")["coef"].to_dict()

    s30, g30 = score(Z, list(p30), p30)
    s52, g52 = score(Z, list(p52), p52)
    log(f"[score] REP30 genes used = {len(g30)}/{len(p30)}")
    log(f"[score] REP52 genes used = {len(g52)}/{len(p52)}")

    # proliferation score
    pg = [g for g in PROLIF if g in Z.index]
    prolif = Z.loc[pg].mean(axis=0).values
    mki67 = Z.loc["MKI67"].values if "MKI67" in Z.index else np.full(len(Z.columns), np.nan)

    res = pd.DataFrame({
        "REP30": s30, "REP52": s52, "prolif": prolif, "MKI67": mki67,
    }, index=Z.columns)
    res = res.join(meta)
    res.to_csv(os.path.join(OUT, "rep30_per_sample.csv"))

    # ------------------------------------------------- comparisons
    def compare(name, maskA, maskB, labelA, labelB, col="REP30"):
        a = res.loc[maskA, col].values
        b = res.loc[maskB, col].values
        d = np.median(a) - np.median(b)
        return {
            "score": col,
            "contrast": name,
            "group_A": labelA, "n_A": len(a),
            "group_B": labelB, "n_B": len(b),
            "median_A": round(float(np.median(a)), 4),
            "median_B": round(float(np.median(b)), 4),
            "delta_median": round(float(d), 4),
            "cliffs_delta": round(cliffs_delta(a, b), 4),
            "p_mwu": mwu(a, b),
        }

    prim = res["group"] == "Primary"
    met = res["group"] == "Metastatic"
    nrm = res["group"] == "Normal"

    rows = []
    rows.append(compare("Primary: Post-treatment vs Pre-treatment",
                        prim & (res["trt"] == "Post-treatment"),
                        prim & (res["trt"] == "Pre-treatment"),
                        "Post-treatment", "Pre-treatment"))
    rows.append(compare("Metastatic vs Primary", met, prim, "Metastatic", "Primary"))
    rows.append(compare("Metastatic vs Normal", met, nrm, "Metastatic", "Normal"))
    rows.append(compare("Primary vs Normal", prim, nrm, "Primary", "Normal"))

    # proliferation-adjusted (linear residuals of REP30 on prolif)
    beta = np.polyfit(res["prolif"].values, res["REP30"].values, 1)
    res["REP30_adj"] = res["REP30"].values - np.polyval(beta, res["prolif"].values)
    for nm, mA, mB, lA, lB in [
        ("[prolif-adjusted] Primary: Post vs Pre",
         prim & (res["trt"] == "Post-treatment"), prim & (res["trt"] == "Pre-treatment"),
         "Post", "Pre"),
        ("[prolif-adjusted] Metastatic vs Primary", met, prim, "Metastatic", "Primary"),
    ]:
        a = res.loc[mA, "REP30_adj"].values
        b = res.loc[mB, "REP30_adj"].values
        rows.append({
            "score": "REP30_adj",
            "contrast": nm, "group_A": lA, "n_A": len(a), "group_B": lB, "n_B": len(b),
            "median_A": round(float(np.median(a)), 4),
            "median_B": round(float(np.median(b)), 4),
            "delta_median": round(float(np.median(a) - np.median(b)), 4),
            "cliffs_delta": round(float(cliffs_delta(a, b)), 4),
            "p_mwu": mwu(a, b)})

    # by metastasis site
    for site in res.loc[met, "tissue"].value_counts().index:
        if (res.loc[met, "tissue"] == site).sum() < 3:
            continue
        rows.append(compare(f"Met site {site} vs Primary", met & (res["tissue"] == site),
                            prim, site, "Primary"))

    # ---- confounder: sample prep (FFPE vs fresh frozen) inside primary ----
    log("\n[confound] primary trt x prep:")
    xt = pd.crosstab(res.loc[prim, "trt"], res.loc[prim, "prep"])
    log(xt.to_string())
    log("[confound] group x prep:")
    xg = pd.crosstab(res.loc[~nrm, "group"], res.loc[~nrm, "prep"])
    log(xg.to_string())
    # prep effect size, measured inside one homogeneous group (pre-treatment primary)
    base = prim & (res["trt"] == "Pre-treatment")
    rows.append(compare("Primary pre-treatment: FFPE vs fresh frozen",
                        base & (res["prep"] == "FFPE"),
                        base & (res["prep"] == "Fresh frozen"),
                        "FFPE", "Fresh frozen"))
    for prep in ["FFPE", "Fresh frozen"]:
        mA = prim & (res["trt"] == "Post-treatment") & (res["prep"] == prep)
        mB = prim & (res["trt"] == "Pre-treatment") & (res["prep"] == prep)
        if mA.sum() >= 3 and mB.sum() >= 3:
            rows.append(compare(f"Primary Post vs Pre [{prep} only]", mA, mB,
                                f"Post-{prep}", f"Pre-{prep}"))
    # ---- prep-matched subsets for the metastasis contrast (met is 96% fresh frozen) ----
    rows.append(compare("Metastatic vs Primary [fresh frozen only]",
                        met & (res["prep"] == "Fresh frozen"),
                        prim & (res["prep"] == "Fresh frozen"),
                        "Met-FF", "Primary-FF"))
    rows.append(compare("Metastatic vs Pre-treatment Primary [fresh frozen only]",
                        met & (res["prep"] == "Fresh frozen"),
                        prim & (res["prep"] == "Fresh frozen") & (res["trt"] == "Pre-treatment"),
                        "Met-FF", "PrePrim-FF"))
    rows.append(compare("Metastatic vs Primary [fresh frozen only]",
                        met & (res["prep"] == "Fresh frozen"),
                        prim & (res["prep"] == "Fresh frozen"),
                        "Met-FF", "Primary-FF", col="prolif"))
    # ---- same contrasts on the proliferation score (specificity control) ----
    for col in ["prolif", "MKI67", "REP52"]:
        rows.append(compare("Primary: Post vs Pre", prim & (res["trt"] == "Post-treatment"),
                            prim & (res["trt"] == "Pre-treatment"),
                            "Post-treatment", "Pre-treatment", col=col))
        rows.append(compare("Metastatic vs Primary", met, prim,
                            "Metastatic", "Primary", col=col))

    comp = pd.DataFrame(rows)
    comp["score"] = comp["score"].fillna("REP30_adj")
    comp["p_bonf"] = [bonferroni(p, len(comp)) for p in comp["p_mwu"]]
    comp = comp[["score", "contrast", "group_A", "n_A", "group_B", "n_B",
                 "median_A", "median_B", "delta_median", "cliffs_delta",
                 "p_mwu", "p_bonf"]]
    comp.to_csv(os.path.join(OUT, "rep30_comparisons.csv"), index=False)
    log("\n[comparisons]")
    log(comp.to_string(index=False))

    # ------------------------------------------------- patient-paired tests
    pairs = []
    piv = pd.DataFrame()          # paired primary<->metastasis table (for plotting)
    # paired primary <-> metastasis
    cp = res[["patient", "group", "REP30", "REP52"]].copy()
    pv = cp[cp["group"].isin(["Primary", "Metastatic"])].groupby(
        ["patient", "group"])["REP30"].median().unstack()
    if {"Primary", "Metastatic"} <= set(pv.columns):
        pv = pv.dropna()
        if len(pv) >= 5:
            piv = pv
            p = wilcoxon(pv["Metastatic"].values, pv["Primary"].values)
            pairs.append({
                "contrast": "PAIRED primary -> metastasis (same patient)",
                "n_pairs": len(pv),
                "median_delta_met_minus_prim": round(float(
                    np.median(pv["Metastatic"] - pv["Primary"])), 4),
                "n_up": int((pv["Metastatic"] > pv["Primary"]).sum()),
                "n_down": int((pv["Metastatic"] < pv["Primary"]).sum()),
                "p_wilcoxon": p})
    # fresh-frozen-only paired version (prep-matched on both sides)
    cpf = res[["patient", "group", "REP30", "prep"]]
    f = cpf[(cpf["group"].isin(["Primary", "Metastatic"]))
            & (cpf["prep"] == "Fresh frozen")]
    pvf = f.groupby(["patient", "group"])["REP30"].median().unstack()
    if {"Primary", "Metastatic"} <= set(pvf.columns):
        pvf = pvf.dropna()
        if len(pvf) >= 5:
            pairs.append({
                "contrast": "PAIRED primary -> metastasis [fresh frozen both]",
                "n_pairs": len(pvf),
                "median_delta_met_minus_prim": round(float(
                    np.median(pvf["Metastatic"] - pvf["Primary"])), 4),
                "n_up": int((pvf["Metastatic"] > pvf["Primary"]).sum()),
                "n_down": int((pvf["Metastatic"] < pvf["Primary"]).sum()),
                "p_wilcoxon": wilcoxon(pvf["Metastatic"].values, pvf["Primary"].values)})
    # paired pre vs post treatment primary
    pr = res[prim & res["trt"].isin(["Pre-treatment", "Post-treatment"])]
    piv2 = pr.groupby(["patient", "trt"])["REP30"].median().unstack()
    if {"Pre-treatment", "Post-treatment"} <= set(piv2.columns) and len(piv2):
        piv2 = piv2.dropna()
        n2 = len(piv2)
        pairs.append({
            "contrast": "PAIRED pre-treatment -> post-treatment primary",
            "n_pairs": n2,
            "median_delta_post_minus_pre": round(float(
                np.median(piv2["Post-treatment"] - piv2["Pre-treatment"])), 4) if n2 else np.nan,
            "n_up": int((piv2["Post-treatment"] > piv2["Pre-treatment"]).sum()) if n2 else 0,
            "n_down": int((piv2["Post-treatment"] < piv2["Pre-treatment"]).sum()) if n2 else 0,
            "p_wilcoxon": wilcoxon(piv2["Post-treatment"].values,
                                   piv2["Pre-treatment"].values) if n2 >= 5 else np.nan})
    else:
        pairs.append({"contrast": "PAIRED pre-treatment -> post-treatment primary",
                      "n_pairs": 0, "median_delta_post_minus_pre": np.nan,
                      "n_up": 0, "n_down": 0, "p_wilcoxon": np.nan})
    pair_df = pd.DataFrame(pairs)
    pair_df.to_csv(os.path.join(OUT, "rep30_paired.csv"), index=False)
    log("\n[paired]")
    log(pair_df.to_string(index=False) if len(pair_df) else "  (no pair with n>=5)")

    # ------------------------------------------------- permutation null
    log("\n[perm] building random 30-gene null (2000 sets) ...")
    span = Z.values.max(axis=1) - Z.values.min(axis=1)
    exp_genes = Z.index.values[span > 0.5]
    rng = np.random.default_rng(0)
    contrasts = {
        "Primary Post vs Pre": (prim & (res["trt"] == "Post-treatment"),
                                prim & (res["trt"] == "Pre-treatment")),
        "Metastatic vs Primary": (met, prim),
    }
    null_rows = []
    coefs30 = np.array([p30[g] for g in g30])
    Zv = Z.values
    idx = {g: i for i, g in enumerate(Z.index)}
    for cname, (mA, mB) in contrasts.items():
        ia = np.where(mA.values)[0]
        ib = np.where(mB.values)[0]
        obs = float(np.median(res.loc[mA, "REP30"]) - np.median(res.loc[mB, "REP30"]))
        # interpretable, scale-matched companion: unweighted mean z of the same panel
        Zm = Z.loc[g30].values
        obs_mean = float(np.median(Zm[:, ia].mean(0)) - np.median(Zm[:, ib].mean(0)))
        nulls = np.empty(2000)
        nulls_mean = np.empty(2000)
        for k in range(2000):
            pick = rng.choice(exp_genes, size=30, replace=False)
            rows_i = np.array([idx[g] for g in pick if g in idx])
            v = Zv[rows_i]
            w = rng.permutation(coefs30)          # same coef magnitudes, random pairing
            sc = (w[:, None] * v).sum(0)
            nulls[k] = np.median(sc[ia]) - np.median(sc[ib])
            vm = v.mean(0)
            nulls_mean[k] = np.median(vm[ia]) - np.median(vm[ib])
        p_emp = float((np.abs(nulls) >= abs(obs)).mean())
        p_emp_mean = float((np.abs(nulls_mean) >= abs(obs_mean)).mean())
        null_rows.append({
            "contrast": cname,
            "observed_delta_median": round(obs, 4),
            "null_sd": round(float(nulls.std()), 4),
            "null_q025": round(float(np.percentile(nulls, 2.5)), 4),
            "null_q975": round(float(np.percentile(nulls, 97.5)), 4),
            "p_empirical": p_emp,
            "observed_mean_z_delta": round(obs_mean, 4),
            "p_empirical_unweighted": p_emp_mean})
        log(f"  {cname}: obs={obs:+.4f} nullSD={nulls.std():.3f} p_emp={p_emp:.4f} | "
            f"mean-z obs={obs_mean:+.4f} p_emp={p_emp_mean:.4f}")
    null_df = pd.DataFrame(null_rows)
    null_df.to_csv(os.path.join(OUT, "permutation_null.csv"), index=False)

    # ------------------------------------------------- OLS: does the group effect survive adjustment?
    log("\n[OLS] Metastatic vs Primary, adjusted for prep / proliferation (tumour samples only)")
    tum = res[~nrm].copy()
    y = tum["REP30"].values
    grp = (tum["group"] == "Metastatic").astype(float).values
    is_ffpe = (tum["prep"] == "FFPE").astype(float).values
    pro = tum["prolif"].values
    models = {
        "M1 group only": [grp],
        "M2 group + prep": [grp, is_ffpe],
        "M3 group + prep + proliferation": [grp, is_ffpe, pro],
    }
    term_names = {0: "Metastatic vs Primary", 1: "prep FFPE vs FreshFrozen", 2: "proliferation"}
    ols_rows = []
    for name, cols in models.items():
        X = np.column_stack([np.ones(len(y))] + cols)
        beta, se, pv = ols(X, y)
        for j in range(len(cols)):
            ols_rows.append({"model": name, "term": term_names[j],
                             "beta": round(float(beta[j + 1]), 4),
                             "se": round(float(se[j + 1]), 4),
                             "t": round(float(beta[j + 1] / se[j + 1]), 3),
                             "p": float(pv[j + 1]), "n": len(y)})
    ols_df = pd.DataFrame(ols_rows)
    ols_df.to_csv(os.path.join(OUT, "rep30_ols_models.csv"), index=False)
    log(ols_df.to_string(index=False))

    # ------------------------------------------------- per-gene: is the met signal a prep artefact?
    gset = [g for g in p30 if g in Z.index]
    m_ffpe = (prim & (res["trt"] == "Pre-treatment") & (res["prep"] == "FFPE")).values
    m_ff = (prim & (res["trt"] == "Pre-treatment") & (res["prep"] == "Fresh frozen")).values
    Zg = Z.loc[gset].values
    d_prep = Zg[:, m_ffpe].mean(1) - Zg[:, m_ff].mean(1)   # prep bias per gene
    d_met = Zg[:, met.values].mean(1) - Zg[:, prim.values].mean(1)
    r_prep = float(np.corrcoef(d_prep, d_met)[0, 1])
    log(f"\n[prep-bias] corr(per-gene prep bias, per-gene met-vs-primary delta) = {r_prep:.3f}")
    pd.DataFrame({"gene": gset, "prep_bias_FFPE_minus_FF": d_prep,
                  "delta_met_minus_primary": d_met}).to_csv(
        os.path.join(OUT, "per_gene_prep_bias_vs_met_delta.csv"), index=False)


    # ------------------------------------------------- per-gene deltas
    def per_gene(mA, mB, tag):
        a = Z.loc[[g for g in p30 if g in Z.index], mA.values]
        b = Z.loc[[g for g in p30 if g in Z.index], mB.values]
        d = pd.DataFrame({
            "gene": a.index,
            "mean_A": a.mean(axis=1).values,
            "mean_B": b.mean(axis=1).values,
        })
        d["delta_z"] = d["mean_A"] - d["mean_B"]
        d["coef"] = [p30[g] for g in d["gene"]]
        d = d.sort_values("delta_z", ascending=False)
        d.to_csv(os.path.join(OUT, f"per_gene_delta_{tag}.csv"), index=False)
        return d
    d_trt = per_gene(prim & (res["trt"] == "Post-treatment"),
                     prim & (res["trt"] == "Pre-treatment"), "treatment")
    d_met = per_gene(met, prim, "metastasis")
    log("\n[per-gene] treatment top5 up:", list(d_trt.head(5)["gene"]))
    log("[per-gene] treatment top5 down:", list(d_trt.tail(5)["gene"]))

    # ------------------------------------------------- plots
    def box(ax, groups, title, ylab="REP30 score"):
        labels = [l for l, _ in groups]
        data = [res.loc[m, "REP30"].values for _, m in groups]
        bp = ax.boxplot(data, showfliers=False, widths=0.55, patch_artist=True)
        for patch, c in zip(bp["boxes"], ["#4C72B0", "#C44E52", "#55A868", "#8172B2"]):
            patch.set_facecolor(c); patch.set_alpha(0.55)
        for i, d in enumerate(data):
            x = np.random.default_rng(1).normal(i + 1, 0.06, len(d))
            ax.scatter(x, d, s=7, alpha=0.55, color="#333333", zorder=3)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.axhline(0, ls=":", c="grey", lw=0.8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    box(axes[0], [("Pre-treat\nPrimary", prim & (res["trt"] == "Pre-treatment")),
                  ("Post-treat\nPrimary", prim & (res["trt"] == "Post-treatment"))],
        "A. Treatment exposure (primary only)")
    box(axes[1], [("Primary", prim), ("Metastatic", met), ("Normal", nrm)],
        "B. Metastasis")
    axes[2].scatter(res["prolif"], res["REP30"], s=16, c="#4C72B0", alpha=0.7)
    r = np.corrcoef(res["prolif"], res["REP30"])[0, 1]
    axes[2].set_xlabel("proliferation score (12-gene mean z)")
    axes[2].set_ylabel("REP30")
    axes[2].set_title(f"C. REP30 vs proliferation (r={r:.2f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_rep30_aurora.png"), dpi=150)
    plt.close(fig)

    # paired plot
    if len(piv) >= 5:
        fig, ax = plt.subplots(figsize=(4.2, 4.6))
        for pid, row in piv.iterrows():
            ax.plot([0, 1], [row["Primary"], row["Metastatic"]],
                    marker="o", ms=4, lw=1, alpha=0.7)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Primary", "Metastasis"])
        ax.set_ylabel("REP30 score")
        ax.set_title(f"Paired, n={len(piv)} patients", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_rep30_paired_primary_met.png"), dpi=150)
        plt.close(fig)

    # ------------------------------------------------- summary
    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("REP30 / REP52 in AURORA US (GSE209998)\n")
        f.write("=" * 60 + "\n")
        f.write(f"samples: {len(meta)}  (Primary {int(prim.sum())}, "
                f"Metastatic {int(met.sum())}, Normal {int(nrm.sum())})\n")
        f.write("Primary split: Pre-treatment %d / Post-treatment %d\n" % (
            int((prim & (res["trt"] == "Pre-treatment")).sum()),
            int((prim & (res["trt"] == "Post-treatment")).sum())))
        f.write(f"REP30 genes scored: {len(g30)}/30 ; REP52: {len(g52)}/52\n")
        f.write(f"corr(REP30, proliferation score) = {r:.3f}\n")
        f.write(f"corr(REP30, MKI67 z)             = {np.corrcoef(res['MKI67'], res['REP30'])[0,1]:.3f}\n")
        f.write(f"corr(REP30, REP52)               = {np.corrcoef(res['REP52'], res['REP30'])[0,1]:.3f}\n\n")
        f.write("Confounder check - primary samples, treatment x prep:\n")
        f.write(xt.to_string() + "\n\n")
        f.write("Confounder check - tumour samples, group x prep:\n")
        f.write(xg.to_string() + "\n\n")
        f.write("Comparisons\n" + comp.to_string(index=False) + "\n\n")
        f.write("Paired\n" + (pair_df.to_string(index=False) if len(pair_df) else "n/a") + "\n\n")
        f.write("Permutation null\n" + null_df.to_string(index=False) + "\n\n")
        f.write("OLS: Metastatic vs Primary adjusted for prep / proliferation\n")
        f.write(ols_df.to_string(index=False) + "\n\n")
        f.write(f"corr(per-gene prep bias, per-gene met-vs-primary delta) = {r_prep:.3f}\n")
    log("\n[done] -> " + OUT)


if __name__ == "__main__":
    main()
    LOG.close()
