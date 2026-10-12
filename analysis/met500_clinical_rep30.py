# -*- coding: utf-8 -*-
"""
MET500 -- does REP30 capture a drug-resistant state, now that the cohort HAS
clinical annotation and per-case genomic alterations?

Why this script exists
----------------------
Earlier MET500 work (analysis/met500_rep30.py) could only show that REP30 partly
tracks proliferation, because MET500 shipped no clinical file. That gap is now
closed with two annotations recovered from the MET500 paper supplement
(Robinson et al. 2017, Nature 548:297, doi:10.1038/nature23306):

  * data/06_metastatic/MET500/MET500_clinical_TableS1.tsv
      Supplementary Table 1 -- Demographics and Clinical Details, 500 cases:
      age / gender / population / cancer type / oncotree code / biopsy site /
      analysis cohort / primary site / interval to biopsy (mo) / PRIOR THERAPY
  * data/06_metastatic/MET500/MET500_SupTable3_alterations.tsv
      Supplementary Table 3 -- Recurrent molecular alterations, 486 cases x 2938
      genes (variant_class: twohit / deleterious / missense / amplification /
      deletion) -> gives the acquired-resistance genomic labels ESR1, RB1, ERBB2 ...

Linkage
-------
expression sample  --(meta.plus sample_source)-->  case_id (MO_/TP_/ES_)
                                                    |
                              clinical_TableS1 <----+----> SupTable3 alterations

Tests (BRCA-primary focus, pan-cohort as secondary)
---------------------------------------------------
 1. REP30 vs ESR1 mutation (the canonical acquired endocrine-resistance label)
 2. REP30 vs RB1 loss (the canonical acquired CDK4/6-resistance label)
 3. REP30 vs ERBB2 amplification, PIK3CA mutation (control labels)
 4. REP30 vs prior therapy exposure                (very unbalanced, caveated)
 5. REP30 vs proliferation (MKI67 module) and the partial correlation after
    removing proliferation -> the key confound test
 6. confound screens: tumour content, number of libraries per case
 7. REP30 by metastatic biopsy site within BRCA

Outputs: results/metastatic/MET500_clinical/
"""
import os
import gzip
import sys
import numpy as np
import pandas as pd
import openpyxl
from scipy import stats

ROOT = r"E:/01.WorkSpace/01.Breast_cancer_gene_panel/resistance_signature"
DAT = os.path.join(ROOT, "data/06_metastatic/MET500")
MX = os.path.join(DAT, "MET500_geneExpression_M.mx.log2.txt.txt")   # gzip, log2(fpkm+0.001)
META = os.path.join(DAT, "MET500_geneExpression_M.meta.plus.txt")
PMAP = os.path.join(DAT, "probeMap_gencode.v23.annotation.gene.probemap")
CLIN = os.path.join(DAT, "MET500_clinical_TableS1.tsv")
SUP3 = r"C:\Users\mao02\AppData\Local\Temp\met500_supp\MOESM3_23306.xlsx"
SUP3_TSV = os.path.join(DAT, "MET500_SupTable3_alterations.tsv")
PANEL30 = os.path.join(ROOT, "results/rep52_trim/rep52_trimmed_panel.csv")
OUT = os.path.join(ROOT, "results/metastatic/MET500_clinical")
os.makedirs(OUT, exist_ok=True)

PROLIF = ["MKI67", "PCNA", "TOP2A", "CCNB1", "BUB1", "AURKA", "PLK1", "MCM2",
          "RRM2", "TYMS", "CDC20", "BIRC5"]

ALIAS = {"MO": "MO", "TP": "TP", "ES": "ES"}

_log = []
def log(s=""):
    print(s)
    _log.append(str(s))


# ----------------------------------------------------------------- statistics
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
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return u, p


def partial_spearman(x, y, z):
    """Spearman of x,y after removing the rank-linear effect of z (z may be 1-D or 2-D)."""
    x, y = stats.rankdata(x), stats.rankdata(y)
    Z = np.asarray(z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    Z = np.column_stack([stats.rankdata(Z[:, j]) for j in range(Z.shape[1])])
    def resid(v, c):
        c = np.column_stack([np.ones(len(c)), c])
        beta, *_ = np.linalg.lstsq(c, v, rcond=None)
        return v - c @ beta
    rx, ry = resid(x, Z), resid(y, Z)
    r, p = stats.pearsonr(rx, ry)
    return r, p


def mwu_resid(score, flag, covars):
    """residualise score on covars (OLS) then Mann-Whitney on the residual."""
    y = np.asarray(score, float)
    C = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in covars])
    beta, *_ = np.linalg.lstsq(C, y, rcond=None)
    r = y - C @ beta
    f = np.asarray(flag, bool)
    if f.sum() < 3 or (~f).sum() < 3:
        return np.nan, np.nan, np.nan
    u, p = stats.mannwhitneyu(r[f], r[~f], alternative="two-sided")
    return cliffs_delta(r[f], r[~f]), p, r


# ----------------------------------------------------------------- load
def load_panel():
    p = pd.read_csv(PANEL30)
    p["gene"] = p["gene"].astype(str).str.strip()
    return dict(zip(p["gene"], p["coef"]))


def load_probemap():
    return pd.read_csv(PMAP, sep="\t")


def ensure_sup3():
    if os.path.exists(SUP3_TSV):
        return
    wb = openpyxl.load_workbook(SUP3, read_only=True, data_only=True)
    ws = wb["Supplementary Table 3"]
    rows = list(ws.iter_rows(values_only=True))
    data = [r for r in rows[3:] if r and r[0]]
    with open(SUP3_TSV, "w", encoding="utf-8", newline="") as f:
        f.write("case_id\tgene\tvariant_class\tinactivating\tclass\n")
        for r in data:
            f.write("\t".join(str(x) for x in r[:5]) + "\n")
    wb.close()
    log(f"wrote {SUP3_TSV}")


def load_expr(genes_needed, id2sym):
    """stream the gzipped log2 matrix, keep only the genes we need"""
    want = {}
    for g in genes_needed:
        want[g] = g
    sym2rows = {g: [] for g in genes_needed}
    with gzip.open(MX, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        samples = header[1:]
        for line in f:
            parts = line.rstrip("\n").split("\t")
            eid = parts[0]
            sym = id2sym.get(eid) or id2sym.get(eid.split(".")[0])
            if sym in sym2rows:
                sym2rows[sym].append(np.array(parts[1:], dtype=float))
    mat = {}
    missing = []
    for g in genes_needed:
        rows = sym2rows[g]
        if not rows:
            missing.append(g)
            continue
        mat[g] = np.nanmean(np.vstack(rows), axis=0)
    return samples, mat, missing


def main():
    panel = load_panel()
    log(f"REP30 panel genes: {len(panel)}")

    pmap = load_probemap()
    id2sym_ver = dict(zip(pmap["id"].astype(str), pmap["gene"].astype(str)))
    id2sym_nv = {k.split(".")[0]: v for k, v in id2sym_ver.items()}
    id2sym = dict(id2sym_nv); id2sym.update(id2sym_ver)

    genes_needed = sorted(set(panel) | set(PROLIF))
    samples, mat, missing = load_expr(genes_needed, id2sym)
    found = [g for g in genes_needed if g in mat]
    log(f"expression samples: {len(samples)}   genes found: {len(found)}/{len(genes_needed)}")
    if missing:
        log(f"  MISSING genes: {missing}")

    X = pd.DataFrame(mat, index=samples)

    # z-score across samples, then REP30 = sum(coef * z)
    Z = (X - X.mean()) / X.std(ddof=0)
    rep_genes = [g for g in panel if g in Z.columns]
    rep30 = sum(panel[g] * Z[g] for g in rep_genes)
    prolif_genes = [g for g in PROLIF if g in Z.columns]
    prolif = Z[prolif_genes].mean(axis=1)
    df = pd.DataFrame({"rep30": rep30, "prolif": prolif, "MKI67": Z.get("MKI67")})

    # ------------------------------------------------------------- annotation
    meta = pd.read_csv(META, sep="\t")
    meta = meta.rename(columns={"Sample_id": "sample"})
    clin = pd.read_csv(CLIN, sep="\t")
    ensure_sup3()
    alt = pd.read_csv(SUP3_TSV, sep="\t")

    df = df.join(meta.set_index("sample")[["sample_source", "cohort", "biopsy_tissue",
                                           "tc", "sample_type"]], how="left")
    df = df.rename(columns={"sample_source": "case_id"})
    df = df.join(clin.set_index("case_id")[["age", "gender", "cancer_type", "oncotree_code",
                                            "biopsy_site", "analysis_cohort", "primary_site",
                                            "interval_mo", "prior_therapy"]], on="case_id")
    log(f"joined clinical: {df['analysis_cohort'].notna().sum()}/{len(df)} samples")
    df["bf"] = df["cohort"].fillna(df["analysis_cohort"])

    # case-level alteration flags
    alt["case_id"] = alt["case_id"].astype(str)
    def flag(gene, vc):
        return set(alt.loc[(alt["gene"] == gene) & (alt["variant_class"].isin(vc)), "case_id"])
    flags = {
        "ESR1_mut": flag("ESR1", ["missense", "deleterious", "twohit"]),
        "RB1_loss": flag("RB1", ["twohit", "deletion", "deleterious"]),
        "ERBB2_amp": flag("ERBB2", ["amplification"]),
        "PIK3CA_mut": flag("PIK3CA", ["missense", "deleterious", "twohit"]),
        "TP53_any": flag("TP53", ["missense", "deleterious", "twohit", "deletion", "amplification"]),
        "BRCA1_mut": flag("BRCA1", ["missense", "deleterious", "twohit"]),
    }
    for k, v in flags.items():
        df[k] = df["case_id"].isin(v)
    log("alteration flag carriers (samples): " + str({k: int(df[k].sum()) for k in flags}))

    df["rep30_hi"] = df["rep30"] > df["rep30"].median()
    df.to_csv(os.path.join(OUT, "met500_clinical_linked.csv"), index=False)

    # ------------------------------------------------------------- tests
    rows = []
    def add(scope, test, n, stat, p, extra=""):
        rows.append(dict(scope=scope, test=test, n=n, statistic=stat, p=p, detail=extra))

    def group_test(scope, sub, label_col, name, score="rep30"):
        a = sub.loc[sub[label_col], score].values
        b = sub.loc[~sub[label_col], score].values
        if len(a) < 3 or len(b) < 3:
            add(scope, name, f"{len(a)}/{len(b)}", np.nan, np.nan, "too few")
            return
        u, p = mwu(a, b)
        d = cliffs_delta(a, b)
        add(scope, name, f"{len(a)}/{len(b)}", round(d, 3), p,
            f"median {np.median(a):.3f} vs {np.median(b):.3f}")
        # proliferation-adjusted
        pr, pp = partial_spearman(sub[score].values, sub[label_col].astype(float).values,
                                  sub["prolif"].values)
        add(scope, name + " [prolif-adjusted]", f"{len(a)}/{len(b)}", round(pr, 3), pp, "partial spearman")
        # proliferation + tumour-content adjusted (the two confounds identified here)
        tc = pd.to_numeric(sub["tc"], errors="coerce")
        ok = tc.notna()
        d3, p3, _ = mwu_resid(sub.loc[ok, score].values, sub.loc[ok, label_col].values,
                              [sub.loc[ok, "prolif"].values, tc[ok].values])
        add(scope, name + " [prolif+tc-adjusted MWU]",
            f"{int(sub.loc[ok, label_col].sum())}/{int((~sub.loc[ok, label_col]).sum())}",
            (round(d3, 3) if d3 == d3 else np.nan), p3, "OLS residual, Cliff's delta")

    scopes = {
        "all-cases": df,
        "BRCA": df[df["bf"] == "BRCA"],
        "BRCA(tc>=0.4)": df[(df["bf"] == "BRCA") & (pd.to_numeric(df["tc"], errors="coerce") >= 0.4)],
    }
    for sname, sub in scopes.items():
        if len(sub) < 10:
            log(f"skip scope {sname} (n={len(sub)})")
            continue
        log(f"\n--- scope {sname}: n={len(sub)} samples / {sub['case_id'].nunique()} cases")
        for lab, nm in [("ESR1_mut", "REP30 vs ESR1 mutation"),
                        ("RB1_loss", "REP30 vs RB1 loss"),
                        ("ERBB2_amp", "REP30 vs ERBB2 amplification"),
                        ("PIK3CA_mut", "REP30 vs PIK3CA mutation"),
                        ("TP53_any", "REP30 vs TP53 alteration")]:
            group_test(sname, sub, lab, nm)

        r, p = stats.spearmanr(sub["rep30"], sub["prolif"])
        add(sname, "REP30 vs proliferation (Spearman)", len(sub), round(r, 3), p)
        r2, p2 = stats.spearmanr(sub["rep30"], sub["MKI67"].astype(float))
        add(sname, "REP30 vs MKI67 (Spearman)", len(sub), round(r2, 3), p2)
        # prior therapy
        a = sub.loc[sub["prior_therapy"] == "Y", "rep30"]
        b = sub.loc[sub["prior_therapy"] == "N", "rep30"]
        if len(b) >= 3:
            u, pp = mwu(a, b)
            add(sname, "REP30 vs prior therapy (Y vs N)", f"{len(a)}/{len(b)}",
                round(cliffs_delta(a, b), 3), pp)
        # confounds
        for c in ["tc", "age", "interval_mo"]:
            if c in sub:
                v = pd.to_numeric(sub[c], errors="coerce")
                ok = v.notna()
                if ok.sum() > 10:
                    rr, pv = stats.spearmanr(sub.loc[ok, "rep30"], v[ok])
                    add(sname, f"REP30 vs {c} (Spearman, confound screen)", int(ok.sum()),
                        round(rr, 3), pv)

        # REP30 across metastatic biopsy sites (the original "does the site matter" question)
        bs = sub["biopsy_site"].fillna("NA")
        keep = [b for b, n in bs.value_counts().items() if n >= 5]
        grps = [sub.loc[bs == b, "rep30"].values for b in keep]
        if len(grps) >= 3:
            h, ph = stats.kruskal(*grps)
            add(sname, "REP30 across biopsy sites (Kruskal-Wallis)", len(sub),
                round(h, 2), ph, "; ".join(f"{b}={len(g)}" for b, g in zip(keep, grps)))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "resistance_label_tests.csv"), index=False)
    log("\n" + res.to_string(index=False))

    # ------------------------------------------------------------- figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        brca = df[df["bf"] == "BRCA"]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, lab, nm in [(axes[0], "ESR1_mut", "ESR1 mutation"),
                            (axes[1], "RB1_loss", "RB1 loss"),
                            (axes[2], "ERBB2_amp", "ERBB2 amplification")]:
            grp = [brca.loc[brca[lab], "rep30"].values, brca.loc[~brca[lab], "rep30"].values]
            try:
                ax.boxplot(grp, tick_labels=[f"{lab}+", "WT"], showfliers=False)
            except TypeError:
                ax.boxplot(grp, labels=[f"{lab}+", "WT"], showfliers=False)
            for i, g in enumerate(grp):
                ax.scatter(np.random.normal(i + 1, 0.05, len(g)), g, s=10, alpha=0.6)
            ax.set_title(f"BRCA mets: {nm}\nn={len(grp[0])} vs {len(grp[1])}")
            ax.set_ylabel("REP30 score")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "brca_resistance_labels.png"), dpi=160)
        plt.close()

        plt.figure(figsize=(5.5, 5))
        plt.scatter(brca["prolif"], brca["rep30"], s=16, alpha=0.7)
        plt.xlabel("proliferation (mean z of 12 genes)")
        plt.ylabel("REP30 score")
        rr, pp = stats.spearmanr(brca["prolif"], brca["rep30"])
        plt.title(f"MET500 BRCA mets (n={len(brca)})\nSpearman r={rr:.3f}, p={pp:.2e}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "brca_rep30_vs_proliferation.png"), dpi=160)
        plt.close()
        log("figures written")
    except Exception as e:
        log(f"figure step skipped: {type(e).__name__}: {e}")

    with open(os.path.join(OUT, "run_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(_log))


if __name__ == "__main__":
    main()
