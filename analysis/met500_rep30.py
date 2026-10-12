"""
MET500 pan-cancer metastatic cohort -- is REP30 a pan-proliferation program?

Context
-------
MET500 (Robinson et al. 2017, Nature; Michigan) = 868 metastatic tumour RNA-seq
samples from 496 patients across 23 tissues. Two properties make it the cleanest
available test bed for the REP30 question:

  * ALL samples are metastases, processed with ONE uniform protocol (TCGA RNA-seq
    pipeline) -> no FFPE-vs-fresh-frozen composition confound of the kind that
    destroyed the AURORA "metastasis raises REP30" signal.
  * Pan-cancer coverage -> lets us ask whether REP30 simply tracks proliferation
    across many lineages, rather than marking a resistance state.

What we test
------------
  1. Pan-cancer corr(REP30, proliferation) / corr(REP30, MKI67); coherence ceiling.
  2. Variance decomposition: proliferation vs lineage (eta^2 of tissue).
  3. Empirical permutation null: 2000 random 30-gene programs, same statistic.
  4. REP30 by tissue / metastatic biopsy site; within-tissue corr with proliferation.
  5. Per-gene: which panel genes carry the signal (and cancel each other).
  6. Secondary, caveated: MET500 breast mets vs TCGA-BRCA primaries (dataset batch
     is collinear with group -> descriptive only).

Inputs   : data/06_metastatic/MET500/{mx.txt.gz, meta.plus.txt, probemap}
Panels   : results/rep52_trim/rep52_trimmed_panel.csv (REP30), results/rep_genes.csv (REP52)
Outputs  : results/metastatic/MET500_rep30/
"""
import os, gzip
import numpy as np
import pandas as pd
from scipy import stats

ROOT = r"E:/01.WorkSpace/01.Breast_cancer_gene_panel/resistance_signature"
DAT = os.path.join(ROOT, "data/06_metastatic/MET500")
MX = os.path.join(DAT, "MET500_geneExpression_M.mx.txt.gz")
META = os.path.join(DAT, "MET500_geneExpression_M.meta.plus.txt")
PMAP = os.path.join(DAT, "probeMap_gencode.v23.annotation.gene.probemap")
PANEL30 = os.path.join(ROOT, "results/rep52_trim/rep52_trimmed_panel.csv")
PANEL52 = os.path.join(ROOT, "results/rep_genes.csv")
TCGA_MX = os.path.join(
    ROOT, "data/01_core/TCGA-BRCA/brca_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem.txt")
OUT = os.path.join(ROOT, "results/metastatic/MET500_rep30")
os.makedirs(OUT, exist_ok=True)

PROLIF = ["MKI67", "PCNA", "TOP2A", "CCNB1", "BUB1", "AURKA", "PLK1", "MCM2",
          "RRM2", "TYMS", "CDC20", "BIRC5"]
rng = np.random.default_rng(0)

_log = []
def log(s=""):
    print(s)
    _log.append(str(s))

def cliffs_delta(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return np.nan
    a = a[:, None]; b = b[None, :]
    return ((a > b).sum() - (a < b).sum()) / (na * nb)

def main():
    # ------------------------------------------------------------------ panels
    p30 = pd.read_csv(PANEL30); p30["gene"] = p30["gene"].astype(str).str.strip()
    p52 = pd.read_csv(PANEL52); p52["gene"] = p52["gene"].astype(str).str.strip()
    coef30 = dict(zip(p30["gene"], p30["coef"]))
    coef52 = dict(zip(p52["gene"], p52["coef"]))
    panel_set = set(p30["gene"]) | set(p52["gene"]) | set(PROLIF) | {"MKI67"}
    log(f"[panel] REP30 n={len(p30)}  REP52 n={len(p52)}")

    # ------------------------------------------------------------- probemap
    ens2sym = {}
    with open(PMAP, "rt", encoding="utf-8", errors="replace") as fh:
        next(fh)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                ens2sym[parts[0]] = parts[1]
    log(f"[probemap] ens2sym entries = {len(ens2sym)}")

    # -------------------------------------------- expression: ALL gene symbols
    # (whole matrix is needed so the permutation null can draw random gene sets)
    allv, dup = {}, {}
    with gzip.open(MX, "rt", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        sample_ids = header[1:]
        n = len(sample_ids)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            ens = parts[0]
            sym = ens2sym.get(ens) or ens2sym.get(ens.split(".")[0])
            if sym is None:
                continue
            try:
                vals = np.array(parts[1:], dtype=float)
            except ValueError:
                continue
            if len(vals) != n:
                continue
            if sym in allv:
                dup.setdefault(sym, []).append(vals)
            else:
                allv[sym] = vals
    for sym, lst in dup.items():
        allv[sym] = np.mean(np.vstack([allv[sym]] + lst), axis=0)
    log(f"[expr] samples = {len(sample_ids)}  gene symbols retained = {len(allv)}")

    Eall = pd.DataFrame(allv, index=sample_ids).T        # genes x samples
    q = np.nanpercentile(Eall.values, [50, 99, 99.9])
    log(f"[expr] value quantiles p50/p99/p99.9 = {q.round(3)}")
    # .mx.txt.gz holds LINEAR FPKM (the companion .log2 file is corrupt);
    # put it on the log2 scale the panel coefficients were built on.
    if q[2] > 100:
        Eall = np.log2(Eall + 0.001)
        q2 = np.nanpercentile(Eall.values, [50, 99, 99.9])
        log(f"[expr] linear FPKM -> log2(FPKM+0.001)  new p50/p99/p99.9 = {q2.round(3)}")
    else:
        log("[expr] already log2 scale -> no transform")

    log(f"[cover] REP30 missing={[g for g in p30['gene'] if g not in Eall.index]}  "
        f"REP52 missing={[g for g in p52['gene'] if g not in Eall.index]}  "
        f"prolif missing={[g for g in PROLIF if g not in Eall.index]}")

    # --------------------------------------------------------------- metadata
    meta = pd.read_csv(META, sep="\t", dtype=str).set_index("Sample_id")
    meta = meta.reindex(Eall.columns)
    log(f"[meta] aligned {meta['tissue'].notna().sum()}/{len(meta)} samples")

    # --------------------------------------------------------------- scoring
    Z = Eall.sub(Eall.mean(axis=1), axis=0).div(
        Eall.std(axis=1).replace(0, np.nan), axis=0).dropna(how="all")

    g30 = [g for g in p30["gene"] if g in Z.index]
    g52 = [g for g in p52["gene"] if g in Z.index]
    gp = [g for g in PROLIF if g in Z.index]
    w30 = np.array([coef30[g] for g in g30])
    w52 = np.array([coef52[g] for g in g52])

    s30 = (Z.loc[g30].T.values * w30).sum(axis=1)
    s52 = (Z.loc[g52].T.values * w52).sum(axis=1)
    prolif = Z.loc[gp].mean(axis=0).values
    mki67 = Z.loc["MKI67"].values if "MKI67" in Z.index else np.full(Z.shape[1], np.nan)

    res = pd.DataFrame({"REP30": s30, "REP52": s52, "prolif": prolif, "MKI67": mki67},
                       index=Z.columns).join(meta)
    log(f"[score] REP30 genes used = {len(g30)}/30  REP52 = {len(g52)}/52  "
        f"prolif = {len(gp)}/{len(PROLIF)}")

    # --------------------------------------------------- 1. global correlation
    r_p = stats.pearsonr(res["REP30"], res["prolif"])
    rho_p = stats.spearmanr(res["REP30"], res["prolif"])
    r_m = stats.pearsonr(res["REP30"], res["MKI67"])
    r_ref = stats.pearsonr(res["MKI67"], res["prolif"])   # coherence ceiling
    beta = np.polyfit(res["prolif"].values, res["REP30"].values, 1)
    res["REP30_resid"] = res["REP30"].values - np.polyval(beta, res["prolif"].values)
    res.to_csv(os.path.join(OUT, "rep30_per_sample.csv"), encoding="utf-8")

    corr_df = pd.DataFrame([
        {"pair": "REP30 vs proliferation(12-gene)", "pearson_r": round(r_p[0], 4),
         "p": r_p[1], "spearman_rho": round(rho_p[0], 4)},
        {"pair": "REP30 vs MKI67", "pearson_r": round(r_m[0], 4), "p": r_m[1],
         "spearman_rho": round(stats.spearmanr(res["REP30"], res["MKI67"])[0], 4)},
        {"pair": "REP52 vs proliferation(12-gene)",
         "pearson_r": round(stats.pearsonr(res["REP52"], res["prolif"])[0], 4),
         "p": stats.pearsonr(res["REP52"], res["prolif"])[1],
         "spearman_rho": round(stats.spearmanr(res["REP52"], res["prolif"])[0], 4)},
        {"pair": "[ceiling] MKI67 vs proliferation(12-gene)",
         "pearson_r": round(r_ref[0], 4), "p": r_ref[1],
         "spearman_rho": round(stats.spearmanr(res["MKI67"], res["prolif"])[0], 4)},
    ])
    corr_df.to_csv(os.path.join(OUT, "correlations.csv"), index=False)
    log("\n[1] pan-cancer correlations (n=%d)" % len(res))
    log(corr_df.to_string(index=False))
    log(f"    R^2(REP30~prolif) = {r_p[0]**2:.3f}   "
        f"R^2(MKI67~prolif) = {r_ref[0]**2:.3f}")

    # ------------------------------ 2. variance decomposition (prolif vs lineage)
    grand = res["REP30"].mean()
    ss_tot = float(((res["REP30"] - grand) ** 2).sum())
    ss_b = float(sum(len(s) * (s["REP30"].mean() - grand) ** 2
                     for _, s in res.groupby("tissue")))
    eta2_tissue = ss_b / ss_tot
    kw = stats.kruskal(*[s["REP30"].values for _, s in res.groupby("tissue")
                         if len(s) >= 8])
    log(f"\n[2] variance decomposition of REP30")
    log(f"    R^2 by proliferation          = {r_p[0]**2:.3f}")
    log(f"    eta^2 by tissue (lineage)     = {eta2_tissue:.3f} "
        f"(one-way ANOVA over 23 tissues)")
    log(f"    Kruskal-Wallis across tissues = H={kw.statistic:.2f} p={kw.pvalue:.2e}")

    # ------------------------------------------- 3. permutation null (30 genes)
    cand = [g for g in Z.index if g not in panel_set]
    log(f"\n[3] permutation null: 2000 random 30-gene programs (candidate pool {len(cand)})")
    null = np.empty(2000)
    for k in range(2000):
        pick = rng.choice(cand, size=30, replace=False)
        sc = Z.loc[pick].mean(axis=0).values
        null[k] = stats.pearsonr(sc, res["prolif"].values)[0]
    p_emp = float((np.abs(null) >= abs(r_p[0])).mean())
    log(f"    null r: mean={null.mean():+.4f} sd={null.std():.4f} "
        f"|r|_q95={np.percentile(np.abs(null), 95):.4f} "
        f"|r|_q99={np.percentile(np.abs(null), 99):.4f}")
    log(f"    REP30 r = {r_p[0]:+.4f} -> empirical p(|null|>=obs) = {p_emp:.4f}")
    pd.DataFrame({"null_r": null}).to_csv(
        os.path.join(OUT, "permutation_null.csv"), index=False)

    # ------------------------------------- 4. REP30 by tissue / biopsy site
    def group_summary(col, min_n=8):
        rows = []
        for k, sub in res.groupby(col):
            if len(sub) < min_n:
                continue
            rows.append({col: k, "n": len(sub),
                         "median_REP30": round(float(sub["REP30"].median()), 4),
                         "median_prolif": round(float(sub["prolif"].median()), 4),
                         "median_REP30_resid": round(float(sub["REP30_resid"].median()), 4)})
        return pd.DataFrame(rows).sort_values("median_REP30", ascending=False)

    tis = group_summary("tissue", 8)
    site = group_summary("biopsy_tissue", 8)
    tis.to_csv(os.path.join(OUT, "rep30_by_tissue.csv"), index=False)
    site.to_csv(os.path.join(OUT, "rep30_by_site.csv"), index=False)
    log("\n[4a] REP30 by primary tissue (>=8)")
    log(tis.to_string(index=False))
    log("\n[4b] REP30 by metastatic biopsy site (>=8)")
    log(site.to_string(index=False))

    # within-tissue correlation with proliferation
    wrows = []
    for t, sub in res.groupby("tissue"):
        if len(sub) >= 20:
            r = stats.pearsonr(sub["REP30"], sub["prolif"])
            wrows.append({"tissue": t, "n": len(sub),
                          "corr_within": round(r[0], 4), "p": r[1]})
    wt = pd.DataFrame(wrows).sort_values("corr_within", ascending=False)
    wt.to_csv(os.path.join(OUT, "within_tissue_corr.csv"), index=False)
    log("\n[4c] within-tissue corr(REP30, proliferation), n>=20")
    log(wt.to_string(index=False))

    # --------------------------------------------- 5. BRCA subset specifics
    brca = res[res["tissue"] == "breast"]
    log("\n[5] BRCA metastatic subset n=%d" % len(brca))
    if len(brca) >= 10:
        rb = stats.pearsonr(brca["REP30"], brca["prolif"])
        log(f"    corr(REP30, prolif) within BRCA = {rb[0]:.3f} (p={rb[1]:.2e})")
        log(f"    BRCA median REP30 = {brca['REP30'].median():.4f} "
            f"(all-cancer {res['REP30'].median():.4f})")

    # ---------------------------------- 6. per-gene association with prolif
    rows = []
    for g in g30:
        r = stats.pearsonr(Z.loc[g].values, res["prolif"].values)
        rows.append({"gene": g, "coef": coef30[g],
                     "corr_with_prolif": round(r[0], 4), "p": r[1]})
    pg = pd.DataFrame(rows).sort_values("corr_with_prolif", ascending=False)
    pg.to_csv(os.path.join(OUT, "per_gene_vs_proliferation.csv"), index=False)
    log("\n[6] per-gene correlation with proliferation (REP30 panel)")
    log(pg.to_string(index=False))

    # ------------------------------- 7. secondary: met(MET500) vs primary(TCGA)
    log("\n[7] secondary: MET500 breast mets vs TCGA-BRCA primaries (CAVEATED)")
    if os.path.exists(TCGA_MX):
        need = set(g30) | set(gp) | {"MKI67"}
        rec = {}
        with open(TCGA_MX, "rt", encoding="utf-8", errors="replace") as fh:
            head = fh.readline().rstrip("\n").split("\t")
            tcols = head[1:]
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                sym = parts[0].strip()
                if sym not in need:
                    continue
                try:
                    vals = np.array(parts[1:], dtype=float)
                except ValueError:
                    continue
                if len(vals) != len(tcols):
                    continue
                rec.setdefault(sym, []).append(vals)
        tE = pd.DataFrame({s: np.mean(np.vstack(v), axis=0) for s, v in rec.items()},
                          index=tcols).T
        shared = [g for g in tE.index if g in Eall.index]
        M = pd.concat([Eall.loc[shared], tE.loc[shared]], axis=1)
        Mz = M.sub(M.mean(axis=1), axis=0).div(
            M.std(axis=1).replace(0, np.nan), axis=0).dropna()
        gg = [g for g in g30 if g in Mz.index]
        ww = np.array([coef30[g] for g in gg])
        sc = (Mz.loc[gg].T.values * ww).sum(axis=1)
        d = pd.DataFrame({"score": sc}, index=Mz.columns)
        d["cohort"] = ["MET500_met" if (c in Eall.columns and meta.loc[c, "tissue"] == "breast")
                       else ("TCGA_primary" if c in tE.columns else "drop") for c in d.index]
        d = d[d["cohort"] != "drop"]
        a = d.loc[d.cohort == "MET500_met", "score"].values
        b = d.loc[d.cohort == "TCGA_primary", "score"].values
        sec = pd.DataFrame([{
            "contrast": "MET500 breast-met vs TCGA primary",
            "n_met": len(a), "n_primary": len(b),
            "median_met": round(float(np.median(a)), 4),
            "median_primary": round(float(np.median(b)), 4),
            "delta_median": round(float(np.median(a) - np.median(b)), 4),
            "cliffs_delta": round(float(cliffs_delta(a, b)), 4),
            "p_mwu": stats.mannwhitneyu(a, b).pvalue}])
        sec.to_csv(os.path.join(OUT, "secondary_met_vs_primary.csv"), index=False)
        log(sec.to_string(index=False))
        log("    NOTE: cohort is collinear with group -> batch-confounded, NOT interpretable "
            "as biology; a near-complete separation (|delta|~1) is the signature of a batch "
            "shift, not a real effect.")
    else:
        log("    TCGA matrix not found -> skipped.")

    # ------------------------------------------------------------------ figure
    if os.environ.get("ENABLE_PLOTS", "1") == "1":
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
            axes[0].scatter(res["prolif"], res["REP30"], s=14, c="#4C72B0", alpha=0.65)
            axes[0].set_xlabel("proliferation score (12-gene mean z)")
            axes[0].set_ylabel("REP30")
            axes[0].set_title(f"A. REP30 vs proliferation (r={r_p[0]:.2f})")
            order = tis["tissue"].tolist()
            axes[1].boxplot([res.loc[res.tissue == t, "REP30"].values for t in order],
                            tick_labels=order, showfliers=False)
            axes[1].tick_params(axis="x", rotation=90, labelsize=7)
            axes[1].set_ylabel("REP30"); axes[1].set_title("B. REP30 by tissue")
            sorder = site["biopsy_tissue"].tolist()
            axes[2].boxplot([res.loc[res.biopsy_tissue == t, "REP30"].values for t in sorder],
                            tick_labels=sorder, showfliers=False)
            axes[2].tick_params(axis="x", rotation=90, labelsize=7)
            axes[2].set_title("C. REP30 by metastatic site")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, "fig_met500_rep30.png"), dpi=150)
            plt.close(fig)
            log("[fig] saved fig_met500_rep30.png")
        except Exception as e:
            log(f"[fig] skipped: {e}")

    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")
    with open(os.path.join(OUT, "run.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")
    log("\n[done] -> " + OUT)


if __name__ == "__main__":
    main()
