# -*- coding: utf-8 -*-
"""
Report on the three-cohort Harmony integration (results/sc/integrated/integrated.h5ad)
with a focus on the REP question: does the REP program mark a resistant / malignant
state at single-cell resolution?

Context
-------
`analysis/sc_harmony_integrate.py` was run on the user's Miniforge (harmonypy) at
2026-09-21 19:58 and produced:
    integrated.h5ad  96,368 cells x 15,416 genes, 29 leiden clusters
    summary.txt / composition_pct.csv / composition_count.csv / rep_by_celltype.csv

Two things have to be repaired / kept in mind before reading any table:

1. obs['study'] is integer-coded 0/1/2, NOT cohort names. `sc.concat` was handed a
   LIST (parts.append(ad)), so label="study" produced 0/1/2. The list order is
   FILES order -> 0=GSE264205 (16,368 cells; samples Pre/Post1/Post2/Meta),
   1=GSE249361 (22 samples, capped at 40,000), 2=GSE161529 (69 samples, capped at
   40,000). Verified here by inspecting obs['sample']. The generator script is
   patched, but THIS h5ad predates the patch, so we remap on the fly.

2. REP_score was scored with sc.tl.score_genes on the log-normalised expression
   matrix. Harmony corrected the PCA space, NOT the expression matrix. Therefore
   REP_score retains per-study / per-sample offsets and cross-study comparisons of
   absolute REP_score are batch-confounded. We therefore also compute a
   within-sample z-scored REP (removes the per-sample offset) and test both.

Outputs: results/sc/integrated/report/
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import anndata as ad

ROOT = r"E:/01.WorkSpace/01.Breast_cancer_gene_panel/resistance_signature"
H5 = os.path.join(ROOT, "results/sc/integrated/integrated.h5ad")
REP_CSV = os.path.join(ROOT, "results/rep_genes.csv")
OUT = os.path.join(ROOT, "results/sc/integrated/report")
os.makedirs(OUT, exist_ok=True)

STUDY_MAP = {"0": "GSE264205", "1": "GSE249361", "2": "GSE161529"}
G264205_ORDER = ["Pre", "Post1", "Post2", "Meta"]

_log = []
def log(s=""):
    print(s)
    _log.append(str(s))


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


def eta_sq(groups):
    """effect size for a one-way layout (groups = list of arrays)"""
    allv = np.concatenate(groups)
    gm = allv.mean()
    ss_b = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
    ss_t = ((allv - gm) ** 2).sum()
    return ss_b / ss_t if ss_t > 0 else np.nan


def perm_p(a, b, n=2000, seed=0):
    """two-sided permutation p on the difference of means"""
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


def main():
    a = ad.read_h5ad(H5, backed="r")
    o = a.obs[["study", "sample", "cell_type_int", "cell_type_old",
               "leiden_int", "REP_score"]].copy()
    o["study"] = o["study"].astype(str).map(STUDY_MAP)
    o["sample"] = o["sample"].astype(str)
    log(f"cells={len(o)}  genes={a.n_vars}")
    log("study mapping: " + str(o["study"].value_counts().to_dict()))
    log("GSE264205 samples: " + str(sorted(o.loc[o.study == 'GSE264205', 'sample'].unique())))

    # ---- REP gene coverage -------------------------------------------------
    rep = pd.read_csv(REP_CSV)
    present = set(a.var_names)
    miss = [g for g in rep["gene"].tolist() if g not in present]
    log(f"REP52 genes present {52 - len(miss)}/52 ; missing: {miss}")

    # ---- batch structure of REP_score --------------------------------------
    o["REP_z_within_sample"] = (o.groupby("sample")["REP_score"]
                                .transform(lambda v: (v - v.mean()) / (v.std(ddof=0) or 1)))
    o["REP_z_within_study"] = (o.groupby("study")["REP_score"]
                               .transform(lambda v: (v - v.mean()) / (v.std(ddof=0) or 1)))

    rows = []
    def add(scope, test, n, stat, p, detail=""):
        rows.append(dict(scope=scope, test=test, n=n, statistic=stat, p=p, detail=detail))

    # 1) how much of REP_score variance is batch vs cell type?
    g_study = [g["REP_score"].values for _, g in o.groupby("study")]
    g_ct = [g["REP_score"].values for _, g in o.groupby("cell_type_int")]
    g_samp = [g["REP_score"].values for _, g in o.groupby("sample") if len(g) >= 20]
    log(f"\neta^2 REP_score ~ study      = {eta_sq(g_study):.4f}")
    log(f"eta^2 REP_score ~ cell_type  = {eta_sq(g_ct):.4f}")
    log(f"eta^2 REP_score ~ sample     = {eta_sq(g_samp):.4f}")
    add("all", "eta^2 REP_score ~ study", len(o), round(eta_sq(g_study), 4), np.nan)
    add("all", "eta^2 REP_score ~ cell_type", len(o), round(eta_sq(g_ct), 4), np.nan)
    add("all", "eta^2 REP_score ~ sample", len(o), round(eta_sq(g_samp), 4), np.nan)

    # 2) study x cell_type means
    tab = o.groupby(["cell_type_int", "study"])["REP_score"].mean().unstack().round(3)
    tab.to_csv(os.path.join(OUT, "rep_by_celltype_study.csv"))
    log("\nmean REP_score by cell type x study:\n" + tab.to_string())

    # consistency: rank of cell types across studies
    rk = tab.rank(ascending=False)
    log("\ncell-type rank within each study (1 = highest REP):\n" + rk.to_string())
    add("all", "Spearman of cell-type REP ranks, GSE249361 vs GSE161529",
        int(tab.notna().sum().min()),
        round(stats.spearmanr(rk["GSE249361"], rk["GSE161529"], nan_policy="omit")[0], 3),
        stats.spearmanr(rk["GSE249361"], rk["GSE161529"], nan_policy="omit")[1])

    # 3) GSE264205 Pre/Post/Meta -- the acquired-resistance contrast
    g = o[o.study == "GSE264205"].copy()
    log("\n--- GSE264205 sample composition (cells per sample x cell type)")
    comp = g.groupby(["sample", "cell_type_int"]).size().unstack(fill_value=0)
    comp = comp.reindex([s for s in G264205_ORDER if s in comp.index])
    comp.to_csv(os.path.join(OUT, "gse264205_composition.csv"))
    log(comp.to_string())
    comp_pct = (comp.div(comp.sum(axis=1), axis=0) * 100).round(1)
    log("\npercent:\n" + comp_pct.to_string())

    big = [c for c in comp.columns if comp[c].min() >= 50]
    log(f"\ncell types present (>=50 cells) in every GSE264205 sample: {big}")
    for ct in big:
        s = g[g.cell_type_int == ct]
        pre = s.loc[s["sample"] == "Pre", "REP_score"].values
        post = s.loc[s["sample"].isin(["Post1", "Post2"]), "REP_score"].values
        meta = s.loc[s["sample"] == "Meta", "REP_score"].values
        d = cliffs_delta(post, pre)
        p = perm_p(post, pre, n=2000)
        add(f"GSE264205/{ct}", "REP Pre vs Post (raw score)", f"{len(pre)}/{len(post)}",
            round(d, 3), p, f"med {np.median(pre):.3f} vs {np.median(post):.3f}")
        # within-sample z version removes the per-sample offset
        zpre = s.loc[s["sample"] == "Pre", "REP_z_within_sample"].values
        zpost = s.loc[s["sample"].isin(["Post1", "Post2"]), "REP_z_within_sample"].values
        dz = cliffs_delta(zpost, zpre)
        pz = perm_p(zpost, zpre, n=2000)
        add(f"GSE264205/{ct}", "REP Pre vs Post (within-sample z)", f"{len(zpre)}/{len(zpost)}",
            round(dz, 3), pz, "per-sample offset removed")
        if len(meta) >= 10:
            dm = cliffs_delta(meta, pre)
            pm = perm_p(meta, pre, n=2000)
            add(f"GSE264205/{ct}", "REP Pre vs Meta (within-sample z)",
                f"{len(pre)}/{len(meta)}", round(cliffs_delta(
                    s.loc[s['sample'] == 'Meta', 'REP_z_within_sample'].values,
                    zpre), 3), pm, "per-sample offset removed")

    # 4) 4-sample omnibus per cell type
    for ct in big:
        s = g[g.cell_type_int == ct]
        grps = [s.loc[s["sample"] == sp, "REP_z_within_sample"].values
                for sp in G264205_ORDER if (s["sample"] == sp).sum() > 0]
        if len(grps) >= 3:
            H, pH = stats.kruskal(*grps)
            add(f"GSE264205/{ct}", "REP across Pre/Post1/Post2/Meta (KW, within-sample z)",
                len(s), round(H, 2), pH, f"eta^2={eta_sq(grps):.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "integrated_rep_tests.csv"), index=False)
    log("\n" + res.to_string(index=False))

    # ---- figure -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        ax = axes[0]
        keep_ct = list(tab.index)
        xs = np.arange(len(keep_ct))
        for i, st in enumerate(["GSE264205", "GSE249361", "GSE161529"]):
            ax.plot(xs, tab[st].reindex(keep_ct).values, "o-", label=st)
        ax.set_xticks(xs)
        ax.set_xticklabels(keep_ct, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("mean REP_score")
        ax.set_title("REP_score by cell type (per-study offset retained)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1]
        ct_plot = "Epithelial/Malignant" if "Epithelial/Malignant" in big else big[0]
        s = g[g.cell_type_int == ct_plot]
        data = [s.loc[s["sample"] == sp, "REP_z_within_sample"].values
                for sp in G264205_ORDER if (s["sample"] == sp).sum() > 0]
        labels = [sp for sp in G264205_ORDER if (s["sample"] == sp).sum() > 0]
        try:
            bp = ax.boxplot(data, tick_labels=labels, showfliers=False)
        except TypeError:
            bp = ax.boxplot(data, labels=labels, showfliers=False)
        for i, d in enumerate(data):
            ax.scatter(np.random.normal(i + 1, 0.05, len(d)), d, s=6, alpha=0.4)
        ax.set_ylabel("REP z (within sample)")
        ax.set_title(f"GSE264205 {ct_plot}: Pre -> Post -> Meta")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "integrated_rep_overview.png"), dpi=160)
        plt.close()
        log("figure written")
    except Exception as e:
        log(f"figure skipped: {type(e).__name__}: {e}")

    with open(os.path.join(OUT, "run_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(_log))


if __name__ == "__main__":
    main()
