"""
GSE264205 gene-level DE on the GENE-ALIGNED rebuild (replaces the invalid
resistance_evolution results produced from the misaligned original h5ad).

Why the contrasts look like this
--------------------------------
With the alignment fixed, the compartment structure is unchanged:
  Pre(s0)  : 2453 malignant, 151 CAF, 486 monocyte, 110 plasma, 489 CD4, 1810 CD8
  Post1(s1):    1 malignant, 521 CAF, 3309 monocyte
  Post2(s2):   16 malignant,   2 CAF, 6575 monocyte
  Meta(s3) :  406 malignant,   0 CAF,   37 monocyte
So a within-malignant Pre-vs-Post DE is impossible (Post malignant ~0). The only
feasible contrasts are:
  Epithelial/Malignant : Pre vs Meta   (primary vs metastatic lesion; confounded)
  Fibroblast/CAF       : Pre vs Post1
  Monocyte             : Pre vs post (Post1+Post2)
Each is a Wilcoxon rank-sum DE (scanpy), top-50 by padj, plus an overlay of the
project's resistance panel / REP52 / REP30 / curated pathways.

Outputs (results/sc/GSE264205/resistance_evolution_fixed/):
  de_<CellType>.csv          top-50 DE genes per contrast
  resistance_focus.csv       panel/REP/pathway genes x compartment (logFC, padj)
  volcano_<CellType>.png
  resistance_heatmap.png
  summary.txt / interpretation.md

Usage:  python sc_gse264205_de_fixed.py
        ENABLE_PLOTS=0 python sc_gse264205_de_fixed.py
"""
import os
import warnings
import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")

ENABLE_PLOTS = os.environ.get("ENABLE_PLOTS", "1") != "0"
plt = None
if ENABLE_PLOTS:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[warn] matplotlib unavailable -> plots skipped ({e})")
        plt = None

ROOT = r"E:\01.WorkSpace\01.Breast_cancer_gene_panel\resistance_signature"
H5 = os.path.join(ROOT, "results/sc/GSE264205/reload/GSE264205_fixed.h5ad")
RES = os.path.join(ROOT, "results/sc/GSE264205/resistance_evolution_fixed")
os.makedirs(RES, exist_ok=True)

RESIST_CSV = os.path.join(ROOT, "resistance_genes.csv")
REP52_CSV = os.path.join(ROOT, "results/rep_genes.csv")
REP30_CSV = os.path.join(ROOT, "results/rep52_trim/rep52_trimmed_panel.csv")

# (cell_type, groupA=reference, groupB=test, list of samples in B)
CONTRASTS = [
    ("Epithelial/Malignant", "Pre", "Meta", ["Meta"]),
    ("Fibroblast/CAF", "Pre", "Post1", ["Post1"]),
    ("Monocyte", "Pre", "post", ["Post1", "Post2"]),
]

PATHWAYS = {
    "PI3K-AKT": ["PIK3CA", "PIK3R1", "AKT1", "PTEN", "MTOR"],
    "CellCycle": ["CCND1", "CCND2", "CCNE1", "CDK4", "CDK6", "RB1", "CDKN2A", "MYC"],
    "EMT": ["CDH1", "CDH2", "SNAI1", "SNAI2", "TWIST1", "ZEB1", "VIM"],
    "ImmuneCheckpoint": ["PDCD1", "CTLA4", "HAVCR2", "LAG3", "TIGIT"],
    "MDRefflux": ["ABCB1", "ABCG2", "ABCC1"],
    "HRpathway": ["ESR1", "PGR", "GATA3", "FOXA1"],
}


def load_genes(path, col="gene"):
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    return [str(g).upper() for g in df[col].tolist()]


def main():
    print(f"[load] {H5}")
    ad = sc.read_h5ad(H5)
    print(f"  {ad.n_obs} cells x {ad.n_vars} genes")
    ad.obs["sample"] = ad.obs["sample"].astype(str)

    panel = load_genes(RESIST_CSV)
    rep52 = load_genes(REP52_CSV)
    rep30 = load_genes(REP30_CSV)
    pathway = sorted({g.upper() for v in PATHWAYS.values() for g in v})
    focus_set = set(panel) | set(rep52) | set(rep30) | set(pathway)

    focus_rows = []
    for ctype, gA, gB, bset in CONTRASTS:
        sub = ad[ad.obs["cell_type"] == ctype].copy()
        maskB = sub.obs["sample"].isin(bset).values
        maskA = (sub.obs["sample"] == gA).values
        nA, nB = int(maskA.sum()), int(maskB.sum())
        if nA < 10 or nB < 10:
            print(f"[skip] {ctype} {gA} vs {gB}: nA={nA} nB={nB}")
            continue
        sub.obs["grp"] = np.where(maskA, gA, np.where(maskB, gB, "other"))
        sub = sub[sub.obs["grp"] != "other"].copy()
        sc.tl.rank_genes_groups(sub, groupby="grp", groups=[gB], reference=gA,
                                method="wilcoxon", use_raw=False,
                                n_genes=min(sub.n_vars, 200))
        de = sc.get.rank_genes_groups_df(sub, group=gB)
        de = de.rename(columns={"names": "gene", "logfoldchanges": "logFC",
                                "pvals": "pval", "pvals_adj": "padj"})
        de.insert(0, "cell_type", ctype)
        de.insert(1, "contrast", f"{gA}_vs_{gB}")
        de.head(50).to_csv(os.path.join(RES, f"de_{ctype.replace('/', '_')}.csv"), index=False)
        print(f"\n[{ctype} {gA} vs {gB}] nA={nA} nB={nB} | top: "
              f"{de.iloc[0]['gene']} logFC={de.iloc[0]['logFC']:.2f} padj={de.iloc[0]['padj']:.1e} "
              f"| sig(padj<0.05)={int((de['padj'] < 0.05).sum())}")

        # focus genes present in DE
        for g in sorted(focus_set):
            hit = de[de["gene"].astype(str).str.upper() == g]
            if len(hit):
                r = hit.iloc[0]
                lab = []
                if g in pathway:
                    lab += [k for k, v in PATHWAYS.items() if g in v]
                if g in panel:
                    lab.append("PANEL")
                if g in rep52:
                    lab.append("REP52")
                if g in rep30:
                    lab.append("REP30")
                focus_rows.append({"gene": g, "labels": ";".join(lab),
                                   "cell_type": ctype, "contrast": f"{gA}_vs_{gB}",
                                   "logFC": round(float(r["logFC"]), 3),
                                   "padj": float(r["padj"]),
                                   "sig": bool(r["padj"] < 0.05)})

        if plt is not None:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(de["logFC"], -np.log10(de["padj"].clip(lower=1e-300)),
                       s=5, c="gray", alpha=0.4)
            hi = de[(de["padj"] < 0.05) & (de["logFC"].abs() > 0.5)]
            ax.scatter(hi["logFC"], -np.log10(hi["padj"].clip(lower=1e-300)),
                       s=8, c="firebrick", alpha=0.6)
            for _, r in de.iterrows():
                if str(r["gene"]).upper() in focus_set and r["padj"] < 0.05:
                    ax.annotate(r["gene"], (r["logFC"], -np.log10(r["padj"])),
                                fontsize=7, alpha=0.85)
            ax.axhline(-np.log10(0.05), color="k", ls="--", lw=0.6)
            ax.axvline(0, color="k", ls="--", lw=0.6)
            ax.set_xlabel(f"log2 FC ({gB}/{gA})"); ax.set_ylabel("-log10 padj")
            ax.set_title(f"GSE264205 (gene-aligned) DE: {ctype}\n{gA} vs {gB}")
            plt.tight_layout()
            plt.savefig(os.path.join(RES, f"volcano_{ctype.replace('/', '_')}.png"), dpi=130)
            plt.close()

    if focus_rows:
        fdf = pd.DataFrame(focus_rows).sort_values(["cell_type", "padj"])
        fdf.to_csv(os.path.join(RES, "resistance_focus.csv"), index=False)
        if plt is not None:
            piv = fdf.pivot_table(index="gene", columns="cell_type", values="logFC")
            fig, ax = plt.subplots(figsize=(max(6, piv.shape[1] * 1.1),
                                            max(6, piv.shape[0] * 0.32)))
            im = ax.imshow(piv.values, aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5)
            ax.set_xticks(range(piv.shape[1]))
            ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(piv.shape[0]))
            ax.set_yticklabels(piv.index, fontsize=7)
            plt.colorbar(im, ax=ax, label="log2 FC (post/Pre)")
            ax.set_title("GSE264205 (gene-aligned) resistance-gene shift")
            plt.tight_layout()
            plt.savefig(os.path.join(RES, "resistance_heatmap.png"), dpi=140)
            plt.close()
        print("\n[focus] resistance/REP/pathway genes with padj<0.05:")
        sig = fdf[fdf["sig"]]
        print(sig.to_string(index=False) if len(sig) else "  (none)")

    with open(os.path.join(RES, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("GSE264205 gene-level DE (gene-ALIGNED rebuild)\n")
        f.write("=" * 60 + "\n")
        f.write("source: results/sc/GSE264205/reload/GSE264205_fixed.h5ad\n")
        f.write("NOTE: replaces the invalid results in resistance_evolution/\n")
        f.write("      (produced from the per-sample gene-misaligned original h5ad).\n\n")
        f.write("feasible contrasts (Post malignant cells ~0):\n")
        f.write("  Epithelial/Malignant : Pre vs Meta (primary vs metastasis)\n")
        f.write("  Fibroblast/CAF       : Pre vs Post1\n")
        f.write("  Monocyte             : Pre vs post (Post1+Post2)\n\n")
        if focus_rows:
            f.write("resistance/REP/pathway genes shifting (padj<0.05):\n")
            f.write(fdf[fdf["sig"]].to_string(index=False) + "\n")
    print(f"\n[DONE] outputs in {RES}")


if __name__ == "__main__":
    main()
