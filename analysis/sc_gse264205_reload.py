"""
Rebuild the GSE264205 AnnData with CORRECT per-sample gene alignment, then
run the REP30 Pre-vs-Post control on the repaired object.

Why this script exists
----------------------
The project's existing results/sc/GSE264205/GSE264205.h5ad was produced by
sc_pipeline.py via sc.concat over the four GSM count matrices. Those matrices
have DIFFERENT gene sets per sample:

    Pre   20445 genes (first: AL627309.1)
    Post1 20766 genes (first: AL627309.1)   <- equals the h5ad's 20766 vars
    Post2 19817 genes (first: AL627309.1)
    Meta  17548 genes (first: AL627309.5)

Because the h5ad carries exactly 20766 vars (= Post1 alone), the pipeline used a
single sample's var_names and stacked the matrices POSITIONALLY. As a result all
samples except Post1 have their gene labels shifted -> housekeeping genes
collapse to ~0 (B2M positive-cell fraction: Pre 99.6% vs Post1/2/Meta ~0.2%),
which is biologically impossible and poisoned every cross-sample expression
comparison.

This script re-reads each sample with ITS OWN genes.tsv + barcodes.tsv, keeps
the gene INTERSECTION across the four samples, and produces a clean object.
It then:
  1. prints a housekeeping sanity table (ACTB/B2M/GAPDH/RPS6 positive fraction
     per sample -- should be high and stable once aligned);
  2. quantifies the compartment shift (malignant cells only in Pre + Meta);
  3. runs the REP30 Pre-vs-{post, Meta} control WITHIN compartments.

Outputs (results/sc/GSE264205/reload/):
  GSE264205_fixed.h5ad          corrected, gene-aligned object (intersection)
  housekeeping_sanity.csv       positive-fraction per sample (bug detector)
  compartment_composition.csv
  rep30_comparisons.csv         per-compartment REP30 contrast
  rep30_pergene_*.csv
  box_*.png
  summary.txt / interpretation.md

Usage:  python sc_gse264205_reload.py
        ENABLE_PLOTS=0 python sc_gse264205_reload.py
"""
import os
import glob
import gzip
import warnings
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
from scipy import stats

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
DATA = os.path.join(ROOT, "data/03_single_cell/GSE264205")
PANEL = os.path.join(ROOT, "results/rep52_trim/rep52_trimmed_panel.csv")
RES = os.path.join(ROOT, "results/sc/GSE264205/reload")
os.makedirs(RES, exist_ok=True)

SAMPLES = [("Pre", "Pre"), ("Post1", "Post1"), ("Post2", "Post2"), ("Meta", "Meta")]
MALIGNANT = "Epithelial/Malignant"

GROUP_MAP = {
    "Pre": ["Pre"], "Post1": ["Post1"], "Post2": ["Post2"], "Meta": ["Meta"],
    "post": ["Post1", "Post2"],
}
CONTRASTS = [
    (MALIGNANT, "Pre", "Meta"),
    ("Fibroblast/CAF", "Pre", "Post1"),
    ("Monocyte", "Pre", "post"),
    ("Monocyte", "Pre", "Meta"),
]

# cell-type assignment is taken from the ORIGINAL h5ad annotation (barcodes map
# 1:1 to the per-sample count matrices), because we only fix GENE alignment.
ORIG_H5 = os.path.join(ROOT, "results/sc/GSE264205/GSE264205.h5ad")

HK = ["ACTB", "GAPDH", "B2M", "RPS6", "RPL13A"]


def read_sample(tag):
    b = glob.glob(os.path.join(DATA, f"*{tag}_barcodes.tsv.gz"))[0]
    g = glob.glob(os.path.join(DATA, f"*{tag}_genes.tsv.gz"))[0]
    m = glob.glob(os.path.join(DATA, f"*{tag}_matrix.mtx.gz"))[0]
    with gzip.open(g, "rt", errors="replace") as fh:
        genes = [ln.split("\t")[0].strip() for ln in fh if ln.strip()]
    with gzip.open(b, "rt", errors="replace") as fh:
        bc = [ln.strip() for ln in fh if ln.strip()]
    M = scipy.io.mmread(m)               # genes x cells
    M = sp.csr_matrix(M).T.tocsr()       # cells x genes
    assert M.shape == (len(bc), len(genes)), (M.shape, len(bc), len(genes))
    a = ad.AnnData(X=M)
    a.obs_names = [f"{tag}_{x}" for x in bc]
    a.var_names = genes
    a.obs["sample"] = tag
    a.var_names_make_unique()
    return a


def cliffs_delta(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.nan
    diff = a[:, None] - b[None, :]
    return float(np.sign(diff).sum()) / (len(a) * len(b))


def main():
    print("[reload] reading four samples with their OWN genes.tsv ...")
    adatas = {}
    for tag, _ in SAMPLES:
        a = read_sample(tag)
        adatas[tag] = a
        print(f"  {tag}: {a.n_obs} cells x {a.n_vars} genes")

    # gene intersection (all four share a common annotation backbone)
    common = set(adatas["Pre"].var_names)
    for tag, _ in SAMPLES[1:]:
        common &= set(adatas[tag].var_names)
    common = sorted(common)
    print(f"[reload] gene intersection = {len(common)}")
    for tag, _ in SAMPLES:
        adatas[tag] = adatas[tag][:, common].copy()
    A = ad.concat(adatas, join="inner", label=None, index_unique=None)
    print(f"[reload] concatenated: {A.n_obs} cells x {A.n_vars} genes")

    A.var_names_make_unique()

    # ---- carry over cell_type annotation from the original h5ad --------------
    # The original object concatenated samples in sorted-glob order
    # [Pre, Post1, Post2, Meta] and contains exactly the same per-sample cell
    # counts, so the cell ORDER is identical -> transfer annotation positionally.
    print("[reload] transferring cell_type from original annotation ...")
    orig = sc.read_h5ad(ORIG_H5)
    assert len(orig.obs) == A.n_obs, (len(orig.obs), A.n_obs)
    assert (orig.obs["sample"].astype(str).values ==
            A.obs["sample"].map({s: str(i) for i, (s, _) in enumerate(SAMPLES)}).values).all(), \
        "sample order mismatch between original and rebuilt object"
    A.obs["cell_type"] = orig.obs["cell_type"].values
    A.obs["leiden"] = orig.obs["leiden"].values
    print("  cell_type counts:", A.obs["cell_type"].value_counts().to_dict())

    # ---- normalisation (per-cell) -------------------------------------------
    sc.pp.normalize_total(A, target_sum=1e4)
    sc.pp.log1p(A)

    # ---- housekeeping sanity (detects the alignment bug) --------------------
    hk = [g for g in HK if g in A.var_names]
    Xh = A[:, hk].X
    Xh = np.asarray(Xh.todense()) if hasattr(Xh, "todense") else np.asarray(Xh)
    hkdf = pd.DataFrame(Xh, columns=hk)
    hkdf["sample"] = A.obs["sample"].values
    frac = hkdf.groupby("sample")[hk].apply(lambda g: (g > 0).mean()).round(3)
    mexp = hkdf.groupby("sample")[hk].mean().round(3)
    frac.to_csv(os.path.join(RES, "housekeeping_sanity.csv"))
    print("\n[housekeeping] positive-cell fraction per sample (should be high & stable):")
    print(frac.to_string())
    print("\n[housekeeping] mean log-expression:")
    print(mexp.to_string())

    A.write_h5ad(os.path.join(RES, "GSE264205_fixed.h5ad"))
    print(f"[saved] {os.path.join(RES, 'GSE264205_fixed.h5ad')}")

    # ---- compartment composition --------------------------------------------
    ct = pd.crosstab(A.obs["sample"], A.obs["cell_type"])
    ct.to_csv(os.path.join(RES, "compartment_composition.csv"))
    print("\n[composition] sample x cell_type:")
    print(ct.to_string())

    # ---- REP30 per compartment ----------------------------------------------
    panel_df = pd.read_csv(PANEL)
    panel_df["gene"] = panel_df["gene"].astype(str)
    coef = dict(zip(panel_df["gene"], panel_df["coef"]))

    rows, all_cell = [], []
    for ctype, gA, gB in CONTRASTS:
        sub = A[A.obs["cell_type"] == ctype].copy()
        mA = sub.obs["sample"].isin(GROUP_MAP[gA]).values
        mB = sub.obs["sample"].isin(GROUP_MAP[gB]).values
        nA, nB = int(mA.sum()), int(mB.sum())
        label = f"{ctype} | {gA} vs {gB}"
        if nA < 10 or nB < 10:
            print(f"[skip] {label}: nA={nA} nB={nB}")
            rows.append({"compartment": ctype, "group_A": gA, "group_B": gB,
                         "n_A": nA, "n_B": nB, "note": "skipped n<10"})
            continue
        genes = [g for g in panel_df["gene"].tolist() if g in sub.var_names]
        X = sub[:, genes].X
        X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
        X = X.astype(float)
        Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
        w = np.array([coef[g] for g in genes]); w = w / np.abs(w).sum()
        score = Z @ w
        a, b = score[mA], score[mB]
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        delta = cliffs_delta(a, b)
        rows.append({
            "compartment": ctype, "group_A": gA, "group_B": gB,
            "n_A": nA, "n_B": nB,
            "median_A": round(float(np.median(a)), 4),
            "median_B": round(float(np.median(b)), 4),
            "delta_median(B-A)": round(float(np.median(b) - np.median(a)), 4),
            "mean_A": round(float(a.mean()), 4), "mean_B": round(float(b.mean()), 4),
            "cliffs_delta": round(float(delta), 4), "mwu_p": p,
            "n_genes_used": len(genes),
        })
        pg = []
        for j, g in enumerate(genes):
            pg.append({"gene": g, "mean_A": round(float(X[mA, j].mean()), 4),
                       "mean_B": round(float(X[mB, j].mean()), 4),
                       "delta_B_minus_A": round(float(X[mB, j].mean() - X[mA, j].mean()), 4)})
        pg = pd.DataFrame(pg).sort_values("delta_B_minus_A")
        pg.to_csv(os.path.join(RES, f"rep30_pergene_{ctype.replace('/', '_')}_{gA}_vs_{gB}.csv"),
                  index=False)
        tmp = sub.obs[["sample", "cell_type"]].copy(); tmp["rep30"] = score
        tmp["contrast"] = label; all_cell.append(tmp)
        print(f"\n[{label}] nA={nA} nB={nB} | med {np.median(a):.4f} vs {np.median(b):.4f} "
              f"(d={np.median(b)-np.median(a):+.4f}) cliffs_d={delta:+.3f} p={p:.2e}")
        print("  top |delta| genes:")
        print(pg.reindex(pg["delta_B_minus_A"].abs().sort_values(ascending=False).index)
              .head(6).to_string(index=False))
        if plt is not None:
            fig, ax = plt.subplots(figsize=(5, 4.5))
            ax.boxplot([a, b], tick_labels=[f"{gA}\n(n={nA})", f"{gB}\n(n={nB})"],
                       showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor="#c8d8f0"),
                       medianprops=dict(color="firebrick"))
            for i, (vals, col) in enumerate([(a, "#1f77b4"), (b, "#d95f02")]):
                jit = (np.random.rand(len(vals)) - 0.5) * 0.25 + (i + 1)
                ax.scatter(jit, vals, s=3, color=col, alpha=0.25)
            ax.set_ylabel("REP30 score")
            ax.set_title(f"REP30 (gene-aligned): {ctype}\n{gA} vs {gB} (p={p:.1e}, d={delta:+.2f})")
            plt.tight_layout()
            plt.savefig(os.path.join(RES, f"box_{ctype.replace('/', '_')}_{gA}_vs_{gB}.png"), dpi=140)
            plt.close()

    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(RES, "rep30_comparisons.csv"), index=False)
    if all_cell:
        pd.concat(all_cell).to_csv(os.path.join(RES, "rep30_by_cell.csv"))

    with open(os.path.join(RES, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("GSE264205 rebuilt (per-sample gene alignment fixed) + REP30 control\n")
        f.write("=" * 68 + "\n")
        f.write("gene intersection = %d ; cells = %d\n" % (A.n_vars, A.n_obs))
        f.write("\n== housekeeping positive-cell fraction (alignment sanity) ==\n")
        f.write(frac.to_string() + "\n")
        f.write("\n== compartment composition ==\n")
        f.write(ct.to_string() + "\n")
        f.write("\n== REP30 per compartment ==\n")
        f.write(comp.to_string(index=False) + "\n")
    print(f"\n[DONE] outputs in {RES}")


if __name__ == "__main__":
    main()
