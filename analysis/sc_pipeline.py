"""
Per-study single-cell RNA-seq basic analysis pipeline (scanpy).

For each GEO study (GSE264205 / GSE249361 / GSE161529):
  raw mtx (genes x barcodes) -> AnnData per sample
  -> per-sample cell filtering (drop empty 10x whitelist barcodes / lowQ)
  -> within-study concatenation
  -> global QC (n_genes, n_counts, mito%)
  -> normalize + log1p -> HVG (seurat_v3, 3000)
  -> optional subsample to MAX_CELLS (memory safety for huge studies)
  -> batch correction: Harmony if harmonypy importable, else ComBat (scanpy built-in)
  -> PCA(50) -> neighbors -> leiden -> UMAP
  -> rank_genes_groups (wilcoxon) per leiden cluster
  -> rule-based cell-type annotation via canonical marker panels
Outputs (results/sc/{study}/):
  {study}.h5ad, annotation_table.csv, cluster_markers.csv,
  qc_violin.png, umap_cluster.png, umap_type.png

Usage:  python sc_pipeline.py GSE264205            (one study)
        python sc_pipeline.py all                  (loop all three)
"""
import os, sys, re, glob, gzip, warnings
import numpy as np
import pandas as pd
import scipy.io as sio
import scanpy as sc
warnings.filterwarnings("ignore")

ROOT = r"E:\01.WorkSpace\01.Breast_cancer_gene_panel\resistance_signature"
DATA = os.path.join(ROOT, "data", "03_single_cell")
RES  = os.path.join(ROOT, "results", "sc")
os.makedirs(RES, exist_ok=True)

MAX_CELLS = 120000          # subsample cap per study (memory safety)
MIN_GENES = 200             # min genes per cell
MIN_COUNTS = 500            # min counts per cell
MAX_MITO = 0.30             # max mito fraction
N_HVG = 3000
N_PCS = 50
LEIDEN_RES = 0.8

# ---------------------------------------------------------------- Harmony availability
try:
    import harmonypy
    USE_HARMONY = True
    print("[batch] harmonypy available -> Harmony integration")
except Exception:
    USE_HARMONY = False
    print("[batch] harmonypy NOT available -> falling back to ComBat (scanpy)")

# ---------------------------------------------------------------- loaders
def find_matrix_files(study_dir):
    """Return list of (matrix_path, barcodes_path, features_path) tuples."""
    matrices = sorted(glob.glob(os.path.join(study_dir, "*matrix.mtx.gz")) +
                      glob.glob(os.path.join(study_dir, "*matrix.mtx")))
    out = []
    for m in matrices:
        name = os.path.basename(m)
        bc = (name.replace("_matrix.mtx.gz", "_barcodes.tsv.gz").replace("-matrix.mtx.gz", "-barcodes.tsv.gz")
                 .replace("_matrix.mtx", "_barcodes.tsv").replace("-matrix.mtx", "-barcodes.tsv"))
        bc_path = os.path.join(study_dir, bc)
        if not os.path.exists(bc_path):
            # shared barcodes? unlikely; skip
            cand = glob.glob(os.path.join(study_dir, os.path.basename(m).split("_matrix")[0].split("-matrix")[0] + "*barcodes*"))
            bc_path = cand[0] if cand else None
        ft = (name.replace("_matrix.mtx.gz", "_features.tsv.gz").replace("-matrix.mtx.gz", "-features.tsv.gz")
                 .replace("_matrix.mtx.gz", "_genes.tsv.gz").replace("-matrix.mtx.gz", "-genes.tsv.gz")
                 .replace("_matrix.mtx", "_features.tsv").replace("-matrix.mtx", "-features.tsv"))
        ft_path = os.path.join(study_dir, ft)
        if not os.path.exists(ft_path):
            shared = glob.glob(os.path.join(study_dir, "*features.tsv.gz")) + glob.glob(os.path.join(study_dir, "*genes.tsv.gz"))
            ft_path = shared[0] if shared else None
        out.append((m, bc_path, ft_path))
    return [(m, b, f) for (m, b, f) in out if b and f]

def read_features(path):
    df = pd.read_csv(path, sep="\t", header=None, low_memory=False,
                     compression="gzip" if path.endswith(".gz") else None)
    if df.shape[1] >= 2:
        gene_id = df[0].astype(str).str.strip().values
        symbol  = df[1].astype(str).str.strip().values
    else:
        gene_id = symbol = df[0].astype(str).str.strip().values
    # prefer symbol when it looks like a real symbol (not Ensembl id)
    use_symbol = np.array([not s.startswith("ENSG") for s in symbol])
    var_names = np.where(use_symbol, symbol, gene_id)
    return var_names, gene_id

def load_sample(matrix_path, barcodes_path, features_path, study, sample_label):
    var_names, gene_id = read_features(features_path)
    with gzip.open(barcodes_path, "rt") if barcodes_path.endswith(".gz") else open(barcodes_path) as fh:
        barcodes = [l.strip() for l in fh if l.strip()]
    M = sio.mmread(matrix_path).tocsc().T.tocsr()   # cells x genes
    # align genes: features file may have different gene count than matrix rows
    ng = M.shape[1]
    if len(var_names) != ng:
        # truncate/pad
        if len(var_names) > ng:
            var_names = var_names[:ng]; gene_id = gene_id[:ng]
        else:
            var_names = np.array(list(var_names) + [f"gene_{i}" for i in range(len(var_names), ng)])
            gene_id = np.array(list(gene_id) + [f"gene_{i}" for i in range(len(gene_id), ng)])
    n_bc = M.shape[0]
    if len(barcodes) != n_bc:
        barcodes = [f"{sample_label}_cell{i}" for i in range(n_bc)]
    ad = sc.AnnData(M)
    ad.var_names = pd.Index(var_names)
    ad.obs_names = pd.Index([f"{sample_label}-{i}" for i in range(n_bc)])
    ad.obs["sample"] = sample_label
    ad.obs["study"] = study
    ad.var["gene_id"] = gene_id
    ad.var_names_make_unique()
    ad.obs_names_make_unique()
    ad.obs["n_genes_raw"] = np.asarray((M > 0).sum(axis=1)).flatten()
    ad.obs["n_counts_raw"] = np.asarray(M.sum(axis=1)).flatten()
    return ad

# ---------------------------------------------------------------- canonical markers for annotation
MARKERS = {
    "Epithelial/Malignant": ["EPCAM","KRT8","KRT18","KRT19","KRT7","CDH1","MUC1","ESRP1","KRT17"],
    "T cell CD8":           ["CD3D","CD3E","CD8A","CD8B","GZMA","GZMK","NKG7","CCL5"],
    "T cell CD4":           ["CD3D","CD3E","CD4","IL7R","FOXP3","ICOS","TIGIT"],
    "Treg":                 ["FOXP3","IL2RA","CTLA4","TIGIT","IKZF2"],
    "B cell":               ["CD79A","CD79B","MS4A1","BANK1","IGHD","CD19"],
    "Plasma cell":          ["SDC1","MZB1","IGHG1","IGHG2","IGHA1","SEC11C","TNFRSF17","JCHAIN"],
    "NK cell":              ["NCAM1","KLRD1","KLRC1","GNLY","NKG7","FCGR3A"],
    "Macrophage":           ["LYZ","CD68","CD163","CSF1R","MARCO","VSIG4","FOLR2"],
    "Monocyte":             ["LYZ","FCN1","S100A8","S100A9","CD14","VCAN"],
    "Dendritic cell":       ["FCER1A","CD1C","CLEC10A","LILRA4","IRF8"],
    "Mast cell":            ["KIT","TPSAB1","TPSB2","CPA3","MS4A2","HDC"],
    "Endothelial":          ["PECAM1","VWF","CLDN5","PLVAP","CDH5","FLT1"],
    "Fibroblast/CAF":       ["COL1A1","COL1A2","PDGFRB","ACTA2","FAP","PDPN","RGS5"],
    "Erythrocyte":          ["HBB","HBA1","HBA2","HBD"],
}

def annotate(adata, clusters):
    """Predict cell type per leiden cluster by mean expression of canonical
    marker panels (cluster-mean z-scored across clusters for specificity).
    Returns (pred_mean, pred_overlap) -- two views; pred_mean assigns a type
    to every cluster (first-pass), pred_overlap is a stricter marker-overlap
    check for manual curation."""
    rg = adata.uns["rank_genes_groups"]
    all_m = sorted({g for v in MARKERS.values() for g in v})
    present = [g for g in all_m if g in adata.var_names]
    present_idx = [list(adata.var_names).index(g) for g in present]
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    sub = X[:, present_idx]
    g2i = {g: i for g, i in zip(present, range(len(present)))}
    pred_mean, pred_overlap = {}, {}
    for cl in clusters:
        m = adata.obs["leiden"] == cl
        cm = sub[m].mean(axis=0)
        cm = (cm - cm.mean()) / (cm.std() + 1e-9)
        best_type, best_score = "Unknown", 0.0
        for t, ms in MARKERS.items():
            vals = [cm[g2i[g]] for g in ms if g in g2i]
            if not vals:
                continue
            s = float(np.mean(vals))
            if s > best_score:
                best_type, best_score = t, s
        pred_mean[cl] = best_type
        # overlap view (top-30 specific markers)
        top = [str(g).upper() for g in rg["names"][cl][:30]]
        bo, bn = "Unknown", 0
        for t, ms in MARKERS.items():
            mset = set(g.upper() for g in ms)
            n = sum(1 for g in top if g in mset)
            if n > bn:
                bo, bn = t, n
        pred_overlap[cl] = bo
    return pred_mean, pred_overlap

# ---------------------------------------------------------------- main per-study
def run_study(study):
    sdir = os.path.join(DATA, study)
    outdir = os.path.join(RES, study)
    os.makedirs(outdir, exist_ok=True)
    print(f"\n{'='*70}\n[{study}] loading samples")
    groups = find_matrix_files(sdir)
    print(f"  found {len(groups)} matrix files")
    adatas = []
    for mi, (m, bc, ft) in enumerate(groups):
        sample_label = re.sub(r"[-_]?(matrix|barcodes|features|genes)\.mtx(\.gz)?$", "",
                               os.path.basename(m))
        sample_label = re.sub(r"\.(gz|tsv)$", "", sample_label)
        ad = load_sample(m, bc, ft, study, sample_label)
        # per-sample cell filter: drop empty whitelist barcodes / lowQ
        keep = (ad.obs["n_genes_raw"].values >= MIN_GENES) & (ad.obs["n_counts_raw"].values >= MIN_COUNTS)
        ad = ad[keep].copy()
        if ad.n_obs == 0:
            continue
        adatas.append(ad)
        if (mi + 1) % 5 == 0 or mi == 0:
            print(f"   sample {mi+1}/{len(groups)} {sample_label}: {ad.n_obs} cells kept")
    if not adatas:
        print(f"[{study}] NO cells after filtering -> abort")
        return
    adata = sc.concat(adatas, join="outer", label="sample", index_unique=None) if len(adatas) > 1 else adatas[0]
    adata.var_names_make_unique()
    print(f"[{study}] concatenated: {adata.n_obs} cells x {adata.n_vars} genes, "
          f"{adata.obs['sample'].nunique()} samples")

    # global QC
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    mito = np.asarray(adata.var_names.str.startswith("MT-"))
    adata.obs["mito"] = np.asarray(adata.X[:, mito].sum(axis=1)).flatten() / adata.obs["total_counts"].values
    adata.obs["mito"] = np.nan_to_num(adata.obs["mito"].values, nan=0.0)
    qc_before = adata.n_obs
    adata = adata[(adata.obs["n_genes_by_counts"] >= MIN_GENES) &
                  (adata.obs["total_counts"] >= MIN_COUNTS) &
                  (adata.obs["mito"] <= MAX_MITO)].copy()
    print(f"[{study}] QC filter: {qc_before} -> {adata.n_obs} cells "
          f"(mito<={MAX_MITO:.0%}, genes>={MIN_GENES})")
    if adata.n_obs == 0:
        print(f"[{study}] all cells removed by QC -> abort"); return

    # optional subsample
    if adata.n_obs > MAX_CELLS:
        rng = np.random.default_rng(0)
        sel = rng.choice(adata.n_obs, MAX_CELLS, replace=False)
        adata = adata[sel].copy()
        print(f"[{study}] subsampled to {adata.n_obs} cells (MAX_CELLS={MAX_CELLS})")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=N_HVG)
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata_hvg, max_value=10)

    # batch correction across samples within study
    if USE_HARMONY:
        sc.tl.pca(adata_hvg, n_comps=N_PCS, svd_solver="auto")
        sc.external.pp.harmony_integrate(adata_hvg, key="sample", basis="X_pca",
                                         adjusted_basis="X_pca_harmony")
        embed = "X_pca_harmony"
    else:
        sc.pp.combat(adata_hvg, key="sample")
        sc.tl.pca(adata_hvg, n_comps=N_PCS, svd_solver="auto")
        embed = "X_pca"
    sc.pp.neighbors(adata_hvg, n_pcs=N_PCS)
    sc.tl.leiden(adata_hvg, resolution=LEIDEN_RES, key_added="leiden")
    sc.tl.umap(adata_hvg)

    # transfer clustering/umap back to full-gene adata
    adata.obsm["X_umap"] = adata_hvg.obsm["X_umap"]
    adata.obsm["X_pca"] = adata_hvg.obsm[embed]
    adata.obs["leiden"] = adata_hvg.obs["leiden"]
    adata.uns["leiden"] = adata_hvg.uns.get("leiden", {})

    # markers per cluster
    sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon",
                            use_raw=False, n_genes=10)
    cl = sorted(adata.obs["leiden"].unique(), key=lambda x: int(x))
    pred_mean, pred_overlap = annotate(adata, cl)
    adata.obs["cell_type"] = adata.obs["leiden"].map(pred_mean).astype(str)
    adata.obs["cell_type_overlap"] = adata.obs["leiden"].map(pred_overlap).astype(str)

    # ---- outputs
    adata.write(os.path.join(outdir, f"{study}.h5ad"))
    # annotation table
    rows = []
    for c in cl:
        msk = adata.obs["leiden"] == c
        top = [g for g in adata.uns["rank_genes_groups"]["names"][c][:10]]
        rows.append({"cluster": c, "n_cells": int(msk.sum()),
                     "predicted_cell_type": pred_mean[c],
                     "predicted_cell_type_overlap": pred_overlap[c],
                     "top_markers": ",".join(top)})
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "annotation_table.csv"), index=False)
    # full marker table
    rg = adata.uns["rank_genes_groups"]
    pct = rg.get("pct_nz", None)
    mrows = []
    for c in cl:
        for i in range(10):
            mrows.append({"cluster": c, "rank": i + 1,
                          "gene": rg["names"][c][i],
                          "logfoldchanges": rg["logfoldchanges"][c][i],
                          "pvals": rg["pvals"][c][i],
                          "pvals_adj": rg["pvals_adj"][c][i],
                          "pct_nz": (pct[c][i] if pct is not None else float("nan"))})
    pd.DataFrame(mrows).to_csv(os.path.join(outdir, "cluster_markers.csv"), index=False)

    # plots (matplotlib direct -> outdir)
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    uv = adata.obsm["X_umap"]
    # UMAP colored by leiden
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    leiden_codes = {c: i for i, c in enumerate(cl)}
    cols = [leiden_codes[c] for c in adata.obs["leiden"]]
    ax[0].scatter(uv[:, 0], uv[:, 1], s=2, c=cols, cmap="tab20", alpha=0.5)
    ax[0].set_title(f"{study} - leiden clusters"); ax[0].set_axis_off()
    # UMAP colored by predicted cell type
    types = sorted(adata.obs["cell_type"].unique())
    tcode = {t: i for i, t in enumerate(types)}
    tc = [tcode[t] for t in adata.obs["cell_type"]]
    ax[1].scatter(uv[:, 0], uv[:, 1], s=2, c=tc, cmap="tab20", alpha=0.5)
    ax[1].set_title(f"{study} - predicted cell type"); ax[1].set_axis_off()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "umap.png"), dpi=120); plt.close()
    # QC histogram
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].hist(adata.obs["n_genes_by_counts"], bins=40); ax[0].set_title("n_genes")
    ax[1].hist(adata.obs["total_counts"], bins=40); ax[1].set_title("n_counts")
    ax[2].hist(adata.obs["mito"], bins=40); ax[2].set_title("mito%")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "qc_hist.png"), dpi=120); plt.close()
    print(f"[{study}] DONE -> {outdir}")
    print(f"  clusters: {len(cl)}, cell types: {adata.obs['cell_type'].value_counts().to_dict()}")

# ---------------------------------------------------------------- run
if __name__ == "__main__":
    studies = sys.argv[1:]
    if not studies or studies == ["all"]:
        studies = ["GSE264205", "GSE249361", "GSE161529"]
    for s in studies:
        run_study(s)
