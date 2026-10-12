"""
Harmony cross-study integration of three breast-cancer scRNA-seq cohorts:
  GSE264205 (Pre/Post paired, 16,368 cells) -- REPLACED by the gene-aligned
            rebuild results/sc/GSE264205/reload/GSE264205_fixed.h5ad
            (the original h5ad had a per-sample gene-misalignment bug)
  GSE249361 (22 samples, 120,000 sampled)
  GSE161529 (69 samples, 120,000 sampled)

Pipeline (run with miniforge python, harmonypy required):
  load -> gene-intersection -> log1p-restore -> per-study subsample ->
  concat -> normalize/log -> HVG -> scale -> PCA -> harmony_integrate(study) ->
  neighbors/UMAP/leiden -> re-annotate (unified marker panel) -> restore full-gene
  matrix + .raw -> REP score on full-gene matrix ->
  write integrated.h5ad (X = all intersection genes, .raw preserved) + CSV + summary.

Plotting is separated into _plot_integrated.py (run with the managed venv that
has matplotlib but no harmonypy). Set ENABLE_PLOTS=1 only on the user's local
Miniforge Prompt where matplotlib works; in the sandbox keep ENABLE_PLOTS=0.
"""
import os, re, sys, warnings, numpy as np, pandas as pd
import scanpy as sc
from scipy import sparse
warnings.filterwarnings("ignore")

ROOT = r"E:/01.WorkSpace/01.Breast_cancer_gene_panel/resistance_signature"
FILES = {
    # NOTE 2026-09-21: the original GSE264205.h5ad had a per-sample GENE
    # MISALIGNMENT bug (four GSMs have different genes.tsv -> positional concat
    # scrambled gene labels; housekeeping genes collapsed). It is replaced here
    # by the rebuilt, gene-aligned object (16328 intersection genes, see
    # analysis/sc_gse264205_reload.py). Do NOT revert to the old path.
    "GSE264205": os.path.join(ROOT, "results/sc/GSE264205/reload/GSE264205_fixed.h5ad"),
    "GSE249361": os.path.join(ROOT, "results/sc/GSE249361/GSE249361.h5ad"),
    "GSE161529": os.path.join(ROOT, "results/sc/GSE161529/GSE161529.h5ad"),
}
RES = os.path.join(ROOT, "results/sc/integrated")
REP_CSV = os.path.join(ROOT, "results/rep_genes.csv")
MAX_PER_STUDY = int(os.environ.get("MAX_PER_STUDY", "40000"))
ENABLE_PLOTS = os.environ.get("ENABLE_PLOTS", "0") != "0"

# unified cell-type marker panel (symbols)
MARKERS = {
    "Epithelial/Malignant": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "ESA"],
    "T cell CD8": ["CD8A", "CD8B", "GZMK", "GZMA", "CCR7"],
    "T cell CD4": ["CD4", "IL7R", "FOXP3", "SAT1"],
    "NK cell": ["NCAM1", "NKG7", "KLRD1", "GNLY", "KLRC1"],
    "B cell": ["MS4A1", "CD79A", "CD79B", "BANK1", "CD19"],
    "Plasma cell": ["SDC1", "MZB1", "SEC11C", "XBP1", "CD138"],
    "Macrophage": ["CD68", "CD163", "MARCO", "FOLR2", "LYZ", "VSIG4"],
    "Monocyte": ["LYZ", "VCAN", "FCN1", "S100A8", "S100A9", "CXCL8"],
    "Dendritic cell": ["FCER1A", "HLA-DQA1", "CD1C", "CLEC10A", "CST3"],
    "Endothelial": ["PECAM1", "VWF", "CLDN5", "PLVAP", "FLT1"],
    "Fibroblast/CAF": ["COL1A1", "COL1A2", "PDGFRB", "ACTA2", "PDPN", "RGS5"],
    "Mast cell": ["TPSAB1", "CPA3", "KIT", "MS4A2", "TPSB2"],
}


def log(*a):
    print(" ".join(str(x) for x in a), flush=True)


def main():
    os.makedirs(RES, exist_ok=True)

    # ---- pass 1: gene intersection ----
    var_sets = {}
    for s, p in FILES.items():
        ad = sc.read_h5ad(p)
        var_sets[s] = set(ad.var_names)
        log(f"[load] {s}: {ad.n_obs} cells, {ad.n_vars} genes")
        del ad
    inter = sorted(set.intersection(*var_sets.values()))
    log(f"[genes] intersection = {len(inter)}")

    # ---- sanity: GSE264205 gene alignment (housekeeping should be ~1.0) ----
    _hk = [g for g in ["ACTB", "GAPDH", "B2M", "RPS6"] if g in inter]
    _ad = sc.read_h5ad(FILES["GSE264205"])
    _X = _ad[:, _hk].X
    _X = np.asarray(_X.todense()) if sparse.issparse(_X) else np.asarray(_X)
    _pos = (_X > 0).mean(axis=0)
    log("[sanity] GSE264205 housekeeping +frac "
        f"{dict(zip(_hk, np.round(_pos, 3)))} (expect >0.8; else gene misalignment)")
    del _ad

    # ---- pass 2: prep + concat ----
    parts = []
    for s, p in FILES.items():
        ad = sc.read_h5ad(p)
        # restore to counts if previously log1p-normalized
        if "log1p" in ad.uns:
            X = ad.X.toarray() if sparse.issparse(ad.X) else np.asarray(ad.X)
            ad.X = sparse.csr_matrix(np.expm1(X))
            ad.uns.pop("log1p", None)
        ad = ad[:, inter].copy()
        ad.obs["study"] = s
        ad.obs["cell_type_old"] = ad.obs["cell_type"].astype(str)
        if ad.n_obs > MAX_PER_STUDY:
            idx = ad.obs.sample(MAX_PER_STUDY, random_state=0).index
            ad = ad[idx].copy()
            log(f"[subsample] {s}: -> {ad.n_obs}")
        else:
            log(f"[subsample] {s}: kept {ad.n_obs} (<= {MAX_PER_STUDY})")
        parts.append(ad)

    adata = sc.concat(parts, join="inner", label="study", index_unique=None)
    adata.var_names_make_unique()  # safety (inter genes already unique)
    # BUGFIX 2026-09-21: sc.concat received a LIST, so `label="study"` produced
    # integer codes 0/1/2 instead of the cohort names -> the cohort identity was
    # lost in obs and every downstream table showed "study 0/1/2". Map the codes
    # back to the FILES keys (list order == FILES order) so obs['study'] always
    # carries real study names.
    _names = list(FILES.keys())
    adata.obs["study"] = pd.Categorical(
        [_names[int(i)] for i in adata.obs["study"].astype(str)],
        categories=_names,
    )
    log(f"[concat] {adata.n_obs} cells x {adata.n_vars} genes; "
        f"studies={adata.obs['study'].value_counts().to_dict()}")

    # ---- harmonypy integration ----
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # keep a full-gene (all-intersection-genes) log-normalized copy for the final
    # object's .X and .raw; downstream scale/PCA/harmony use the HVG subset only
    # (scaling all ~18.8k genes over 96k cells would OOM the sandbox/most laptops).
    adata_full = adata.copy()
    adata.raw = adata.copy()
    sc.pp.highly_variable_genes(adata, batch_key="study", flavor="seurat_v3",
                                n_top_genes=3000, inplace=True)
    log(f"[hvg] {int(adata.var['highly_variable'].sum())} highly variable genes")
    adata = adata[:, adata.var["highly_variable"]].copy()
    log("[scale] start")
    sc.pp.scale(adata, max_value=10)
    log("[scale] done")
    log("[pca] start")
    sc.tl.pca(adata, n_comps=50, svd_solver="randomized", random_state=0)
    log("[pca] done")
    log("[harmony] start")
    import harmonypy as hm
    import pandas as pd
    pcs = np.asarray(adata.obsm["X_pca"])              # (n_obs, n_comps)
    log(f"[harmony] X_pca {pcs.shape}; n_obs {adata.n_obs}")
    # harmonypy.run_harmony expects data_mat of shape (n_comps, n_obs) and a
    # DataFrame meta_data with the covariate column(s) named in vars_use.
    m_h = pcs.T                                         # (n_comps, n_obs)
    meta = pd.DataFrame({"study": np.asarray(adata.obs["study"].values)})
    ho = hm.run_harmony(m_h, meta, vars_use=["study"])
    Z = np.asarray(ho.Z_corr)                           # (n_comps, n_obs)
    if Z.ndim == 1:                                    # defensive reshape
        Z = Z.reshape(pcs.shape[1], adata.n_obs)
    Zout = Z if Z.shape[0] == adata.n_obs else Z.T      # ensure (n_obs, n_comps)
    adata.obsm["X_pca_harmony"] = Zout
    log(f"[harmony] done; X_pca_harmony {Zout.shape}")
    sc.pp.neighbors(adata, use_rep="X_pca_harmony", n_neighbors=30, random_state=0)
    sc.tl.umap(adata, random_state=0)
    sc.tl.leiden(adata, resolution=0.8, key_added="leiden_int", random_state=0)
    log(f"[leiden] {adata.obs['leiden_int'].nunique()} clusters")

    # ---- re-annotate (unified marker panel) ----
    sc.tl.rank_genes_groups(adata, groupby="leiden_int", method="wilcoxon",
                            n_genes=100, use_raw=False)
    assign = {}
    for cl in adata.obs["leiden_int"].cat.categories:
        # Use the official helper (robust across scanpy versions). Newer scanpy
        # (>=1.10) stores rank_genes_groups results as a structured array
        # indexed by GROUP NAME, so rg["names"][int_pos] raises
        # "too many indices for array" -> avoid poking uns directly.
        df = sc.get.rank_genes_groups_df(adata, group=cl)
        top = set(df["names"].head(50).tolist())
        scores = {ct: len(top & set(m)) for ct, m in MARKERS.items()}
        best = max(scores, key=scores.get)
        assign[cl] = best if scores[best] > 0 else "Unknown"
    adata.obs["cell_type_int"] = adata.obs["leiden_int"].map(assign).astype(str)
    # also store overlap view
    adata.obs["cell_type_overlap"] = adata.obs["cell_type_int"]
    log("[annotate] unified cell types:")
    log("  " + str(adata.obs["cell_type_int"].value_counts().to_dict()))

    # ---- restore full-gene expression matrix + preserve .raw ----
    # adata currently holds only the 3000 HVG (required for scale/PCA/harmony);
    # re-attach the all-genes log-normalized matrix from adata_full, keeping the
    # computed obs annotations (study/cell_type_int/leiden_int) and embeddings.
    full = adata_full.copy()
    full.obs = adata.obs.copy()
    full.obsm = adata.obsm.copy()
    full.uns = adata.uns.copy()
    adata = full
    adata.raw = adata.copy()   # .raw = all-genes log-normalized
    log(f"[restore] final X = {adata.n_vars} genes (all intersection genes); "
        f".raw preserved ({adata.raw.shape})")

    # ---- REP score (pan-subtype 52-gene), now on the full-gene matrix ----
    rep = pd.read_csv(REP_CSV)
    rep_genes = [g for g in rep["gene"].tolist() if g in adata.var_names]
    log(f"[REP] {len(rep_genes)}/{len(rep)} REP genes present; scoring...")
    sc.tl.score_genes(adata, rep_genes, score_name="REP_score")

    # ---- tables ----
    comp = (adata.obs.groupby(["study", "cell_type_int"]).size()
            .unstack(fill_value=0))
    comp_pct = comp.div(comp.sum(axis=1), axis=0) * 100
    comp_pct.to_csv(os.path.join(RES, "composition_pct.csv"))
    comp.to_csv(os.path.join(RES, "composition_count.csv"))

    rep_tab = (adata.obs.groupby(["study", "cell_type_int"])["REP_score"]
               .agg(["mean", "median", "count"]).reset_index())
    rep_tab.to_csv(os.path.join(RES, "rep_by_celltype.csv"), index=False)

    # ---- summary ----
    with open(os.path.join(RES, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("Harmony three-cohort breast-cancer scRNA-seq integration\n")
        f.write("GSE264205 source: reload/GSE264205_fixed.h5ad (gene-aligned rebuild)\n")
        f.write(f"gene intersection: {len(inter)}\n")
        f.write(f"max_per_study (subsample cap): {MAX_PER_STUDY}\n")
        f.write(f"total cells integrated: {adata.n_obs}\n")
        f.write(f"per study: {adata.obs['study'].value_counts().to_dict()}\n")
        f.write(f"leiden clusters: {adata.obs['leiden_int'].nunique()}\n")
        f.write(f"REP genes used: {len(rep_genes)}/{len(rep)}\n")
        f.write("\ncell_type_int composition (%):\n")
        f.write(comp_pct.round(1).to_string() + "\n")
        f.write("\nREP_score mean by study x cell_type:\n")
        f.write(rep_tab.round(3).to_string(index=False) + "\n")
        f.write("\nNOTE: GSE264205 Pre/Post composition is severely skewed "
                "(Monocyte 8.8%->91.3%); treat its cell-type proportions as "
                "tissue-source confounded, not pure treatment effect.\n")

    adata.write(os.path.join(RES, "integrated.h5ad"))
    log(f"[done] wrote {os.path.join(RES, 'integrated.h5ad')}")

    if ENABLE_PLOTS:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sc.pl.umap(adata, color=["study", "cell_type_int", "REP_score"],
                   ncols=1, show=False,
                   save="_overview.png", wspace=0.4)
        log("[plots] overview umap saved")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
