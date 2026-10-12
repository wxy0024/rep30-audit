"""
GSE264205 resistance-evolution analysis (single-cell, pre vs post treatment).

GSE264205 is a longitudinal breast-cancer scRNA-seq cohort (treatment-naive
baseline + on-/post-treatment timepoints) -- the study most directly relevant
to DRUG-RESISTANCE evolution in this project. This script consumes the
already-cleaned AnnData produced by sc_pipeline.py and answers:

  1. Cell-composition shift: which cell types expand / contract from Pre -> Post
     (per-sample proportions, two-proportion z-test Pre vs pooled-Post).
  2. Differential expression Pre vs Post, computed WITHIN each major cell type
     (controls for composition changes) via scanpy wilcoxon.
  3. Resistance focus: overlay the 29-gene resistance panel + REP genes +
     curated resistance pathways (PI3K/AKT, cell cycle, EMT, immune-checkpoint,
     MDR efflux) on the DE results -> which resistance programs are gained/lost
     along treatment, and in which cellular compartment.

Outputs (results/sc/GSE264205/resistance_evolution/):
  phase_map.csv            sample -> inferred Pre/Post phase + raw GEO text
  composition_shift.csv     per-cell-type Pre% vs Post% + z-test p
  composition_bar.png
  de_<CellType>.csv         top DE genes (Post vs Pre) per major type
  resistance_focus.csv      panel/pathway genes x cell-type (logFC, padj)
  resistance_heatmap.png    logFC heatmap (cell types x resistance genes)
  volcano_<CellType>.png    DE volcano highlighting resistance genes
  summary.txt

Usage:  python sc_resistance_evolution.py
"""
import os, re, glob, gzip, warnings
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
warnings.filterwarnings("ignore")

# ---- plotting is optional: matplotlib.pyplot import is blocked under some
#      sandboxes (SIGKILL). Set ENABLE_PLOTS=0 to skip figures and still get
#      all CSV / summary outputs. On a normal machine (Miniforge Prompt) it
#      defaults to on. ----
ENABLE_PLOTS = os.environ.get("ENABLE_PLOTS", "1") != "0"
plt = None
if ENABLE_PLOTS:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as _e:  # pragma: no cover - only hits when backend is blocked
        print(f"[warn] matplotlib.pyplot unavailable -> plots skipped ({_e})")
        plt = None

ROOT = r"E:\01.WorkSpace\01.Breast_cancer_gene_panel\resistance_signature"
H5 = os.path.join(ROOT, "results/sc/GSE264205/GSE264205.h5ad")
SERIES_DIR = os.path.join(ROOT, "data/03_single_cell/GSE264205")
RES = os.path.join(ROOT, "results/sc/GSE264205/resistance_evolution")
os.makedirs(RES, exist_ok=True)

RESIST_CSV = os.path.join(ROOT, "resistance_genes.csv")
REP_CSV = os.path.join(ROOT, "results/rep_genes.csv")

MAJOR_TYPES = ["Epithelial/Malignant", "T cell CD8", "T cell CD4", "Macrophage",
               "Monocyte", "Fibroblast/CAF", "Plasma cell", "B cell", "NK cell",
               "Dendritic cell", "Endothelial", "Mast cell"]

# ---- curated resistance pathway genes (uppercase symbols) ----
PATHWAYS = {
    "PI3K-AKT": ["PIK3CA", "PIK3R1", "PIK3R2", "AKT1", "AKT2", "AKT3", "PTEN", "MTOR"],
    "CellCycle": ["CCND1", "CCND2", "CCNE1", "CDK2", "CDK4", "CDK6", "RB1", "CDKN2A", "MYC", "FZR1"],
    "EMT": ["CDH1", "CDH2", "SNAI1", "SNAI2", "TWIST1", "ZEB1", "VIM"],
    "ImmuneCheckpoint": ["PDCD1", "PDCD1LG2", "CTLA4", "HAVCR2", "LAG3", "TIGIT"],
    "MDRefflux": ["ABCB1", "ABCG2", "ABCC1"],
    "HRpathway": ["ESR1", "PGR", "GATA3", "FOXA1"],
}


# ---------------------------------------------------------------- phase parsing
def infer_phase_map(series_dir, present_samples):
    """Infer integer-sample -> phase from the RAW data dir.

    sc_pipeline.py concatenated per-sample AnnData with sc.concat(label='sample',
    keys=None), which OVERWRITES the sample column with the list POSITION
    (0,1,2,...) of `sorted(glob.glob('*matrix.mtx*'))`. So the h5ad's integer
    sample id i == i-th sorted matrix file == the filename token. We recover the
    phase directly from the directory (a GEO series matrix would carry GSM ids
    that do NOT match these integer ids, so it cannot be used directly here).

    Filename token: 'Pre' -> Pre (treatment-naive baseline);
                   'Post*' -> Post (on/post-treatment);
                   'Meta*' -> metastatic/relapsed lesion -> folded into Post pool
                              for the naive-vs-evolved contrast, but recorded
                              separately so the assumption is auditable.

    Returns (phase_map, raw_token, raw_name) keyed by the integer sample string.
    """
    mats = sorted(glob.glob(os.path.join(series_dir, "*matrix.mtx.gz")) +
                  glob.glob(os.path.join(series_dir, "*matrix.mtx")))
    token_of, name_of = {}, {}
    for i, m in enumerate(mats):
        nm = os.path.basename(m).lower()
        if re.search(r"(^|[-_])pre", nm):
            tk = "Pre"
        elif re.search(r"(^|[-_])post", nm):
            tk = "Post"
        elif "meta" in nm:
            tk = "Meta"
        else:
            tk = "Unknown"
        token_of[i] = tk
        name_of[i] = os.path.basename(m)
    phase_map, raw_token, raw_name = {}, {}, {}
    for s in present_samples:
        si = int(s)
        tk = token_of.get(si, "Unknown")
        # Meta folded into Post for the naive-vs-evolved contrast
        phase_map[s] = "Post" if tk in ("Post", "Meta") else tk
        raw_token[s] = tk
        raw_name[s] = name_of.get(si, "")
    return phase_map, raw_token, raw_name


# ---------------------------------------------------------------- two-prop z-test
def two_prop_z(n1, x1, n2, x2):
    """Returns (z, two-sided p). x = successes, n = trials."""
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p2 - p1) / se
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


def main():
    print(f"[load] {H5}")
    ad = sc.read_h5ad(H5)
    print(f"  {ad.n_obs} cells x {ad.n_vars} genes; samples={ad.obs['sample'].nunique()}")

    # ---- infer Pre/Post phase from raw data dir (integer sample ids) ----
    present = list(ad.obs["sample"].unique())
    phase_map, raw_token, raw_name = infer_phase_map(SERIES_DIR, present)
    order_map = {s: (0 if phase_map[s] == "Pre" else 1) for s in present}
    # save phase_map.csv (auditable: integer id, inferred phase, raw filename token)
    pm_rows = []
    for s in sorted(present, key=lambda x: int(x)):
        pm_rows.append({"sample": s, "phase": phase_map[s],
                        "raw_token": raw_token[s], "filename": raw_name[s],
                        "n_cells": int((ad.obs["sample"] == s).sum())})
    pd.DataFrame(pm_rows).to_csv(os.path.join(RES, "phase_map.csv"), index=False)
    print("[phase] inferred from data-dir filename tokens (sc_pipeline concat order):")
    for r in pm_rows:
        print(f"    sample {r['sample']}: {r['phase']}  (raw='{r['raw_token']}')  "
              f"cells={r['n_cells']}  <- {r['filename']}")

    ad.obs["phase"] = ad.obs["sample"].map(phase_map).astype(str)
    ad.obs["phase_order"] = ad.obs["sample"].map(order_map).astype(int)
    pre_samples = [s for s in present if phase_map[s] == "Pre"]
    post_samples = [s for s in present if phase_map[s] == "Post"]
    print(f"[phase] Pre samples ({len(pre_samples)}): {pre_samples}")
    print(f"[phase] Post samples ({len(post_samples)}): {post_samples}")
    if not pre_samples or not post_samples:
        print("[FATAL] need >=1 Pre and >=1 Post sample; abort")
        return

    # ---------------------------------------------------------------- 1. composition shift
    ct = pd.crosstab(ad.obs["phase"], ad.obs["cell_type"])
    prop = ct.div(ct.sum(axis=1), axis=0) * 100.0
    rows = []
    for ct_ in ct.columns:
        if ct_ not in MAJOR_TYPES and ct_ == "Unknown":
            pass
        n_pre = int(ct.loc["Pre", ct_]) if "Pre" in ct.index else 0
        n_post = int(ct.loc["Post", ct_]) if "Post" in ct.index else 0
        tot_pre = int(ct.loc["Pre"].sum()) if "Pre" in ct.index else 0
        tot_post = int(ct.loc["Post"].sum()) if "Post" in ct.index else 0
        p_pre = n_pre / tot_pre * 100 if tot_pre else 0.0
        p_post = n_post / tot_post * 100 if tot_post else 0.0
        z, pval = two_prop_z(tot_pre, n_pre, tot_post, n_post)
        rows.append({"cell_type": ct_, "n_pre": n_pre, "n_post": n_post,
                     "pct_pre": round(p_pre, 2), "pct_post": round(p_post, 2),
                     "delta_pct": round(p_post - p_pre, 2),
                     "fold_change": round((p_post / p_pre) if p_pre > 0 else np.nan, 2),
                     "z": round(z, 2), "p_two_prop": pval})
    comp = pd.DataFrame(rows).sort_values("delta_pct", key=lambda s: s.abs(), ascending=False)
    comp.to_csv(os.path.join(RES, "composition_shift.csv"), index=False)
    print("\n[composition shift] top expanding/contracting cell types:")
    print(comp.head(12).to_string(index=False))

    # composition bar plot
    if plt is not None:
        order_types = comp["cell_type"].tolist()
        x = np.arange(len(order_types))
        pre_p = comp["pct_pre"].values
        post_p = comp["pct_post"].values
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - 0.2, pre_p, 0.4, label="Pre", color="#1f77b4")
        ax.bar(x + 0.2, post_p, 0.4, label="Post", color="#d95f02")
        ax.set_xticks(x); ax.set_xticklabels(order_types, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("% of cells"); ax.set_title("GSE264205 cell-type composition: Pre vs Post")
        ax.legend()
        plt.tight_layout(); plt.savefig(os.path.join(RES, "composition_bar.png"), dpi=130); plt.close()
    else:
        print("[plot] composition_bar.png skipped (ENABLE_PLOTS=0)")

    # ---------------------------------------------------------------- 2+3. DE per cell type + resistance focus
    resist = pd.read_csv(RESIST_CSV)
    panel_genes = [str(g).upper() for g in resist["gene"].tolist()]
    rep_genes = []
    if os.path.exists(REP_CSV):
        rep = pd.read_csv(REP_CSV)
        if "gene" in rep.columns:
            rep_genes = [str(g).upper() for g in rep["gene"].tolist()]
    pathway_genes = sorted({g.upper() for v in PATHWAYS.values() for g in v})
    focus_set = set(panel_genes) | set(rep_genes) | set(pathway_genes)

    focus_rows = []  # gene, pathway, cell_type, logfc, padj, present
    n_pre_total = int((ad.obs["phase"] == "Pre").sum())
    n_post_total = int((ad.obs["phase"] == "Post").sum())

    for ctype in MAJOR_TYPES:
        sub = ad[ad.obs["cell_type"] == ctype]
        if sub.n_obs < 30:
            continue
        n_pre_c = int((sub.obs["phase"] == "Pre").sum())
        n_post_c = int((sub.obs["phase"] == "Post").sum())
        if n_pre_c < 10 or n_post_c < 10:
            print(f"[DE] {ctype}: skip (pre={n_pre_c}, post={n_post_c} < 10)")
            continue
        sc.tl.rank_genes_groups(sub, groupby="phase", groups=["Post"],
                                reference="Pre", method="wilcoxon",
                                use_raw=False, n_genes=min(sub.n_vars, 200))
        rg = sub.uns["rank_genes_groups"]
        de = pd.DataFrame({
            "gene": rg["names"]["Post"],
            "logfoldchanges": rg["logfoldchanges"]["Post"],
            "pvals": rg["pvals"]["Post"],
            "pvals_adj": rg["pvals_adj"]["Post"],
            "pct_nz": rg.get("pct_nz", {}).get("Post", [np.nan] * len(rg["names"]["Post"])),
        })
        de["logFC"] = de["logfoldchanges"]  # scanpy already returns log2 fold-change
        de = de.sort_values("pvals_adj").head(50)
        de.insert(0, "cell_type", ctype)
        de.to_csv(os.path.join(RES, f"de_{ctype.replace('/', '_')}.csv"), index=False)
        print(f"[DE] {ctype}: pre={n_pre_c} post={n_post_c}, top gene={de.iloc[0]['gene']} "
              f"logFC={de.iloc[0]['logFC']:.2f} padj={de.iloc[0]['pvals_adj']:.1e}")

        # volcano
        full = pd.DataFrame({
            "gene": rg["names"]["Post"],
            "logFC": rg["logfoldchanges"]["Post"],
            "padj": rg["pvals_adj"]["Post"],
        })
        if plt is not None:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(full["logFC"], -np.log10(full["padj"].clip(lower=1e-300)),
                       s=4, c="gray", alpha=0.4)
            hi = full[(full["padj"] < 0.05) & (full["logFC"].abs() > 0.5)]
            ax.scatter(hi["logFC"], -np.log10(hi["padj"].clip(lower=1e-300)),
                       s=6, c="firebrick", alpha=0.6)
            # label resistance genes
            for _, r in full.iterrows():
                if str(r["gene"]).upper() in focus_set and r["padj"] < 0.05:
                    ax.annotate(r["gene"], (r["logFC"], -np.log10(r["padj"])),
                                fontsize=7, alpha=0.8)
            ax.axhline(-np.log10(0.05), color="k", ls="--", lw=0.6)
            ax.axvline(0, color="k", ls="--", lw=0.6)
            ax.set_xlabel("log2 FC (Post/Pre)"); ax.set_ylabel("-log10 padj")
            ax.set_title(f"GSE264205 DE: {ctype} (Post vs Pre)")
            plt.tight_layout()
            plt.savefig(os.path.join(RES, f"volcano_{ctype.replace('/', '_')}.png"), dpi=120)
            plt.close()

        # focus genes
        for g in focus_set:
            if g in set(full["gene"].astype(str).str.upper()):
                rec = full[full["gene"].astype(str).str.upper() == g].iloc[0]
                pw = [k for k, v in PATHWAYS.items() if g in v]
                pwlabel = ";".join(pw) if pw else ""
                if g in panel_genes:
                    pwlabel = (pwlabel + ";PANEL").strip(";")
                if g in rep_genes:
                    pwlabel = (pwlabel + ";REP").strip(";")
                focus_rows.append({"gene": g, "pathway": pwlabel, "cell_type": ctype,
                                   "logFC": round(float(rec["logFC"]), 3),
                                   "padj": float(rec["padj"]),
                                   "sig": bool(rec["padj"] < 0.05)})

        focus = pd.DataFrame(focus_rows)
    if not focus.empty:
        focus = focus.sort_values(["cell_type", "padj"])
        focus.to_csv(os.path.join(RES, "resistance_focus.csv"), index=False)
        # heatmap cell_type x focus gene (logFC)
        if plt is not None:
            piv = focus.pivot_table(index="gene", columns="cell_type", values="logFC")
            fig, ax = plt.subplots(figsize=(max(8, piv.shape[1] * 0.6), max(6, piv.shape[0] * 0.35)))
            im = ax.imshow(piv.values, aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5)
            ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=7)
            plt.colorbar(im, ax=ax, label="log2 FC (Post/Pre)")
            ax.set_title("GSE264205 resistance-gene program shift (Pre -> Post)")
            plt.tight_layout()
            plt.savefig(os.path.join(RES, "resistance_heatmap.png"), dpi=130); plt.close()
        else:
            print("[plot] resistance_heatmap.png skipped (ENABLE_PLOTS=0)")
        print("\n[resistance focus] panel/pathway genes changing Pre->Post (padj<0.05):")
        print(focus[focus["sig"]].to_string(index=False))

    # ---------------------------------------------------------------- summary
    with open(os.path.join(RES, "summary.txt"), "w") as f:
        f.write("GSE264205 resistance-evolution summary\n")
        f.write(f"Pre samples: {pre_samples}\nPost samples: {post_samples}\n")
        f.write(f"cells Pre={n_pre_total}, Post={n_post_total}\n")
        f.write("NOTE: sample ids are sc_pipeline concat positions (0..3) matching "
                "sorted glob of *matrix.mtx* in the raw dir.\n")
        f.write("  Pre=treatment-naive baseline; Post1/Post2=on/post-treatment; "
                "Meta=metastatic/relapsed lesion folded into Post pool (443 cells, ~2.7%).\n")
        f.write("  Re-run with Meta excluded if a lesion-type confounder is a concern.\n\n")
        f.write("== Cell-composition shift (Pre vs Post) ==\n")
        f.write(comp.to_string(index=False) + "\n\n")
        if not focus.empty:
            f.write("== Resistance genes shifted (padj<0.05) ==\n")
            f.write(focus[focus["sig"]].to_string(index=False) + "\n")
    print(f"\n[DONE] outputs in {RES}")


if __name__ == "__main__":
    main()
