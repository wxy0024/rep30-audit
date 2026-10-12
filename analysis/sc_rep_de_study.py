"""
Within-cell-type DE of high- vs low-REP cells, for an arbitrary study in
integrated.h5ad.  Replicates the GSE161529 analysis on another queue so the
REP resistance program can be cross-validated.

Run with managed venv (no harmonypy needed):
  python analysis/sc_rep_de_study.py --study 1 --gse GSE249361

Outputs to results/sc/integrated/rep_de/<gse>/
  de_<CellType>.csv          (top-50 DE genes, with module + in_REP52 tags)
  heatmap_<CellType>.png     (scaled expr, low|high blocks)
  summary.txt
  overlap_vs_GSE161529.csv   (only if GSE161529 DE present at rep_de/ top level)
"""
import os, argparse, numpy as np, pandas as pd, scanpy as sc

ENABLE_PLOTS = os.environ.get("ENABLE_PLOTS", "1") != "0"
ROOT = r"E:/01.WorkSpace/01.Breast_cancer_gene_panel/resistance_signature"
RES  = os.path.join(ROOT, "results/sc/integrated")

# biologically meaningful gene modules for tagging DE hits (broader than
# narrow canonical pathways, so the real resistance-relevant signal is captured)
MODULES = {
    "Proliferation_Translation": ["RPL3","RPL4","RPL5","RPL6","RPL7","RPL7A","RPL9","RPL13","RPL14",
        "RPL18","RPL18A","RPL23","RPL24","RPL26","RPL28","RPL29","RPL30","RPL34","RPL35A","RPL36",
        "RPL37","RPL37A","RPL39","RPS3","RPS3A","RPS4X","RPS6","RPS7","RPS8","RPS9","RPS11","RPS12",
        "RPS18","RPS23","RPS24","RPS26","RPS27A","RPS28","EEF1A1","EEF1B2","EEF2","TPT1","NPM1",
        "FAU","RACK1","PFDN5","CCT5","CCT8","ANAPC16","MKI67","PCNA"],
    "Basal_Stem": ["KRT5","KRT14","KRT6A","KRT6B","KRT17","KRT15","TP63","NOTCH1",
        "NOTCH3","CD44","ALDH1A1"],
    "Luminal_Keratin": ["KRT8","KRT18","KRT19","KRT7","KRT20"],
    "Stress_DrugTolerance": ["GSTP1","CRYAB","SLPI","HSPB1","HSPA1A","HSPA5","SOD2","TXN",
        "CYP1B1","HSP90AA1","HSPD1"],
    "CAF_niche": ["PDGFRA","CXCL12","ACTA2","COL1A1","COL1A2","FAP","S100A4","VIM",
        "TGFB1","TGFBI","COL3A1"],
    "Invasion_Matrix": ["MMP7","MMP9","MMP2","MMP1","ANXA1","ANXA2","MFGE8","S100A11",
        "S100A10","LGALS3","SPARC","FN1","MMP14"],
    "Wnt": ["SFRP1","WNT5A","CTNNB1","WNT11"],
    "Onco_Prolif": ["YBX1","YBX3","MYC","CCND1","CCNE1","CDK1","CDK4"],
}
MODULE_UPPER = {m: {g.upper() for g in gs} for m, gs in MODULES.items()}

# REP 52-gene set (for tagging)
rep_csv = os.path.join(ROOT, "results/rep_genes.csv")
REP52 = set(pd.read_csv(rep_csv)["gene"].tolist()) if os.path.exists(rep_csv) else set()

def tag_pathways(gene):
    g = str(gene).upper()
    return ";".join(m for m, S in MODULE_UPPER.items() if g in S)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, help='study index string, e.g. "1"')
    ap.add_argument("--gse", required=True, help='GSE id, e.g. GSE249361')
    ap.add_argument("--cells", default="Epithelial/Malignant,Fibroblast/CAF",
                    help="comma-separated cell types to stratify")
    ap.add_argument("--qhi", type=float, default=0.75, help="top-quartile cutoff (0-1)")
    args = ap.parse_args()

    TARGET_STUDY = args.study
    GSE = args.gse
    TARGET_CTS = [c.strip() for c in args.cells.split(",")]
    Q_HI = args.qhi
    Q_LO = 1.0 - Q_HI

    OUT = os.path.join(RES, "rep_de", GSE)
    os.makedirs(OUT, exist_ok=True)
    logf = os.path.join(OUT, "run.log")
    def log(*a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        with open(logf, "a") as f:
            f.write(line + "\n")

    log(f"[start] study={TARGET_STUDY} gse={GSE} cells={TARGET_CTS} qhi={Q_HI}")

    ad = sc.read_h5ad(os.path.join(RES, "integrated.h5ad"))
    log(f"[load] {ad.n_obs} cells x {ad.n_vars} genes; REP_score={ 'REP_score' in ad.obs }")
    log("[study levels] " + ", ".join(str(x) for x in ad.obs["study"].unique()))

    sub = ad[ad.obs["study"] == TARGET_STUDY].copy()
    log(f"[subset] -> {sub.n_obs} cells")
    if sub.n_obs == 0:
        log("[ERROR] no cells for this study; check --study value"); return

    summ, de_frames = [], {}
    for ct in TARGET_CTS:
        s = sub[sub.obs["cell_type_int"] == ct].copy()
        if s.n_obs < 200:
            log(f"[skip] {ct}: only {s.n_obs} cells"); continue
        rep = s.obs["REP_score"]
        hi = rep >= rep.quantile(Q_HI)
        lo = rep <= rep.quantile(Q_LO)
        s.obs["rep_group"] = np.where(hi, "high", np.where(lo, "low", "mid"))
        s = s[s.obs["rep_group"].isin(["high", "low"])].copy()
        log(f"[{ct}] high={int((s.obs['rep_group']=='high').sum())} "
            f"low={int((s.obs['rep_group']=='low').sum())} (quartile split)")

        sc.tl.rank_genes_groups(s, groupby="rep_group", groups=["high"],
                                reference="low", method="wilcoxon",
                                use_raw=False, n_genes=50)
        df = sc.get.rank_genes_groups_df(s, group="high")
        df = df.rename(columns={"names": "gene", "logfoldchanges": "logFC",
                                "pvals": "pval", "pvals_adj": "padj"})
        df["module"] = df["gene"].map(tag_pathways).fillna("")
        df["in_REP52"] = df["gene"].isin(REP52)
        df = df.sort_values("padj").reset_index(drop=True)
        out_csv = os.path.join(OUT, f"de_{ct.replace('/', '_')}.csv")
        df.to_csv(out_csv, index=False)
        de_frames[ct] = df
        log(f"[{ct}] DE -> {out_csv} (top: {', '.join(df['gene'].head(8).tolist())})")

        sig = df[df["padj"] < 0.05]
        pw_counts = {m: int(sig["module"].str.contains(m).sum()) for m in MODULES}
        n_rep52 = int(sig["in_REP52"].sum())
        summ.append((ct, s.n_obs, len(sig), n_rep52, pw_counts))

        if ENABLE_PLOTS:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            top = [g for g in df["gene"].head(25).tolist() if g in s.var_names]
            hs = s[s.obs["rep_group"] == "high"]; ls = s[s.obs["rep_group"] == "low"]
            nh = min(400, hs.n_obs); nl = min(400, ls.n_obs)
            hs = hs[np.random.RandomState(0).choice(hs.n_obs, nh, replace=False)]
            ls = ls[np.random.RandomState(1).choice(ls.n_obs, nl, replace=False)]
            hm = sc.concat([hs, ls], label="rep_group", keys=["high", "low"])
            sc.pp.scale(hm, max_value=3)
            Xs = np.asarray(hm.X)
            genes = list(hm.var_names)
            present = [g for g in top if g in genes]
            gi = [genes.index(g) for g in present]
            low_idx = np.where(hm.obs["rep_group"].values == "low")[0]
            high_idx = np.where(hm.obs["rep_group"].values == "high")[0]
            cols = np.concatenate([low_idx, high_idx])
            mat = Xs[:, gi].T
            mat = mat[:, cols]
            fig, ax = plt.subplots(figsize=(7, max(6, len(present) * 0.32)))
            im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5)
            ax.set_yticks(range(len(present))); ax.set_yticklabels(present, fontsize=7)
            ax.set_xticks([])
            ax.set_title(f"{ct}: high- vs low-REP ({GSE}, scaled expr)")
            ax.axvline(len(low_idx) - 0.5, color="k", lw=1)
            plt.colorbar(im, ax=ax, label="scaled expr")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT, f"heatmap_{ct.replace('/', '_')}.png"), dpi=130)
            plt.close()
            log(f"[{ct}] heatmap saved ({len(present)} genes)")

    # ---- summary ----
    log("\n==== summary (high- vs low-REP, " + GSE + ") ====")
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write(f"Within-cell-type DE: high- vs low-REP ({GSE} / study {TARGET_STUDY})\n")
        f.write(f"REP52 genes available for tagging: {len(REP52)}\n\n")
        for ct, n, nsig, n52, pw in summ:
            line = (f"{ct} (n={n}): {nsig} DE genes padj<0.05; "
                    f"{n52} in REP52; pathways: " +
                    ", ".join(f"{p}={c}" for p, c in pw.items() if c > 0) + "\n")
            log(line); f.write(line)

    # ---- cross-study overlap vs GSE161529 (top-level rep_de/ de_*.csv) ----
    ref_dir = os.path.join(RES, "rep_de")
    ref_tag = "GSE161529"
    rows = []
    have_ref = False
    for ct in de_frames:
        ref_csv = os.path.join(ref_dir, f"de_{ct.replace('/', '_')}.csv")
        if not os.path.exists(ref_csv):
            continue
        have_ref = True
        rdf = pd.read_csv(ref_csv)
        cur_sig = set(de_frames[ct][de_frames[ct]["padj"] < 0.05]["gene"])
        ref_sig = set(rdf[rdf["padj"] < 0.05]["gene"])
        if not cur_sig or not ref_sig:
            continue
        inter = cur_sig & ref_sig
        union = cur_sig | ref_sig
        jac = len(inter) / len(union) if union else 0.0
        cur_rep52 = set(de_frames[ct][de_frames[ct]["in_REP52"]]["gene"])
        ref_rep52 = set(rdf[rdf["in_REP52"]]["gene"])
        top10_cur = set(de_frames[ct]["gene"].head(10))
        top10_ref = set(rdf["gene"].head(10))
        rows.append({
            "cell_type": ct,
            "n_sig_current": len(cur_sig),
            "n_sig_ref": len(ref_sig),
            "overlap_genes": len(inter),
            "jaccard": round(jac, 3),
            "overlap_list": ";".join(sorted(inter)) if inter else "",
            "shared_REP52": ";".join(sorted(cur_rep52 & ref_rep52)) if (cur_rep52 & ref_rep52) else "",
            "shared_top10": ";".join(sorted(top10_cur & top10_ref)) if (top10_cur & top10_ref) else "",
        })
    if have_ref and rows:
        ov = pd.DataFrame(rows)
        ov.to_csv(os.path.join(OUT, f"overlap_vs_{ref_tag}.csv"), index=False)
        log(f"\n==== overlap vs {ref_tag} ====")
        for _, r in ov.iterrows():
            log(f"[{r['cell_type']}] jaccard={r['jaccard']} "
                f"overlap={r['overlap_genes']}/{r['n_sig_current']} cur / {r['n_sig_ref']} ref "
                f"| shared_REP52={r['shared_REP52']} | shared_top10={r['shared_top10']}")
    log("DONE")

if __name__ == "__main__":
    main()
