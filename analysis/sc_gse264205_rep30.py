"""
GSE264205 single-cell REP30 Pre-vs-Post control (within-compartment).

Context / why this script looks the way it does
------------------------------------------------
GSE264205 is the one longitudinal scRNA-seq cohort in this project with paired
baseline + on/post-treatment samples. It was originally expected to support a
"REP30 high/low along treatment" comparison inside malignant cells. A direct
probe of the cleaned AnnData shows that is NOT feasible:

  sample 0 (Pre)   : 2453 malignant / 151 CAF / 486  monocyte
  sample 1 (Post1) :    1 malignant / 521 CAF / 3309 monocyte
  sample 2 (Post2) :   16 malignant /   2 CAF / 6575 monocyte
  sample 3 (Meta)  :  406 malignant /   0 CAF /   37 monocyte

Post1/Post2 are monocyte-dominated and carry essentially ZERO malignant cells
-- i.e. the treatment timepoints sampled a different tissue compartment. So a
within-malignant Pre-vs-Post contrast is impossible; the only post-like group
that still contains malignant cells is Meta (a metastatic/relapsed lesion,
n=406). We therefore:

  1. Quantify the compartment shift (leiden x sample, cell_type x sample) and
     report it explicitly so the confound is on the record.
  2. Within MALIGNANT (Epithelial/Malignant): Pre (s0) vs Meta (s3). This is
     the only malignant-cell contrast available; it is CONFOUNDED by lesion
     type (primary vs metastasis), so it is reported as such, not as a clean
     treatment effect.
  3. Within CAF:  Pre (s0) vs Post1 (s1).
  4. Within Monocyte: Pre (s0) vs Post (s1+s2) and Pre (s0) vs Meta (s3).
  5. Per comparison: REP30 score distribution (Mann-Whitney U + Cliff's delta
     + median/mean diff), per-gene log-expression delta (which panel genes
     drive it), and a boxplot.

REP30 score = coef-weighted sum of per-gene z-scores (z within the compartment
subset), matching the bulk backtest convention (Sigma coef * z).

Outputs (results/sc/GSE264205/rep30_prepost/):
  compartment_composition.csv     sample x cell_type counts + % (the confound)
  rep30_comparisons.csv           one row per compartment Pre-vs-Post contrast
  rep30_pergene_<comp>.csv        per-gene mean-log delta for each contrast
  rep30_by_cell.csv               per-cell score + compartment + group
  box_<comp>.png                  score boxplot per contrast
  summary.txt / interpretation.md

Usage:  python sc_gse264205_rep30.py
        ENABLE_PLOTS=0 python sc_gse264205_rep30.py     # skip figures
"""
import os
import warnings
import numpy as np
import pandas as pd
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
H5 = os.path.join(ROOT, "results/sc/GSE264205/GSE264205.h5ad")
PANEL = os.path.join(ROOT, "results/rep52_trim/rep52_trimmed_panel.csv")
RES = os.path.join(ROOT, "results/sc/GSE264205/rep30_prepost")
os.makedirs(RES, exist_ok=True)

# integer sample id -> biological group (see module docstring)
# NOTE: obs['sample'] is stored as STRING ('0'..'3') in this object -> key by str.
SAMPLE_GROUP = {"0": "Pre", "1": "Post1", "2": "Post2", "3": "Meta"}

# cell_type label used for the malignant compartment in this object
MALIGNANT = "Epithelial/Malignant"

# per-compartment contrast definitions: (compartment cell_type, groupA, groupB)
CONTRASTS = [
    # malignant cells exist only in Pre + Meta -> the one malignant contrast
    (MALIGNANT, "Pre", "Meta"),
    # stromal / immune compartments where Pre AND a real Post exist
    ("Fibroblast/CAF", "Pre", "Post1"),
    ("Monocyte", "Pre", "post"),
    ("Monocyte", "Pre", "Meta"),
]

GROUP_MAP = {
    "Pre": ["Pre"],
    "Post1": ["Post1"],
    "Post2": ["Post2"],
    "Meta": ["Meta"],
    "post": ["Post1", "Post2"],   # pooled treatment timepoints
}


def cliffs_delta(a, b):
    """Vectorised Cliff's delta (a vs b). Positive => a tends larger."""
    if len(a) == 0 or len(b) == 0:
        return np.nan
    diff = a[:, None] - b[None, :]
    return float(np.sign(diff).sum()) / (len(a) * len(b))


def score_rep30(sub, panel_df):
    """Return a per-cell REP30 score for `sub` (coef-weighted z-sum)."""
    genes = [g for g in panel_df["gene"].tolist() if g in sub.var_names]
    coef = dict(zip(panel_df["gene"], panel_df["coef"]))
    X = sub[:, genes].X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    X = X.astype(float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    w = np.array([coef[g] for g in genes], dtype=float)
    w = w / np.abs(w).sum()
    return Z @ w, genes


def per_gene_delta(sub, genes, gA_idx, gB_idx):
    X = sub[:, genes].X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    X = X.astype(float)
    rows = []
    for j, g in enumerate(genes):
        a = X[gA_idx, j]
        b = X[gB_idx, j]
        rows.append({"gene": g,
                     "mean_A": round(float(a.mean()), 4),
                     "mean_B": round(float(b.mean()), 4),
                     "delta_B_minus_A": round(float(b.mean() - a.mean()), 4)})
    return pd.DataFrame(rows).sort_values("delta_B_minus_A")


def main():
    print(f"[load] {H5}")
    ad = sc.read_h5ad(H5)
    print(f"  {ad.n_obs} cells x {ad.n_vars} genes")

    # normalisation check (X should be log-normalised; guard otherwise)
    Xmax = float(ad.X.max())
    print(f"  X.max()={Xmax:.2f} -> {'log-normalised' if Xmax < 30 else 'COUNTS?'}")
    if Xmax >= 30:
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
        print("  [fix] applied normalize_total + log1p")

    ad.obs["group"] = ad.obs["sample"].astype(str).map(SAMPLE_GROUP)
    print("[group] value counts:",
          ad.obs["group"].value_counts(dropna=False).to_dict())
    panel_df = pd.read_csv(PANEL)
    panel_df["gene"] = panel_df["gene"].astype(str)

    # ---------------------------------------------------- 1. compartment confound
    ct = pd.crosstab(ad.obs["sample"], ad.obs["cell_type"])
    pct = (ct.div(ct.sum(axis=1), axis=0) * 100).round(1)
    ct.to_csv(os.path.join(RES, "compartment_composition.csv"))
    print("\n[composition] sample x cell_type (counts):")
    print(ct.to_string())
    print("\n[composition] sample x cell_type (%):")
    print(pct.to_string())

    if plt is not None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        pct.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
        ax.set_ylabel("% of cells"); ax.set_xlabel("sample (0=Pre,1=Post1,2=Post2,3=Meta)")
        ax.set_title("GSE264205 compartment shift across samples")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(RES, "compartment_composition.png"), dpi=140)
        plt.close()

    # leiden x sample (batch/compartment structure check)
    lx = pd.crosstab(ad.obs["sample"], ad.obs["leiden"])
    lx.to_csv(os.path.join(RES, "leiden_by_sample.csv"))

    # ---------------------------------------------------- 2. per-compartment REP30
    comp_rows = []
    all_cell = []
    for ctype, gA, gB in CONTRASTS:
        sub = ad[ad.obs["cell_type"] == ctype].copy()
        setA, setB = GROUP_MAP[gA], GROUP_MAP[gB]
        mA = sub.obs["group"].isin(setA).values
        mB = sub.obs["group"].isin(setB).values
        nA, nB = int(mA.sum()), int(mB.sum())
        label = f"{ctype} | {gA} vs {gB}"
        if nA < 10 or nB < 10:
            print(f"[skip] {label}: nA={nA} nB={nB} (<10)")
            comp_rows.append({"compartment": ctype, "group_A": gA, "group_B": gB,
                              "n_A": nA, "n_B": nB, "note": "skipped n<10"})
            continue

        score, genes = score_rep30(sub, panel_df)
        sub.obs["rep30"] = score
        a = score[mA]; b = score[mB]
        try:
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            u, p = np.nan, np.nan
        delta = cliffs_delta(a, b)
        # signed logFC-style delta (mean of z-weighted diff)
        comp_rows.append({
            "compartment": ctype, "group_A": gA, "group_B": gB,
            "n_A": nA, "n_B": nB,
            "median_A": round(float(np.median(a)), 4),
            "median_B": round(float(np.median(b)), 4),
            "delta_median(B-A)": round(float(np.median(b) - np.median(a)), 4),
            "mean_A": round(float(a.mean()), 4),
            "mean_B": round(float(b.mean()), 4),
            "cliffs_delta": round(float(delta), 4),
            "mwu_p": p,
            "n_genes_used": len(genes),
        })
        # per-gene delta
        pg = per_gene_delta(sub, genes, mA, mB)
        pg.to_csv(os.path.join(RES, f"rep30_pergene_{ctype.replace('/', '_')}_{gA}_vs_{gB}.csv"),
                  index=False)
        # per-cell dump
        tmp = sub.obs[["sample", "group", "cell_type", "rep30"]].copy()
        tmp["contrast"] = label
        all_cell.append(tmp)

        print(f"\n[{label}] nA={nA} nB={nB} | median {np.median(a):.4f} vs "
              f"{np.median(b):.4f} (d={np.median(b)-np.median(a):+.4f}) | "
              f"cliffs_d={delta:+.3f} p={p:.2e}")
        print("  top |delta| genes:")
        pg2 = pg.reindex(pg["delta_B_minus_A"].abs().sort_values(ascending=False).index).head(8)
        print(pg2.to_string(index=False))

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
            ax.set_title(f"REP30: {ctype}\n{gA} vs {gB}  (p={p:.1e}, d={delta:+.2f})")
            plt.tight_layout()
            plt.savefig(os.path.join(RES, f"box_{ctype.replace('/', '_')}_{gA}_vs_{gB}.png"),
                        dpi=140)
            plt.close()

    comp = pd.DataFrame(comp_rows)
    comp.to_csv(os.path.join(RES, "rep30_comparisons.csv"), index=False)
    if all_cell:
        pd.concat(all_cell).to_csv(os.path.join(RES, "rep30_by_cell.csv"))

    # ---------------------------------------------------- summary
    with open(os.path.join(RES, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("GSE264205 single-cell REP30 Pre-vs-Post control\n")
        f.write("=" * 60 + "\n")
        f.write("sample id map: 0=Pre, 1=Post1, 2=Post2, 3=Meta(metastatic lesion)\n")
        f.write("REP30 score = coef-weighted z-sum (z within compartment subset)\n\n")
        f.write("== compartment confound (sample x cell_type, %) ==\n")
        f.write(pct.to_string() + "\n\n")
        f.write("== REP30 comparison per compartment ==\n")
        f.write(comp.to_string(index=False) + "\n")
    print(f"\n[DONE] outputs in {RES}")


if __name__ == "__main__":
    main()
