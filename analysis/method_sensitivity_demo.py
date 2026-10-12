# -*- coding: utf-8 -*-
"""Method-sensitivity experiment.

Question posed by the user: "when comparing groups, will choosing a
different algorithm give a different result?"  Instead of answering from
theory, this script re-analyses the SAME comparisons with every method
family we have and tabulates where the verdict flips.

Three deliberately different regimes are used:
  R1  binary outcome, zero cell, n=59          (DFCI TP53 / RB1 / ESR1)
  R2  continuous outcome, n=3 in one arm       (DFCI CDKN2A treatment days)
  R3  continuous outcome, n=123 NON-INDEPENDENT (AURORA: several biopsies
      per patient -> independent-samples tests vs mixed model vs paired)
Plus correlation-family sensitivity on the AURORA treatment-exposure axis.
"""
import io
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stats_toolkit as st  # noqa: E402

import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

BASE = r"E:\01.WorkSpace\01.Breast_cancer_gene_panel\resistance_signature"
CB = os.path.join(BASE, "data", "06_metastatic", "cbioportal")
OUTD = os.path.join(BASE, "results", "method_sensitivity")
os.makedirs(OUTD, exist_ok=True)

LOG = []


def log(*a):
    LOG.append(" ".join(str(x) for x in a))


def flush():
    with io.open(os.path.join(OUTD, "run_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))


def load_cbio(path):
    return pd.read_csv(path, sep="\t", comment="#", low_memory=False)


ALL = []  # every comparison x method row


def push(regime, comparison, rows, extra=None):
    for r in rows:
        d = dict(regime=regime, comparison=comparison)
        d.update(r)
        if extra:
            d.update(extra)
        ALL.append(d)


NONCODING = {"Silent", "Intron", "IGR", "RNA", "3'UTR", "5'UTR", "3'Flank",
             "5'Flank", "lincRNA", "Targeted_Region"}

# ==========================================================================
# R1 / R2 : DFCI
# ==========================================================================
log("=" * 78)
log("R1  binary outcome with zero cells, n=59 (DFCI)")
log("=" * 78)

D = os.path.join(CB, "brca_dfci_2020")
sp = load_cbio(os.path.join(D, "data_clinical_sample.txt"))
pp = load_cbio(os.path.join(D, "data_clinical_patient.txt"))
pp.columns = [c.strip() for c in pp.columns]
mut = load_cbio(os.path.join(D, "data_mutations.txt"))
dmg = mut[~mut["Variant_Classification"].isin(NONCODING)].copy()

samples = sp["SAMPLE_ID"].tolist()
gi = pd.crosstab(dmg["Tumor_Sample_Barcode"], dmg["Hugo_Symbol"]).clip(upper=1)
gi = gi.reindex(index=samples, fill_value=0).fillna(0).astype(int)

sp = sp.set_index("SAMPLE_ID").reindex(samples)
sp["BRR"] = pp.set_index("PATIENT_ID")["BRR"].reindex(sp["PATIENT_ID"].values).values
sp["treat_days"] = pd.to_numeric(
    pp.set_index("PATIENT_ID")["TREATMENT _DURATION"].reindex(sp["PATIENT_ID"].values),
    errors="coerce").values
sp["purity"] = pd.to_numeric(sp["PURITY"], errors="coerce")
resistant = (sp["BX_PHENOTYPE"] != "Sensitive").values.astype(float)

for gene in ["TP53", "RB1", "ESR1", "PIK3CA", "CDH1", "GATA3"]:
    if gene not in gi.columns:
        continue
    g = gi[gene].values.astype(float)
    rows, table = st.compare_binary(g[resistant == 1], g[resistant == 0])
    push("R1-binary/zero-cell", f"DFCI {gene} mut vs resistance", rows,
         {"n_case": int((resistant == 1).sum()), "n_ctrl": int((resistant == 0).sum()),
          "n_carriers": int(g.sum()), "table": str(table)})
    log(f"{gene}: 2x2 = {table}")

log("")
log("--- R1 method matrix: DFCI TP53 (21/41 vs 0/18) ---")
for r in [x for x in ALL if x["regime"].startswith("R1") and "TP53" in x["comparison"]]:
    log(f"  [{r['family']:<11}] {r['method']:<48} p={r['p'] if pd.notna(r['p']) else float('nan'):.5f}"
        f"  stat={r['stat'] if pd.notna(r['stat']) else float('nan'):.4g}  {r.get('note','')}")

# covariate-adjusted version of the TP53 contrast (is TP53 just purity / ER?)
log("")
log("--- R1b: is the TP53 signal explained by purity / receptor status? ---")
sub = sp.copy()
sub["gene"] = gi["TP53"].values
sub["resistant"] = resistant
sub = sub.dropna(subset=["purity"])
try:
    m = smf.logit("resistant ~ gene", data=sub).fit(disp=0)
    log(f"  unadjusted:      OR(gene)={np.exp(m.params['gene']):.3f}  p={m.pvalues['gene']:.4g}")
    m2 = smf.logit("resistant ~ gene + purity", data=sub).fit(disp=0)
    log(f"  + purity:        OR(gene)={np.exp(m2.params['gene']):.3f}  p={m2.pvalues['gene']:.4g}")
    sub2 = sub[sub["BX_ER"].isin(["Positive", "Negative"])]
    m3 = smf.logit("resistant ~ gene + C(BX_ER)", data=sub2).fit(disp=0)
    log(f"  + ER status:     OR(gene)={np.exp(m3.params['gene']):.3f}  p={m3.pvalues['gene']:.4g}")
    # Firth on the adjusted model (separation-safe)
    X = np.column_stack([np.ones(len(sub2)),
                         sub2["gene"].values.astype(float),
                         (sub2["BX_ER"] == "Positive").values.astype(float)])
    y = sub2["resistant"].values.astype(float)
    beta, se = st.firth_logit(X, y)
    z = beta[1] / se[1]
    log(f"  + ER (Firth):    OR(gene)={np.exp(beta[1]):.3f}  p={2*(1-stats.norm.cdf(abs(z))):.4g}")
except Exception as e:
    log("  adjustment failed: " + str(e))

log("")
log("=" * 78)
log("R2  continuous outcome, tiny arm (DFCI treatment duration)")
log("=" * 78)

comp_list = []
for gene in ["TP53", "RB1", "CDKN2A", "ESR1", "PIK3CA"]:
    if gene not in gi.columns:
        continue
    a = sp.loc[gi[gene] == 1, "treat_days"].dropna().values.astype(float)
    b = sp.loc[gi[gene] == 0, "treat_days"].dropna().values.astype(float)
    if len(a) < 2 or len(b) < 2:
        continue
    rows = st.compare_continuous(a, b, nperm=20000, seed=1)
    push("R2-continuous/tiny-arm", f"DFCI treatment days ~ {gene} mut", rows,
         {"n_case": len(a), "n_ctrl": len(b),
          "median_case": float(np.median(a)), "median_ctrl": float(np.median(b))})
    comp_list.append((gene, a, b))
    log(f"{gene}: n={len(a)} vs {len(b)}; median {np.median(a):.0f} vs {np.median(b):.0f} d")

log("")
for gene, a, b in comp_list:
    if gene != "CDKN2A":
        continue
    log(f"--- R2 method matrix: DFCI CDKN2A mut (n={len(a)}!) vs WT treatment days ---")
    log(f"    carriers: {np.sort(a)}")
    for r in [x for x in ALL if x["regime"].startswith("R2") and gene in x["comparison"]]:
        pv = r["p"]
        ptxt = f"{pv:.6f}" if pd.notna(pv) else "   n/a "
        eff = (f"   {r['effect_name']}={r['effect']:.4g}"
               if pd.notna(r.get("effect")) else "")
        log(f"  [{r['family']:<11}] {r['method']:<48} p={ptxt}{eff}")
log("")
log("--- R2 for comparison: DFCI TP53 mut vs treatment days ---")
for r in [x for x in ALL if x["regime"].startswith("R2") and "TP53" in x["comparison"]]:
    pv = r["p"]
    log(f"  [{r['family']:<11}] {r['method']:<48} p={pv:.6f}"
        + (f"   effect={r['effect_name']}={r['effect']:.4g}" if pd.notna(r.get("effect")) else ""))

# ==========================================================================
# R3 : AURORA, non-independent samples
# ==========================================================================
log("")
log("=" * 78)
log("R3  continuous outcome with repeated biopsies per patient (AURORA)")
log("=" * 78)

AU = os.path.join(BASE, "results", "metastatic", "AURORA_cBioPortal", "aurora_linked.csv")
au = pd.read_csv(AU, index_col=0)
log(f"AURORA linked: {au.shape}; sample types={au['SAMPLE_TYPE'].value_counts().to_dict()}")

au["is_met"] = (au["SAMPLE_TYPE"] == "Metastatic").astype(int)
a = au.loc[au["is_met"] == 1, "rep30"].astype(float).values
b = au.loc[au["is_met"] == 0, "rep30"].astype(float).values
rows = st.compare_continuous(a, b, nperm=20000, seed=2)
push("R3-continuous/repeated", "AURORA REP30: metastatic vs primary (all biopsies)", rows,
     {"n_case": len(a), "n_ctrl": len(b)})
log(f"metastatic n={len(a)} vs primary n={len(b)}")

# mixed model: patient random intercept (accounts for repeated biopsies)
try:
    md = smf.mixedlm("rep30 ~ is_met", au, groups=au["PATIENT_ID"])
    mdf = md.fit(reml=True)
    log(f"  MixedLM (patient random intercept): beta={mdf.params['is_met']:.4f} "
        f"p={mdf.pvalues['is_met']:.4f}")
    ALL.append(dict(regime="R3-continuous/repeated",
                    comparison="AURORA REP30: metastatic vs primary (all biopsies)",
                    family="model", method="Linear mixed model (patient random intercept)",
                    stat=float(mdf.params["is_met"]), p=float(mdf.pvalues["is_met"]),
                    effect_name="beta", effect=float(mdf.params["is_met"]),
                    note="uses ALL biopsies, accounts for within-patient correlation",
                    n_case=len(a), n_ctrl=len(b), table=""))
except Exception as e:
    log("  MixedLM failed: " + str(e))

# paired subset
piv = au.pivot_table(index="PATIENT_ID", columns="is_met", values="rep30", aggfunc="mean")
piv = piv.dropna()
if len(piv):
    d = (piv[1] - piv[0]).values.astype(float)
    w = stats.wilcoxon(d)
    t = stats.ttest_rel(piv[1].values, piv[0].values)
    log(f"  paired subset n={len(piv)} patients: median delta={np.median(d):+.4f} "
        f"Wilcoxon p={w.pvalue:.4f} paired-t p={t.pvalue:.4f}")
    ALL.append(dict(regime="R3-continuous/repeated",
                    comparison="AURORA REP30: metastatic vs primary (paired patients only)",
                    family="paired", method="Wilcoxon signed-rank (paired)",
                    stat=float(w.statistic), p=float(w.pvalue), effect_name="median delta",
                    effect=float(np.median(d)), note="within-patient design", table="",
                    n_case=len(piv), n_ctrl=len(piv)))
    ALL.append(dict(regime="R3-continuous/repeated",
                    comparison="AURORA REP30: metastatic vs primary (paired patients only)",
                    family="paired", method="Paired t-test",
                    stat=float(t.statistic), p=float(t.pvalue), effect_name="mean delta",
                    effect=float(d.mean()), note="within-patient design", table="",
                    n_case=len(piv), n_ctrl=len(piv)))

# covariate-adjusted (proliferation, tumour content, preservation)
adj = au.dropna(subset=["rep30", "prolif", "PERCENT_TUMOR_CELLS", "PRESERVATION"])
forms = {
    "OLS, no covariate": "rep30 ~ is_met",
    "OLS + proliferation": "rep30 ~ is_met + prolif",
    "OLS + prolif + tumour content": "rep30 ~ is_met + prolif + PERCENT_TUMOR_CELLS",
    "OLS + prolif + tumour content + preservation":
        "rep30 ~ is_met + prolif + PERCENT_TUMOR_CELLS + C(PRESERVATION)",
    "OLS, full + receiver status":
        "rep30 ~ is_met + prolif + PERCENT_TUMOR_CELLS + C(PRESERVATION) + C(PRIM_ER)",
}
for label, f in forms.items():
    try:
        m = smf.ols(f, data=adj).fit(cov_type="HC3")
        ALL.append(dict(regime="R3-continuous/repeated",
                        comparison="AURORA REP30: metastatic vs primary (all biopsies)",
                        family="model", method=label,
                        stat=float(m.params["is_met"]), p=float(m.pvalues["is_met"]),
                        effect_name="beta", effect=float(m.params["is_met"]),
                        note=f"n={int(m.nobs)}, HC3 robust SE", table="",
                        n_case=len(a), n_ctrl=len(b)))
        log(f"  {label:<48} beta={m.params['is_met']:+.4f} p={m.pvalues['is_met']:.4f}")
    except Exception as e:
        log(f"  {label} failed: {e}")

# quantile regression: does the conclusion hold away from the mean?
try:
    for q in [0.25, 0.5, 0.75]:
        qr = smf.quantreg("rep30 ~ is_met", data=adj).fit(q=q)
        ALL.append(dict(regime="R3-continuous/repeated",
                        comparison="AURORA REP30: metastatic vs primary (all biopsies)",
                        family="model", method=f"Quantile regression (q={q})",
                        stat=float(qr.params["is_met"]), p=float(qr.pvalues["is_met"]),
                        effect_name="beta_q", effect=float(qr.params["is_met"]),
                        note="shift at this quantile, not at the mean", table="",
                        n_case=len(a), n_ctrl=len(b)))
        log(f"  Quantile regression q={q:.2f}: beta={qr.params['is_met']:+.4f} p={qr.pvalues['is_met']:.4f}")
except Exception as e:
    log("  quantreg failed: " + str(e))

# resistance-label contrasts inside AURORA (binary outcome, moderate n)
for label, col in [("ESR1 mutation", "ESR1_mut"), ("RB1 loss", "RB1_loss"),
                   ("PIK3CA mutation", "PIK3CA_mut"), ("ERBB2 amplification", "ERBB2_amp")]:
    if col not in au.columns:
        continue
    f = au[[col, "rep30"]].dropna()
    g1 = f.loc[f[col] == True, "rep30"].astype(float).values
    g0 = f.loc[f[col] == False, "rep30"].astype(float).values
    if len(g1) < 3:
        continue
    rows = st.compare_continuous(g1, g0, nperm=20000, seed=3)
    push("R3-continuous/label", f"AURORA REP30 ~ {label} (metastatic biopsies only)", rows,
         {"n_case": len(g1), "n_ctrl": len(g0)})

# ==========================================================================
# correlation family (treatment-exposure axis)
# ==========================================================================
log("")
log("=" * 78)
log("R4  correlation family: REP30 vs treatment exposure in metastatic biopsies")
log("=" * 78)
mm = au[(au["is_met"] == 1) & au["rep30"].notna()]
for xcol, xname in [("met_lines_before_biopsy", "prior lines of therapy"),
                    ("days_on_met_tx", "cumulative days on metastatic therapy"),
                    ("days_since_met_dx", "days since metastatic diagnosis")]:
    if xcol not in mm.columns:
        continue
    d = mm[[xcol, "rep30"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 10:
        continue
    cr = st.compare_correlation(d[xcol].values, d["rep30"].values)
    push("R4-correlation", f"AURORA REP30 vs {xname}", cr,
         {"n_case": len(d), "n_ctrl": len(d)})
    log(f"  {xname} (n={len(d)}):")
    for r in cr:
        log(f"    [{r['method']:<44}] r={r['stat']:+.4f} p={r['p']:.5f} {r['note']}")

# ==========================================================================
# consistency summary
# ==========================================================================
df = pd.DataFrame(ALL)
df.to_csv(os.path.join(OUTD, "method_matrix_all.csv"), index=False, encoding="utf-8")

srows = []
for (reg, comp), grp in df.groupby(["regime", "comparison"]):
    pv = grp["p"].dropna()
    if len(pv) == 0:
        continue
    sig = (pv < 0.05).sum()
    srows.append(dict(regime=reg, comparison=comp, n_methods=len(pv),
                      n_sig= int(sig), pct_sig=100 * sig / len(pv),
                      p_min=float(pv.min()), p_max=float(pv.max()),
                      p_median=float(pv.median()),
                      spread_order_of_magnitude=float(np.log10(pv.max() / max(pv.min(), 1e-300))),
                      verdict="consistent" if (sig == 0 or sig == len(pv)) else "FLIPS"))
summ = pd.DataFrame(srows).sort_values("pct_sig")
summ.to_csv(os.path.join(OUTD, "method_verdict_summary.csv"), index=False, encoding="utf-8")

log("")
log("=" * 78)
log("SUMMARY  where does the verdict depend on the algorithm?")
log("=" * 78)
for _, r in summ.iterrows():
    log(f"  {r['verdict']:<10} {r['n_sig']:>2}/{r['n_methods']:<2} significant  "
        f"p range {r['p_min']:.5f}-{r['p_max']:.5f}  ({r['spread_order_of_magnitude']:.1f} orders)  {r['comparison']}")

# ==========================================================================
# figure
# ==========================================================================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = df.dropna(subset=["p"]).copy()
    order = list(summ.sort_values("pct_sig")["comparison"])
    fig, ax = plt.subplots(figsize=(9.0, max(4.0, 0.42 * len(order))))
    families = sorted(plot_df["family"].unique())
    cmap = {"parametric": "#c0392b", "rank": "#2c6fbb", "resampling": "#4c8c4a",
            "exact": "#8e44ad", "large-sample": "#d9a441", "model": "#16a085",
            "paired": "#e67e22", "effect": "#7f8c8d"}
    ypos = {c: i for i, c in enumerate(order)}
    for fam in families:
        s = plot_df[plot_df["family"] == fam]
        ax.scatter(s["p"], [ypos[c] + np.random.default_rng(0).normal(0, 0.06)
                            for c in s["comparison"]],
                   s=26, label=fam, color=cmap.get(fam, "#555555"), alpha=0.85)
    ax.axvline(0.05, color="black", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("p-value (log scale) -- dashed line = 0.05")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([c if len(c) < 62 else c[:59] + "..." for c in order], fontsize=7)
    ax.legend(fontsize=7, ncol=4, loc="lower right")
    ax.set_title("Same comparison, different algorithms: p-value landscape", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTD, "method_pvalue_landscape.png"), dpi=200)
    plt.close(fig)
    log("")
    log("figure written")
except Exception as e:
    log("")
    log(f"figure step skipped: {type(e).__name__}: {e}")

flush()
print("OK")
