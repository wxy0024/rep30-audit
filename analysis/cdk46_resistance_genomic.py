# -*- coding: utf-8 -*-
"""Genomic-layer (DNA / RAS-like) analysis of CDK4/6 inhibitor resistance.

Two cohorts, both DNA-only (neither ships an expression matrix):

  * brca_dfci_2020  (Wander et al., Cancer Discov 2020; 59 CDK4/6i-EXPOSED
    metastatic breast tumours, WES).  Carries a per-biopsy resistance
    phenotype (Intrinsic Resistance / Acquired Resistance / Sensitive),
    receptor status at biopsy, best radiographic response and treatment
    duration -> this is the only package here that can actually test a
    resistance PHENOTYPE.

  * breast_ink4_msk_2021  (Qing et al., Cancer Discov 2022; 1365 metastatic
    breast tumours, MSK-IMPACT targeted DNA: mutations + discrete CNA + SV).
    No treatment/response fields -> usable only for alteration
    co-occurrence, i.e. what the "resistance genome" looks like.

Since REP30 is a Cox-weighted EXPRESSION score, it cannot be computed in
either cohort.  This script therefore answers the genomic half of the
question: which somatic alterations track with CDK4/6i resistance, and how
coherent is the resistance genome in metastatic disease.
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

BASE = r"E:\01.WorkSpace\01.Breast_cancer_gene_panel\resistance_signature"
CB = os.path.join(BASE, "data", "06_metastatic", "cbioportal")
OUTD = os.path.join(BASE, "results", "metastatic", "cdk46_resistance")
os.makedirs(OUTD, exist_ok=True)

LOG = []


def log(*a):
    LOG.append(" ".join(str(x) for x in a))


def flush():
    with io.open(os.path.join(OUTD, "run_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))


def load_cbio(path):
    return pd.read_csv(path, sep="\t", comment="#", low_memory=False)


NONCODING = {"Silent", "Intron", "IGR", "RNA", "3'UTR", "5'UTR", "3'Flank",
             "5'Flank", "lincRNA", "Targeted_Region"}
TRUNCATING = {"Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
              "Splice_Site", "Translation_Start_Site", "Nonstop_Mutation"}
HOTSPOT_ESR1 = {"p.Y537S", "p.D538G", "p.Y537C", "p.Y537N", "p.E380Q",
                "p.L536H", "p.L536P", "p.Y537H", "p.V422del", "p.E380G"}


def bin_matrix(long_df, sample_col, gene_col, samples):
    piv = pd.crosstab(long_df[sample_col], long_df[gene_col])
    piv = piv.clip(upper=1)
    piv = piv.reindex(index=samples, fill_value=0).fillna(0).astype(int)
    return piv


def enrichment(bits, focus):
    """bits: 0/1 Series over samples; focus: bool Series (True = focal group)."""
    ok = bits.notna() & focus.notna()
    gb = bits[ok].astype(int)
    fc = focus[ok].astype(bool)
    a = int(((gb == 1) & fc).sum())
    b = int(((gb == 1) & ~fc).sum())
    c = int(((gb == 0) & fc).sum())
    d = int(((gb == 0) & ~fc).sum())
    if (a + c) == 0 or (b + d) == 0:
        return None
    or_, p2 = stats.fisher_exact([[a, b], [c, d]])
    lo, hi, note = st._or_ci_woolf(a, b, c, d)
    rd_lo, rd_hi = st._rd_ci_newcombe(a, b, c, d)
    return dict(case_pos=a, case_n=a + c, ctrl_pos=b, ctrl_n=b + d,
                freq_case=a / (a + c), freq_ctrl=b / (b + d),
                odds_ratio=or_, or_lo=lo, or_hi=hi, ci_note=note,
                risk_diff=a / (a + c) - b / (b + d), rd_lo=rd_lo, rd_hi=rd_hi,
                fisher_p=p2)


def cochran_armitage(bits, score):
    """Trend test across ordered categories given by `score` (per sample)."""
    ok = bits.notna() & score.notna()
    b = bits[ok].astype(int).values
    s = score[ok].astype(float).values
    cats = np.unique(s)
    n_i = np.array([(s == c).sum() for c in cats], float)
    x_i = np.array([b[s == c].sum() for c in cats], float)
    N = n_i.sum()
    R = x_i.sum()
    T = float((x_i * cats).sum())
    E = R / N * float((n_i * cats).sum())
    var = R * (N - R) / (N * (N - 1)) * (
        float((n_i * cats ** 2).sum()) - float((n_i * cats).sum()) ** 2 / N)
    if var <= 0:
        return np.nan, np.nan
    z = (T - E) / np.sqrt(var)
    return float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


# ==========================================================================
# PART 1 - DFCI: alteration vs CDK4/6i resistance phenotype
# ==========================================================================
log("=" * 78)
log("PART 1  brca_dfci_2020 - CDK4/6i resistance phenotype vs alteration")
log("=" * 78)

D = os.path.join(CB, "brca_dfci_2020")
sp = load_cbio(os.path.join(D, "data_clinical_sample.txt"))
pp = load_cbio(os.path.join(D, "data_clinical_patient.txt"))
pp.columns = [c.strip() for c in pp.columns]
mut = load_cbio(os.path.join(D, "data_mutations.txt"))

log(f"samples={len(sp)} patients={len(pp)} mutation rows={len(mut)}")
log("phenotype: " + str(sp["BX_PHENOTYPE"].value_counts().to_dict()))
log("BRR: " + str(pp["BRR"].value_counts().to_dict()))

samples = sp["SAMPLE_ID"].tolist()
dmg = mut[~mut["Variant_Classification"].isin(NONCODING)].copy()
dmg = dmg[dmg["Tumor_Sample_Barcode"].isin(samples)]
trunc = dmg[dmg["Variant_Classification"].isin(TRUNCATING)]

M = bin_matrix(dmg, "Tumor_Sample_Barcode", "Hugo_Symbol", samples)
MT = bin_matrix(trunc, "Tumor_Sample_Barcode", "Hugo_Symbol", samples)
log(f"damaging gene matrix: {M.shape[0]} samples x {M.shape[1]} genes")
log(f"truncating gene matrix: {MT.shape[1]} genes")

sp = sp.set_index("SAMPLE_ID").reindex(samples)
brr = pp.set_index("PATIENT_ID")["BRR"].reindex(sp["PATIENT_ID"].values)
dur = pd.to_numeric(pp.set_index("PATIENT_ID")["TREATMENT _DURATION"]
                    .reindex(sp["PATIENT_ID"].values), errors="coerce")
sp["BRR"] = brr.values
sp["treat_days"] = dur.values

phen = sp["BX_PHENOTYPE"]
log("treatment duration (days): " + str(pd.Series(sp["treat_days"]).describe().round(1).to_dict()))
log("ER at biopsy: " + str(sp["BX_ER"].value_counts().to_dict()))

# cohort is overwhelmingly HR+/HER2-
hr_pos = (sp["BX_ER"] == "Positive") & (sp["BX_HER2OVERALL"] != "Positive")
log(f"HR+/HER2- biopsies: {int(hr_pos.sum())}/{len(sp)}")

PANEL = ["ESR1", "RB1", "TP53", "PIK3CA", "NF1", "AKT1", "ERBB2", "PTEN",
         "CDH1", "BRCA1", "BRCA2", "GATA3", "FAT1", "MAP3K1", "ARID1A",
         "KMT2C", "CBFB", "RUNX1", "TBX3", "CDKN2A", "CDKN2B", "SMAD4",
         "ATM", "CHEK2", "MYC", "EGFR", "ERBB3", "FGFR1", "CCND1", "CCNE1",
         "CDK4", "CDK6", "MDM2", "KRAS", "BRAF", "STK11"]
PANEL = [g for g in PANEL if g in M.columns]
log(f"candidate panel present: {len(PANEL)}")

CONTRASTS = {
    "resistant(Intrinsic+Acquired) vs Sensitive":
        (phen != "Sensitive") & phen.notna(),
    "Intrinsic Resistance vs others":
        (phen == "Intrinsic Resistance"),
    "Acquired Resistance vs Sensitive":
        (phen == "Acquired Resistance"),
    "Acquired Resistance vs Intrinsic Resistance":
        (phen == "Acquired Resistance"),
}
# for the last contrast the reference is restricted to Intrinsic below
rows = []
for cname, focus in CONTRASTS.items():
    ref_mask = None
    if cname.startswith("Acquired Resistance vs Intrinsic"):
        ref_mask = (phen == "Intrinsic Resistance")
    for g in PANEL:
        bits = M[g]
        if ref_mask is not None:
            ok = ref_mask | focus
            r = enrichment(bits[ok], focus[ok])
        else:
            r = enrichment(bits, focus)
        if r is None:
            continue
        r.update(dict(contrast=cname, gene=g, n_carriers=int(bits.sum())))
        z, pt = cochran_armitage(bits, sp["BX_PHENOTYPE"].map(
            {"Sensitive": 1, "Acquired Resistance": 2, "Intrinsic Resistance": 3}))
        r.update(dict(trend_z=z, trend_p=pt))
        rows.append(r)
df1 = pd.DataFrame(rows)
df1["fisher_q"] = st.bh_fdr(df1["fisher_p"].values)
df1 = df1[["contrast", "gene", "n_carriers", "case_pos", "case_n", "ctrl_pos",
           "ctrl_n", "freq_case", "freq_ctrl", "odds_ratio", "or_lo", "or_hi",
           "risk_diff", "rd_lo", "rd_hi", "fisher_p", "fisher_q",
           "trend_z", "trend_p", "ci_note"]]
df1.to_csv(os.path.join(OUTD, "dfci_gene_vs_phenotype.csv"), index=False,
           encoding="utf-8")
log("")
log("--- DFCI candidate panel, resistant vs sensitive (top by p) ---")
show = df1[df1["contrast"] == "resistant(Intrinsic+Acquired) vs Sensitive"].copy()
show = show.sort_values("fisher_p")
for _, r in show.head(12).iterrows():
    log(f"  {r['gene']:<8} carriers={int(r['n_carriers']):>2}  "
        f"{int(r['case_pos'])}/{int(r['case_n'])} vs {int(r['ctrl_pos'])}/{int(r['ctrl_n'])}  "
        f"OR={r['odds_ratio']:.2f} [{r['or_lo']:.2f},{r['or_hi']:.2f}]  p={r['fisher_p']:.4f} q={r['fisher_q']:.3f}")

log("")
log("--- ESR1 / RB1 detail across the three phenotypes ---")
for g in ["ESR1", "RB1", "TP53", "PIK3CA"]:
    if g not in M.columns:
        continue
    tab = pd.crosstab(M[g], phen)
    log(f"  {g}: " + str({str(k): int(v) for k, v in M[g].value_counts().items()}))
    log("     carriers by phenotype: " +
        str({str(c): int(tab[c].get(1, 0)) for c in tab.columns if str(c) != "nan"}))

# ---- ESR1 hotspot-only (acquired endocrine resistance) ----
esr_hs = set(dmg[(dmg["Hugo_Symbol"] == "ESR1") &
                 (dmg["HGVSp_Short"].isin(HOTSPOT_ESR1))]["Tumor_Sample_Barcode"])
hs = pd.Series(0, index=samples)
hs.loc[hs.index.isin(esr_hs)] = 1
log("")
log(f"ESR1 canonical hotspot carriers: {int(hs.sum())}/{len(samples)}")
tab = pd.crosstab(hs, phen)
log("  by phenotype: " + str({str(c): int(tab[c].get(1, 0)) for c in tab.columns if str(c) != "nan"}))
r_hs = enrichment(hs, (phen != "Sensitive") & phen.notna())
if r_hs:
    log(f"  resistant vs sensitive: {r_hs['case_pos']}/{r_hs['case_n']} vs "
        f"{r_hs['ctrl_pos']}/{r_hs['ctrl_n']} OR={r_hs['odds_ratio']:.2f} p={r_hs['fisher_p']:.4f}")
    pd.DataFrame([dict(gene="ESR1_hotspot_only", **r_hs)]).to_csv(
        os.path.join(OUTD, "dfci_esr1_hotspot.csv"), index=False, encoding="utf-8")

# ---- treatment duration vs alteration ----
log("")
log("--- DFCI treatment duration (days on CDK4/6i) vs alteration ---")
rows = []
for g in PANEL:
    a = sp.loc[M[g] == 1, "treat_days"].dropna().values.astype(float)
    b = sp.loc[M[g] == 0, "treat_days"].dropna().values.astype(float)
    if len(a) < 3 or len(b) < 3:
        continue
    mw = st.mannwhitney(a, b)
    tw = st.t_welch(a, b)
    rows.append(dict(gene=g, n_carriers=len(a), n_wt=len(b),
                     median_carriers=float(np.median(a)), median_wt=float(np.median(b)),
                     cliffs_delta=st.cliffs_delta(a, b),
                     hodges_lehmann=st.hodges_lehmann(a, b),
                     mwu_p=mw["p"], welch_p=tw["p"],
                     cohens_d=st.cohens_d(a, b)))
df2 = pd.DataFrame(rows)
df2["mwu_q"] = st.bh_fdr(df2["mwu_p"].values)
df2 = df2.sort_values("mwu_p")
df2.to_csv(os.path.join(OUTD, "dfci_gene_vs_treatment_duration.csv"), index=False,
           encoding="utf-8")
for _, r in df2.head(10).iterrows():
    log(f"  {r['gene']:<8} n={int(r['n_carriers']):>2}  median "
        f"{r['median_carriers']:.0f} vs {r['median_wt']:.0f} d  "
        f"delta={r['cliffs_delta']:+.3f}  MWU p={r['mwu_p']:.4f} "
        f"(Welch t p={r['welch_p']:.4f})  q={r['mwu_q']:.3f}")

# ---- best radiographic response ----
log("")
log("--- DFCI best radiographic response (Progression vs Response/Stable) ---")
prog = (sp["BRR"] == "Progression")
rows = []
for g in PANEL:
    r = enrichment(M[g], prog)
    if r is None:
        continue
    r.update(dict(gene=g, n_carriers=int(M[g].sum())))
    rows.append(r)
df3 = pd.DataFrame(rows).sort_values("fisher_p")
df3["fisher_q"] = st.bh_fdr(df3["fisher_p"].values)
df3.to_csv(os.path.join(OUTD, "dfci_gene_vs_best_response.csv"), index=False,
           encoding="utf-8")
for _, r in df3.head(8).iterrows():
    log(f"  {r['gene']:<8} {int(r['case_pos'])}/{int(r['case_n'])} vs "
        f"{int(r['ctrl_pos'])}/{int(r['ctrl_n'])} OR={r['odds_ratio']:.2f} "
        f"p={r['fisher_p']:.4f} q={r['fisher_q']:.3f}")

# ---- unbiased exome-wide scan (low power, reported for honesty) ----
log("")
log("--- DFCI exome-wide scan (exploratory, BH-controlled) ---")
focus = (phen != "Sensitive") & phen.notna()
min_carriers = 4
scan = []
for g in M.columns:
    if int(M[g].sum()) < min_carriers:
        continue
    r = enrichment(M[g], focus)
    if r is None:
        continue
    r.update(dict(gene=g, n_carriers=int(M[g].sum())))
    scan.append(r)
df4 = pd.DataFrame(scan)
if len(df4):
    df4["q"] = st.bh_fdr(df4["fisher_p"].values)
    df4 = df4.sort_values("fisher_p")[["gene", "n_carriers", "case_pos", "case_n",
                                       "ctrl_pos", "ctrl_n", "odds_ratio",
                                       "or_lo", "or_hi", "fisher_p", "q"]]
    df4.to_csv(os.path.join(OUTD, "dfci_exomewide_scan.csv"), index=False, encoding="utf-8")
    log(f"  genes tested: {len(df4)}")
    for _, r in df4.head(12).iterrows():
        log(f"  {r['gene']:<10} carriers={int(r['n_carriers']):>2} OR={r['odds_ratio']:.2f} "
            f"p={r['fisher_p']:.4f} q={r['q']:.3f}")
    log(f"  genes with q<0.10: {int((df4['q'] < 0.10).sum())}; "
        f"q<0.25: {int((df4['q'] < 0.25).sum())}")

sp.to_csv(os.path.join(OUTD, "dfci_sample_table.csv"), encoding="utf-8")

# ==========================================================================
# PART 2 - INK4/MSK: co-occurrence of the "resistance genome"
# ==========================================================================
log("")
log("=" * 78)
log("PART 2  breast_ink4_msk_2021 - metastatic resistance-genome co-occurrence")
log("=" * 78)

D2 = os.path.join(CB, "breast_ink4_msk_2021")
sp2 = load_cbio(os.path.join(D2, "data_clinical_sample.txt"))
m2 = load_cbio(os.path.join(D2, "data_mutations.txt"))
c2 = load_cbio(os.path.join(D2, "data_cna.txt"))
log(f"samples={len(sp2)}  mutation rows={len(m2)}  cna rows={c2.shape[0]}")

s2 = sp2["SAMPLE_ID"].tolist()
dmg2 = m2[~m2["Variant_Classification"].isin(NONCODING)].copy()
dmg2 = dmg2[dmg2["Tumor_Sample_Barcode"].isin(s2)]
M2 = bin_matrix(dmg2, "Tumor_Sample_Barcode", "Hugo_Symbol", s2)

cna = c2.set_index("Hugo_Symbol")
cna = cna[[c for c in cna.columns if c in set(s2)]]
cna = cna.apply(pd.to_numeric, errors="coerce")
log(f"targeted mutation matrix: {M2.shape}; CNA matrix: {cna.shape}")

ALT = {}


def add_mut(name, gene, mat=M2):
    if gene in mat.columns:
        ALT[name] = mat[gene].reindex(s2).fillna(0).astype(int)


def add_cna(name, gene, direction):
    if gene in cna.index:
        v = cna.loc[gene].reindex(s2)
        ALT[name] = ((v > 0) if direction == "amp" else (v < 0)).astype(int)


for gene in ["ESR1", "RB1", "TP53", "PIK3CA", "AKT1", "ERBB2", "NF1", "PTEN",
             "CDH1", "GATA3", "BRCA1", "BRCA2", "CDKN2A", "CDKN2B", "ARID1A",
             "MAP3K1", "FAT1", "KMT2C", "TBX3", "RUNX1", "MYC", "CCND1",
             "CCNE1", "CDK4", "CDK6", "MDM2", "FGFR1", "EGFR", "ERBB3",
             "KRAS", "BRAF", "ATM", "CHEK2", "SMAD4", "STK11", "AKT2",
             "IGF1R", "SYK", "TSC1", "TSC2", "PIK3R1"]:
    add_mut(f"{gene}_mut", gene)
for gene, direction in [("ESR1", "amp"), ("RB1", "del"), ("CCND1", "amp"),
                        ("CDK4", "amp"), ("CDK6", "amp"), ("CCNE1", "amp"),
                        ("FGFR1", "amp"), ("ERBB2", "amp"), ("MDM2", "amp"),
                        ("PTEN", "del"), ("NF1", "del"), ("CDKN2A", "del"),
                        ("CDKN2B", "del"), ("MYC", "amp"), ("ARID1A", "del"),
                        ("ATM", "del"), ("BRCA2", "del"), ("MAP2K4", "del"),
                        ("AKT2", "amp"), ("IGF1R", "amp")]:
    add_cna(f"{gene}_{direction}", gene, direction)

ALT = pd.DataFrame(ALT).reindex(s2).fillna(0).astype(int)
log(f"alteration matrix: {ALT.shape[0]} samples x {ALT.shape[1]} alterations")
freq = ALT.mean().sort_values(ascending=False)
log("alteration frequencies (top 25):")
for k, v in freq.head(25).items():
    log(f"  {k:<14} {v*100:5.1f}%  (n={int(ALT[k].sum())})")
ALT.to_csv(os.path.join(OUTD, "ink4_alteration_matrix.csv"), encoding="utf-8")
freq.rename("frequency").to_csv(os.path.join(OUTD, "ink4_alteration_frequency.csv"),
                                encoding="utf-8")

# ---- co-occurrence anchored on ESR1 mutation ----
anchor = "ESR1_mut"
log("")
log(f"--- co-occurrence with {anchor} (n={int(ALT[anchor].sum())}) ---")
rows = []
for name in ALT.columns:
    if name == anchor:
        continue
    a = int(((ALT[anchor] == 1) & (ALT[name] == 1)).sum())
    b = int(((ALT[anchor] == 1) & (ALT[name] == 0)).sum())
    c = int(((ALT[anchor] == 0) & (ALT[name] == 1)).sum())
    d = int(((ALT[anchor] == 0) & (ALT[name] == 0)).sum())
    if (a + b) == 0 or (c + d) == 0:
        continue
    or_, p = stats.fisher_exact([[a, b], [c, d]])
    lo, hi, _ = st._or_ci_woolf(a, b, c, d)
    rows.append(dict(partner=name, n_partner=int(ALT[name].sum()),
                     both=a, anchor_only=b, partner_only=c, neither=d,
                     freq_in_anchor=a / (a + b) if (a + b) else np.nan,
                     freq_in_rest=c / (c + d) if (c + d) else np.nan,
                     odds_ratio=or_, or_lo=lo, or_hi=hi, fisher_p=p))
dc = pd.DataFrame(rows)
dc["q"] = st.bh_fdr(dc["fisher_p"].values)
dc = dc.sort_values("fisher_p")
dc.to_csv(os.path.join(OUTD, "ink4_cooccurrence_with_ESR1mut.csv"), index=False,
          encoding="utf-8")
for _, r in dc.head(15).iterrows():
    log(f"  {r['partner']:<16} n={int(r['n_partner']):>4}  in-ESR1mut "
        f"{r['freq_in_anchor']*100:5.1f}% vs rest {r['freq_in_rest']*100:5.1f}%  "
        f"OR={r['odds_ratio']:.2f} [{r['or_lo']:.2f},{r['or_hi']:.2f}] "
        f"p={r['fisher_p']:.3g} q={r['q']:.3g}")

# ---- pairwise co-occurrence matrix among canonical CDK4/6-axis genes ----
AXIS = [k for k in ["RB1_mut", "RB1_del", "CCND1_amp", "CDK4_amp", "CDK6_amp",
                    "CCNE1_amp", "FGFR1_amp", "MDM2_amp", "ERBB2_amp",
                    "ERBB2_mut", "ESR1_mut", "ESR1_amp", "AKT1_mut",
                    "NF1_mut", "PTEN_del", "PTEN_mut", "TP53_mut",
                    "PIK3CA_mut", "CDKN2A_del", "MYC_amp"] if k in ALT.columns]
or_mat = pd.DataFrame(np.nan, index=AXIS, columns=AXIS)
p_mat = pd.DataFrame(np.nan, index=AXIS, columns=AXIS)
for i, gi in enumerate(AXIS):
    for j, gj in enumerate(AXIS):
        if j <= i:
            continue
        a = int(((ALT[gi] == 1) & (ALT[gj] == 1)).sum())
        b = int(((ALT[gi] == 1) & (ALT[gj] == 0)).sum())
        c = int(((ALT[gi] == 0) & (ALT[gj] == 1)).sum())
        d = int(((ALT[gi] == 0) & (ALT[gj] == 0)).sum())
        if min(a + b, c + d, a + c, b + d) == 0:
            continue
        or_, p = stats.fisher_exact([[a, b], [c, d]])
        or_mat.loc[gi, gj] = or_mat.loc[gj, gi] = or_
        p_mat.loc[gi, gj] = p_mat.loc[gj, gi] = p
or_mat.to_csv(os.path.join(OUTD, "ink4_pairwise_OR.csv"), encoding="utf-8")
p_mat.to_csv(os.path.join(OUTD, "ink4_pairwise_p.csv"), encoding="utf-8")

log("")
log("--- notable pairwise co-occurrence (BH q<0.05) ---")
flat = []
for i, gi in enumerate(AXIS):
    for j, gj in enumerate(AXIS):
        if j <= i:
            continue
        if np.isfinite(p_mat.loc[gi, gj]):
            flat.append(dict(a=gi, b=gj, OR=or_mat.loc[gi, gj], p=p_mat.loc[gi, gj],
                             n_a=int(ALT[gi].sum()), n_b=int(ALT[gj].sum())))
dfp = pd.DataFrame(flat)
dfp["q"] = st.bh_fdr(dfp["p"].values)
dfp = dfp.sort_values("p")
dfp.to_csv(os.path.join(OUTD, "ink4_pairwise_tests.csv"), index=False, encoding="utf-8")
for _, r in dfp.head(15).iterrows():
    log(f"  {r['a']:<12} x {r['b']:<12} OR={r['OR']:6.2f} p={r['p']:.3g} q={r['q']:.3g}")

# ---- combined "CDK4/6-resistance alteration burden" ----
BURDEN = [k for k in ["RB1_mut", "RB1_del", "CCND1_amp", "CDK4_amp", "CDK6_amp",
                      "CCNE1_amp", "FGFR1_amp", "MDM2_amp", "AKT1_mut", "NF1_mut",
                      "NF1_del", "PTEN_del", "PTEN_mut"] if k in ALT.columns]
burden = ALT[BURDEN].sum(axis=1)
log("")
log(f"--- CDK4/6-axis alteration burden ({len(BURDEN)} alterations) ---")
log("  distribution: " + str(burden.value_counts().sort_index().to_dict()))
log(f"  mean={burden.mean():.3f} median={burden.median():.0f} max={int(burden.max())}")
burden.rename("cdk46_axis_burden").to_csv(os.path.join(OUTD, "ink4_cdk46_burden.csv"),
                                          encoding="utf-8")
for name in ["ESR1_mut", "ESR1_amp"]:
    if name in ALT.columns:
        a = burden[ALT[name] == 1].values
        b = burden[ALT[name] == 0].values
        mw = st.mannwhitney(a, b)
        log(f"  burden | {name}+ (n={len(a)}) = {a.mean():.3f} vs "
            f"{name}- (n={len(b)}) = {b.mean():.3f}  delta={st.cliffs_delta(a,b):+.3f} "
            f"MWU p={mw['p']:.3g}")

# ==========================================================================
# figures
# ==========================================================================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 1: dfci forest on the RISK-DIFFERENCE scale.
    # Deliberate choice: the odds ratio is unbounded/infinite wherever a
    # stratum has zero carriers (TP53, RB1, CDKN2A, EGFR here), so an
    # OR forest cannot be drawn at all. Newcombe's risk-difference CI is
    # finite for every row, which is exactly the point of this exercise.
    s1 = df1[df1["contrast"] == "resistant(Intrinsic+Acquired) vs Sensitive"].copy()
    s1 = s1.sort_values("risk_diff")
    fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.28 * len(s1))))
    y = np.arange(len(s1))
    lo = np.clip(s1["risk_diff"] - s1["rd_lo"], 0, None)
    hi = np.clip(s1["rd_hi"] - s1["risk_diff"], 0, None)
    ax.errorbar(s1["risk_diff"], y, xerr=[lo, hi], fmt="o", ms=4, lw=1,
                color="#c0392b", ecolor="#888888")
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(y)
    labels = [f"{g} [{int(cp)}/{int(cn)} vs {int(tp)}/{int(tn)}]"
              for g, cp, cn, tp, tn in zip(s1["gene"], s1["case_pos"], s1["case_n"],
                                           s1["ctrl_pos"], s1["ctrl_n"])]
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Risk difference (Newcombe 95% CI): frequency in resistant minus sensitive")
    ax.set_title("DFCI CDK4/6i cohort (n=59): alteration enrichment\n"
                 "resistant (Intrinsic+Acquired, n=41) vs sensitive (n=18)", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTD, "dfci_forest_resistant_vs_sensitive.png"), dpi=200)
    plt.close(fig)

    # Fig 1b: phenotype vs treatment duration (internal validity check)
    order = ["Sensitive", "Acquired Resistance", "Intrinsic Resistance"]
    data = [pd.to_numeric(sp.loc[phen == o, "treat_days"], errors="coerce").dropna().values
            for o in order]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    bp = ax.boxplot(data, positions=[1, 2, 3], showfliers=False,
                    patch_artist=True, widths=0.55)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([o.replace(" Resistance", "\nResistance") for o in order], fontsize=8)
    for b, c in zip(bp["boxes"], ["#4c8c4a", "#d9a441", "#c0392b"]):
        b.set_facecolor(c)
        b.set_alpha(0.55)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        ax.scatter(i + 1 + rng.normal(0, 0.045, len(d)), d, s=9, color="#333333", alpha=0.6)
    ax.set_ylabel("days on CDK4/6 inhibitor")
    ax.set_title("DFCI: treatment duration by resistance phenotype", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTD, "dfci_duration_by_phenotype.png"), dpi=200)
    plt.close(fig)
    log("")
    log("--- internal validity: treatment duration by phenotype ---")
    for o in order:
        d = pd.to_numeric(sp.loc[phen == o, "treat_days"], errors="coerce").dropna()
        log(f"  {o:<22} n={len(d):>2} median={d.median():.0f} d  mean={d.mean():.0f} d")
    a, b = data[0], data[2]
    log(f"  Sensitive vs Intrinsic: delta={st.cliffs_delta(a, b):+.3f} "
        f"MWU p={st.mannwhitney(a, b)['p']:.4f}")
    a, b = data[1], data[2]
    log(f"  Acquired vs Intrinsic: delta={st.cliffs_delta(a, b):+.3f} "
        f"MWU p={st.mannwhitney(a, b)['p']:.4f}")

    # Fig 2: ink4 co-occurrence with ESR1 mut
    s2f = dc[np.isfinite(dc["odds_ratio"]) & (dc["odds_ratio"] > 0)].head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y = np.arange(len(s2f))
    ax.errorbar(s2f["odds_ratio"], y,
                xerr=[np.clip(s2f["odds_ratio"] - s2f["or_lo"], 0, None),
                      np.clip(s2f["or_hi"] - s2f["odds_ratio"], 0, None)],
                fmt="o", ms=4, lw=1, color="#2c6fbb", ecolor="#888888")
    ax.axvline(1, color="black", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{p} (n={int(n)})" for p, n in zip(s2f["partner"], s2f["n_partner"])],
                       fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio of co-occurring with ESR1 mutation (log scale)")
    ax.set_title("MSK INK4 metastatic cohort (n=1365): co-occurrence with ESR1 mutation", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTD, "ink4_cooccurrence_ESR1mut.png"), dpi=200)
    plt.close(fig)

    # Fig 3: alteration frequency bar
    f = freq.head(28).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, max(3.0, 0.25 * len(f))))
    ax.barh(np.arange(len(f)), f.values * 100, color="#4c8c4a")
    ax.set_yticks(np.arange(len(f)))
    ax.set_yticklabels(f.index, fontsize=7)
    ax.set_xlabel("carrier frequency (%)")
    ax.set_title("MSK INK4 metastatic cohort (n=1365): alteration frequencies", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTD, "ink4_alteration_frequency.png"), dpi=200)
    plt.close(fig)
    log("")
    log("figures written")
except Exception as e:
    log("")
    log(f"figure step skipped: {type(e).__name__}: {e}")

flush()
print("OK")
