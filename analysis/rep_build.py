"""
Layer 2 - REP (Resistance Expression Program) signature builder + cross-cohort validation.

Input : TCGA-BRCA (RSEM RNA-seq) + METABRIC (illumina microarray), gene-level expression
        + cBioPortal clinical (Overall Survival Status / Months).
Method: (1) vectorized univariate Cox score test (Breslow) across all genes -> FDR screening;
        (2) penalized multivariate Cox on candidates -> Cox-coefficient-weighted risk score;
        (3) internal cross-cohort validation: TRAIN = METABRIC, TEST = TCGA, plus pooled.
Output: results/rep_genes.csv, rep_summary.csv, rep_*_km.png, rep_*_cindex.csv
"""
import os, io, gzip, warnings, math
import numpy as np
import pandas as pd
from scipy import stats
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

ROOT = r"E:\01.WorkSpace\01.Breast_cancer_gene_panel\resistance_signature"
DATA = os.path.join(ROOT, "data")
RES  = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

TCGA_DIR = os.path.join(DATA, "01_core", "TCGA-BRCA", "brca_tcga_pan_can_atlas_2018")
MET_DIR  = os.path.join(DATA, "01_core", "METABRIC", "brca_metabric")
GENE_CSV = os.path.join(ROOT, "resistance_genes.csv")

# ---------------------------------------------------------------- loaders
def load_expr(path):
    """Return gene(>=1) x patient DataFrame (z-scored later). Aggregates dup Hugo + dup patient."""
    raw = pd.read_csv(path, sep="\t", header=0, low_memory=False, dtype=None)
    raw = raw.rename(columns={raw.columns[0]: "Hugo", raw.columns[1]: "Entrez"})
    raw["Hugo"] = raw["Hugo"].fillna("").astype(str).str.strip()
    raw = raw[(raw["Hugo"] != "") & (~raw["Hugo"].str.startswith("?"))]
    expr = raw.groupby("Hugo", sort=False)[raw.columns[2:]].mean()
    # sample -> patient: TCGA barcodes "TCGA-XXX-01" -> "TCGA-XXX"; keep others as-is
    def col_to_patient(c):
        if c.startswith("TCGA-") and c.count("-") >= 3:
            return c.rsplit("-", 1)[0]
        return c
    expr.columns = [col_to_patient(c) for c in expr.columns]
    expr = expr.T.groupby(level=0).mean().T
    return expr.astype(float)

def load_clinical(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [l for l in f if not l.startswith("#")]
    hdr = next(i for i, l in enumerate(lines) if "PATIENT_ID" in l)
    return pd.read_csv(io.StringIO("".join(lines[hdr:])), sep="\t")

def get_os(clin):
    cols = list(clin.columns)
    cands_status = [c for c in cols if "OS" in c.upper() and "STATUS" in c.upper()]
    if not cands_status:
        cands_status = [c for c in cols if "VITAL" in c.upper()]
    status_col = cands_status[0]
    cands_time = [c for c in cols if "OS" in c.upper() and "MONTH" in c.upper()]
    if not cands_time:
        cands_time = [c for c in cols if "OS" in c.upper() and "DAY" in c.upper()]
    time_col = cands_time[0]
    st = clin[status_col].astype(str).str.strip().str.upper()
    # cBioPortal encodes as "1:DECEASED" / "0:LIVING"
    event = st.str.contains("DECEASED").astype(int)
    if event.sum() == 0:
        event = (pd.to_numeric(st, errors="coerce").fillna(0).astype(int) == 1).astype(int)
    time = pd.to_numeric(clin[time_col], errors="coerce")
    pat = clin["PATIENT_ID"].astype(str)
    out = pd.DataFrame({"PATIENT_ID": pat, "time": time, "event": event})
    out = out.dropna(subset=["time", "event"])
    out = out[(out["time"] > 0)]
    return out

# ---------------------------------------------------------------- univariate Cox score test (vectorized, Breslow)
def cox_screen(time, event, X):
    """X: (n,G) z-scored. Returns z-stat and two-sided p per gene."""
    order = np.argsort(time, kind="mergesort")
    t = time[order]; e = event[order].astype(int); Xs = X[order]
    n, G = Xs.shape
    cs_x  = np.cumsum(Xs[::-1], axis=0)[::-1]
    cs_x2 = np.cumsum((Xs ** 2)[::-1], axis=0)[::-1]
    cs_n  = np.cumsum(np.ones(n)[::-1])[::-1]
    U = np.zeros(G); I = np.zeros(G)
    for tk in np.unique(t[e == 1]):
        idx = np.where(t == tk)[0]
        p0 = idx.min()
        nR = cs_n[p0]; Sx = cs_x[p0]; Sx2 = cs_x2[p0]
        meanR = Sx / nR
        varR = Sx2 / nR - meanR ** 2
        dk = len(idx)
        U += Xs[idx].sum(axis=0) - dk * meanR
        I += dk * varR
    z = U / np.sqrt(I + 1e-12)
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return z, p

def benjamini_hochberg(p):
    p = np.asarray(p); n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(q, 0, 1)
    return out

# ---------------------------------------------------------------- assemble cohort
def build_cohort(expr_path, clin_path, name):
    expr = load_expr(expr_path)
    clin = get_os(load_clinical(clin_path))
    common = list(set(expr.columns) & set(clin["PATIENT_ID"]))
    expr = expr[common]
    clin = clin.set_index("PATIENT_ID").loc[common]
    # z-score per gene
    X = expr.values.T  # (n_samples, n_genes)
    mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    Xz = np.nan_to_num(Xz, nan=0.0)
    time = clin["time"].values.astype(float)
    event = clin["event"].values.astype(int)
    print(f"[{name}] n={len(common)} events={int(event.sum())} genes={expr.shape[0]}")
    return {"name": name, "genes": expr.index.values, "Xz": Xz, "time": time, "event": event}

train = build_cohort(os.path.join(MET_DIR,  "data_mrna_illumina_microarray.txt"),
                     os.path.join(MET_DIR,  "data_clinical_patient.txt"), "METABRIC")
test  = build_cohort(os.path.join(TCGA_DIR, "data_mrna_seq_v2_rsem.txt"),
                     os.path.join(TCGA_DIR, "data_clinical_patient.txt"), "TCGA-BRCA")

# align gene sets
genes_tr = set(train["genes"]); genes_te = set(test["genes"])
common_genes = sorted(genes_tr & genes_te)
print(f"[genes] common to both cohorts: {len(common_genes)}")

def restrict(c, gset):
    idx = [list(c["genes"]).index(g) for g in gset]
    return c["Xz"][:, idx]

Xtr = restrict(train, common_genes)
Xte = restrict(test,  common_genes)

# ---------------------------------------------------------------- screen + select (TRAIN)
z, p = cox_screen(train["time"], train["event"], Xtr)
q = benjamini_hochberg(p)
screen = pd.DataFrame({"gene": common_genes, "uni_z": z, "uni_p": p, "qvalue": q})
screen = screen.sort_values("uni_p")
cands = screen[screen["qvalue"] < 0.05].copy()
print(f"[screen] METABRIC FDR<0.05 genes: {len(cands)}")
# keep direction-consistent candidates with reasonable effect (|z|>2)
cands = cands[cands["uni_z"].abs() > 2.0]
# cap candidate set to top 300 by |z| to avoid collinearity in multivariate step
cands = cands.reindex(cands["uni_z"].abs().sort_values(ascending=False).index).head(300)
print(f"[screen] after |z|>2 + top300 cap: {len(cands)}")

# ---------------------------------------------------------------- multivariate Cox (penalized) on candidates
cand_genes = cands["gene"].tolist()
ci = [list(common_genes).index(g) for g in cand_genes]
Xtr_c = Xtr[:, ci]; Xte_c = Xte[:, ci]
df_tr = pd.DataFrame(Xtr_c, columns=cand_genes)
df_tr["time"], df_tr["event"] = train["time"], train["event"]
cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.0)
cph.fit(df_tr, duration_col="time", event_col="event")
coefs = cph.params_
hr = np.exp(coefs)
multivar = pd.DataFrame({"gene": cand_genes, "coef": coefs.values,
                         "HR": hr.values})
# keep parsimonious: |coef| meaningful
keep = multivar[multivar["coef"].abs() > 0.05]["gene"].tolist()
print(f"[multivar] candidate genes={len(cand_genes)}; kept |coef|>0.05: {len(keep)}")
ki = [list(cand_genes).index(g) for g in keep]
coef_keep = coefs.values[ki]
Xtr_k = Xtr_c[:, ki]; Xte_k = Xte_c[:, ki]

def risk_score(X, coef):
    return X @ coef

# ---------------------------------------------------------------- validation helper
def evaluate(name, time, event, score):
    med = np.median(score)
    grp = (score > med).astype(int)
    if np.std(score) == 0 or np.isnan(score).any():
        return {"cohort": name, "n": len(time), "events": int(event.sum()),
                "HR_high_vs_low": float("nan"), "p_high_vs_low": float("nan"),
                "logrank_p": float("nan"), "Cindex": float("nan")}, grp, med
    kmf_h = KaplanMeierFitter(); kmf_l = KaplanMeierFitter()
    # logrank via lifelines fit + compute
    from lifelines.statistics import logrank_test
    lr = logrank_test(time[grp == 0], time[grp == 1],
                      event_observed_A=event[grp == 0], event_observed_B=event[grp == 1])
    # continuous Cox
    df = pd.DataFrame({"time": time, "event": event, "score": score})
    c = CoxPHFitter(); c.fit(df, duration_col="time", event_col="event")
    hr_cont = math.exp(c.params_["score"])
    p_cont = c.summary["p"][0]
    ci_val = concordance_index(time, -score, event)
    return {"cohort": name, "n": len(time), "events": int(event.sum()),
            "HR_high_vs_low": hr_cont, "p_high_vs_low": p_cont,
            "logrank_p": lr.p_value, "Cindex": ci_val}, grp, med

rows = []
# TRAIN
r_tr, g_tr, m_tr = evaluate("METABRIC(train)", train["time"], train["event"], risk_score(Xtr_k, coef_keep))
rows.append(r_tr)
r_te, g_te, m_te = evaluate("TCGA-BRCA(test)", test["time"], test["event"], risk_score(Xte_k, coef_keep))
rows.append(r_te)
# pooled (z per cohort already; combine, add cohort covariate)
score_tr = risk_score(Xtr_k, coef_keep); score_te = risk_score(Xte_k, coef_keep)
pool_time = np.concatenate([train["time"], test["time"]])
pool_event = np.concatenate([train["event"], test["event"]])
pool_score = np.concatenate([score_tr, score_te])
pool_cohort = np.concatenate([np.zeros(len(score_tr)), np.ones(len(score_te))])
dfp = pd.DataFrame({"time": pool_time, "event": pool_event, "score": pool_score, "cohort": pool_cohort})
c = CoxPHFitter(); c.fit(dfp, duration_col="time", event_col="event", formula="score + C(cohort)")
hr_pool = math.exp(c.params_["score"]); p_pool = c.summary["p"]["score"]
ci_pool = concordance_index(pool_time, -pool_score, pool_event)
# pooled KM by median of pooled score
medp = np.median(pool_score); grp_p = (pool_score > medp).astype(int)
from lifelines.statistics import logrank_test
lrp = logrank_test(pool_time[grp_p == 0], pool_time[grp_p == 1],
                   event_observed_A=pool_event[grp_p == 0], event_observed_B=pool_event[grp_p == 1])
rows.append({"cohort": "POOLED(MET+TCGA)", "n": len(pool_time), "events": int(pool_event.sum()),
             "HR_high_vs_low": hr_pool, "p_high_vs_low": p_pool,
             "logrank_p": lrp.p_value, "Cindex": ci_pool})

summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(RES, "rep_summary.csv"), index=False)
print("\n=== REP validation summary ===")
print(summ.to_string(index=False))

# ---------------------------------------------------------------- plots
def km_plot(time, event, grp, title, fname):
    plt.figure(figsize=(6, 5))
    kmf = KaplanMeierFitter()
    for lab, g in [("Low risk", 0), ("High risk", 1)]:
        m = grp == g
        if m.sum() == 0: continue
        kmf.fit(time[m], event_observed=event[m], label=lab)
        kmf.plot_survival_function(ci_show=False)
    plt.title(title); plt.xlabel("Months"); plt.ylabel("OS probability")
    plt.tight_layout(); plt.savefig(os.path.join(RES, fname), dpi=130); plt.close()

km_plot(train["time"], train["event"], g_tr, "REP risk - METABRIC (train)", "rep_METABRIC_km.png")
km_plot(test["time"],  test["event"],  g_te, "REP risk - TCGA-BRCA (test)",  "rep_TCGA-BRCA_km.png")
km_plot(pool_time, pool_event, grp_p, "REP risk - POOLED", "rep_POOLED_km.png")

# ---------------------------------------------------------------- save signature + overlap with curated resistance genes
sig = multivar[multivar["gene"].isin(keep)].sort_values("coef", key=lambda s: s.abs(), ascending=False)
sig.to_csv(os.path.join(RES, "rep_genes.csv"), index=False)
curated = pd.read_csv(GENE_CSV)
cur_genes = set(curated["gene"].str.upper())
sig["in_curated_resistance_set"] = sig["gene"].str.upper().isin(cur_genes)
overlap = sig[sig["in_curated_resistance_set"]]
print(f"\n[overlap] REP signature genes={len(sig)}; overlap with curated resistance set={len(overlap)}")
print(overlap[["gene", "coef", "HR"]].to_string(index=False))
overlap.to_csv(os.path.join(RES, "rep_resistance_overlap.csv"), index=False)

# top 20 table for quick view
print("\n=== Top 20 REP genes by |coef| ===")
print(sig.head(20)[["gene", "coef", "HR"]].to_string(index=False))
print("\nDONE. Outputs in", RES)
