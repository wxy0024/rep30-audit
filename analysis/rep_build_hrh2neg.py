"""
Layer 2b - REP rebuilt WITHIN the HR+/HER2- (luminal) subgroup.

Motivation: the original REP (rep_build.py) was trained on all BC subtypes and is
therefore partly driven by the good-vs-poor prognosis split between luminal / HER2 /
basal tumors. To obtain a resistance signature that is *informative inside the
endocrine-treated HR+/HER2- population* (the clinically relevant cohort for
endocrine + CDK4/6i resistance), we restrict both training and testing to
HR+/HER2- tumors defined by intrinsic molecular subtype:
    TCGA     : SUBTYPE in {BRCA_LumA, BRCA_LumB}
    METABRIC : CLAUDIN_SUBTYPE in {LumA, LumB}

Method mirrors rep_build.py:
    (1) vectorized univariate Breslow Cox screen (FDR<0.05) within HR+/HER2-;
    (2) penalized multivariate Cox on METABRIC-HR+ -> Cox-weighted risk score;
    (3) TRAIN=METABRIC-HR+, TEST=TCGA-HR+, plus pooled HR+.
Bonus comparisons:
    (B) original 52-gene REP (rep_genes.csv) applied to the HR+/HER2- subset;
    (C) new HR+-specific REP applied to the FULL (all-subtype) cohort.
Output: results/rep_genes_hrh2neg.csv, rep_hrh2neg_summary.csv, rep_hrh2neg_*.png
"""
import os, io, gzip, warnings, math
import numpy as np
import pandas as pd
from scipy import stats
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
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
ORIG_REP = os.path.join(RES, "rep_genes.csv")

# ---------------------------------------------------------------- loaders (copied from rep_build.py)
def load_expr(path):
    raw = pd.read_csv(path, sep="\t", header=0, low_memory=False, dtype=None)
    raw = raw.rename(columns={raw.columns[0]: "Hugo", raw.columns[1]: "Entrez"})
    raw["Hugo"] = raw["Hugo"].fillna("").astype(str).str.strip()
    raw = raw[(raw["Hugo"] != "") & (~raw["Hugo"].str.startswith("?"))]
    expr = raw.groupby("Hugo", sort=False)[raw.columns[2:]].mean()
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
    event = st.str.contains("DECEASED").astype(int)
    if event.sum() == 0:
        event = (pd.to_numeric(st, errors="coerce").fillna(0).astype(int) == 1).astype(int)
    time = pd.to_numeric(clin[time_col], errors="coerce")
    pat = clin["PATIENT_ID"].astype(str)
    out = pd.DataFrame({"PATIENT_ID": pat, "time": time, "event": event})
    out = out.dropna(subset=["time", "event"])
    out = out[(out["time"] > 0)]
    return out

def hrh2neg_patients(pat_clin, source):
    """Return set of PATIENT_ID that are HR+/HER2- by intrinsic subtype."""
    if source == "TCGA":
        sub = pat_clin["SUBTYPE"].astype(str)
        mask = sub.isin(["BRCA_LumA", "BRCA_LumB"])
    else:  # METABRIC
        sub = pat_clin["CLAUDIN_SUBTYPE"].astype(str)
        mask = sub.isin(["LumA", "LumB"])
    return set(pat_clin.loc[mask, "PATIENT_ID"].astype(str)), int(mask.sum())

# ---------------------------------------------------------------- univariate Cox screen (vectorized, Breslow)
def cox_screen(time, event, X):
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

# ---------------------------------------------------------------- assemble cohort (optionally restricted to HR+/HER2-)
def build_cohort(expr_path, clin_path, pat_clin_path, source, name, keep=None):
    expr = load_expr(expr_path)
    clin = get_os(load_clinical(clin_path))
    common = list(set(expr.columns) & set(clin["PATIENT_ID"]))
    if keep is not None:
        common = [c for c in common if c in keep]
    expr = expr[common]
    clin = clin.set_index("PATIENT_ID").loc[common]
    X = expr.values.T
    mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    Xz = np.nan_to_num(Xz, nan=0.0)
    time = clin["time"].values.astype(float)
    event = clin["event"].values.astype(int)
    print(f"[{name}] n={len(common)} events={int(event.sum())} genes={expr.shape[0]}")
    return {"name": name, "genes": expr.index.values, "Xz": Xz, "time": time, "event": event}

def restrict(c, gset):
    idx = [list(c["genes"]).index(g) for g in gset]
    return c["Xz"][:, idx]

def risk_score(X, coef):
    return X @ coef

def evaluate(name, time, event, score):
    if np.std(score) == 0 or np.isnan(score).any():
        return {"cohort": name, "n": len(time), "events": int(event.sum()),
                "HR_perSD": float("nan"), "p": float("nan"),
                "logrank_p": float("nan"), "Cindex": float("nan")}
    grp = (score > np.median(score)).astype(int)
    lr = logrank_test(time[grp == 0], time[grp == 1],
                      event_observed_A=event[grp == 0], event_observed_B=event[grp == 1])
    df = pd.DataFrame({"time": time, "event": event, "score": score})
    c = CoxPHFitter(); c.fit(df, duration_col="time", event_col="event")
    hr = math.exp(c.params_["score"]); p = c.summary["p"][0]
    ci = concordance_index(time, -score, event)
    return {"cohort": name, "n": len(time), "events": int(event.sum()),
            "HR_perSD": hr, "p": p, "logrank_p": lr.p_value, "Cindex": ci}

def km_plot(time, event, score, title, fname):
    grp = (score > np.median(score)).astype(int)
    plt.figure(figsize=(6, 5))
    kmf = KaplanMeierFitter()
    for lab, g in [("Low risk", 0), ("High risk", 1)]:
        m = grp == g
        if m.sum() == 0: continue
        kmf.fit(time[m], event_observed=event[m], label=lab)
        kmf.plot_survival_function(ci_show=False)
    plt.title(title); plt.xlabel("Months"); plt.ylabel("OS probability")
    plt.tight_layout(); plt.savefig(os.path.join(RES, fname), dpi=130); plt.close()

# ---------------------------------------------------------------- 1) define HR+/HER2- sets
tcga_pat = load_clinical(os.path.join(TCGA_DIR, "data_clinical_patient.txt"))
met_pat  = load_clinical(os.path.join(MET_DIR,  "data_clinical_patient.txt"))
tcga_keep, tcga_n = hrh2neg_patients(tcga_pat, "TCGA")
met_keep,  met_n  = hrh2neg_patients(met_pat,  "METABRIC")
print(f"[subtype] TCGA HR+/HER2- (LumA+LumB): {tcga_n}; METABRIC HR+/HER2- (LumA+LumB): {met_n}")

# ---------------------------------------------------------------- 2) build HR+/HER2- cohorts
train = build_cohort(os.path.join(MET_DIR,  "data_mrna_illumina_microarray.txt"),
                     os.path.join(MET_DIR,  "data_clinical_patient.txt"), MET_DIR,
                     "METABRIC", "METABRIC-HR+", keep=met_keep)
test  = build_cohort(os.path.join(TCGA_DIR, "data_mrna_seq_v2_rsem.txt"),
                     os.path.join(TCGA_DIR, "data_clinical_patient.txt"), TCGA_DIR,
                     "TCGA", "TCGA-HR+", keep=tcga_keep)

genes_tr = set(train["genes"]); genes_te = set(test["genes"])
common_genes = sorted(genes_tr & genes_te)
print(f"[genes] common to both HR+ cohorts: {len(common_genes)}")
Xtr = restrict(train, common_genes)
Xte = restrict(test,  common_genes)

# ---------------------------------------------------------------- 3) screen + select (TRAIN = METABRIC-HR+)
z, p = cox_screen(train["time"], train["event"], Xtr)
q = benjamini_hochberg(p)
screen = pd.DataFrame({"gene": common_genes, "uni_z": z, "uni_p": p, "qvalue": q}).sort_values("uni_p")
cands = screen[screen["qvalue"] < 0.05].copy()
cands = cands[cands["uni_z"].abs() > 2.0]
cands = cands.reindex(cands["uni_z"].abs().sort_values(ascending=False).index).head(300)
print(f"[screen] METABRIC-HR+ FDR<0.05 & |z|>2 & top300: {len(cands)}")

cand_genes = cands["gene"].tolist()
ci = [list(common_genes).index(g) for g in cand_genes]
Xtr_c = Xtr[:, ci]; Xte_c = Xte[:, ci]
df_tr = pd.DataFrame(Xtr_c, columns=cand_genes)
df_tr["time"], df_tr["event"] = train["time"], train["event"]
cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.0)
cph.fit(df_tr, duration_col="time", event_col="event")
coefs = cph.params_
hr = np.exp(coefs)
multivar = pd.DataFrame({"gene": cand_genes, "coef": coefs.values, "HR": hr.values})
keep = multivar[multivar["coef"].abs() > 0.05]["gene"].tolist()
print(f"[multivar] kept |coef|>0.05: {len(keep)}")
ki = [list(cand_genes).index(g) for g in keep]
coef_keep = coefs.values[ki]
Xtr_k = Xtr_c[:, ki]; Xte_k = Xte_c[:, ki]

# ---------------------------------------------------------------- 4) evaluate new HR+-specific signature
rows = []
r_tr = evaluate("METABRIC-HR+(train)", train["time"], train["event"], risk_score(Xtr_k, coef_keep))
rows.append(r_tr)
r_te = evaluate("TCGA-HR+(test)",      test["time"],  test["event"],  risk_score(Xte_k, coef_keep))
rows.append(r_te)
# pooled
s_tr = risk_score(Xtr_k, coef_keep); s_te = risk_score(Xte_k, coef_keep)
pt = np.concatenate([train["time"], test["time"]]); pe = np.concatenate([train["event"], test["event"]])
ps = np.concatenate([s_tr, s_te]); pc = np.concatenate([np.zeros(len(s_tr)), np.ones(len(s_te))])
dfp = pd.DataFrame({"time": pt, "event": pe, "score": ps, "cohort": pc})
cp = CoxPHFitter(); cp.fit(dfp, duration_col="time", event_col="event", formula="score + C(cohort)")
hr_pool = math.exp(cp.params_["score"]); p_pool = cp.summary["p"]["score"]
ci_pool = concordance_index(pt, -ps, pe)
rows.append({"cohort": "POOLED-HR+", "n": len(pt), "events": int(pe.sum()),
             "HR_perSD": hr_pool, "p": p_pool, "logrank_p": float("nan"), "Cindex": ci_pool})

# ---------------------------------------------------------------- 5) comparison B: original 52-gene REP in HR+ subset
if os.path.exists(ORIG_REP):
    orig = pd.read_csv(ORIG_REP)
    og = [g for g in orig["gene"] if g in common_genes]
    oi_tr = [list(common_genes).index(g) for g in og]
    ocoef = np.array([orig.loc[orig["gene"] == g, "coef"].values[0] for g in og])
    oi_te = oi_tr
    rB_tr = evaluate("ORIG-REP in METABRIC-HR+", train["time"], train["event"], risk_score(Xtr[:, oi_tr], ocoef))
    rB_te = evaluate("ORIG-REP in TCGA-HR+",      test["time"],  test["event"],  risk_score(Xte[:, oi_te], ocoef))
    rows += [rB_tr, rB_te]

# ---------------------------------------------------------------- save
summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(RES, "rep_hrh2neg_summary.csv"), index=False)
print("\n=== REP (HR+/HER2- rebuilt) summary ===")
print(summ.to_string(index=False))

sig = multivar[multivar["gene"].isin(keep)].sort_values("coef", key=lambda s: s.abs(), ascending=False)
sig.to_csv(os.path.join(RES, "rep_genes_hrh2neg.csv"), index=False)
curated = pd.read_csv(GENE_CSV)
cur_genes = set(curated["gene"].str.upper())
sig["in_curated_resistance_set"] = sig["gene"].str.upper().isin(cur_genes)
overlap = sig[sig["in_curated_resistance_set"]]
print(f"\n[overlap] HR+-REP genes={len(sig)}; overlap with curated resistance set={len(overlap)}")
overlap.to_csv(os.path.join(RES, "rep_hrh2neg_resistance_overlap.csv"), index=False)

# plots
km_plot(train["time"], train["event"], risk_score(Xtr_k, coef_keep), "REP-HR+ risk - METABRIC-HR+ (train)", "rep_hrh2neg_METABRIC_km.png")
km_plot(test["time"],  test["event"],  risk_score(Xte_k, coef_keep), "REP-HR+ risk - TCGA-HR+ (test)",      "rep_hrh2neg_TCGA_km.png")

print("\nDONE. Outputs in", RES)
