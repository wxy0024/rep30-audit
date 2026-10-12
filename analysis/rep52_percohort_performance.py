"""
REP52 panel: per-cohort performance in TCGA-BRCA and METABRIC.

For each of the 52 genes that form the overall-survival panel (results/rep_genes.csv),
report performance separately in:
    - METABRIC (microarray, the cohort the penalised multivariate Cox was fitted on)
    - TCGA-BRCA (RNA-seq, held out)
    - POOLED (per-cohort z-scores concatenated, cohort included as covariate)

Per cohort, per gene we report:
    * univariate Cox HR per 1 SD, 95% CI, z, p, BH q
    * penalised multivariate Cox coefficient over the 52-gene set (contribution)

Also writes the trimmed REP30 table with Cox coefficient, direction, per-cohort
coverage and the same per-cohort HRs.

Run with the managed venv (has lifelines):
  C:/Users/mao02/.workbuddy/binaries/python/envs/default/Scripts/python.exe
     analysis/rep52_percohort_performance.py
"""
import io
import os
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = r"E:/01.WorkSpace/01.Breast_cancer_gene_panel/resistance_signature"
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "rep52_trim")
os.makedirs(OUT, exist_ok=True)

TCGA_DIR = os.path.join(DATA, "01_core", "TCGA-BRCA", "brca_tcga_pan_can_atlas_2018")
MET_DIR = os.path.join(DATA, "01_core", "METABRIC", "brca_metabric")


# ------------------------------------------------------------------ loaders
def read_header(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.readline().rstrip("\n").split("\t")


def load_expr(path, keep_genes):
    """Gene x sample matrix, restricted to keep_genes (chunked: files are 150-700 MB)."""
    cols = read_header(path)
    samples = cols[2:]
    want = set(keep_genes)
    chunks = []
    for ch in pd.read_csv(path, sep="\t", header=0, names=cols, skiprows=1,
                          chunksize=20000, low_memory=False,
                          dtype={cols[0]: str, cols[1]: str}):
        ch = ch.rename(columns={cols[0]: "Hugo", cols[1]: "Entrez"})
        ch["Hugo"] = ch["Hugo"].fillna("").astype(str).str.strip()
        ch = ch[(ch["Hugo"] != "") & (~ch["Hugo"].str.startswith("?"))]
        ch = ch[ch["Hugo"].isin(want)]
        if len(ch):
            chunks.append(ch)
    raw = pd.concat(chunks, ignore_index=True)
    expr = raw.groupby("Hugo", sort=False)[samples].mean()

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
    out = pd.DataFrame({"PATIENT_ID": clin["PATIENT_ID"].astype(str),
                        "time": time, "event": event})
    out = out.dropna(subset=["time", "event"])
    out = out[out["time"] > 0]
    return out


def benjamini_hochberg(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def build(expr_path, clin_path, genes, name):
    expr = load_expr(expr_path, genes)
    clin = get_os(load_clinical(clin_path))
    common = [c for c in expr.columns if c in set(clin["PATIENT_ID"])]
    expr = expr[common]
    clin = clin.set_index("PATIENT_ID").loc[common]
    X = expr.reindex(genes).values.T.astype(float)     # (n_samples, n_genes)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0
    Xz = np.nan_to_num((X - mu) / sd, nan=0.0)
    print(f"[{name}] n={len(common)} events={int(clin['event'].sum())} genes={len(genes)}")
    return {"name": name, "Xz": Xz, "time": clin["time"].values.astype(float),
            "event": clin["event"].values.astype(int)}


# ------------------------------------------------------------------ data
rep52 = pd.read_csv(os.path.join(RES, "rep_genes.csv"))
genes = rep52["gene"].tolist()
print(f"[panel] {len(genes)} genes loaded from rep_genes.csv")

met = build(os.path.join(MET_DIR, "data_mrna_illumina_microarray.txt"),
            os.path.join(MET_DIR, "data_clinical_patient.txt"), genes, "METABRIC")
tcg = build(os.path.join(TCGA_DIR, "data_mrna_seq_v2_rsem.txt"),
            os.path.join(TCGA_DIR, "data_clinical_patient.txt"), genes, "TCGA-BRCA")


# ------------------------------------------------------------------ univariate per cohort
def univariate(cohort):
    """Univariate Cox per gene: HR per 1 SD, 95% CI, z, p. Returns dict of arrays."""
    t, e, X = cohort["time"], cohort["event"], cohort["Xz"]
    n = X.shape[1]
    hr = np.zeros(n); lo = np.zeros(n); hi = np.zeros(n)
    z = np.zeros(n); pv = np.zeros(n)
    for j in range(n):
        df = pd.DataFrame({"time": t, "event": e, "g": X[:, j]})
        try:
            cph = CoxPHFitter()
            cph.fit(df, duration_col="time", event_col="event")
            s = cph.summary.loc["g"]
            hr[j] = np.exp(s["coef"])
            lo[j] = np.exp(s["coef lower 95%"])
            hi[j] = np.exp(s["coef upper 95%"])
            z[j] = s["z"]
            pv[j] = s["p"]
        except Exception:
            hr[j] = lo[j] = hi[j] = z[j] = np.nan
            pv[j] = np.nan
    return {"hr": hr, "lo": lo, "hi": hi, "z": z, "p": pv,
            "q": benjamini_hochberg(pv)}


print("[fit] univariate Cox per gene ...")
u_met = univariate(met)
u_tcg = univariate(tcg)

# pooled: z-scores already per cohort -> concatenate, cohort as covariate
pool_time = np.concatenate([met["time"], tcg["time"]])
pool_event = np.concatenate([met["event"], tcg["event"]])
pool_X = np.vstack([met["Xz"], tcg["Xz"]])
pool_coh = np.concatenate([np.zeros(len(met["time"])), np.ones(len(tcg["time"]))])
pool = {"name": "POOLED", "Xz": pool_X, "time": pool_time, "event": pool_event}
u_pool = {"hr": np.zeros(len(genes)), "lo": np.zeros(len(genes)), "hi": np.zeros(len(genes)),
          "z": np.zeros(len(genes)), "p": np.zeros(len(genes)), "q": np.zeros(len(genes))}
for j in range(len(genes)):
    df = pd.DataFrame({"time": pool_time, "event": pool_event,
                       "g": pool_X[:, j], "coh": pool_coh})
    try:
        cph = CoxPHFitter()
        cph.fit(df, duration_col="time", event_col="event")
        s = cph.summary.loc["g"]
        u_pool["hr"][j] = np.exp(s["coef"])
        u_pool["lo"][j] = np.exp(s["coef lower 95%"])
        u_pool["hi"][j] = np.exp(s["coef upper 95%"])
        u_pool["z"][j] = s["z"]
        u_pool["p"][j] = s["p"]
    except Exception:
        u_pool["hr"][j] = u_pool["lo"][j] = u_pool["hi"][j] = np.nan
        u_pool["z"][j] = np.nan
        u_pool["p"][j] = np.nan
u_pool["q"] = benjamini_hochberg(u_pool["p"])


# ------------------------------------------------------------------ multivariate per cohort
def multivariate(cohort):
    df = pd.DataFrame(cohort["Xz"], columns=genes)
    df["time"] = cohort["time"]
    df["event"] = cohort["event"]
    cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.0)
    cph.fit(df, duration_col="time", event_col="event")
    return cph.params_.reindex(genes).values


print("[fit] penalised multivariate Cox (52 genes) per cohort ...")
mv_met = multivariate(met)
mv_tcg = multivariate(tcg)
dfp = pd.DataFrame(pool_X, columns=genes)
dfp["time"] = pool_time
dfp["event"] = pool_event
dfp["coh"] = pool_coh
cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.0)
cph.fit(dfp, duration_col="time", event_col="event")
mv_pool = cph.params_.reindex(genes).values


# ------------------------------------------------------------------ assemble
assess = pd.read_csv(os.path.join(OUT, "rep52_trim_assessment.csv"))
assess = assess.set_index("gene")

rows = []
for rank, (_, r) in enumerate(rep52.iterrows(), start=1):
    g = r["gene"]
    a = assess.loc[g]
    rows.append({
        "rank": rank,
        "gene": g,
        "panel_coef": round(float(r["coef"]), 4),
        "panel_HR": round(float(r["HR"]), 4),
        "METABRIC_HR_perSD": round(float(u_met["hr"][rank - 1]), 4),
        "METABRIC_CI95": "[%.3f-%.3f]" % (u_met["lo"][rank - 1], u_met["hi"][rank - 1]),
        "METABRIC_p": float(u_met["p"][rank - 1]),
        "METABRIC_q": float(u_met["q"][rank - 1]),
        "METABRIC_multivar_coef": round(float(mv_met[rank - 1]), 4),
        "TCGA_HR_perSD": round(float(u_tcg["hr"][rank - 1]), 4),
        "TCGA_CI95": "[%.3f-%.3f]" % (u_tcg["lo"][rank - 1], u_tcg["hi"][rank - 1]),
        "TCGA_p": float(u_tcg["p"][rank - 1]),
        "TCGA_q": float(u_tcg["q"][rank - 1]),
        "TCGA_multivar_coef": round(float(mv_tcg[rank - 1]), 4),
        "POOLED_HR_perSD": round(float(u_pool["hr"][rank - 1]), 4),
        "POOLED_CI95": "[%.3f-%.3f]" % (u_pool["lo"][rank - 1], u_pool["hi"][rank - 1]),
        "POOLED_p": float(u_pool["p"][rank - 1]),
        "POOLED_q": float(u_pool["q"][rank - 1]),
        "POOLED_multivar_coef": round(float(mv_pool[rank - 1]), 4),
        "direction_concordant": "yes" if (u_met["hr"][rank - 1] - 1) * (u_tcg["hr"][rank - 1] - 1) > 0 else "no",
        "top_tertile_contribution": int(a["abs_coef_top_tertile"]),
        "confirmed_in_HRplus_rebuild": int(a["in_HRplus108"]),
        "sc_hit_GSE161529": int(a["sc_hit_GSE161529"]),
        "sc_hit_GSE249361": int(a["sc_hit_GSE249361"]),
        "support_lines": int(a["support"]),
        "tier": a["tier"],
        "in_REP30": "yes" if a["tier"] in ("A_core", "B_keep") else "no",
    })

tab52 = pd.DataFrame(rows)
p52 = os.path.join(OUT, "REP52_percohort_performance.csv")
tab52.to_csv(p52, index=False)
print("\n[write]", p52)


# ------------------------------------------------------------------ REP30 table
s7 = pd.read_csv(os.path.join(ROOT, "manuscript", "tables", "TableS1_rep30_genes.csv"))
idx = {g: i for i, g in enumerate(genes)}
s7 = s7.copy()
for prefix, u in (("METABRIC", u_met), ("TCGA", u_tcg), ("POOLED", u_pool)):
    s7[prefix + "_HR_perSD"] = [round(float(u["hr"][idx[g]]), 3) for g in s7["gene"]]
    s7[prefix + "_CI95"] = ["[%.2f-%.2f]" % (u["lo"][idx[g]], u["hi"][idx[g]]) for g in s7["gene"]]
    s7[prefix + "_p"] = [float(u["p"][idx[g]]) for g in s7["gene"]]
    s7[prefix + "_q"] = [float(u["q"][idx[g]]) for g in s7["gene"]]

order = ["gene", "cox_coefficient", "direction"] + \
        ["METABRIC_HR_perSD", "METABRIC_CI95", "METABRIC_p", "METABRIC_q",
         "TCGA_HR_perSD", "TCGA_CI95", "TCGA_p", "TCGA_q",
         "POOLED_HR_perSD", "POOLED_CI95", "POOLED_p", "POOLED_q"] + \
        [c for c in s7.columns if c not in
         ("gene", "cox_coefficient", "direction") and not c.startswith(("METABRIC_", "TCGA_", "POOLED_"))]
s7 = s7[order]
p30 = os.path.join(OUT, "REP30_panel_table.csv")
s7.to_csv(p30, index=False)
print("[write]", p30)

# Excel bundle
try:
    px = os.path.join(OUT, "REP52_and_REP30_tables.xlsx")
    with pd.ExcelWriter(px, engine="openpyxl") as w:
        tab52.to_excel(w, sheet_name="REP52_percohort", index=False)
        s7.to_excel(w, sheet_name="REP30_panel", index=False)
    print("[write]", px)
except Exception as ex:
    print("[warn] xlsx not written:", ex)

# ------------------------------------------------------------------ console summary
print("\n=== counts ===")
print("52-gene panel:", len(tab52))
print("retained in REP30:", int((tab52["in_REP30"] == "yes").sum()))
print("dropped:", int((tab52["in_REP30"] == "no").sum()))
print("\n=== direction concordance (METABRIC vs TCGA univariate sign) ===")
print(tab52["direction_concordant"].value_counts().to_dict())
print("\n=== top 15 by |panel coef| ===")
print(tab52.head(15)[["rank", "gene", "panel_coef", "panel_HR",
                      "METABRIC_HR_perSD", "METABRIC_q",
                      "TCGA_HR_perSD", "TCGA_q", "tier"]].to_string(index=False))
