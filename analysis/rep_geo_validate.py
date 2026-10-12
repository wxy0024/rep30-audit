"""
rep_geo_validate.py  -  External validation of the 52-gene REP expression signature
on public endocrine-treated breast cancer GEO cohorts.

What it does:
  1. Parses each GEO series matrix (probe-level) -> gene-level expression
     (probe->gene via the GPL annotation tables in data/05_gpl/, auto platform-match).
  2. Extracts survival outcome (time + event) per cohort-specific characteristic fields.
  3. Computes the REP signature score per sample (REP coefs from results/rep_genes.csv),
     z-scoring each gene within cohort for platform robustness.
  4. Validates: (a) KM log-rank high vs low; (b) univariate Cox HR of continuous score;
     (c) resistance enrichment - compare score between patients who RELAPSED within 5y
     on endocrine therapy (resistant) vs those censored >=5y (sensitive): Mann-Whitney
     + ROC AUC. This is the "resistance-specific" signal.
  5. Saves results/geo_validation_summary.csv + per-cohort KM plots + a cross-cohort
     forest plot of the signature's Cox HR.

NOTE on fixes vs the original draft:
  - GEO .annot tables begin with '^Annotation' / '!Annotation_*' / '!platform_table_begin'
    section lines; the real header is the line right after '!platform_table_begin'.
    We skip to it (comment="#" would NOT skip '^'/'!' lines).
  - All '!Sample_characteristics_ch*' lines share the SAME line label; each line is a
    DIFFERENT characteristic. We must accumulate every line (not key by the label) and
    build each sample's dict from all of them.
  - Characteristic strings may contain a colon inside parentheses, e.g.
    'metastasis (1: yes, 0: no/censored): 0'. A naive split(":",1) breaks the key, so we
    split on the first ':' that is OUTSIDE parentheses (split_keyval).

RUN (sandbox can't fetch, but the data is already local):
  python rep_geo_validate.py
"""
import os, io, re, glob, gzip
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test
    HAVE_LIFELINES = True
except Exception:
    HAVE_LIFELINES = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "02_acquired_resistance")
GPLDIR = os.path.join(ROOT, "data", "05_gpl")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

# cohort -> (series_matrix_file, outcome extractor key)
# GSE6532-GPL96 is EXCLUDED: its series matrix carries only grade/ggi/sample-name
# characteristics (no survival), so it cannot be outcome-validated from the matrix alone.
COHORTS = {
    "GSE9195":  ("GSE9195_series_matrix.txt.gz",  "rfs_days"),
    "GSE26971": ("GSE26971_series_matrix.txt.gz", "mfi_months"),
    "GSE17705": ("GSE17705_series_matrix.txt.gz", "dr_years"),
    "GSE45255": ("GSE45255_series_matrix.txt.gz", "dmfs_years"),
}

# cohorts where not every sample is endocrine-treated. NOTE: GSE45255 mixes
# endocrine and chemo regimens; its 'characteristics: endocrine? (0=no, 1=yes)'
# field name itself contains a colon, so it is NOT used to subset. The resistance
# enrichment contrast below is computed on all samples and flagged as exploratory
# for GSE45255 (GSE9195/26971/17705 are uniformly endocrine-treated).


# ============================================================ helpers
def split_keyval(s):
    """Split 'key: value' on the first ':' that is OUTSIDE parentheses."""
    s = s.strip().strip('"')
    depth = 0
    for i, c in enumerate(s):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == ":" and depth == 0:
            return s[:i].strip(), s[i + 1:].strip()
    return s, ""


# ============================================================ GPL annot loaders
def load_gpl_annots():
    """Return dict gpl -> {probe: gene} for every available annot file."""
    out = {}
    for gp in sorted(glob.glob(os.path.join(GPLDIR, "*.annot"))):
        gpl = os.path.splitext(os.path.basename(gp))[0]
        # locate the real table header (line right after '!platform_table_begin')
        header_idx = 0
        with open(gp, "r", encoding="utf-8", errors="replace") as fh:
            for i, l in enumerate(fh):
                if l.startswith("!platform_table_begin"):
                    header_idx = i + 1
                    break
        df = pd.read_csv(gp, sep="\t", skiprows=header_idx, header=0,
                         dtype=str, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        keep = [c for c in df.columns if c.lower() in ("id", "id_ref", "gene symbol")]
        if not keep:
            print(f"[gpl] {gpl}: no id/gene-symbol column found, skip")
            continue
        df = df[keep]
        df.columns = [c.lower() for c in df.columns]
        probe_col = "id" if "id" in df.columns else "id_ref"
        sym_col = "gene symbol"
        m = {}
        for _, row in df.iterrows():
            probe = str(row[probe_col]).strip()
            sym = str(row[sym_col]).strip() if sym_col in row else ""
            if probe and sym and sym not in ("", "NA", "---", "nan"):
                gene = re.split(r"\s+///\s+|\s+//\s+|\s+/\s+", sym)[0].strip()
                if gene:
                    m[probe] = gene
        out[gpl] = m
        print(f"[gpl] {gpl}: {len(m)} probe->gene mappings (header_idx={header_idx})")
    return out


# ============================================================ series matrix parser
def parse_series(gse, fname, gpl_annots):
    """Return (expr: genes x samples DataFrame, samples: list of meta dicts)."""
    path = os.path.join(DATA, gse, fname)
    with gzip.open(path, "rt") as f:
        lines = f.readlines()
    begin = None
    for i, l in enumerate(lines):
        if l.startswith("!series_matrix_table_begin"):
            begin = i
            break
    if begin is None:
        raise RuntimeError(f"{gse}: no !series_matrix_table_begin")
    header = lines[begin + 1].rstrip("\n").split("\t")
    # accumulate EVERY '!Sample_characteristics_ch*' line (each is one characteristic)
    ch_lines = []
    for l in lines[:begin]:
        if l.startswith("!Sample_characteristics_ch"):
            parts = l.rstrip("\n").split("\t")
            ch_lines.append([v.strip().strip('"') for v in parts[1:]])
    nsamp = len(header) - 1
    samples = []
    for j in range(nsamp):
        d = {}
        for vals in ch_lines:
            if j < len(vals) and ":" in vals[j]:
                kk, vv = split_keyval(vals[j])
                if kk:
                    d[kk] = vv
        samples.append(d)
    # expression table
    tbl = "".join(lines[begin + 1:])
    expr = pd.read_csv(io.StringIO(tbl), sep="\t", index_col=0, low_memory=False)
    expr.columns = [c.strip().strip('"') for c in expr.columns]
    expr.index = [str(i).strip().strip('"') for i in expr.index]
    expr = expr.apply(pd.to_numeric, errors="coerce")
    # platform auto-detect: maximize the FRACTION of the series' own probes that
    # are annotated by the platform (overlap / n_series_probes). This avoids a
    # small sub-platform (e.g. GPL201) winning with "coverage 1.0" just because all
    # its probes happen to sit inside a larger array. Tie-break by (absolute
    # overlap, then closeness of platform size to the series size) to pick the true
    # platform among Affymetrix siblings (GPL96 vs GPL570).
    probes = set(expr.index)
    n_probes = len(probes)
    best, best_ov, best_key = None, 0, (-1.0, 0, -10**9)
    for gpl, m in gpl_annots.items():
        ov = len(probes & set(m.keys()))
        series_cov = ov / n_probes if n_probes else 0.0
        size_diff = -abs(len(m) - n_probes)
        key = (series_cov, ov, size_diff)
        if key > best_key:
            best_key, best_ov, best = key, ov, gpl
    best_ratio = best_key[0]
    if best is None or best_ov < 50:
        raise RuntimeError(f"{gse}: no platform with enough probe overlap (best={best} ov={best_ov})")
    gmap = gpl_annots[best]
    print(f"[parse] {gse}: {nsamp} samples, {expr.shape[0]} probes, platform={best} "
          f"(overlap {best_ov}, series_coverage {best_ratio:.2f})")
    gene_rows = {}
    for probe, row in expr.iterrows():
        g = gmap.get(probe)
        if g:
            gene_rows.setdefault(g, []).append(row)
    if not gene_rows:
        raise RuntimeError(f"{gse}: no probes mapped to genes via {best}")
    gexpr = pd.DataFrame({g: pd.concat(v, axis=1).mean(axis=1) for g, v in gene_rows.items()})
    gexpr.columns = [c.strip().strip('"') for c in gexpr.columns]
    gexpr = gexpr.T  # genes (rows) x samples (cols) - matches rep_build.py convention
    return gexpr, samples


# ============================================================ outcome extraction
def extract_outcome(gse, key, samples):
    """Return (time array, event array, unit) per cohort-specific fields."""
    times, events = [], []
    for d in samples:
        t = ev = None
        if key == "rfs_days":
            t = _num(d.get("t.rfs")); ev = _event01(d.get("e.rfs"))
        elif key == "mfi_months":
            t = _num(d.get("metastasis free interval (months)"))
            ev = _event01(d.get("metastasis (1: yes, 0: no/censored)"))
        elif key == "dr_years":
            t = _num(d.get("event time (years)"))
            ev = _event01(d.get("distant relapse (1=dr, 0 censored)"))
        elif key == "dmfs_years":
            t = _num(d.get("dmfs time"))
            ev = _event01(d.get("dmfs event (defined as distant metastasis or death from breast cancer)"))
        times.append(t); events.append(ev)
    return np.array(times, float), np.array(events, float), key


def _num(x):
    try:
        return float(str(x).strip().strip('"'))
    except Exception:
        return np.nan


def _event01(x):
    if x is None:
        return np.nan
    s = str(x).strip().strip('"').lower()
    if s in ("1", "yes", "1:yes", "1:dr", "1=dr"):
        return 1.0
    if s in ("0", "no", "0:no/censored", "0 censored", "0=no"):
        return 0.0
    return np.nan


# ============================================================ signature scoring
def load_rep():
    df = pd.read_csv(os.path.join(RES, "rep_genes.csv"))
    return dict(zip(df["gene"], df["coef"]))


def score_cohort(gexpr, coefs):
    genes = [g for g in coefs if g in gexpr.index]
    if len(genes) < 5:
        return None, genes
    sub = gexpr.loc[genes]
    z = (sub - sub.mean(axis=1, skipna=True).values.reshape(-1, 1)) / \
        sub.std(axis=1, skipna=True).values.reshape(-1, 1)
    z = z.fillna(0.0)
    score = np.zeros(z.shape[1])
    for g in genes:
        score += coefs[g] * z.loc[g].values
    return score, genes


# ============================================================ KM / Cox / enrichment
def km_plot(gse, time, event, score, out_png):
    if not HAVE_LIFELINES:
        return
    med = np.median(score)
    kmf_h = KaplanMeierFitter(); kmf_l = KaplanMeierFitter()
    hi = score > med; lo = ~hi
    plt.figure(figsize=(5, 4))
    if lo.sum() > 0:
        kmf_l.fit(time[lo], event_observed=event[lo], label="Low REP")
        kmf_l.plot_survival_function(color="tab:blue", ci_show=False)
    if hi.sum() > 0:
        kmf_h.fit(time[hi], event_observed=event[hi], label="High REP")
        kmf_h.plot_survival_function(color="tab:red", ci_show=False)
    plt.title(f"{gse}  REP high vs low")
    plt.tight_layout(); plt.savefig(out_png, dpi=120); plt.close()


def analyze():
    gpl_annots = load_gpl_annots()
    if not gpl_annots:
        print("[FATAL] no GPL annot loaded"); return
    coefs = load_rep()
    print(f"[rep] {len(coefs)} signature genes loaded")
    rows = []
    forest = []
    for gse, (fname, okey) in COHORTS.items():
        try:
            gexpr, samples = parse_series(gse, fname, gpl_annots)
        except Exception as e:
            print(f"[SKIP] {gse}: parse error {e}")
            continue
        time, event, key = extract_outcome(gse, okey, samples)
        mask = np.isfinite(time) & np.isfinite(event) & (time > 0)
        time, event = time[mask], event[mask]
        score_all, genes = score_cohort(gexpr, coefs)
        if score_all is None:
            print(f"[SKIP] {gse}: only {len(genes)} signature genes mapped")
            continue
        score = score_all[mask]
        n = len(time); nev = int(event.sum())
        print(f"[validate] {gse}: n={n} events={nev} sig_genes={len(genes)}")
        if n == 0 or nev == 0:
            print(f"  -> no usable outcomes, skip")
            continue
        # (a) log-rank
        lr_p = np.nan
        if HAVE_LIFELINES and nev > 0:
            med = np.median(score)
            try:
                res = logrank_test(time[score > med], time[score <= med],
                                   event_observed_A=event[score > med],
                                   event_observed_B=event[score <= med])
                lr_p = res.p_value
            except Exception as e:
                print(f"  logrank err {e}")
        # (b) univariate Cox (continuous, per SD)
        cox_hr = cox_p = np.nan
        if HAVE_LIFELINES and nev > 5 and np.std(score) > 0:
            try:
                sd = (score - score.mean()) / score.std()
                cdf = pd.DataFrame({"T": time, "E": event.astype(int), "S": sd})
                cph = CoxPHFitter(penalizer=0.01)
                cph.fit(cdf, duration_col="T", event_col="E")
                cox_hr = float(np.exp(cph.params_["S"]))
                cox_p = float(cph.summary.loc["S", "p"])
            except Exception as e:
                print(f"  cox err {e}")
        # (c) resistance enrichment: relapse <=5y (resistant) vs censored >=5y (sensitive)
        cutoff = {"rfs_days": 1825.0, "mfi_months": 60.0, "dr_years": 5.0,
                  "dmfs_years": 5.0}[key]
        resi = (event == 1) & (time <= cutoff)
        sens = (event == 0) & (time >= cutoff)
        mw_p = np.nan; auc = np.nan
        if resi.sum() >= 5 and sens.sum() >= 5:
            a = score[resi]; b = score[sens]
            try:
                mw_p = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
            except Exception:
                pass
            try:
                from sklearn.metrics import roc_auc_score
                auc = float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))],
                                          np.r_[a, b]))
            except Exception:
                pass
        rows.append({
            "cohort": gse, "outcome": okey, "n": n, "events": nev,
            "sig_genes_used": len(genes),
            "km_logrank_p": lr_p, "cox_hr_perSD": cox_hr, "cox_p": cox_p,
            "resistant_n": int(resi.sum()), "sensitive_n": int(sens.sum()),
            "resistance_MWU_p": mw_p, "resistance_AUC": auc,
        })
        forest.append((gse, cox_hr, cox_p, nev))
        km_plot(gse, time, event, score,
                os.path.join(RES, f"geo_{gse}_km.png"))
        print(f"  -> logrank_p={lr_p:.2e}  cox_HR={cox_hr:.2f}(p={cox_p:.2e})  "
              f"resist AUC={auc:.2f}(p={mw_p:.2e}) "
              f"[resist_n={int(resi.sum())} sens_n={int(sens.sum())}]")
    if rows:
        summ = pd.DataFrame(rows)
        summ.to_csv(os.path.join(RES, "geo_validation_summary.csv"), index=False)
        print("\n[summary]\n", summ.to_string(index=False))
        # ---- inverse-variance weighted fixed-effect meta of the Cox HRs ----
        meta = _meta(forest)
        if meta:
            hr, lo, hi, p = meta
            print(f"\n[meta] pooled Cox HR = {hr:.2f} (95% CI {lo:.2f}-{hi:.2f}), p = {p:.2e}")
            meta_row = {"cohort": "POOLED", "outcome": "cox_meta", "n": int(summ["n"].sum()),
                        "events": int(summ["events"].sum()), "sig_genes_used": "",
                        "km_logrank_p": np.nan, "cox_hr_perSD": hr, "cox_p": p,
                        "resistant_n": "", "sensitive_n": "",
                        "resistance_MWU_p": np.nan, "resistance_AUC": np.nan}
            pd.DataFrame([meta_row]).to_csv(os.path.join(RES, "geo_meta.csv"), index=False)
            forest.append(("POOLED", hr, p, int(summ["events"].sum())))
        _forest(forest, os.path.join(RES, "geo_cox_forest.png"))
    else:
        print("[summary] NO cohort produced results")


def _meta(forest):
    """Inverse-variance weighted fixed-effect meta of per-SD Cox HRs.
    forest entries: (label, hr, p, n_events). Returns (hr, lo, hi, p) or None."""
    hrs, ses, labels = [], [], []
    for lab, hr, p, _ in forest:
        if not (hr == hr and p == p and p > 0 and hr > 0):
            continue
        z = stats.norm.ppf(1 - p / 2)
        se = abs(np.log(hr)) / z if z != 0 else np.nan
        if not (se == se):
            continue
        hrs.append(hr); ses.append(se); labels.append(lab)
    if len(hrs) < 2:
        return None
    w = 1.0 / np.array(ses) ** 2
    pool_loghr = float(np.sum(w * np.log(hrs)) / np.sum(w))
    pool_se = float(1.0 / np.sqrt(np.sum(w)))
    pool_z = pool_loghr / pool_se
    pool_p = float(2 * (1 - stats.norm.cdf(abs(pool_z))))
    pool_hr = float(np.exp(pool_loghr))
    return (pool_hr, float(np.exp(pool_loghr - 1.96 * pool_se)),
            float(np.exp(pool_loghr + 1.96 * pool_se)), pool_p)


def _forest(forest, png):
    forest = [f for f in forest if f[1] == f[1] and f[2] == f[2]]
    if not forest:
        return
    labels = [f[0] for f in forest]
    hrs = [f[1] for f in forest]
    ps = [f[2] for f in forest]
    cis = []
    for hr, p in zip(hrs, ps):
        if p > 0:
            z = stats.norm.ppf(1 - p / 2)
            se = abs(np.log(hr)) / z if z != 0 else np.nan
            cis.append((np.exp(np.log(hr) - 1.96 * se), np.exp(np.log(hr) + 1.96 * se)))
        else:
            cis.append((np.nan, np.nan))
    plt.figure(figsize=(6, 0.6 * len(labels) + 1.5))
    for i, (lab, hr, ci) in enumerate(zip(labels, hrs, cis)):
        plt.hlines(i, ci[0], ci[1], color="k", lw=1)
        plt.plot(hr, i, "o", color="tab:red")
    plt.axvline(1.0, color="gray", ls="--")
    plt.yticks(range(len(labels)), labels)
    plt.xscale("log")
    plt.xlabel("Cox HR of REP score (per SD)")
    plt.title("GEO external validation - REP signature")
    plt.tight_layout(); plt.savefig(png, dpi=120); plt.close()


if __name__ == "__main__":
    analyze()
