#!/usr/bin/env python
"""
Backtest of the trimmed REP52 panel.

Purpose
----
REP52 is a Cox-weighted pan-subtype prognostic signature (coefficients from a pooled TCGA-BRCA + METABRIC Cox).
rep52_trimmed_panel.csv is the 30-gene trimmed panel after removing 22 genes "without independent evidence" (original coefficients kept).
This script recomputes REP scores and fits Cox models on *the same pair of cohorts used to build the signature*, comparing:
    full-52  (rep_genes.csv, 52 genes)
    trimmed-30 (rep52_trimmed_panel.csv, 30 genes)
Question: does the composite HR collapse after trimming (performance retention)?

Score definition
--------
REP_score = sum_i coef_i * z_i      # z_i = precomputed per-gene z-score from cBioPortal
(standard Cox-weighted gene-signature scoring; the z-score matrices ship with the data, no re-standardization needed)

Cox models (lifelines)
---------------------
  1) continuous: HR per 1 unit of score
  2) standardized: HR per 1 SD of score (best comparability)
  3) dichotomized: HR for high vs low (median split; most intuitive)
For two populations:
  A) whole cohort (pan-subtype, same protocol as signature construction; primary analysis)
  B) HR+ subgroup (METABRIC clinical ER+/HER2-; TCGA ESR1 z>0 surrogate; secondary robustness)

Usage
--------
  python analysis/rep52_backtest.py            # default paths, METABRIC + TCGA
  python analysis/rep52_backtest.py --hrp      # additionally run the HR+ subgroup
Outputs
----
  results/rep52_trim/backtest_results.csv
  results/rep52_trim/backtest_summary.md
"""
import os, sys, argparse
import numpy as np
import pandas as pd

ROOT = r"E:\01.WorkSpace\01.Breast_cancer_gene_panel\resistance_signature"
MB_DIR = os.path.join(ROOT, "data/01_core/METABRIC/brca_metabric")
TC_DIR = os.path.join(ROOT, "data/01_core/TCGA-BRCA/brca_tcga_pan_can_atlas_2018")
FULL_CSV = os.path.join(ROOT, "results/rep_genes.csv")
TRIM_CSV = os.path.join(ROOT, "results/rep52_trim/rep52_trimmed_panel.csv")
OUT = os.path.join(ROOT, "results/rep52_trim")


def load_panel(path):
    d = pd.read_csv(path)
    return dict(zip(d["gene"], d["coef"]))  # gene -> coef


def load_z_subset(matrix_path, wanted):
    """Manually scan a large matrix line by line, keeping only wanted gene rows; returns a gene x sample DataFrame (float)."""
    rows = {}
    order = []
    with open(matrix_path, "rt", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        samples = header[2:]  # skip Hugo_Symbol, Entrez_Gene_Id
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue
            g = parts[0]
            if g in wanted and g not in rows:
                vals = [float(x) if x not in ("", "NA", "NaN") else np.nan for x in parts[2:]]
                rows[g] = np.array(vals, dtype=float)
                order.append(g)
    mat = pd.DataFrame({g: rows[g] for g in order}, index=samples).T  # gene x sample
    return mat


def rep_score(mat, coef_map):
    genes = [g for g in coef_map if g in mat.index]
    sub = mat.loc[genes]
    coefs = np.array([coef_map[g] for g in genes])
    score = (sub.values * coefs[:, None]).sum(axis=0)  # weighted sum over genes -> per sample
    return pd.Series(score, index=mat.columns)


def _read_cbio_clinical(path):
    # cBioPortal clinical_*: the first lines start with # (attribute names + descriptions), followed by the real header + data
    import io
    with open(path, "rt", encoding="utf-8-sig", errors="replace") as fh:
        lines = fh.read().splitlines()
    data = [ln for ln in lines if ln and not ln.startswith("#")]
    # data[0] = the real header, data[1:] = data rows
    df = pd.read_csv(io.StringIO("\n".join(data)), sep="\t", low_memory=False)
    return df


def load_clinical_mb(path):
    df = _read_cbio_clinical(path)
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame()
    out["patient"] = df["PATIENT_ID"]
    out["os_months"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    # OS_STATUS looks like "0:LIVING" / "1:DECEASED" or "LIVING"/"DECEASED"
    st = df["OS_STATUS"].astype(str).str.split(":").str[-1].str.strip().str.upper()
    out["os_status"] = st
    out["er"] = df["ER_IHC"].astype(str).str.strip()      # includes the "Positve" spelling variant
    out["her2"] = df["HER2_SNP6"].astype(str).str.strip()
    return out


def load_clinical_tc(path):
    df = _read_cbio_clinical(path)
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame()
    out["patient"] = df["PATIENT_ID"]
    out["os_months"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    st = df["OS_STATUS"].astype(str).str.split(":").str[-1].str.strip().str.upper()
    out["os_status"] = st
    return out


def to_event(status):
    s = status.astype(str).str.strip().str.upper()
    return (s.isin(["DECEASED", "1", "TRUE"]) | s.str.startswith("DEAD")).astype(int)


def cox_models(score, clin, label_prefix):
    """Run three Cox variants for a given score series + clin (with os_months, os_status, patient); returns rows.
    score.index are sample ids; clin is matched by patient (METABRIC sample==patient; TCGA patient=col[:12]).
    """
    from lifelines import CoxPHFitter
    # align
    sc = score.rename("score")
    sc.index = sc.index.astype(str)
    clin = clin.copy()
    clin["pid"] = clin["patient"].astype(str)
    # TCGA: sample id has -01 suffix; match by first 12 chars
    sc_pid = sc.index.str.replace(r"-01[ABC]$", "", regex=True).str[:12]
    sc2 = pd.DataFrame({"score": sc.values}, index=sc_pid)
    merged = clin.merge(sc2, left_on="pid", right_index=True, how="inner")
    merged = merged.dropna(subset=["os_months", "os_status", "score"])
    merged = merged[merged["os_months"] > 0]
    merged["event"] = to_event(merged["os_status"])
    if merged.shape[0] < 20:
        return [], merged.shape[0]
    out = []
    base = {"time": merged["os_months"], "event": merged["event"]}

    # 1) continuous per 1 unit
    df = pd.DataFrame({"time": base["time"], "event": base["event"], "score": merged["score"]})
    try:
        cph = CoxPHFitter().fit(df, duration_col="time", event_col="event")
        hr = float(np.exp(cph.params_["score"]))
        lo, hi = np.exp(cph.summary.loc["score", ["coef lower 95%", "coef upper 95%"]])
        p = float(cph.summary.loc["score", "p"])
        out.append({"model": "continuous_per1unit", "HR": hr, "lower": lo, "upper": hi, "p": p})
    except Exception as e:
        out.append({"model": "continuous_per1unit", "HR": np.nan, "lower": np.nan, "upper": np.nan, "p": np.nan, "err": str(e)[:60]})

    # 2) standardized per 1 SD
    z = (merged["score"] - merged["score"].mean()) / merged["score"].std()
    df = pd.DataFrame({"time": base["time"], "event": base["event"], "z": z.values})
    try:
        cph = CoxPHFitter().fit(df, duration_col="time", event_col="event")
        hr = float(np.exp(cph.params_["z"]))
        lo, hi = np.exp(cph.summary.loc["z", ["coef lower 95%", "coef upper 95%"]])
        p = float(cph.summary.loc["z", "p"])
        out.append({"model": "standardized_per1SD", "HR": hr, "lower": lo, "upper": hi, "p": p})
    except Exception as e:
        out.append({"model": "standardized_per1SD", "HR": np.nan, "lower": np.nan, "upper": np.nan, "p": np.nan, "err": str(e)[:60]})

    # 3) dichotomized high vs low (median)
    med = merged["score"].median()
    grp = (merged["score"] > med).astype(int)
    df = pd.DataFrame({"time": base["time"], "event": base["event"], "grp": grp.values})
    try:
        cph = CoxPHFitter().fit(df, duration_col="time", event_col="event")
        hr = float(np.exp(cph.params_["grp"]))
        lo, hi = np.exp(cph.summary.loc["grp", ["coef lower 95%", "coef upper 95%"]])
        p = float(cph.summary.loc["grp", "p"])
        out.append({"model": "high_vs_low_median", "HR": hr, "lower": lo, "upper": hi, "p": p})
    except Exception as e:
        out.append({"model": "high_vs_low_median", "HR": np.nan, "lower": np.nan, "upper": np.nan, "p": np.nan, "err": str(e)[:60]})

    for r in out:
        r["n"] = merged.shape[0]
        r["prefix"] = label_prefix
    return out, merged.shape[0]


def process_cohort(name, mb_or_tc, zpath, clin, full_coef, trim_coef, do_hrp, esr1_mat=None):
    wanted = set(full_coef) | set(trim_coef)
    if do_hrp and esr1_mat is not None:
        wanted |= {"ESR1"}
    print(f"[{name}] reading z-score matrix (subset {len(wanted)} genes)...")
    mat = load_z_subset(zpath, wanted)
    print(f"[{name}] matrix shape gene x sample = {mat.shape}")
    full_score = rep_score(mat, full_coef)
    trim_score = rep_score(mat, trim_coef)

    rows = []
    r, n = cox_models(full_score, clin, f"{name}|all|full52")
    rows += r
    r, n = cox_models(trim_score, clin, f"{name}|all|trim30")
    rows += r
    print(f"[{name}|all] full52 n={rows[0]['n'] if rows else '?'}")

    if do_hrp:
        if name == "METABRIC":
            # METABRIC HER2_SNP6: GAIN(17.5%) = HER2 amplification = HER2+; NEUTRAL/LOSS = HER2-; UNDEF/nan unknown
            er_pos = clin["er"].astype(str).str.contains("pos", case=False, na=False)  # "Positve" spelling variant
            her2_neg = clin["her2"].astype(str).str.upper().isin(["NEUTRAL", "LOSS"])
            mask = er_pos & her2_neg
            sub = clin[mask].copy()
            tag = "HR+ (ER+/HER2-)"
        else:
            # TCGA: ESR1 z>0 as the ER+ surrogate
            if esr1_mat is None or "ESR1" not in esr1_mat.index:
                print(f"[{name}|HR+] no ESR1, skipping")
                sub = None
                tag = "HR+ (ESR1>0)"
            else:
                # TCGA clinical has no ER/HER2 fields -> ESR1 z-score>0 as the ER+ surrogate
                esr1 = esr1_mat.loc["ESR1"]
                esr1.index = esr1.index.astype(str)
                pos_samples = esr1[esr1 > 0].index
                pos_pid = pd.Index(pos_samples.str.replace(r"-01[ABC]$", "", regex=True).str[:12])
                sub = clin[clin["patient"].isin(pos_pid)].copy()
                tag = "HR+ (ESR1 z>0)"
        if sub is not None and sub.shape[0] >= 20:
            r, n = cox_models(full_score, sub, f"{name}|HR+|full52")
            rows += r
            r, n = cox_models(trim_score, sub, f"{name}|HR+|trim30")
            rows += r
            print(f"[{name}|HR+] n={n} ({tag})")
        else:
            print(f"[{name}|HR+] subgroup too small or missing, skipping")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb-dir", default=MB_DIR)
    ap.add_argument("--tcga-dir", default=TC_DIR)
    ap.add_argument("--full", default=FULL_CSV)
    ap.add_argument("--panel", default=TRIM_CSV)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--hrp", action="store_true", help="additionally run the HR+ subgroup")
    args = ap.parse_args()

    full_coef = load_panel(args.full)
    trim_coef = load_panel(args.panel)

    all_rows = []

    # METABRIC
    mb_z = os.path.join(args.mb_dir, "data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt")
    mb_clin = load_clinical_mb(os.path.join(args.mb_dir, "data_clinical_patient.txt"))
    all_rows += process_cohort("METABRIC", "mb", mb_z, mb_clin, full_coef, trim_coef, args.hrp)

    # TCGA
    tc_z = os.path.join(args.tcga_dir, "data_mrna_seq_v2_rsem_zscores_ref_diploid_samples.txt")
    tc_clin = load_clinical_tc(os.path.join(args.tcga_dir, "data_clinical_patient.txt"))
    # prefetch the ESR1 row for the HR+ surrogate
    esr1_mat = None
    if args.hrp:
        esr1_mat = load_z_subset(tc_z, {"ESR1"})
    all_rows += process_cohort("TCGA", "tc", tc_z, tc_clin, full_coef, trim_coef, args.hrp, esr1_mat=esr1_mat)

    res = pd.DataFrame(all_rows)
    res.to_csv(os.path.join(args.out, "backtest_results.csv"), index=False)

    # summary
    lines = ["# Backtest of the trimmed REP52 panel (composite HR retention)\n"]
    lines.append("> Cohorts = TCGA-BRCA + METABRIC as used for signature construction (cBioPortal); score = sum coef*z (z = precomputed per-gene z-scores).\n")
    lines.append("> full-52 vs trimmed-30. The high_vs_low_median HR is the intuitive performance anchor.\n")
    lines.append("\n## Summary table\n")
    cols = ["prefix", "model", "n", "HR", "lower", "upper", "p"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, rr in res.iterrows():
        def fmt(x):
            if pd.isna(x):
                return ""
            if not isinstance(x, float):
                return str(x)
            return f"{x:.2e}" if abs(x) < 1e-3 else f"{x:.3f}"
        lines.append("| " + " | ".join(fmt(rr[c]) for c in cols) + " |")

    # retention verdict (high_vs_low_median)
    lines.append("\n## Performance retention (high_vs_low_median)\n")
    for cohort in ["METABRIC", "TCGA"]:
        sub = res[(res["prefix"].str.startswith(cohort)) & (res["model"] == "high_vs_low_median")]
        full_r = sub[sub["prefix"].str.endswith("full52")]
        trim_r = sub[sub["prefix"].str.endswith("trim30")]
        if len(full_r) and len(trim_r):
            f = full_r.iloc[0]; t = trim_r.iloc[0]
            ratio = (t["HR"] / f["HR"]) if f["HR"] else np.nan
            lines.append(f"- **{cohort} whole cohort**: full52 HR={f['HR']:.2f} (p={f['p']:.1e}) vs trimmed30 HR={t['HR']:.2f} (p={t['p']:.1e}); retention (HR ratio)={ratio:.2f}")
            verdict = "OK: not collapsed (same direction and still significant)" if (t["p"] < 0.05 and f["HR"] > 1 and t["HR"] > 1) else "WARN: needs review"
            lines.append(f"  - verdict: {verdict}")
    lines.append("\n## Notes\n")
    lines.append("- The whole-cohort protocol matches signature construction (pan-subtype); this directly tests whether trimming loses signal.\n")
    lines.append("- Because the coefficients were fitted on these same cohorts, full-52 is near in-sample and its HR is optimistic; the key point is that trimmed-30 remains significant with a consistent direction.\n")
    lines.append("- The HR+ subgroup is a secondary robustness check (METABRIC clinical ER+/HER2-; TCGA ESR1 z>0 surrogate).\n")
    lines.append("- If trimmed-30 keeps p<0.05 with the same HR direction as full52 in all cohorts -> the 30-gene panel can be finalized.\n")

    with open(os.path.join(args.out, "backtest_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n=== backtest done ===")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
