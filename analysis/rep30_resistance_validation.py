#!/usr/bin/env python
"""
REP30 independent validation of the resistant state (cross-layer: expression signature vs mutations / CNV / treatment exposure)

Question
----
Does REP30 (a 30-gene Cox-weighted expression signature) capture the "resistant state"?
An expression signature cannot see mutations directly, so we validate cross-layer:
  resistance labels come from the genomic layer (ESR1 mutation/amplification, RB1 loss) and the clinical layer (treatment exposure, progression on treatment)
  and we test the difference in REP30 scores between label groups and controls (direction + effect size + significance).

Data (local copies; not a dedicated metastatic cohort -> see boundaries)
----
  TCGA-BRCA (pan-can-atlas 2018): data_mutations.txt / data_cna.txt / data_timeline_treatment.txt
  METABRIC: data_cna.txt
  expression: cBioPortal per-gene z-score matrices (same protocol as signature construction)
  scoring: REP30 = sum_i coef_i * z_i   (results/rep52_trim/rep52_trimmed_panel.csv)

Prior direction (from the GSE-cohort conclusions): resistant/progressed states have lower REP -> expect label-group REP30 < controls.

Usage
----
  python analysis/rep30_resistance_validation.py
Outputs
----
  results/rep30_resistance_validation/label_comparison.csv
  results/rep30_resistance_validation/summary.md
  results/rep30_resistance_validation/fig_rep30_by_label.png
"""
import os, csv, argparse, collections, re
import numpy as np
import pandas as pd

ROOT = r"E:\01.WorkSpace\01.Breast_cancer_gene_panel\resistance_signature"
TC = os.path.join(ROOT, "data/01_core/TCGA-BRCA/brca_tcga_pan_can_atlas_2018")
MB = os.path.join(ROOT, "data/01_core/METABRIC/brca_metabric")
PANEL = os.path.join(ROOT, "results/rep52_trim/rep52_trimmed_panel.csv")
OUT = os.path.join(ROOT, "results/rep30_resistance_validation")

SILENT = {"Silent", "Intron", "3'UTR", "5'UTR", "3'Flank", "5'Flank", "IGR", "RNA",
          "lincRNA", "Targeted_Region", "Intron,Silent"}


# ---------------- basic utilities ----------------
def load_panel(path):
    d = pd.read_csv(path)
    return dict(zip(d["gene"], d["coef"]))


def load_z_subset(matrix_path, wanted):
    rows = {}; order = []
    with open(matrix_path, "rt", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        samples = header[2:]
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if not p:
                continue
            g = p[0]
            if g in wanted and g not in rows:
                rows[g] = np.array([float(x) if x not in ("", "NA", "NaN") else np.nan
                                    for x in p[2:]], float)
                order.append(g)
    return pd.DataFrame({g: rows[g] for g in order}, index=samples).T  # gene x sample


def rep_score(mat, coef_map):
    genes = [g for g in coef_map if g in mat.index]
    coefs = np.array([coef_map[g] for g in genes])
    return pd.Series((mat.loc[genes].values * coefs[:, None]).sum(axis=0), index=mat.columns)


def patient_of(s):
    s = re.sub(r"-[0-9]{2}[A-Z]?$", "", str(s))  # strip sample suffixes such as -01 / -01A
    return s[:12]


# ---------------- label readers ----------------
def mut_patients(path, gene):
    hits = set()
    with open(path, "rt", encoding="utf-8", errors="replace") as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        for row in rdr:
            if row.get("Hugo_Symbol") != gene:
                continue
            vc = (row.get("Variant_Classification") or "").strip()
            if vc in SILENT:
                continue
            hits.add(patient_of(row.get("Tumor_Sample_Barcode")))
    return hits


def cna_map(path, genes):
    """Returns {gene: {sample: int_value}} plus the sample list"""
    res = {}; samples = None
    with open(path, "rt", encoding="utf-8", errors="replace") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        samples = hdr[2:]
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if p and p[0] in genes:
                res[p[0]] = dict(zip(samples, p[2:]))
    return res, samples


def treatment_labels(path):
    """Returns patient -> dict(treated/hormone/chemo/targeted/immuno/PD/CR)"""
    lab = collections.defaultdict(lambda: dict(treated=0, hormone=0, chemo=0,
                                               targeted=0, immuno=0, PD=0, CR=0))
    with open(path, "rt", encoding="utf-8", errors="replace") as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        for row in rdr:
            pid = (row.get("PATIENT_ID") or "").strip()
            if not pid:
                continue
            tt = (row.get("TREATMENT_TYPE") or "")
            resp = (row.get("MEASURE_OF_RESPONSE") or "")
            d = lab[pid]
            d["treated"] = 1
            if "Hormone" in tt:
                d["hormone"] = 1
            if "Chemotherapy" in tt:
                d["chemo"] = 1
            if "Targeted" in tt:
                d["targeted"] = 1
            if "Immunotherapy" in tt:
                d["immuno"] = 1
            if "Progressive" in resp:
                d["PD"] = 1
            if "Complete Response" in resp:
                d["CR"] = 1
    return lab


# ---------------- statistics ----------------
def cliffs_delta(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    diff = a[:, None] - b[None, :]
    gt = (diff > 0).sum(); lt = (diff < 0).sum()
    return (gt - lt) / (len(a) * len(b))


def compare(score, sample_labels, label_name, cohort, pos_val=1):
    """score: Series(index=sample). sample_labels: dict sample->0/1. pos=1 is the label group."""
    common = [s for s in score.index if s in sample_labels]
    if len(common) < 10:
        return None
    grp = np.array([sample_labels[s] == pos_val for s in common])
    vals = score.loc[common].values
    a = vals[grp]      # label group
    b = vals[~grp]     # control
    if len(a) < 3 or len(b) < 3:
        return None
    from scipy.stats import mannwhitneyu
    try:
        U, p = mannwhitneyu(a, b, alternative="two-sided")
    except Exception:
        p = np.nan
    return dict(cohort=cohort, label=label_name, n_label=len(a), n_ctrl=len(b),
                median_label=float(np.median(a)), median_ctrl=float(np.median(b)),
                mean_label=float(np.mean(a)), mean_ctrl=float(np.mean(b)),
                delta=float(np.median(a) - np.median(b)),
                cliffs_delta=cliffs_delta(a, b), p=float(p))


# ---------------- main pipeline ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    coef = load_panel(PANEL)
    rows = []
    box_data = []

    # ---------- TCGA ----------
    tc_z = os.path.join(TC, "data_mrna_seq_v2_rsem_zscores_ref_diploid_samples.txt")
    print("[TCGA] scoring REP30 ...")
    tc_mat = load_z_subset(tc_z, set(coef))
    tc_score = rep_score(tc_mat, coef)
    tc_pat = pd.Series([patient_of(s) for s in tc_score.index], index=tc_score.index)
    tc_score_by_pat = pd.Series(tc_score.values, index=tc_pat.values)  # duplicated index possible, ok

    label_dicts = {}
    # mutations
    esr1_mut = mut_patients(os.path.join(TC, "data_mutations.txt"), "ESR1")
    rb1_mut = mut_patients(os.path.join(TC, "data_mutations.txt"), "RB1")
    # CNA
    cna, cna_samples = cna_map(os.path.join(TC, "data_cna.txt"), {"ESR1", "RB1"})
    # CNA keys are sample-level IDs -> convert to patient level (same convention as the patient_of mapping for METABRIC)
    esr1_amp = {patient_of(s) for s, v in cna["ESR1"].items() if v == "2"}
    rb1_deepdel = {patient_of(s) for s, v in cna["RB1"].items() if v == "-2"}
    rb1_loss = {patient_of(s) for s, v in cna["RB1"].items() if v in ("-2", "-1")}
    # treatment
    treat = treatment_labels(os.path.join(TC, "data_timeline_treatment.txt"))

    def mk(sample_set):
        return {s: (1 if patient_of(s) in sample_set else 0) for s in tc_score.index}

    def mk_pat_flag(flags):
        return {s: (1 if treat.get(patient_of(s), {}).get(flags, 0) == 1 else 0)
                for s in tc_score.index}

    label_dicts["TCGA:ESR1_mutation"] = mk(esr1_mut)
    label_dicts["TCGA:RB1_mutation"] = mk(rb1_mut)
    label_dicts["TCGA:RB1_inactivated(mut or deepdel)"] = mk(rb1_mut | rb1_deepdel)
    label_dicts["TCGA:ESR1_amplification"] = mk(esr1_amp)
    label_dicts["TCGA:RB1_deep_deletion"] = mk(rb1_deepdel)
    label_dicts["TCGA:RB1_any_loss"] = mk(rb1_loss)
    label_dicts["TCGA:treated_any"] = mk_pat_flag("treated")
    label_dicts["TCGA:hormone_therapy"] = mk_pat_flag("hormone")
    label_dicts["TCGA:chemotherapy"] = mk_pat_flag("chemo")
    label_dicts["TCGA:targeted_therapy"] = mk_pat_flag("targeted")
    label_dicts["TCGA:immunotherapy"] = mk_pat_flag("immuno")

    # resistance phenotype: PD vs CR (mutually exclusive)
    pd_pid = set(p for p, d in treat.items() if d["PD"] == 1)
    cr_pid = set(p for p, d in treat.items() if d["CR"] == 1 and d["PD"] == 0)
    tc_short = tc_score.copy()
    tc_short.index = [patient_of(s) for s in tc_score.index]
    pd_vals = tc_short[tc_short.index.isin(pd_pid)].values
    cr_vals = tc_short[tc_short.index.isin(cr_pid)].values
    if len(pd_vals) >= 3 and len(cr_vals) >= 3:
        from scipy.stats import mannwhitneyu
        U, p = mannwhitneyu(pd_vals, cr_vals, alternative="two-sided")
        rows.append(dict(cohort="TCGA", label="treatment_PD_vs_CR",
                         n_label=len(pd_vals), n_ctrl=len(cr_vals),
                         median_label=float(np.median(pd_vals)), median_ctrl=float(np.median(cr_vals)),
                         mean_label=float(np.mean(pd_vals)), mean_ctrl=float(np.mean(cr_vals)),
                         delta=float(np.median(pd_vals) - np.median(cr_vals)),
                         cliffs_delta=cliffs_delta(pd_vals, cr_vals), p=float(p)))
        box_data.append(("TCGA", "treatment_PD_vs_CR", pd_vals, cr_vals))

    for name, ld in label_dicts.items():
        r = compare(tc_score, ld, name.split(":", 1)[1], "TCGA")
        if r:
            rows.append(r)
            vals = tc_score.loc[[s for s in tc_score.index if ld[s] == 1]].values
            ctrl = tc_score.loc[[s for s in tc_score.index if ld[s] == 0]].values
            box_data.append(("TCGA", name.split(":", 1)[1], vals, ctrl))

    # ---------- METABRIC (CNA only) ----------
    mb_z = os.path.join(MB, "data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt")
    print("[METABRIC] scoring REP30 ...")
    mb_mat = load_z_subset(mb_z, set(coef))
    mb_score = rep_score(mb_mat, coef)
    cna2, _ = cna_map(os.path.join(MB, "data_cna.txt"), {"ESR1", "RB1"})
    mb_esr1_amp = set(s for s, v in cna2["ESR1"].items() if v == "2")
    mb_rb1_del = set(s for s, v in cna2["RB1"].items() if v == "-2")
    mb_rb1_loss = set(s for s, v in cna2["RB1"].items() if v in ("-2", "-1"))

    def mk_mb(sample_set):
        return {s: (1 if patient_of(s) in {patient_of(x) for x in sample_set} else 0)
                for s in mb_score.index}

    for lname, sset in [("ESR1_amplification", mb_esr1_amp),
                        ("RB1_deep_deletion", mb_rb1_del),
                        ("RB1_any_loss", mb_rb1_loss)]:
        ld = mk_mb(sset)
        r = compare(mb_score, ld, lname, "METABRIC")
        if r:
            rows.append(r)
            box_data.append(("METABRIC", lname,
                             mb_score.loc[[s for s in mb_score.index if ld[s] == 1]].values,
                             mb_score.loc[[s for s in mb_score.index if ld[s] == 0]].values))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "label_comparison.csv"), index=False)

    # ---------- summary md ----------
    lines = ["# REP30 independent validation of the resistant state (cross-layer)\n"]
    lines.append("> REP30 = sum coef*z (30 genes, results/rep52_trim/rep52_trimmed_panel.csv)\n")
    lines.append("> Labels come from the genomic layer (ESR1/RB1 mutations and CNA) and the clinical layer (treatment exposure / progression on treatment).\n")
    lines.append("> Prior direction: resistant/progressed states have lower REP -> expect delta<0, cliffs_delta<0.\n")
    lines.append("\n## Summary table\n")
    cols = ["cohort", "label", "n_label", "n_ctrl", "median_label", "median_ctrl",
            "delta", "cliffs_delta", "p"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in df.iterrows():
        def f(x):
            if isinstance(x, float):
                return f"{x:.2e}" if abs(x) < 1e-3 else f"{x:.3f}"
            return str(x)
        lines.append("| " + " | ".join(f(r[c]) for c in cols) + " |")
    lines.append("\n## Notes\n")
    lines.append("- cliffs_delta: effect size for label vs control; negative = the label group has a lower REP30.\n")
    lines.append("- With small n (ESR1 mutation ~15) power is low; directional consistency outweighs single-point p values.\n")
    lines.append("- TCGA/METABRIC are primary tumours; ESR1 mutation is a treatment-acquired event -> rare in primaries; this is a power ceiling.\n")
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    # ---------- figure ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(box_data)
        if n:
            ncol = 3
            nrow = int(np.ceil(n / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow))
            axes = np.array(axes).reshape(-1)
            for i, (cohort, lname, a, b) in enumerate(box_data):
                ax = axes[i]
                ax.boxplot([a, b], tick_labels=["label+", "label-"], showfliers=False)
                ax.set_title(f"{cohort}\n{lname}", fontsize=8)
                ax.axhline(0, color="grey", lw=0.5, ls="--")
                ax.tick_params(labelsize=7)
            for j in range(n, len(axes)):
                axes[j].axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, "fig_rep30_by_label.png"), dpi=200)
            plt.close(fig)
        print("[plot] saved")
    except Exception as e:
        print("[plot] skipped:", str(e)[:80])

    print("\n=== done ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
