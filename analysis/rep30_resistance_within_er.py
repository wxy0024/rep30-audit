#!/usr/bin/env python
"""
REP30 resistant-state validation -- within-ER+ controls (controlling subtype/proliferation confounding)

Motivation
----
The pooled analysis showed: the ESR1-amplified / RB1-deleted groups have a *higher* REP30.
But ESR1 amplification (genome-wide instability) and RB1 loss (enriched in basal/TNBC) co-occur with *high-proliferation subtypes*,
and REP30 is a proliferation programme -> the pooled comparison is dominated by subtype confounding.

Key control: repeat **within the ER+ subgroup** (acquired ESR1 mutations / RB1 loss arise exactly in the ER+ endocrine-treatment setting).
  - if the association with these labels persists within ER+ (stable direction) -> supports "REP30 tracks resistance-associated states"
  - if it disappears/flips within ER+ -> the pooled signal is mainly driven by subtype confounding

Samples / labels
---------
  METABRIC (n~1825 ER+): ER_STATUS=="Positive"; labels ESR1_amp(+2)/RB1_del(-2)/RB1_any_loss
  TCGA (ER+ = ESR1 z>0):  labels ESR1_mut(nonsilent)/RB1_mut/RB1_del(-2)

Usage: python analysis/rep30_resistance_within_er.py
Outputs: results/rep30_resistance_validation/within_er/{label_comparison.csv, summary.md, fig.png}
"""
import os, sys, io, csv
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rep30_resistance_validation import (load_panel, load_z_subset, rep_score,
                                         patient_of, mut_patients, cna_map,
                                         cliffs_delta, compare, PANEL, TC, MB)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "results", "rep30_resistance_validation", "within_er")
OUT = os.path.abspath(OUT)


def read_cbio_table(path):
    with open(path, "rt", encoding="utf-8-sig", errors="replace") as fh:
        lines = fh.read().splitlines()
    data = [l for l in lines if l and not l.startswith("#")]
    return pd.read_csv(io.StringIO("\n".join(data)), sep="\t", low_memory=False)


def main():
    os.makedirs(OUT, exist_ok=True)
    coef = load_panel(PANEL)
    rows = []; box = []

    # ---------- within METABRIC ER+ ----------
    mb_z = os.path.join(MB, "data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt")
    print("[METABRIC] scoring REP30 ...")
    mb_mat = load_z_subset(mb_z, set(coef))
    mb_score = rep_score(mb_mat, coef)
    cna2, _ = cna_map(os.path.join(MB, "data_cna.txt"), {"ESR1", "RB1"})
    clin = read_cbio_table(os.path.join(MB, "data_clinical_sample.txt"))
    er_pos = set(clin[clin["ER_STATUS"].astype(str).str.strip() == "Positive"]["PATIENT_ID"].astype(str))
    mb_er = [s for s in mb_score.index if patient_of(s) in er_pos]
    score_er = mb_score.loc[mb_er]
    print(f"[METABRIC] ER+ samples = {len(score_er)}")

    for lname, gene, vals in [("ESR1_amplification", "ESR1", ["2"]),
                              ("RB1_deep_deletion", "RB1", ["-2"]),
                              ("RB1_any_loss", "RB1", ["-2", "-1"])]:
        pos_pat = {patient_of(s) for s, v in cna2[gene].items() if v in vals}
        ld = {s: (1 if patient_of(s) in pos_pat else 0) for s in score_er.index}
        r = compare(score_er, ld, lname, "METABRIC")
        if r:
            r["label"] = lname + "|ER+"
            rows.append(r)
            box.append(("METABRIC ER+", lname,
                        score_er.loc[[s for s in score_er.index if ld[s] == 1]].values,
                        score_er.loc[[s for s in score_er.index if ld[s] == 0]].values))

    # ---------- within TCGA ER+ (ESR1 z>0) ----------
    tc_z = os.path.join(TC, "data_mrna_seq_v2_rsem_zscores_ref_diploid_samples.txt")
    print("[TCGA] scoring REP30 (+ESR1) ...")
    tc_mat = load_z_subset(tc_z, set(coef) | {"ESR1"})
    tc_score = rep_score(tc_mat, coef)
    esr1 = tc_mat.loc["ESR1"]
    tc_er = [s for s in tc_score.index if float(esr1.get(s, np.nan)) > 0]
    score_er_tc = tc_score.loc[tc_er]
    print(f"[TCGA] ER+ (ESR1 z>0) samples = {len(score_er_tc)}")

    esr1_mut = mut_patients(os.path.join(TC, "data_mutations.txt"), "ESR1")
    rb1_mut = mut_patients(os.path.join(TC, "data_mutations.txt"), "RB1")
    cna, _ = cna_map(os.path.join(TC, "data_cna.txt"), {"RB1"})
    rb1_del = set(s for s, v in cna["RB1"].items() if v == "-2")
    for lname, sset in [("ESR1_mutation", esr1_mut), ("RB1_mutation", rb1_mut),
                        ("RB1_deep_deletion", rb1_del)]:
        ld = {s: (1 if patient_of(s) in sset else 0) for s in score_er_tc.index}
        r = compare(score_er_tc, ld, lname, "TCGA")
        if r:
            r["label"] = lname + "|ER+"
            rows.append(r)
            box.append(("TCGA ER+", lname,
                        score_er_tc.loc[[s for s in score_er_tc.index if ld[s] == 1]].values,
                        score_er_tc.loc[[s for s in score_er_tc.index if ld[s] == 0]].values))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "label_comparison.csv"), index=False)

    lines = ["# REP30 resistance-label comparison -- within ER+ (controlling subtype/proliferation confounding)\n"]
    lines.append("> Against the pooled results: if the pooled comparison is positive but the association disappears/flips within ER+ -> the pooled signal is subtype-confounded.\n")
    lines.append("\n| cohort | label | n_label | n_ctrl | median_label | median_ctrl | delta | cliffs_delta | p |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        def f(x):
            if isinstance(x, float):
                return f"{x:.2e}" if abs(x) < 1e-3 else f"{x:.3f}"
            return str(x)
        lines.append("| " + " | ".join(f(r[c]) for c in
                     ["cohort", "label", "n_label", "n_ctrl", "median_label",
                      "median_ctrl", "delta", "cliffs_delta", "p"]) + " |")
    lines.append("\n> a negative cliffs_delta = the label group has a lower REP30 (consistent with the 'low REP in resistant states' prior).\n")
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(box)
        fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.4)) if n else (None, [])
        if n == 1:
            axes = [axes]
        for ax, (cohort, lname, a, b) in zip(np.array(axes).reshape(-1), box):
            ax.boxplot([a, b], tick_labels=["label+", "label-"], showfliers=False)
            ax.set_title(f"{cohort}\n{lname}", fontsize=8)
            ax.axhline(0, color="grey", lw=0.5, ls="--")
            ax.tick_params(labelsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_within_er.png"), dpi=200)
        plt.close(fig)
        print("[plot] saved")
    except Exception as e:
        print("[plot] skipped:", str(e)[:80])

    print("\n=== within-ER done ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
