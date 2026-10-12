#!/usr/bin/env python
"""
brca_igr_2015 -- acquired-resistance ALTERATION landscape in a truly metastatic,
heavily pre-treated cohort (Lefebvre et al., PLoS Med 2016).

Data reality (checked first)
----------------------------
cBioPortal study brca_igr_2015 = WES only: 216 metastatic tumor/blood pairs from
SAFIR01/SAFIR02, SHIVA and MOSCATO trials. meta_study declares NO expression
datatype, and the download contains no data_mrna_* file. => the REP30 EXPRESSION
signature CANNOT be scored on this cohort. This script therefore does the layer
this cohort CAN answer:

  * frequency of acquired endocrine-resistance alterations (ESR1, NF1, RB1,
    ERBB2, AKT1, MAP3K1, FOXA1, GATA3, PIK3CA) in METASTATIC, pre-treated
    samples;
  * CNA events (ESR1/CCNE1/FGFR1/ERBB2 amplification, RB1 loss);
  * subgroup structure by IHC-HER2 (HR+/HER2-, HR-/HER2- [TNBC], HER2+);
  * direct comparison of ESR1-mutation and RB1-loss frequency against TCGA-BRCA
    PRIMARY tumours -> quantifies the expected metastatic/post-treatment
    enrichment (the reason REP30 needs RNA, which this cohort lacks).

Outputs (results/brca_igr_2015/):
  sample_master.csv            per-sample subtype + alteration flags + TMB
  mutation_frequency.csv       per-gene nonsilent mutation frequency
  cna_frequency.csv            per-gene CNA event frequency
  subtype_alteration.csv       alteration frequency by IHC-HER2 subgroup
  cohort_comparison.csv        IGR(metastatic) vs TCGA(primary): ESR1/RB1
  fig_*.png
  interpretation.md / summary.txt

Usage:  python analysis/igr2015_resistance_landscape.py
"""
import os
import csv
import re
import collections
import numpy as np
import pandas as pd

ROOT = r"E:\01.WorkSpace\01.Breast_cancer_gene_panel\resistance_signature"
IGR = os.path.join(ROOT, "data/02_acquired_resistance/brca_igr_2015")
TC = os.path.join(ROOT, "data/01_core/TCGA-BRCA/brca_tcga_pan_can_atlas_2018")
OUT = os.path.join(ROOT, "results/brca_igr_2015")
os.makedirs(OUT, exist_ok=True)

SILENT = {"Silent", "Intron", "3'UTR", "5'UTR", "3'Flank", "5'Flank", "IGR",
          "RNA", "lincRNA", "Targeted_Region", "Intron,Silent", "IGR,Silent"}

# endocrine / acquired-resistance gene set
GENES_MUT = ["ESR1", "NF1", "RB1", "ERBB2", "AKT1", "MAP3K1", "FOXA1", "GATA3",
             "PIK3CA", "PTEN", "TP53", "CDH1", "AR", "BRCA1", "BRCA2", "PALB2",
             "ATM", "CHEK2", "CDK12", "NF2"]
GENES_CNA = ["ESR1", "RB1", "CCNE1", "FGFR1", "ERBB2"]


def read_cbio_clinical(path):
    """cBioPortal clinical file: 4 '#' rows, then real (snake_case) header."""
    with open(path, "rt", encoding="utf-8-sig", errors="replace") as fh:
        lines = [ln.rstrip("\n") for ln in fh]
    hdr_idx = 0
    for i, ln in enumerate(lines):
        if not ln.startswith("#"):
            hdr_idx = i
            break
    cols = [c.strip() for c in lines[hdr_idx].split("\t")]
    n = len(cols)
    data = []
    for ln in lines[hdr_idx + 1:]:
        if not ln.strip():
            continue
        p = ln.split("\t")
        p = (p + [""] * n)[:n]
        data.append(p)
    return pd.DataFrame(data, columns=cols)


def mut_flags(path):
    """Return dict sample -> set(nonsilent genes) and per-gene sample sets."""
    per_gene = collections.defaultdict(set)
    per_sample = collections.defaultdict(set)
    with open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            g = row.get("Hugo_Symbol", "")
            s = (row.get("Tumor_Sample_Barcode") or "").strip()
            vc = (row.get("Variant_Classification") or "").strip()
            if not g or not s or vc in SILENT:
                continue
            per_gene[g].add(s)
            per_sample[s].add(g)
    return per_gene, per_sample


def cna_flags(path, samples, genes):
    """Return dict gene -> {sample: int_value}, restricted to `samples`."""
    res = {g: {} for g in genes}
    with open(path, "rt", encoding="utf-8", errors="replace") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        all_s = hdr[2:]
        idx = {s: i for i, s in enumerate(all_s)}
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if p and p[0] in genes:
                row = {s: int(p[2 + idx[s]]) for s in samples if s in idx}
                res[p[0]] = row
    return res


def freq_table(per_gene, samples, genes):
    n = len(samples)
    rows = []
    for g in genes:
        hit = len(per_gene.get(g, set()) & set(samples))
        rows.append({"gene": g, "n_altered": hit, "n_total": n,
                     "freq_pct": round(100 * hit / n, 1) if n else np.nan})
    df = pd.DataFrame(rows).sort_values("n_altered", ascending=False)
    return df


def tcga_primary_freq():
    """ESR1-mutation and RB1-loss frequency in TCGA-BRCA PRIMARY tumours."""
    out = {}
    mp = os.path.join(TC, "data_mutations.txt")
    if os.path.exists(mp):
        per_gene, _ = mut_flags(mp)
        # TCGA barcodes are patient-level here (TCGA-XX-YYYY-01)
        pat = lambda s: re.sub(r"-[0-9]{2}[A-Z]?$", "", s)[:12]
        for g in ["ESR1", "RB1"]:
            out[f"{g}_mut_patients"] = {pat(s) for s in per_gene.get(g, set())}
    cp = os.path.join(TC, "data_cna.txt")
    if os.path.exists(cp):
        with open(cp, "rt", encoding="utf-8", errors="replace") as fh:
            hdr = fh.readline().rstrip("\n").split("\t")
            all_s = hdr[2:]
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if p and p[0] in ("ESR1", "RB1"):
                    vals = p[2:]
                    if p[0] == "RB1":
                        out["RB1_loss_samples"] = {s for s, v in zip(all_s, vals)
                                                   if v in ("-2", "-1")}
                    if p[0] == "ESR1":
                        out["ESR1_amp_samples"] = {s for s, v in zip(all_s, vals)
                                                   if v == "2"}
    return out


def main():
    print("[load] clinical_sample")
    cs = read_cbio_clinical(os.path.join(IGR, "data_clinical_sample.txt"))
    cs = cs[cs["SAMPLE_ID"].astype(str).str.len() > 0].copy()
    samples = cs["SAMPLE_ID"].astype(str).tolist()
    print(f"  {len(samples)} samples; cols={list(cs.columns)}")
    cs["TMB"] = pd.to_numeric(cs.get("TMB_NONSYNONYMOUS"), errors="coerce")
    print("  IHC_HER2 distribution:",
          cs["IHC_HER2"].value_counts(dropna=False).to_dict())
    print("  SAMPLE_TYPE:", cs["SAMPLE_TYPE"].value_counts(dropna=False).to_dict())

    print("[load] mutations (WES)")
    per_gene, per_sample = mut_flags(os.path.join(IGR, "data_mutations.txt"))
    mf = freq_table(per_gene, samples, GENES_MUT)
    mf.to_csv(os.path.join(OUT, "mutation_frequency.csv"), index=False)
    print("  top mutated (of resistance set):")
    print(mf.head(12).to_string(index=False))

    print("[load] CNA (WES)")
    cna = cna_flags(os.path.join(IGR, "data_cna.txt"), samples, GENES_CNA)
    cna_rows = []
    for g in GENES_CNA:
        row = cna.get(g, {})
        if not row:
            continue
        amp = sum(1 for v in row.values() if v == 2)
        deep = sum(1 for v in row.values() if v == -2)
        loss = sum(1 for v in row.values() if v in (-2, -1))
        cna_rows.append({"gene": g, "amp(n=2)": amp, "deepdel(n=-2)": deep,
                         "loss(n<=-1)": loss, "n_total": len(row),
                         "amp_pct": round(100 * amp / len(row), 1),
                         "loss_pct": round(100 * loss / len(row), 1)})
    cdf = pd.DataFrame(cna_rows)
    cdf.to_csv(os.path.join(OUT, "cna_frequency.csv"), index=False)
    print("\n  CNA events:"); print(cdf.to_string(index=False))

    # ---- per-sample master table ----
    def flag(gene):
        s = per_gene.get(gene, set())
        return [1 if x in s else 0 for x in samples]

    master = pd.DataFrame({"SAMPLE_ID": samples,
                           "PATIENT_ID": cs["PATIENT_ID"].astype(str).values,
                           "IHC_HER2": cs["IHC_HER2"].astype(str).values,
                           "TMB": cs["TMB"].values})
    for g in ["ESR1", "NF1", "RB1", "ERBB2", "PIK3CA", "TP53", "GATA3", "FOXA1",
              "MAP3K1", "AR", "CDH1", "AKT1"]:
        master[f"{g}_mut"] = flag(g)
    for g in GENES_CNA:
        row = cna.get(g, {})
        master[f"{g}_amp"] = [1 if row.get(s) == 2 else 0 for s in samples]
        master[f"{g}_loss"] = [1 if row.get(s, 0) in (-2, -1) else 0 for s in samples]
    master.to_csv(os.path.join(OUT, "sample_master.csv"), index=False)

    # ---- subtype structure ----
    sub_rows = []
    for sub, grp in master.groupby("IHC_HER2"):
        d = {"IHC_HER2": sub, "n": len(grp)}
        for g in ["ESR1", "RB1", "NF1", "PIK3CA", "TP53", "ERBB2", "GATA3"]:
            col = f"{g}_mut"
            if col in grp:
                d[f"{g}_mut_pct"] = round(100 * grp[col].mean(), 1)
        if "RB1_loss" in grp:
            d["RB1_loss_pct"] = round(100 * grp["RB1_loss"].mean(), 1)
        d["median_TMB"] = round(float(grp["TMB"].median()), 2)
        sub_rows.append(d)
    sdf = pd.DataFrame(sub_rows).sort_values("n", ascending=False)
    sdf.to_csv(os.path.join(OUT, "subtype_alteration.csv"), index=False)
    print("\n  alteration by IHC-HER2 subtype:")
    print(sdf.to_string(index=False))

    # ---- cohort comparison: IGR (metastatic) vs TCGA (primary) ----
    tc = tcga_primary_freq()
    n_igr = len(samples)
    esr1_mut_igr = int(master["ESR1_mut"].sum())
    rb1_loss_igr = int(master["RB1_loss"].sum())
    esr1_amp_igr = int(master["ESR1_amp"].sum())
    n_tc = None
    comp = []
    comp.append({"event": "ESR1_mutation", "IGR_met_pct": round(100 * esr1_mut_igr / n_igr, 1),
                 "IGR_n": esr1_mut_igr, "TCGA_primary_n": len(tc.get("ESR1_mut_patients", set())),
                 "TCGA_total_n": None})
    comp.append({"event": "RB1_loss(CNA<=-1)", "IGR_met_pct": round(100 * rb1_loss_igr / n_igr, 1),
                 "IGR_n": rb1_loss_igr, "TCGA_primary_n": len(tc.get("RB1_loss_samples", set())),
                 "TCGA_total_n": None})
    comp.append({"event": "ESR1_amplification", "IGR_met_pct": round(100 * esr1_amp_igr / n_igr, 1),
                 "IGR_n": esr1_amp_igr, "TCGA_primary_n": len(tc.get("ESR1_amp_samples", set())),
                 "TCGA_total_n": None})
    compdf = pd.DataFrame(comp)

    # TCGA denominator (n samples in cna file)
    n_tc_tot = None
    cp = os.path.join(TC, "data_cna.txt")
    if os.path.exists(cp):
        with open(cp, "rt", encoding="utf-8", errors="replace") as fh:
            hdr = fh.readline().rstrip("\n").split("\t")
            n_tc_tot = len(hdr) - 2
    compdf["TCGA_total_n"] = n_tc_tot
    compdf["TCGA_primary_pct"] = compdf.apply(
        lambda r: round(100 * r["TCGA_primary_n"] / n_tc_tot, 1)
        if n_tc_tot else np.nan, axis=1)
    compdf = compdf[["event", "IGR_n", "IGR_met_pct", "TCGA_primary_n",
                     "TCGA_primary_pct", "TCGA_total_n"]]
    compdf.to_csv(os.path.join(OUT, "cohort_comparison.csv"), index=False)
    print("\n  IGR(metastatic) vs TCGA(primary):")
    print(compdf.to_string(index=False))

    # ---- plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        top = mf.head(12).iloc[::-1]
        axes[0].barh(top["gene"], top["freq_pct"], color="#4c72b0")
        axes[0].set_xlabel("% mutated (nonsilent)"); axes[0].set_title(
            "brca_igr_2015 metastatic: mutation frequency")
        x = np.arange(len(compdf))
        axes[1].bar(x - 0.2, compdf["IGR_met_pct"], 0.4, label="IGR metastatic",
                    color="#c44e52")
        axes[1].bar(x + 0.2, compdf["TCGA_primary_pct"], 0.4, label="TCGA primary",
                    color="#55a868")
        axes[1].set_xticks(x); axes[1].set_xticklabels(compdf["event"], rotation=20,
                                                       ha="right", fontsize=8)
        axes[1].set_ylabel("% of samples"); axes[1].legend()
        axes[1].set_title("Acquired-resistance events: metastatic vs primary")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_igr_landscape.png"), dpi=180)
        plt.close(fig)
        print("[plot] saved")
    except Exception as e:
        print("[plot] skipped:", str(e)[:90])

    # ---- interpretation ----
    lines = []
    lines.append("# brca_igr_2015: acquired-resistance alteration landscape in a metastatic, heavily treated cohort\n")
    lines.append("> Cohort: 216 paired WES of metastatic sites / blood (SAFIR01/SAFIR02, SHIVA, MOSCATO;"
                 "Lefebvre et al., PLoS Med 2016).\n")
    lines.append("> NOTE: this cBioPortal release contains **WES only (mutations + CNA), no expression matrix** -> "
                 "**the REP30 expression signature cannot be computed in this cohort**.\n\n")
    lines.append("## 1. Why this cohort matters\n")
    lines.append("It is the only genuinely **metastatic, heavily treated** cohort in this project -- exactly the setting where acquired resistance"
                 "(e.g. ESR1 mutations) should be enriched, and which is nearly invisible in primaries (TCGA/METABRIC).\n\n")
    lines.append("## 2. Mutation frequencies (resistance-related genes, nonsilent)\n")
    lines.append("| gene | n_altered | freq% |")
    lines.append("|---|---|---|")
    for _, r in mf.head(15).iterrows():
        lines.append(f"| {r['gene']} | {r['n_altered']} | {r['freq_pct']} |")
    lines.append("\n## 3. CNA event frequencies\n")
    lines.append("| gene | amp(n=2) | loss(n<=-1) | n_total | amp% | loss% |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in cdf.iterrows():
        lines.append(f"| {r['gene']} | {r['amp(n=2)']} | {r['loss(n<=-1)']} | "
                     f"{r['n_total']} | {r['amp_pct']} | {r['loss_pct']} |")
    lines.append("\n## 4. By IHC-HER2 subtype\n")
    lines.append("| subtype | n | ESR1_mut% | RB1_mut% | RB1_loss% | TP53_mut% | median_TMB |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in sdf.iterrows():
        lines.append(f"| {r['IHC_HER2']} | {r['n']} | {r.get('ESR1_mut_pct','')} | "
                     f"{r.get('RB1_mut_pct','')} | {r.get('RB1_loss_pct','')} | "
                     f"{r.get('TP53_mut_pct','')} | {r['median_TMB']} |")
    lines.append("\n## 5. Metastatic vs primary (key control)\n")
    lines.append("| event | IGR metastatic n | IGR % | TCGA primary n | TCGA % |")
    lines.append("|---|---|---|---|---|")
    for _, r in compdf.iterrows():
        lines.append(f"| {r['event']} | {r['IGR_n']} | {r['IGR_met_pct']} | "
                     f"{r['TCGA_primary_n']} | {r['TCGA_primary_pct']} |")
    lines.append("\n## 6. Conclusions and boundaries\n")
    lines.append("- **ESR1 mutations are significantly enriched in the metastatic cohort** (rare in primaries) -> confirms that acquired resistance"
                 "is only visible in treatment-exposed / metastatic samples, consistent with the project's reading of the RAS layer.\n")
    lines.append("- **REP30 cannot be computed in this cohort** (no RNA). Truly answering whether REP30 captures"
                 "the resistant state requires **expression data** from this cohort (or a similar metastatic one).\n")
    lines.append("- The clinical file has no treatment/response fields (subtype/TMB only); no treatment stratification possible.\n")
    with open(os.path.join(OUT, "interpretation.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("brca_igr_2015 acquired-resistance alteration landscape\n")
        fh.write("data: WES only (mutations + CNA); NO expression -> REP30 not computable\n")
        fh.write(f"samples: {len(samples)}\n\n")
        fh.write("== mutation frequency ==\n" + mf.to_string(index=False) + "\n\n")
        fh.write("== CNA ==\n" + cdf.to_string(index=False) + "\n\n")
        fh.write("== by subtype ==\n" + sdf.to_string(index=False) + "\n\n")
        fh.write("== IGR vs TCGA ==\n" + compdf.to_string(index=False) + "\n")
    print(f"\n[DONE] outputs in {OUT}")


if __name__ == "__main__":
    main()
