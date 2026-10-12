#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer 1 RAS (Resistance Alteration Score) survival analysis
=================================================
Inputs (local cBioPortal format):
  data_mutations.txt, data_cna.txt, data_clinical_patient.txt for TCGA-BRCA / METABRIC
  resistance_genes.csv  (alt_type determines, per gene, whether SNV / amplification / deletion is counted)

Workflow:
  1. Use alt_type to split genes into the "genomic layer" (SNV/amplification/deletion) and the "expression layer" (skipped here; handled by REP).
  2. Parse mutations (and CNA) -> per sample, per resistance gene, presence of a deleterious alteration -> aggregate into RAS (number of altered genes).
  3. Parse clinical OS -> aggregate per patient -> Kaplan-Meier + Cox (RAS+ vs RAS-).
Outputs:
  results/ras_<cohort>_summary.csv, results/ras_<cohort>_km.png, stdout report.
"""
import os, sys, glob
import numpy as np
import pandas as pd

BASE = r"E:\01.WorkSpace\01.Breast_cancer_gene_panel\resistance_signature"
GENE_CSV = os.path.join(BASE, "resistance_genes.csv")
OUTDIR = os.path.join(BASE, "results")
os.makedirs(OUTDIR, exist_ok=True)

# nonsilent mutations (counted as deleterious)
SILENT = {"Silent", "Intron", "3'UTR", "5'UTR", "3'Flank", "5'Flank", "IGR",
          "RNA", "Targeted_Region", "lincRNA", "5'UTR", "Splice_Region",
          "5KB_UPSTREAM", "2KB_DOWNSTREAM", "3'UTR", ""}

def classify_genes(df_genes):
    """Returns genomic_genes: {gene: {'mut':bool,'amp':bool,'del':bool}}, genomic-layer genes only."""
    out = {}
    for _, r in df_genes.iterrows():
        g = r["gene"].strip()
        at = str(r["alt_type"])
        has_mut = ("SNV" in at) or ("mut" in at)
        has_amp = "amp" in at
        has_del = "del" in at
        if has_mut or has_amp or has_del:
            out[g] = {"mut": has_mut, "amp": has_amp, "del": has_del}
    return out

def parse_clinical(path):
    df = pd.read_csv(path, sep="\t", header=0, dtype=str, keep_default_na=False,
                     low_memory=False)
    df.columns = [c.lstrip("#") for c in df.columns]
    df = df[~df.iloc[:, 0].str.startswith("#")].copy()
    cols = df.columns.tolist()
    idc = cols[0]
    # locate the OS column
    status_col = next((c for c in cols if c.strip().lower() == "overall survival status"), None)
    if status_col is None:
        status_col = next((c for c in cols if "vital status" in c.lower()), None)
    months_col = next((c for c in cols if c.strip().lower() == "overall survival (months)"), None)
    age_col = next((c for c in cols if "age" in c.lower() and "diagnosis" in c.lower()), None)
    if age_col is None:
        age_col = next((c for c in cols if "age" in c.lower()), None)
    sub = df[[idc]].copy()
    sub["os_status"] = df[status_col] if status_col else ""
    sub["os_months"] = df[months_col] if months_col else ""
    sub["age"] = df[age_col] if age_col else ""
    return sub, idc

def load_mutations(path, genomic):
    # keep only the needed columns, filtered to resistance genes
    use = ["Hugo_Symbol", "Variant_Classification", "Tumor_Sample_Barcode"]
    df = pd.read_csv(path, sep="\t", comment="#", header=0, usecols=use,
                     dtype=str, keep_default_na=False, low_memory=False)
    df = df[df["Hugo_Symbol"].isin(set(genomic.keys()))]
    df = df[~df["Variant_Classification"].isin(SILENT)]
    # sample -> patient
    df["pid"] = df["Tumor_Sample_Barcode"].str.split("-").str[:3].str.join("-")
    # mark a mutation for each (pid, gene)
    mut_patients = set(zip(df["pid"], df["Hugo_Symbol"]))
    return mut_patients

def load_cna(path, genomic):
    df = pd.read_csv(path, sep="\t", comment="#", header=0, low_memory=False)
    df = df[df["Hugo_Symbol"].isin(set(genomic.keys()))].copy()
    genes = df["Hugo_Symbol"].tolist()
    sample_cols = [c for c in df.columns if c not in ("Hugo_Symbol", "Entrez_Gene_Id")]
    amp = {}  # (pid, gene)
    dele = {}
    for gi, g in enumerate(genes):
        if not (genomic[g]["amp"] or genomic[g]["del"]):
            continue
        vals = pd.to_numeric(df.iloc[gi][sample_cols], errors="coerce")
        for sid, v in vals.items():
            if pd.isna(v):
                continue
            iv = int(v)
            if genomic[g]["amp"] and iv >= 2:
                amp[(sid, g)] = 1
            if genomic[g]["del"] and iv <= -2:
                dele[(sid, g)] = 1
    return amp, dele

def build_ras(cohort_dir, name):
    genomic = classify_genes(pd.read_csv(GENE_CSV))
    mut_path = os.path.join(cohort_dir, "data_mutations.txt")
    cna_path = os.path.join(cohort_dir, "data_cna.txt")
    clin_path = os.path.join(cohort_dir, "data_clinical_patient.txt")

    mut_patients = load_mutations(mut_path, genomic) if os.path.exists(mut_path) else set()
    amp, dele = load_cna(cna_path, genomic) if os.path.exists(cna_path) else ({}, {})

    # per patient RAS: number of altered resistance genes (mutation/amplification/deletion, any one counts; deduplicated to gene)
    from collections import defaultdict
    altered_genes = defaultdict(set)
    for (pid, g) in mut_patients:
        altered_genes[pid].add(g)
    for (sid, g) in amp:
        altered_genes[sid].add(g)
    for (sid, g) in dele:
        altered_genes[sid].add(g)

    clin, idc = parse_clinical(clin_path)
    clin[idc] = clin[idc].str.strip()
    ras_map = {pid: len(gs) for pid, gs in altered_genes.items()}
    clin["RAS"] = clin[idc].map(ras_map).fillna(0).astype(int)
    # RAS+ = at least 1 altered resistance gene; RAS_high = >=2
    clin["RAS_pos"] = (clin["RAS"] >= 1).astype(int)
    clin["RAS_high"] = (clin["RAS"] >= 2).astype(int)

    # clean the survival data
    clin["os_months"] = pd.to_numeric(clin["os_months"], errors="coerce")
    clin["os_event"] = clin["os_status"].str.upper().str.contains("DECEASED|DEAD|1") & \
                       ~clin["os_status"].str.upper().str.contains("LIVING|ALIVE|0")
    clin["age"] = pd.to_numeric(clin["age"], errors="coerce")

    keep = clin.dropna(subset=["os_months"]).copy()
    keep = keep[keep["os_months"] > 0]
    print(f"\n==== {name} ====")
    print(f"  genomic-layer resistance genes: {len(genomic)} -> {sorted(genomic.keys())}")
    print(f"  clinical samples (with OS): {len(keep)}; RAS+ patients: {(keep['RAS_pos']==1).sum()}, RAS_high: {(keep['RAS_high']==1).sum()}")
    print(f"  events (deaths): {int(keep['os_event'].sum())}")
    return clin, keep, name, sorted(genomic.keys())

def cox_km(keep, name):
    from lifelines import CoxPHFitter, KaplanMeierFitter
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # keep only rows with covariates available
    sub = keep.copy()
    res = {}
    # 1) univariable Cox: RAS_pos
    if sub["RAS_pos"].nunique() > 1 and sub["os_event"].sum() > 5:
        try:
            cph = CoxPHFitter()
            cph.fit(sub[["os_months", "os_event", "RAS_pos"]], duration_col="os_months",
                    event_col="os_event")
            hr = float(np.exp(cph.params_["RAS_pos"]))
            ci = np.exp(cph.confidence_intervals_.loc["RAS_pos"]).tolist()
            p = float(cph.summary.loc["RAS_pos", "p"])
            res["RAS_pos_HR"] = (hr, ci, p)
            print(f"  [Cox] RAS+ vs RAS- : HR={hr:.2f} (95%CI {ci[0]:.2f}-{ci[1]:.2f}), p={p:.1e}")
        except Exception as e:
            print(f"  [Cox] RAS_pos failed: {e}")
    # 2) multivariable Cox: RAS_pos + age
    if sub["age"].notna().sum() > 10:
        sub2 = sub.dropna(subset=["age"])
        if sub2["RAS_pos"].nunique() > 1:
            try:
                cph = CoxPHFitter()
                cph.fit(sub2[["os_months", "os_event", "RAS_pos", "age"]], duration_col="os_months",
                        event_col="os_event")
                hr = float(np.exp(cph.params_["RAS_pos"]))
                ci = np.exp(cph.confidence_intervals_.loc["RAS_pos"]).tolist()
                p = float(cph.summary.loc["RAS_pos", "p"])
                res["RAS_pos_adj_HR"] = (hr, ci, p)
                print(f"  [Cox*age] RAS+ : HR={hr:.2f} (95%CI {ci[0]:.2f}-{ci[1]:.2f}), p={p:.1e}")
            except Exception as e:
                print(f"  [Cox*age] failed: {e}")
    # 3) KM plot
    try:
        plt.figure(figsize=(6, 5))
        for label, grp in [(f"RAS- (n={int((sub['RAS_pos']==0).sum())})", sub[sub.RAS_pos==0]),
                           (f"RAS+ (n={int((sub['RAS_pos']==1).sum())})", sub[sub.RAS_pos==1])]:
            if len(grp) == 0:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(grp["os_months"], event_observed=grp["os_event"], label=label)
            kmf.plot_survival_function()
        plt.title(f"{name}: OS by RAS status")
        plt.xlabel("Months"); plt.ylabel("Overall survival")
        plt.tight_layout()
        fp = os.path.join(OUTDIR, f"ras_{name}_km.png")
        plt.savefig(fp, dpi=130); plt.close()
        print(f"  [KM] figure saved: {fp}")
    except Exception as e:
        print(f"  [KM] failed: {e}")
    return res

def main():
    cohorts = {
        "TCGA-BRCA": os.path.join(BASE, "data", "01_core", "TCGA-BRCA", "brca_tcga_pan_can_atlas_2018"),
        "METABRIC":  os.path.join(BASE, "data", "01_core", "METABRIC", "brca_metabric"),
    }
    allres = {}
    for name, d in cohorts.items():
        clin, keep, name, genes = build_ras(d, name)
        res = cox_km(keep, name)
        allres[name] = (keep, res, genes)
        # save the RAS score table for each cohort
        out = clin[[clin.columns[0], "RAS", "RAS_pos", "RAS_high", "os_months", "os_status", "age"]]
        out.columns = ["patient_id", "RAS", "RAS_pos", "RAS_high", "OS_months", "OS_status", "age"]
        out.to_csv(os.path.join(OUTDIR, f"ras_{name}_summary.csv"), index=False)

    # small cross-cohort frequency table
    print("\n==== per resistance gene: frequency of altered patients (mutation/CNV) in TCGA / METABRIC ====")
    # simplified: count the absolute number of altered patients among RAS genes
    for name, (keep, res, genes) in allres.items():
        n = len(keep)
        raspos = int((keep["RAS_pos"] == 1).sum())
        print(f"  {name}: N={n}, RAS+={raspos} ({100*raspos/n:.1f}%)")

    print("\nDone. Results in:", OUTDIR)

if __name__ == "__main__":
    main()
