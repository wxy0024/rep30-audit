#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per resistance gene, the frequency of altered patients (mutation+CNV) in TCGA / METABRIC. Outputs results/gene_freq.csv."""
import os
import pandas as pd
from collections import defaultdict

BASE = r"E:\01.WorkSpace\01.Breast_cancer_gene_panel\resistance_signature"
GENE_CSV = os.path.join(BASE, "resistance_genes.csv")
OUT = os.path.join(BASE, "results", "gene_freq.csv")
SILENT = {"Silent","Intron","3'UTR","5'UTR","3'Flank","5'Flank","IGR","RNA",
          "Targeted_Region","lincRNA","Splice_Region","",""}

def classify(df_genes):
    out={}
    for _,r in df_genes.iterrows():
        g=r["gene"].strip(); at=str(r["alt_type"])
        hm=("SNV" in at) or ("mut" in at); ha="amp" in at; hd="del" in at
        if hm or ha or hd: out[g]={"mut":hm,"amp":ha,"del":hd}
    return out

def mut_per_gene(path, genomic):
    use=["Hugo_Symbol","Variant_Classification","Tumor_Sample_Barcode"]
    df=pd.read_csv(path,sep="\t",comment="#",header=0,usecols=use,dtype=str,
                   keep_default_na=False,low_memory=False)
    df=df[df["Hugo_Symbol"].isin(set(genomic.keys()))]
    df=df[~df["Variant_Classification"].isin(SILENT)]
    df["pid"]=df["Tumor_Sample_Barcode"].str.split("-").str[:3].str.join("-")
    cnt=defaultdict(set)
    for g,pid in zip(df["Hugo_Symbol"],df["pid"]):
        if genomic[g]["mut"]: cnt[g].add(pid)
    return cnt

def cna_per_gene(path, genomic):
    df=pd.read_csv(path,sep="\t",comment="#",header=0,low_memory=False)
    df=df[df["Hugo_Symbol"].isin(set(genomic.keys()))].copy()
    genes=df["Hugo_Symbol"].tolist()
    cols=[c for c in df.columns if c not in ("Hugo_Symbol","Entrez_Gene_Id")]
    amp=defaultdict(set); dele=defaultdict(set)
    for gi,g in enumerate(genes):
        if not (genomic[g]["amp"] or genomic[g]["del"]): continue
        vals=pd.to_numeric(df.iloc[gi][cols],errors="coerce")
        for sid,v in vals.items():
            if pd.isna(v): continue
            iv=int(v)
            if genomic[g]["amp"] and iv>=2: amp[g].add(sid)
            if genomic[g]["del"] and iv<=-2: dele[g].add(sid)
    return amp,dele

def cohort_freq(cohort_dir, genomic):
    m=mut_per_gene(os.path.join(cohort_dir,"data_mutations.txt"),genomic) if os.path.exists(os.path.join(cohort_dir,"data_mutations.txt")) else {}
    a,d=cna_per_gene(os.path.join(cohort_dir,"data_cna.txt"),genomic) if os.path.exists(os.path.join(cohort_dir,"data_cna.txt")) else ({},{})
    # total number of patients (clinical)
    clin=os.path.join(cohort_dir,"data_clinical_patient.txt")
    cdf=pd.read_csv(clin,sep="\t",header=0,dtype=str,keep_default_na=False,low_memory=False)
    cdf.columns=[c.lstrip("#") for c in cdf.columns]
    cdf=cdf[~cdf.iloc[:,0].str.startswith("#")]
    total=len(cdf)
    rows=[]
    for g in genomic:
        altered=m.get(g,set())|a.get(g,set())|d.get(g,set())
        rows.append((g,len(altered)))
    return total, dict(rows)

def main():
    gdf=pd.read_csv(GENE_CSV)
    genomic=classify(gdf)
    cohorts={
        "TCGA-BRCA":os.path.join(BASE,"data","01_core","TCGA-BRCA","brca_tcga_pan_can_atlas_2018"),
        "METABRIC":os.path.join(BASE,"data","01_core","METABRIC","brca_metabric"),
    }
    out=pd.DataFrame({"gene":list(genomic.keys())})
    out["evidence"]=out["gene"].map(gdf.set_index("gene")["evidence"])
    out["alt_type"]=out["gene"].map(gdf.set_index("gene")["alt_type"])
    for name,d in cohorts.items():
        tot,rows=cohort_freq(d,genomic)
        out[f"{name}_n"]=out["gene"].map(lambda g:rows.get(g,0))
        out[f"{name}_pct"]=out[f"{name}_n"]/tot*100
        print(f"{name}: total patients={tot}")
    out=out.sort_values("METABRIC_pct",ascending=False)
    out.to_csv(OUT,index=False)
    print("\n",out.to_string(index=False))
    print("\nSaved:",OUT)

if __name__=="__main__":
    main()
