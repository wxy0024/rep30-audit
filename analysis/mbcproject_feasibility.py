# -*- coding: utf-8 -*-
"""Feasibility check for brca_mbcproject_2022 / wagle_2017:
can we compute REP30 (needs RNA) AND assign ESR1/RB1 status (needs DNA)
AND restrict to HR+/metastatic/treated (needs clinical)?"""
import io
import json
import os
import time
import urllib.request

import pandas as pd

BASE = r"E:\01.WorkSpace\01.Breast_cancer_gene_panel\resistance_signature"
OUTD = os.path.join(BASE, "results", "cohort_survey")
UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
log = []


def L(*a):
    log.append(" ".join(str(x) for x in a))


def get(url, timeout=300, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:
            if i == tries - 1:
                return {"__error__": f"{type(e).__name__}: {e}"}
            time.sleep(3)


SAMPLE_ATTRS = ["BX_ER", "BX_PR", "BX_HER2OVERALL", "CALC_MET_SETTING",
                "CALC_TREATMENT_NAIVE", "SAMPLE_TYPE", "CANCER_TYPE_DETAILED",
                "CALC_TIME_TO_METS_DX_MONTHS"]
PATIENT_ATTRS = ["HORMONE_RECEPTOR_STATUS", "HER2_STATUS_REPORTED",
                 "EVER_METS_SITES", "AGE_AT_DIAGNOSIS"]

rows = []
for SID in ["brca_mbcproject_2022", "brca_mbcproject_wagle_2017"]:
    L("=" * 78)
    L("STUDY:", SID)
    L("=" * 78)

    # RNA sample list
    rna = get(f"https://www.cbioportal.org/api/sample-lists/{SID}_rna_seq_v2_mrna")
    rna_ids = set(rna.get("sampleIds", [])) if isinstance(rna, dict) else set()
    L(f"RNA samples: {len(rna_ids)}")

    seq = get(f"https://www.cbioportal.org/api/sample-lists/{SID}_sequenced")
    seq_ids = set(seq.get("sampleIds", [])) if isinstance(seq, dict) else set()
    L(f"sequenced samples: {len(seq_ids)}")

    allsl = get(f"https://www.cbioportal.org/api/sample-lists/{SID}_all")
    all_ids = set(allsl.get("sampleIds", [])) if isinstance(allsl, dict) else set()
    L(f"all samples: {len(all_ids)}")

    # clinical attributes
    clin = {}
    for a in SAMPLE_ATTRS:
        r = get(f"https://www.cbioportal.org/api/studies/{SID}/clinical-data"
                f"?clinicalDataType=SAMPLE&attributeId={a}")
        if isinstance(r, list):
            clin[a] = {x["sampleId"]: x["value"] for x in r}
            L(f"  {a}: {len(clin[a])} values; dist="
              f"{pd.Series(list(clin[a].values())).value_counts().head(8).to_dict()}")
        else:
            L(f"  {a}: FAILED {str(r)[:80]}")
    for a in PATIENT_ATTRS:
        r = get(f"https://www.cbioportal.org/api/studies/{SID}/clinical-data"
                f"?clinicalDataType=PATIENT&attributeId={a}")
        if isinstance(r, list):
            clin[a] = {x["patientId"]: x["value"] for x in r}
            L(f"  {a} (patient): {len(clin[a])} values; dist="
              f"{pd.Series(list(clin[a].values())).value_counts().head(6).to_dict()}")

    # mutations (filter to genes of interest)
    m = get(f"https://www.cbioportal.org/api/molecular-profiles/{SID}_mutations"
            f"/mutations?sampleListId={SID}_sequenced&projection=SUMMARY",
            timeout=600)
    esr1 = set()
    gene_cnt = {}
    n_mut_records = 0
    if isinstance(m, list):
        n_mut_records = len(m)
        for rec in m:
            g = rec.get("hugoGeneSymbol") or rec.get("gene", {}).get("hugoGeneSymbol")
            s = rec.get("sampleId")
            if g:
                gene_cnt.setdefault(g, set()).add(s)
            if g == "ESR1":
                esr1.add(s)
    else:
        L("  mutation pull FAILED: " + str(m)[:200])
    L(f"mutation records: {n_mut_records}; genes hit: {len(gene_cnt)}")
    for g in ["ESR1", "RB1", "TP53", "PIK3CA", "AKT1", "NF1", "ERBB2", "CDH1", "GATA3"]:
        if g in gene_cnt:
            L(f"  {g}: {len(gene_cnt[g])} samples")

    usable = rna_ids & seq_ids
    L(f"RNA AND sequenced: {len(usable)}")
    er = clin.get("BX_ER", {})
    hrpos = {s for s in usable if str(er.get(s, "")).upper().startswith("POS")}
    L(f"  of which ER+ at biopsy: {len(hrpos)}")
    met_setting = clin.get("CALC_MET_SETTING", {})
    in_met = {s for s in usable if str(met_setting.get(s, "")).upper() in ("YES", "TRUE")}
    L(f"  of which sampled in metastatic setting: {len(in_met)}")
    rna_esr1 = esr1 & usable
    L(f"  ESR1-mutant among RNA samples: {len(rna_esr1)}")
    rna_hrpos_esr1 = rna_esr1 & hrpos
    L(f"  ESR1-mutant AND ER+ AND RNA: {len(rna_hrpos_esr1)}")

    rows.append(dict(studyId=SID, n_rna=len(rna_ids), n_sequenced=len(seq_ids),
                     n_all=len(all_ids), n_rna_and_seq=len(usable),
                     n_erpos=len(hrpos), n_met_setting=len(in_met),
                     n_esr1_mut_rna=len(rna_esr1),
                     n_esr1_pos_erpos_rna=len(rna_hrpos_esr1),
                     n_mutation_records=n_mut_records))
    L("")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUTD, "mbcproject_feasibility.csv"), index=False, encoding="utf-8")
with io.open(os.path.join(OUTD, "mbcproject_feasibility_log.txt"), "w",
             encoding="utf-8") as f:
    f.write("\n".join(log))
print("OK")
