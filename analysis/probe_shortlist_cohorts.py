# -*- coding: utf-8 -*-
"""Probe the shortlisted cBioPortal breast cohorts for REP30-vs-ESR1 feasibility."""
import io
import json
import os
import time
import urllib.request

BASE = r"E:\01.WorkSpace\01.Breast_cancer_gene_panel\resistance_signature"
OUTD = os.path.join(BASE, "results", "cohort_survey")
os.makedirs(OUTD, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0 Safari/537.36",
      "Content-Type": "application/json", "Accept": "application/json"}
log = []


def L(*a):
    log.append(" ".join(str(x) for x in a))


def get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except Exception as e:
            if i == tries - 1:
                return {"__error__": str(e)}
            time.sleep(2)


def post(url, body, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                         headers=UA, method="POST")
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)
        except Exception as e:
            if i == tries - 1:
                return {"__error__": str(e)}
            time.sleep(2)


STUDIES = ["brca_mbcproject_2022", "brca_mbcproject_wagle_2017",
           "brca_aurora_2023", "brca_dldccc_2022"]
summary = []

for sid in STUDIES:
    L("=" * 78)
    L("STUDY:", sid)
    L("=" * 78)
    profs = get(f"https://www.cbioportal.org/api/studies/{sid}/molecular-profiles")
    mrna = []
    mutprof = None
    if isinstance(profs, list):
        for p in profs:
            t = p.get("molecularAlterationType", "")
            if t == "MRNA_EXPRESSION":
                mrna.append((p.get("molecularProfileId"), p.get("name"),
                             p.get("datatype")))
            if t == "MUTATION_EXTENDED":
                mutprof = p.get("molecularProfileId")
    L("mRNA profiles:")
    for m in mrna:
        L(f"   {m[0]}  |  {m[1]}  |  {m[2]}")
    L(f"mutation profile: {mutprof}")

    attrs = get(f"https://www.cbioportal.org/api/studies/{sid}/clinical-attributes")
    keys = []
    if isinstance(attrs, list):
        for a in attrs:
            keys.append((a.get("clinicalAttributeId"), a.get("displayName"),
                         a.get("datatype")))
    L(f"clinical attributes: {len(keys)}")
    interesting = [k for k in keys if any(t in (k[0] + " " + (k[1] or "")).lower()
                                          for t in ["er_", "estrogen", "receptor", "her2",
                                                    "pr_", "metast", "sample_type",
                                                    "treatment", "therapy", "line",
                                                    "age", "stage", "subtype"])]
    for k in interesting[:40]:
        L(f"   {k[0]:<34} | {str(k[1])[:44]:<44} | {k[2]}")

    sids = get(f"https://www.cbioportal.org/api/studies/{sid}/sample-lists")
    lists = {}
    if isinstance(sids, list):
        lists = {s.get("sampleListId"): s.get("name") for s in sids}
    L("sample lists: " + str(list(lists.keys())[:8]))

    # ESR1 / RB1 / TP53 / PIK3CA mutation counts via fetch API
    if mutprof:
        body = {"entrezGeneIds": [2099, 5925, 7157, 5290, 1956, 367],
                "molecularProfileIds": [mutprof],
                "sampleListIds": [f"{sid}_all"]}
        res = post("https://www.cbioportal.org/api/mutations/fetch", body)
        if isinstance(res, list):
            import collections
            cnt = collections.Counter()
            allsamp = set()
            for m in res:
                cnt[m.get("hugoGeneSymbol") or m.get("entrezGeneId")] += 1
                allsamp.add(m.get("sampleId"))
            L(f"mutations in ESR1/RB1/TP53/PIK3CA/EGFR/AKT1: {dict(cnt)}")
            L(f"  distinct samples carrying any of these: {len(allsamp)}")
            summary.append(dict(studyId=sid, n_mrna_profiles=len(mrna),
                                mrna_profile=mrna[0][0] if mrna else "",
                                mut_profile=mutprof or "",
                                esr1_mut=cnt.get("ESR1", 0),
                                rb1_mut=cnt.get("RB1", 0),
                                tp53_mut=cnt.get("TP53", 0),
                                pik3ca_mut=cnt.get("PIK3CA", 0),
                                n_clinical_attrs=len(keys)))
        else:
            L("  mutation fetch failed: " + str(res)[:200])
    L("")

import pandas as pd  # noqa: E402
if summary:
    pd.DataFrame(summary).to_csv(os.path.join(OUTD, "cohort_probe_summary.csv"),
                                 index=False, encoding="utf-8")
with io.open(os.path.join(OUTD, "cohort_probe_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log))
print("OK")
