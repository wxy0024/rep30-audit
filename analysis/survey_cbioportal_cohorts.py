# -*- coding: utf-8 -*-
"""Survey cBioPortal for breast-cancer studies that could test REP30 vs ESR1 status.

Requirement traceability (why each filter exists):
  (1) must be BREAST cancer            -> cancer type filter
  (2) must have MRNA_EXPRESSION        -> REP30 is a Cox-weighted EXPRESSION score
  (3) must have mutations              -> ESR1 / RB1 status needs DNA calls
  (4) metastatic or treated setting    -> the question is about acquired resistance
"""
import io
import json
import os
import time
import urllib.request

BASE = r"E:\01.WorkSpace\01.Breast_cancer_gene_panel\resistance_signature"
OUTD = os.path.join(BASE, "results", "cohort_survey")
os.makedirs(OUTD, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0 Safari/537.36",
      "Accept": "application/json"}


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


def get_text(url, tries=2):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA["User-Agent"]})
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            if i == tries - 1:
                return "__ERROR__" + str(e)
            time.sleep(2)


log = []


def L(*a):
    log.append(" ".join(str(x) for x in a))


L("fetching study list ...")
studies = get("https://www.cbioportal.org/api/studies?projection=SUMMARY")
if isinstance(studies, dict) and "__error__" in studies:
    L("FAILED: " + studies["__error__"])
    with io.open(os.path.join(OUTD, "survey_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    raise SystemExit(1)

L(f"total studies: {len(studies)}")
ct = get("https://www.cbioportal.org/api/cancer-types")
ctmap = {}
if isinstance(ct, list):
    ctmap = {c.get("cancerTypeId"): c.get("name", "") for c in ct}
    L(f"cancer types: {len(ctmap)}")

# candidate breast studies
cand = []
for s in studies:
    sid = s.get("studyId", "")
    name = (s.get("name") or "")
    desc = (s.get("description") or "")
    cid = s.get("cancerTypeId") or ""
    cname = ctmap.get(cid, "")
    blob = f"{sid} {name} {desc} {cid} {cname}".lower()
    if ("breast" in blob) or cid.startswith("brca") or cid in (
            "idc", "ilc", "brca", "mbc", "bcs", "brcnos"):
        cand.append(dict(studyId=sid, name=name, cancerTypeId=cid,
                         cancerTypeName=cname, description=desc[:300],
                         allSampleCount=s.get("allSampleCount"),
                         sequenceSampleCount=s.get("sequenceSampleCount")))
L(f"breast-related studies: {len(cand)}")

# for each, fetch molecular profiles
rows = []
for i, c in enumerate(cand):
    sid = c["studyId"]
    profs = get(f"https://www.cbioportal.org/api/studies/{sid}/molecular-profiles")
    types = []
    if isinstance(profs, list):
        types = sorted({p.get("molecularAlterationType", "") for p in profs})
    has_mrna = any(t == "MRNA_EXPRESSION" for t in types)
    has_mut = any(t == "MUTATION_EXTENDED" for t in types)
    has_cna = any(t == "COPY_NUMBER_ALTERATION" or t == "DISCRETE copy number"
                  for t in types) or "COPY_NUMBER_ALTERATION" in types
    blobby = f"{sid} {c['name']} {c['description']}".lower()
    metish = any(k in blobby for k in
                 ["metasta", "advanced", "resist", "relapse", "refractory",
                  "pretreated", "cdk4/6", "endocrine"])
    err = ""
    if isinstance(profs, dict) and "__error__" in profs:
        err = profs["__error__"][:80]
    rows.append(dict(studyId=sid, name=c["name"], cancerTypeId=c["cancerTypeId"],
                     n_samples=c.get("allSampleCount"),
                     n_sequenced=c.get("sequenceSampleCount"),
                     has_mRNA=int(has_mrna), has_mutation=int(has_mut),
                     has_CNA=int(has_cna), metastatic_or_treated=int(metish),
                     alteration_types=";".join(types)[:200], error=err))
    if (i + 1) % 20 == 0:
        L(f"  profiled {i+1}/{len(cand)} studies")

import pandas as pd  # noqa: E402
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUTD, "cbioportal_breast_studies.csv"), index=False, encoding="utf-8")

# --- shortlist: breast + mRNA + mutation ---
sl = df[(df["has_mRNA"] == 1) & (df["has_mutation"] == 1)].copy()
sl = sl.sort_values(["metastatic_or_treated", "n_samples"], ascending=False)
sl.to_csv(os.path.join(OUTD, "shortlist_mRNA_and_mutation.csv"), index=False, encoding="utf-8")

L("")
L("=" * 78)
L("SHORTLIST: breast studies with BOTH mRNA expression and mutation data")
L("=" * 78)
for _, r in sl.iterrows():
    L(f"  {r['studyId']:<38} n={str(r['n_samples']):>6}  met/treated={r['metastatic_or_treated']}  "
      f"{str(r['name'])[:60]}")

L("")
L(f"total shortlisted: {len(sl)}")
with io.open(os.path.join(OUTD, "survey_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log))
print("OK", len(df), len(sl))
