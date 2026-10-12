# -*- coding: utf-8 -*-
"""
Extract MET500 Supplementary Table 1 (Demographics and Clinical Details) from the
Nature 2017 MET500 cohort paper supplementary PDF
(MOESM1, doi:10.1038/nature23306).

Source column order:
  CASE ID | AGE | GENDER | POPULATION | CANCER TYPE | ONCOTREE CODE |
  BIOPSY SITE | ANALYSIS COHORT | PRIMARY SITE | INTERVAL (mo) |
  PRIOR THERAPY | OTHER CANCER HISTORY

Parsing strategy
----------------
1. head anchor  : CASE_ID AGE GENDER
2. population   : closed vocabulary
3. tail anchor  : the first "<interval> <Y|N|NA>" pair after population
                  (cancer type / biopsy site / primary site contain no digits)
4. cohort anchor: the last token that is one of the 22 MET500 analysis cohorts
5. code anchor  : the first ALL-CAPS token between cancer type and biopsy site
6. biopsy site  : bootstrapped vocabulary harvested from the rows that DO carry an
                  oncotree code, then applied as a suffix match to the rows that do not.
7. primary site : everything after the cohort token

Outputs
-------
MET500_clinical_TableS1.tsv         (case level, 12 clinical columns)
MET500_clinical_TableS1.report.txt  (parse QA report)
"""
import os
import re
import collections

PDF = r"C:\Users\mao02\AppData\Local\Temp\met500_supp\MOESM1_23306.pdf"
OUTDIR = r"E:\01.WorkSpace\01.Breast_cancer_gene_panel\resistance_signature\data\06_metastatic\MET500"
OUT = os.path.join(OUTDIR, "MET500_clinical_TableS1.tsv")
REPORT = os.path.join(OUTDIR, "MET500_clinical_TableS1.report.txt")

COHORTS = {"ACC", "BLCA", "BRCA", "CHOL", "COLO", "ESCA", "GBM", "HCC", "HNSC",
           "KDNY", "LUNG", "MISC", "OV", "PAAD", "PRAD", "SARC", "SECR", "SKCM",
           "STAD", "TGCT", "THCA", "THYM"}

POPS = ["African Am.", "Hispanic Am.", "Native American", "Caucasian", "African",
        "Hispanic", "Asian", "Am. Indian", "Other", "Unknown", "Not Reported"]

# fallback biopsy vocabulary (from MET500 expression meta 'biopsy_tissue')
FALLBACK_BIOPSY = ["Lymph Node", "Bone Marrow", "Soft Tissue", "Subcut. Nodule",
                   "Abdom. Wall", "Abdom. Mass", "Chest Wall", "Adrenal", "Bladder",
                   "Brain", "Breast", "Cervix", "Colon", "Liver", "Lung", "Oral",
                   "Pancreas", "Prostate", "Skin", "Thyroid", "Bone", "Ovary",
                   "Omentum", "Pleura", "Peritoneum", "Retroperitoneum", "Unknown"]

HEAD_RE = re.compile(
    r"^(?:\d+\s+)?(?P<case>(?:ES|MO|TP)_\d+)\s+(?P<age>\d{1,3})\s+(?P<sex>[MF])\s+(?P<rest>.+)$")
TAIL_RE = re.compile(r"(?<![\w.])(>?\d{1,4}|NA)\s+(Y|N|NA)(?=\s|$)")
UPPER_RE = re.compile(r"^[A-Z][A-Z0-9]{1,7}$")
CODE_STOP = {"NA", "Y", "N"}


def pdf_pages(pdf, max_page=14):
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader
    rd = PdfReader(pdf)
    return [(i + 1, rd.pages[i].extract_text() or "") for i in range(min(max_page, len(rd.pages)))]


def split_pop(rest):
    for p in POPS:
        if rest.startswith(p):
            return p, rest[len(p):].strip()
    tok = rest.split(" ", 1)
    return tok[0], (tok[1] if len(tok) > 1 else "")


def main():
    records = []
    issues = []
    for pg, txt in pdf_pages(PDF):
        for line in txt.split("\n"):
            line = line.strip()
            m = HEAD_RE.match(line)
            if not m:
                continue
            pop, rest = split_pop(m.group("rest"))
            t = TAIL_RE.search(rest)
            if not t:
                issues.append(("no_tail", m.group("case"), rest))
                continue
            head = rest[:t.start()].strip()
            interval, prior = t.group(1), t.group(2)
            other = rest[t.end():].strip()
            records.append(dict(case_id=m.group("case"), age=m.group("age"),
                                gender=m.group("sex"), population=pop,
                                head=head, interval_mo=interval,
                                prior_therapy=prior, other_cancer_history=other,
                                page=pg))

    # --- de-duplicate (same case may appear on a continued page header) ---
    byc = {}
    for r in records:
        byc.setdefault(r["case_id"], r)
    records = list(byc.values())
    # --- keep only Table 1 region: must have a cohort token OR a plausible head ---
    records = [r for r in records if any(tok in COHORTS for tok in r["head"].split())
               or len(r["head"].split()) <= 8]

    # --- structural split ---
    pending = []
    for r in records:
        toks = r["head"].split()
        cohort, ci = "NA", None
        for j in range(len(toks) - 1, -1, -1):
            if toks[j] in COHORTS:
                cohort, ci = toks[j], j
                break
        if ci is None:
            r["analysis_cohort"] = "NA"
            r["primary_site"] = ""
            pre = toks
        else:
            r["analysis_cohort"] = cohort
            r["primary_site"] = " ".join(toks[ci + 1:])
            pre = toks[:ci]
        # oncotree code = first ALL-CAPS token in pre, but "... , NOS" belongs to the
        # cancer name (e.g. "Adenocarcinoma, NOS ADNOS Liver" -> code = ADNOS)
        code_i = None
        for j, tk in enumerate(pre):
            if UPPER_RE.match(tk) and tk not in CODE_STOP:
                if tk == "NOS":          # "..., NOS" is part of the cancer name
                    continue
                code_i = j
                break
        if code_i is not None:
            r["cancer_type"] = " ".join(pre[:code_i]).strip().rstrip(",").strip()
            r["oncotree_code"] = pre[code_i]
            r["biopsy_site"] = " ".join(pre[code_i + 1:])
        else:
            r["oncotree_code"] = "NA"
            r["cancer_type"] = " ".join(pre).strip().rstrip(",").strip()
            r["biopsy_site"] = ""
            pending.append(r)

    # --- biopsy vocabulary bootstrap ---
    vocab = collections.Counter()
    for r in records:
        if r["biopsy_site"]:
            vocab[r["biopsy_site"]] += 1
    bio_vocab = set(vocab) | set(FALLBACK_BIOPSY)
    # suffixes that never belong to a cancer type
    def is_suffix(phrase, pre_str):
        return pre_str == phrase or pre_str.endswith(" " + phrase)

    resolved = 0
    for r in pending:
        pre_str = r["cancer_type"]
        cands = [p for p in bio_vocab if is_suffix(p, pre_str)]
        if cands:
            best = max(cands, key=len)
            r["biopsy_site"] = best
            r["cancer_type"] = pre_str[:-len(best)].strip().rstrip(",")
            resolved += 1
    print(f"cases: {len(records)}  bio_vocab:{len(bio_vocab)}  pending resolved:{resolved}/{len(pending)}")

    # --- manual overrides for rows the grammar cannot resolve (hand-checked
    #     against the raw PDF line; only 2 of 500 cases) ---
    MANUAL_FIX = {
        # "64 MO_1079 53 M Caucasian Sarcomatoid Carcinoma Mandible MISC Mandible 0 N"
        "MO_1079": dict(cancer_type="Sarcomatoid Carcinoma", oncotree_code="NA",
                        biopsy_site="Mandible", analysis_cohort="MISC",
                        primary_site="Mandible"),
        # "112 MO_1142 72 F Caucasian Clear Cell Carcinoma Lung Tongue 161 Y ..."
        # (analysis cohort left blank in the source table)
        "MO_1142": dict(cancer_type="Clear Cell Carcinoma", oncotree_code="NA",
                        biopsy_site="Lung", analysis_cohort="NA", primary_site="Tongue"),
    }
    for r in records:
        if r["case_id"] in MANUAL_FIX:
            r.update(MANUAL_FIX[r["case_id"]])

    header = ["case_id", "age", "gender", "population", "cancer_type", "oncotree_code",
              "biopsy_site", "analysis_cohort", "primary_site", "interval_mo",
              "prior_therapy", "other_cancer_history"]
    os.makedirs(OUTDIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for r in sorted(records, key=lambda x: x["case_id"]):
            f.write("\t".join(str(r.get(h, "")).strip() for h in header) + "\n")

    rep = []
    rep.append(f"MET500 Supplementary Table 1 parse report")
    rep.append(f"cases parsed            : {len(records)}")
    rep.append(f"biopsy vocab size       : {len(bio_vocab)}")
    rep.append(f"pending rows resolved   : {resolved}/{len(pending)}")
    rep.append(f"head-anchor failures    : {len(issues)}")
    rep.append("")
    rep.append("cohort distribution     : " + str(collections.Counter(r['analysis_cohort'] for r in records).most_common()))
    rep.append("prior therapy           : " + str(collections.Counter(r['prior_therapy'] for r in records).most_common()))
    rep.append("top biopsy sites        : " + str(collections.Counter(r['biopsy_site'] for r in records).most_common(25)))
    rep.append("top oncotree codes      : " + str(collections.Counter(r['oncotree_code'] for r in records).most_common(25)))
    bad = [r for r in records if not r["biopsy_site"] or r["analysis_cohort"] == "NA"]
    rep.append("")
    rep.append(f"rows needing manual review: {len(bad)}")
    for r in bad:
        rep.append("   " + str({k: r.get(k) for k in ["case_id", "page", "head", "cancer_type", "biopsy_site", "analysis_cohort", "primary_site"]}))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(rep))
    print("written:", OUT)
    print("\n".join(rep[:14]))


if __name__ == "__main__":
    main()
