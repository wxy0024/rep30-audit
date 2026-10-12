#!/usr/bin/env python
"""
Fetch metastatic / acquired-resistance RNA cohorts for REP30 validation.

Deliberately network-only helper (sandbox can reach ftp.ncbi.nlm.nih.gov and
ucscpublic.xenahubs.net, but NOT www.ncbi.nlm.nih.gov).

Cohorts
-------
AURORA_US : GSE209998 (AUR_129 UQN + raw counts + series matrix metadata)
            GSE193103 (salmon gene matrix + platform series matrices)
            = 55 MBC patients, 51 primaries + 102 metastases (paired design)
              Garcia-Recio et al., Nat Cancer 2023.
MET500    : UCSC Xena public hub (probe a set of candidate filenames).

Usage
-----
python analysis/fetch_metastatic_rna.py --cohort aurora
python analysis/fetch_metastatic_rna.py --cohort met500
python analysis/fetch_metastatic_rna.py --cohort aurora --cohort met500
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import urllib.error

ROOT = r"E:/01.WorkSpace/01.Breast_cancer_gene_panel/resistance_signature"
DEST = os.path.join(ROOT, "data", "06_metastatic")

GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series"

AURORA_FILES = [
    # (url,        relative local path)
    (f"{GEO}/GSE209nnn/GSE209998/suppl/GSE209998_AUR_129_UQN.txt.gz",
     "AURORA_US/GSE209998_AUR_129_UQN.txt.gz"),
    (f"{GEO}/GSE209nnn/GSE209998/suppl/GSE209998_AUR_129_raw_counts.txt.gz",
     "AURORA_US/GSE209998_AUR_129_raw_counts.txt.gz"),
    (f"{GEO}/GSE209nnn/GSE209998/matrix/GSE209998_series_matrix.txt.gz",
     "AURORA_US/GSE209998_series_matrix.txt.gz"),
    (f"{GEO}/GSE193nnn/GSE193103/suppl/GSE193103_salmon_gene_normalized.matrix_RAP101_plus_Normals24.txt.gz",
     "AURORA_US/GSE193103_salmon_gene_normalized.matrix_RAP101_plus_Normals24.txt.gz"),
    (f"{GEO}/GSE193nnn/GSE193103/matrix/GSE193103-GPL11154_series_matrix.txt.gz",
     "AURORA_US/GSE193103-GPL11154_series_matrix.txt.gz"),
    (f"{GEO}/GSE193nnn/GSE193103/matrix/GSE193103-GPL16791_series_matrix.txt.gz",
     "AURORA_US/GSE193103-GPL16791_series_matrix.txt.gz"),
]

# UCSC Xena public hub - candidate filenames for MET500 (probe until one hits)
XENA = "https://ucscpublic.xenahubs.net/download/"
MET500_CANDIDATES = [
    "MET500.htseq_fpkm.tsv",
    "MET500.htseq_fpkm.tsv.gz",
    "MET500.htseq_fpkm-uq.tsv",
    "MET500.expression.tsv",
    "MET500.gene_expression.tsv",
    "MET500.fpkm.tsv",
    "MET500.phenotype.tsv",
    "MET500.clinical.tsv",
    "MET500.sampleMap/HiSeqV2",
    "gencode.v23.annotation.gene.probemap",
    "gencode.v23.annotation.transcript.probemap",
    "gencode.v36.annotation.gene.probemap",
]


def log(msg):
    print(msg, flush=True)


def head_ok(url, timeout=25):
    """Return (ok, size_or_err) via a Range GET of 1 byte."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0",
                                               "Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            cr = r.headers.get("Content-Range", "")
            total = cr.split("/")[-1] if "/" in cr else r.headers.get("Content-Length", "?")
            return True, f"HTTP {r.status} size={total}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:80]}"


def download(url, dest, min_bytes=1000):
    if os.path.exists(dest) and os.path.getsize(dest) > min_bytes:
        log(f"  [skip] already present ({os.path.getsize(dest):,} B)  {os.path.basename(dest)}")
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as fh:
            total = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                total += len(chunk)
        os.replace(tmp, dest)
        log(f"  [ok]   {total:>12,} B  {os.path.basename(dest)}")
        return True
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        log(f"  [FAIL] {url}\n         {type(e).__name__}: {str(e)[:140]}")
        return False


def do_aurora():
    log("\n=== AURORA_US (GSE209998 / GSE193103) ===")
    ok = 0
    for url, rel in AURORA_FILES:
        dest = os.path.join(DEST, rel)
        if download(url, dest):
            ok += 1
    log(f"AURORA_US: {ok}/{len(AURORA_FILES)} files present")


def do_met500():
    log("\n=== MET500 (UCSC Xena public hub, probing candidates) ===")
    os.makedirs(os.path.join(DEST, "MET500"), exist_ok=True)
    for name in MET500_CANDIDATES:
        url = XENA + name
        ok, info = head_ok(url)
        log(f"  {'HIT ' if ok else 'miss'} {info:<28} {name}")
        if ok:
            dest = os.path.join(DEST, "MET500", os.path.basename(name))
            download(url, dest, min_bytes=500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", action="append", choices=["aurora", "met500"],
                    help="repeatable: aurora, met500")
    ap.add_argument("--probe-only", action="store_true")
    args = ap.parse_args()
    cohorts = args.cohort or ["aurora"]

    os.makedirs(DEST, exist_ok=True)
    if args.probe_only:
        for c in cohorts:
            (do_aurora if c == "aurora" else do_met500)()
        return
    for c in cohorts:
        if c == "aurora":
            do_aurora()
        else:
            do_met500()
    log(f"\nDEST = {DEST}")
    for dirpath, _, files in os.walk(DEST):
        for f in sorted(files):
            p = os.path.join(dirpath, f)
            log(f"  {os.path.getsize(p):>12,} B  {os.path.relpath(p, DEST)}")


if __name__ == "__main__":
    main()
