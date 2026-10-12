#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download repair: re-download only the 2 problematic GEO datasets (run locally; the sandbox cannot reach NCBI).
  - GSE6532: the GPL96 / GPL570 series matrices were truncated -> re-download
              (GPL97 is intact and unused; GPL96=HG-U133A is the signature Loi tamoxifen platform)
  - GSE47561: no standard series matrix, only a log2_RMA expression matrix (truncated) + sample_characteristics -> re-download
Usage:
  python download_fix.py
"""
import os, gzip, time, ssl, urllib.request

BASE = r"E:\01.WorkSpace\01.Breast_cancer_gene_panel\resistance_signature\data"
CTX = ssl.create_default_context()
UA = {"User-Agent": "Mozilla/5.0 (download_fix.py)"}

# (GSE, numeric prefix nnn, target subdir 02_acquired_resistance/<GSE>, file-URL suffix or suppl filename)
JOBS = [
    ("GSE6532", "GSE6nnn", "GSE6532-GPL96_series_matrix.txt.gz",
     "matrix/GSE6532-GPL96_series_matrix.txt.gz"),
    ("GSE6532", "GSE6nnn", "GSE6532-GPL570_series_matrix.txt.gz",
     "matrix/GSE6532-GPL570_series_matrix.txt.gz"),
    ("GSE47561", "GSE47nnn", "GSE47561_log2_RMA_renormalized_data.txt.gz",
     "suppl/GSE47561_log2_RMA_renormalized_data.txt.gz"),
    ("GSE47561", "GSE47nnn", "GSE47561_sample_characteristics.txt.gz",
     "suppl/GSE47561_sample_characteristics.txt.gz"),
]

def get(url, dest, retries=5):
    for i in range(1, retries+1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, context=CTX, timeout=300) as r, open(dest, "wb") as f:
                while True:
                    b = r.read(1024*1024)
                    if not b: break
                    f.write(b)
            return True
        except Exception as e:
            if i < retries:
                time.sleep(3); continue
            print(f"    FAILED {url}: {e}"); return False
    return False

def gz_ok(p):
    try:
        with gzip.open(p, "rb") as f:
            while f.read(1024*1024): pass
        return True
    except Exception:
        return False

def main():
    for gse, pfx, fname, suffix in JOBS:
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{pfx}/{gse}/{suffix}"
        dest = os.path.join(BASE, "02_acquired_resistance", gse, fname)
        print(f">> {gse} / {fname}")
        print(f"   {url}")
        if get(url, dest):
            ok = gz_ok(dest)
            print(f"   saved size={os.path.getsize(dest)} gzip_valid={ok}")
        else:
            print(f"   !! download failed; fetch manually from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse}")
    print("\nDone. After re-downloading, re-run _verify_data.py to confirm everything is green.")

if __name__ == "__main__":
    main()
