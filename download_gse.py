#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Breast-cancer resistance analysis - GEO dataset download script (local-machine version, Python)

Works on any machine with Python 3.8+ (Windows / macOS / Linux)
Note: the sandbox cannot reach NCBI FTP; run this script on your local machine.
      The GEO FTP directory rule GSE<NNN>nnn (NNN = numeric part // 1000) was verified in the sandbox.

Dependencies: Python standard library only (urllib / ssl / gzip / re); no pip install needed.

Usage:
    python download_gse.py
    python download_gse.py --base "E:/01.WorkSpace/01.Breast_cancer_gene_panel/resistance_signature/data"
"""

import argparse
import gzip
import os
import re
import ssl
import sys
import time
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
DEFAULT_BASE = r"E:\01.WorkSpace\01.Breast_cancer_gene_panel\resistance_signature\data"

# 8 bulk endocrine / acquired-resistance expression series
BULK = ["GSE9195", "GSE6532", "GSE26971", "GSE17705",
        "GSE45255", "GSE47561", "GSE26459", "GSE33366"]
# 3 single-cell scRNA-seq series (supplementary files downloaded)
SINGLE = ["GSE264205", "GSE161529", "GSE249361"]

# single-line inline ssl configuration (a normal local network does not need --insecure; kept as a compatibility switch)
CTX = ssl.create_default_context()
# if your local machine also hits certificate problems, uncomment the next line to skip verification:
# CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE

UA = {"User-Agent": "Mozilla/5.0 (download_gse.py; contact: local-user)"}


# ---------------------------------------------------------------------------
# utility functions
# ---------------------------------------------------------------------------
def gse_prefix(gse: str) -> str:
    """GEO FTP directory rule: GSE + floor(number/1000) + 'nnn'.
    e.g. GSE9195 -> GSE9nnn; GSE264205 -> GSE264nnn; GSE26459 -> GSE26nnn"""
    n = int(gse[3:])
    return "GSE" + str(n // 1000) + "nnn"


def http_get(url: str, dest: str, retries: int = 3, retry_delay: int = 2) -> bool:
    """Download url to dest; returns success. With retries."""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, context=CTX, timeout=120) as resp:
                # stream to disk to avoid holding large files in memory
                with open(dest, "wb") as fh:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as e:
            if attempt < retries:
                print(f"     retry {attempt}/{retries} ({e}); waiting {retry_delay}s ...")
                time.sleep(retry_delay)
            else:
                print(f"     failed: {e}")
    return False


def verify_gzip(path: str) -> bool:
    """Rough check that the download is a valid gzip (magic bytes)."""
    try:
        with open(path, "rb") as fh:
            return fh.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def fetch_text(url: str, retries: int = 3, retry_delay: int = 2) -> str:
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, context=CTX, timeout=120) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as e:
            if attempt < retries:
                print(f"     retry {attempt}/{retries} ({e})")
                time.sleep(retry_delay)
            else:
                print(f"     index fetch failed: {e}")
    return ""


# ---------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="download the GEO datasets for the breast-cancer resistance analysis")
    ap.add_argument("--base", default=DEFAULT_BASE,
                    help="data root directory (default: %(default)s)")
    ap.add_argument("--bulk-only", action="store_true", help="download bulk series only")
    ap.add_argument("--single-only", action="store_true", help="download single-cell series only")
    args = ap.parse_args()

    base = os.path.abspath(args.base)
    os.makedirs(base, exist_ok=True)

    # ---------- 1) Bulk: series matrices (expression + sample annotations) ----------
    if not args.single_only:
        for g in BULK:
            pfx = gse_prefix(g)
            url = (f"https://ftp.ncbi.nlm.nih.gov/geo/series/{pfx}/{g}/"
                   f"matrix/{g}_series_matrix.txt.gz")
            dest = os.path.join(base, "02_acquired_resistance", g,
                                f"{g}_series_matrix.txt.gz")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            print(f">> {g}  ({pfx})")
            print(f"   {url}")
            ok = http_get(url, dest)
            if ok and os.path.exists(dest):
                size = os.path.getsize(dest)
                valid = verify_gzip(dest)
                print(f"   OK size={size}  gzip_valid={valid}")
            else:
                print(f"   FAILED")

    # ---------- 2) Single cell: supplementary files (count matrices etc.) ----------
    if not args.bulk_only:
        for g in SINGLE:
            pfx = gse_prefix(g)
            suppl = (f"https://ftp.ncbi.nlm.nih.gov/geo/series/{pfx}/{g}/suppl/")
            out_dir = os.path.join(base, "03_single_cell", g)
            os.makedirs(out_dir, exist_ok=True)
            print(f">> {g}  ({pfx})  suppl index -> {suppl}")
            html = fetch_text(suppl)
            if not html:
                print(f"   index fetch failed; download the supplementary files manually from "
                      f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={g}")
                continue
            # parse data files in the suppl directory (excluding parent-directory / directory links)
            files = re.findall(r'href="([^"]+)"', html)
            wanted = [f for f in files
                      if re.search(r"\.(gz|txt|mtx|csv|h5|tar|tsv|tsv\.gz)$", f)
                      and not f.endswith("/")]
            if not wanted:
                print(f"   no data file found in the index; download the supplementary files from the GEO web page")
                continue
            for f in wanted:
                name = f.rsplit("/", 1)[-1]
                print(f"   download {name}")
                http_get(suppl + f, os.path.join(out_dir, name))

    print("\nDone. If the suppl parsing of a series came up empty, download the supplementary files manually from "
          "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=<GSE>.")


if __name__ == "__main__":
    main()
