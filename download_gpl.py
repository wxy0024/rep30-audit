"""
download_gpl.py  -  Fetch GEO Platform (GPL) annotation tables for probe->gene mapping.

WHY: The GEO bulk series in data/02_acquired_resistance are probe-level (Affymetrix).
      To validate the REP expression signature on those cohorts we must map
      probeset IDs -> Hugo gene symbols. The annotation lives in each platform's
      GPL annot.gz file, which is NOT bundled with the series matrix.

CORRECT SOURCE: NCBI FTP  geo/platforms/<STUB>/<GPL>/annot/<GPL>.annot.gz
      where STUB = "drop the last 3 digits of the GPL id, append 'nnn'".
      e.g. GPL96   -> GPLnnn     (id<1000: stub is just 'nnn')
           GPL570  -> GPLnnn
           GPL2570 -> GPL2nnn
           GPL6244 -> GPL6nnn
      NOTE: the earlier version used the geo/download?acc=..&format=annot CGI, which
      returns an XML ERROR PAGE (not the annot). That is why the old run produced
      0-byte .annot files. This version uses the FTP path and validates gzip magic.

RUN ON YOUR LOCAL MACHINE (sandbox blocks NCBI FTP):
    python download_gpl.py
    python download_gpl.py --gpl GPL96 GPL570        # only the platforms you need

Output: data/05_gpl/<GPL>.annot.gz  (+ extracted <GPL>.annot, only if valid gzip)
Then run:  python analysis/rep_geo_validate.py
"""
import os, re, gzip, argparse, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data", "05_gpl")
os.makedirs(OUT, exist_ok=True)

# Platforms used by the downloaded GEO series (bulk + single-cell)
DEFAULT_GPLS = ["GPL96", "GPL570", "GPL97", "GPL201", "GPL8300",
                "GPL6244", "GPL6947", "GPL1352", "GPL6102"]

def stub_for(gpl):
    nums = re.sub(r"\D", "", gpl)          # digits only, e.g. "96"
    if len(nums) >= 3:
        return nums[:-3] + "nnn"           # "2570" -> "2nnn"
    return "nnn"                            # "96"  -> "nnn"

def candidate_dirs(gpl):
    nums = re.sub(r"\D", "", gpl)
    s = stub_for(gpl)
    cands = [f"GPL{s}"]
    if len(nums) < 3:
        cands.append("GPL0nnn")             # some mirrors zero-pad
    cands.append(f"GPL{nums}")              # very-low ids sometimes bare
    return cands

def is_real_gzip(path, min_bytes=100_000):
    try:
        with open(path, "rb") as f:
            head = f.read(2)
        if head != b"\x1f\x8b":
            return False
        return os.path.getsize(path) >= min_bytes
    except Exception:
        return False

def fetch(gpl):
    dest_gz = os.path.join(OUT, f"{gpl}.annot.gz")
    dest_txt = os.path.join(OUT, f"{gpl}.annot")
    # remove any stale error-page / 0-byte files
    for p in (dest_gz, dest_txt):
        if os.path.exists(p):
            os.remove(p)
    for d in candidate_dirs(gpl):
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{d}/{gpl}/annot/{gpl}.annot.gz"
        print(f"[try] {gpl} <- {url}")
        try:
            urllib.request.urlretrieve(url, dest_gz)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        if is_real_gzip(dest_gz):
            # extract
            try:
                with gzip.open(dest_gz, "rb") as f, open(dest_txt, "wb") as o:
                    o.write(f.read())
                print(f"  OK  -> {dest_gz}  ({os.path.getsize(dest_gz):,} B)  extracted {dest_txt}")
                return True
            except Exception as e:
                print(f"  downloaded but extract failed: {e}")
                continue
        else:
            print(f"  not a valid annot (error page or too small) -> try next stub")
            if os.path.exists(dest_gz):
                os.remove(dest_gz)
    print(f"  FAILED to fetch a valid {gpl}.annot.gz from any candidate path.")
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpl", nargs="*", default=DEFAULT_GPLS, help="GPL accession(s)")
    args = ap.parse_args()
    ok, fail = [], []
    for g in args.gpl:
        (ok if fetch(g) else fail).append(g)
    print("\n==== summary ====")
    print("OK  :", ok)
    print("FAIL:", fail)

if __name__ == "__main__":
    main()
