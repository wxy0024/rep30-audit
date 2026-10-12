#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEO series matrices -> genuine resistant / sensitive (or endocrine-treatment exposure) comparisons.

Workflow:
  1. Parse the GEO series matrix (expression matrix + !Sample_* sample annotations; multiple characteristics lines are concatenated)
  2. Extract labels using dataset-specific rules:
       - GSE26459: !Sample_source_name_ch1 contains Resistant / Sensitive (MCF-7 tamoxifen/4OHT resistant vs sensitive cell lines, 12 vs 12) -> a true resistance binary
       - GSE33366: !Sample_characteristics_ch1 treatment group: Tamoxifen/Letrozole (endocrine-treatment exposure) vs control (PDX drug response)
       - GSE45255: !Sample_characteristics_ch1 er status: ER- (intrinsic endocrine-resistance surrogate) vs ER+ (endocrine-sensitive surrogate)
       - Survival cohorts GSE6532/9195/17705 have no relapse/resistance labels in their annotations -> NA; clinical metadata needed separately
  3. DE(Mann-Whitney / log2FC)
  4. Score each sample with REP52 (or the trimmed panel) and test whether the two groups separate
     -> the key validation of "can the REP programme separate resistance / endocrine-exposure states"
  5. Affymetrix platform ID_REFs are probe IDs; GPL annotation (05_gpl/GPLxxx.annot.gz) is needed to map to gene symbols before scoring

Data already downloaded (local project directory):
  python analysis/gse_resistance_compare.py --gse GSE26459 --matrix data/02_acquired_resistance/GSE26459/GSE26459_series_matrix.txt.gz
  python analysis/gse_resistance_compare.py --all data/02_acquired_resistance
  python analysis/gse_resistance_compare.py --selftest

Dependencies: pandas, numpy, scipy (installed in the managed venv)
"""
import os, sys, argparse, gzip, re, json, glob
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = r"E:/01.WorkSpace/01.Breast_cancer_gene_panel/resistance_signature"
RES = os.path.join(ROOT, "results")
REP_CSV = os.path.join(RES, "rep_genes.csv")
GPL_DIR = os.path.join(ROOT, "data", "05_gpl")

# dataset-specific label rules
DATASET_RULES = {
    "GSE26459": {"field": "!Sample_source_name_ch1", "resistant": r"resistant",
                 "sensitive": r"sensitive", "note": "MCF-7 tamoxifen (4OHT) resistant vs sensitive cell lines, true binary"},
    "GSE33366": {"field": "!Sample_characteristics_ch1", "resistant": r"tamoxifen|letrozole",
                 "sensitive": r"control", "note": "PDX endocrine treatment (Tam/Letro) exposure vs control (drug response, not intrinsic resistance)"},
    "GSE45255": {"field": "!Sample_characteristics_ch1", "resistant": r"er status:\s*er-",
                 "sensitive": r"er status:\s*er\+", "note": "ER- (intrinsic tamoxifen-resistance surrogate) vs ER+ (endocrine-sensitive surrogate), both within the tamoxifen cohort"},
}
# survival cohorts default to NA (no relapse labels)
NA_RULES = ["GSE6532", "GSE9195", "GSE17705"]


def parse_series_matrix(path):
    meta = defaultdict(list)
    data = []
    sample_ids = None
    idref_seen = False
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("!") or line.startswith("#"):
                parts = line.split("\t")
                meta[parts[0]].append(parts[1:])
            else:
                parts = [p.strip().strip('"') for p in line.split("\t")]
                if not idref_seen and parts[0] == "ID_REF":
                    sample_ids = parts[1:]
                    idref_seen = True
                elif idref_seen:
                    gid = parts[0]
                    vals = parts[1:]
                    try:
                        arr = [float(x) for x in vals]
                    except ValueError:
                        continue  # skip non-numeric rows (transposed/anomalous rows)
                    data.append((gid, arr))
    if sample_ids is None:
        raise ValueError("ID_REF row not found in series matrix")
    n = len(sample_ids)
    mrows = {}
    for k, sublists in meta.items():
        if not sublists:
            continue
        norm = []
        for sl in sublists:
            if len(sl) == n:
                norm.append(sl)
            elif len(sl) == 1:
                norm.append(sl * n)
            else:
                norm.append((sl + [""] * n)[:n])
        if len(norm) == 1:
            mrows[k] = norm[0]
        else:
            mrows[k] = [" | ".join(str(norm[r][i]) for r in range(len(norm))) for i in range(n)]
    meta_df = pd.DataFrame(mrows, index=sample_ids)
    ids = [d[0] for d in data]
    arr = np.array([d[1] for d in data], dtype=float)
    expr = pd.DataFrame(arr, index=ids, columns=sample_ids)
    return expr, meta_df


def parse_gpl_annot(path):
    cols = []
    probe2gene = {}
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                m = re.match(r"#\s*([^=]+?)\s*=", line)
                if m:
                    cols.append(m.group(1).strip())
                continue
            if line.startswith("^") or line.startswith("!"):
                continue
            if not cols:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            probe = parts[0].strip()
            gi = cols.index("Gene symbol") if "Gene symbol" in cols else 2
            if gi < len(parts):
                sym = parts[gi].strip().split("///")[0].split(" ")[0].strip()
                if probe and sym and sym != "---":
                    probe2gene[probe] = sym
    return probe2gene


def extract_labels(meta_df, rule):
    field = rule.get("field", "!Sample_characteristics_ch1")
    text = ""
    if field in meta_df.columns:
        text = meta_df[field].fillna("").astype(str)
    else:
        cols = [c for c in meta_df.columns if c.startswith("!Sample_")]
        text = meta_df[cols].fillna("").astype(str).agg(" ".join, axis=1)
    text = text.str.lower()
    grp = pd.Series("NA", index=meta_df.index)
    r_res = re.compile(rule.get("resistant", "resistant"), re.I)
    r_sen = re.compile(rule.get("sensitive", "sensitive"), re.I)
    grp = grp.where(~text.str.contains(r_sen), "sensitive")
    grp = grp.where(~text.str.contains(r_res), "resistant")
    both = text.str.contains(r_res) & text.str.contains(r_sen)
    grp[both] = "resistant"
    return grp


def rep_score_probe(expr, rep_csv, probe2gene=None, genes=None):
    rep = pd.read_csv(rep_csv)
    if genes is not None:
        rep = rep[rep["gene"].isin(genes)]
    if probe2gene is None:
        sub = expr.copy()
    else:
        gene_of = pd.Series({p: probe2gene.get(p) for p in expr.index})
        sub = expr.copy()
        sub.index = gene_of.values
        sub = sub[sub.index.notna()]
        if sub.shape[0] == 0:
            return None, 0
        sub = sub.groupby(level=0).mean()
    common = [g for g in rep["gene"] if g in sub.index]
    if not common:
        return None, 0
    sub = sub.loc[common]
    z = (sub - sub.mean(axis=1).values.reshape(-1, 1)) / (sub.std(axis=1).replace(0, 1).values.reshape(-1, 1))
    coef = rep.set_index("gene").loc[common, "coef"].values
    score = (z * coef.reshape(-1, 1)).sum(axis=0)
    return score, len(common)


def analyze(gse, matrix_path, out_dir, labels_csv=None, panel=None):
    expr, meta = parse_series_matrix(matrix_path)
    rule = DATASET_RULES.get(gse, {"field": "!Sample_characteristics_ch1",
                                   "resistant": "resistant", "sensitive": "sensitive", "note": ""})
    grp = extract_labels(meta, rule)
    if labels_csv and os.path.exists(labels_csv):
        lab = pd.read_csv(labels_csv)
        lab = lab.set_index(lab.columns[0])["group"].astype(str).str.lower()
        grp.update(lab)
        grp = grp.where(grp.isin(["resistant", "sensitive"]), "NA")
    meta = meta.copy()
    meta["group"] = grp.values
    os.makedirs(out_dir, exist_ok=True)
    meta.to_csv(os.path.join(out_dir, "sample_labels.csv"))

    r = grp.value_counts().to_dict()
    print(f"[{gse}] samples={expr.shape[1]} genes(probes)={expr.shape[0]} "
          f"labels={r}  note={rule.get('note','')}")
    if r.get("resistant", 0) < 3 or r.get("sensitive", 0) < 3:
        print(f"[{gse}] too few resistant/sensitive samples -> skipping DE/scoring (provide relapse/resistance labels via --labels)")
        with open(os.path.join(out_dir, "summary.txt"), "w") as f:
            f.write(f"{gse}: automatic labels insufficient (labels={r}); manual --labels required for comparison.\n")
        return

    use = grp[grp.isin(["resistant", "sensitive"])].index
    ex = expr[use]
    gl = grp[use]
    res = ex.columns[gl.values == "resistant"]
    sen = ex.columns[gl.values == "sensitive"]

    # DE
    rows = []
    Xr = ex[res].values
    Xs = ex[sen].values
    for i, gid in enumerate(ex.index):
        a = Xr[i]
        b = Xs[i]
        try:
            u, p = mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            p = 1.0
        lfc = np.mean(a) - np.mean(b)
        rows.append((gid, np.mean(a), np.mean(b), lfc, p))
    de = pd.DataFrame(rows, columns=["gene", "mean_resistant", "mean_sensitive", "log2FC", "pval"])
    try:
        from statsmodels.stats.multitest import multipletests
        de["padj"] = multipletests(de["pval"], method="fdr_bh")[1]
    except Exception:
        de["padj"] = (de["pval"] * len(de)).clip(upper=1.0)
    de = de.sort_values("padj").reset_index(drop=True)
    de.to_csv(os.path.join(out_dir, "de_resistant_vs_sensitive.csv"), index=False)

    # REP score (GPL mapping)
    platform = None
    for c in meta.columns:
        if c == "!Sample_platform_id":
            platform = str(meta[c].iloc[0]).strip().strip('"')
    probe2gene = None
    annot_path = os.path.join(GPL_DIR, platform + ".annot.gz") if platform else None
    annot_used = "no"
    if annot_path and os.path.exists(annot_path):
        probe2gene = parse_gpl_annot(annot_path)
        annot_used = platform
    score_full, n_full = rep_score_probe(ex, REP_CSV, probe2gene)
    score_trim, n_trim = rep_score_probe(ex, REP_CSV, probe2gene, genes=(panel if panel else None))
    sc = pd.DataFrame({"sample": ex.columns, "group": gl.values,
                       "REP52_score": score_full.values if score_full is not None else np.nan,
                       "REPtrim_score": score_trim.values if score_trim is not None else np.nan})
    sc.to_csv(os.path.join(out_dir, "rep_score_by_group.csv"), index=False)
    stat = {}
    for col in ["REP52_score", "REPtrim_score"]:
        a = sc.loc[sc.group == "resistant", col].dropna()
        b = sc.loc[sc.group == "sensitive", col].dropna()
        if len(a) > 2 and len(b) > 2:
            u, p = mannwhitneyu(a, b, alternative="two-sided")
            stat[col] = {"mean_resistant": round(a.mean(), 3), "mean_sensitive": round(b.mean(), 3),
                         "MWU_p": round(p, 4), "n_res": len(a), "n_sen": len(b)}
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"{gse} resistant(n={r.get('resistant')}) vs sensitive(n={r.get('sensitive')})\n")
        f.write(f"label_semantics: {rule.get('note','')}\n")
        f.write(f"platform={platform}; GPL_annot={annot_used}; REP52 genes mapped={n_full}; trimmed mapped={n_trim}\n")
        f.write("REP score separation (resistant - sensitive):\n")
        f.write(json.dumps(stat, indent=2) + "\n")
        f.write(f"DE genes padj<0.05: {int((de['padj']<0.05).sum())}\n")
    print(f"[{gse}] done. REP sep: {stat}")


def selftest():
    import tempfile
    txt = (
        "!series_matrix_table_begin\n"
        "!sample_table_begin\n"
        "!Sample_geo_accession\tGSM1\tGSM2\tGSM3\tGSM4\tGSM5\tGSM6\n"
        "!Sample_title\tMCF7-R\tMCF7-R\tMCF7-R\tMCF7-S\tMCF7-S\tMCF7-S\n"
        "!Sample_source_name_ch1\tResistant 4OHT\tResistant estradiol\tResistant control\tSensitive 4OHT\tSensitive estradiol\tSensitive control\n"
        "ID_REF\tGSM1\tGSM2\tGSM3\tGSM4\tGSM5\tGSM6\n"
        "RPL5\t8.0\t8.2\t8.1\t6.0\t6.1\t5.9\n"
        "CFL1\t7.5\t7.6\t7.4\t5.5\t5.4\t5.6\n"
        "GSTP1\t5.0\t5.1\t5.0\t3.0\t3.1\t2.9\n"
        "!sample_table_end\n"
    )
    fd, p = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    with open(p, "w") as f:
        f.write(txt)
    out = os.path.join(RES, "gse_resistance", "_selftest")
    rep_tmp = os.path.join(tempfile.gettempdir(), "rep_selftest.csv")
    pd.DataFrame({"gene": ["RPL5", "CFL1", "GSTP1"], "coef": [0.06, 0.07, 0.05], "HR": [1.06, 1.07, 1.05]}).to_csv(rep_tmp, index=False)
    global REP_CSV
    REP_CSV = rep_tmp
    analyze("GSE26459", p, out)
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gse")
    ap.add_argument("--matrix")
    ap.add_argument("--all", help="batch directory (containing <GSE>/<GSE>*_series_matrix.txt.gz)")
    ap.add_argument("--labels", help="manual label CSV (sample_id,group)")
    ap.add_argument("--panel", help="optional: trimmed-panel gene CSV (gene column only)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    panel = pd.read_csv(args.panel)["gene"].tolist() if args.panel else None
    if args.all:
        for gdir in sorted(os.listdir(args.all)):
            gpath = os.path.join(args.all, gdir)
            if not os.path.isdir(gpath) or not gdir.startswith("GSE"):
                continue
            mats = glob.glob(os.path.join(gpath, "*_series_matrix.txt.gz"))
            if not mats:
                print(f"[skip] {gdir}: no series matrix found")
                continue
            analyze(gdir, mats[0], os.path.join(RES, "gse_resistance", gdir),
                    labels_csv=args.labels, panel=panel)
        return
    if not (args.gse and args.matrix):
        print("need --gse + --matrix, or --all <dir>, or --selftest")
        sys.exit(1)
    out = args.out or os.path.join(RES, "gse_resistance", args.gse)
    analyze(args.gse, args.matrix, out, labels_csv=args.labels, panel=panel)


if __name__ == "__main__":
    main()
