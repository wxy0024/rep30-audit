import pandas as pd, gzip, re, os
GPL = os.path.join(r"E:\01.WorkSpace\01.Breast_cancer_gene_panel\resistance_signature\data\05_gpl", "GPL96.annot.gz")

def parse_gpl(path):
    cols = []
    m = {}
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                mm = re.match(r"#\s*([^=]+?)\s*=", line)
                if mm:
                    cols.append(mm.group(1).strip())
                continue
            if line.startswith("^") or line.startswith("!"):
                continue
            if not cols:
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            gi = cols.index("Gene symbol") if "Gene symbol" in cols else 2
            if gi < len(p):
                s = p[gi].strip().split("///")[0].split(" ")[0].strip()
                if p[0].strip() and s and s != "---":
                    m[p[0].strip()] = s
    return m

gpl = parse_gpl(GPL)
de = pd.read_csv(r"E:\01.WorkSpace\01.Breast_cancer_gene_panel\resistance_signature\results\gse_resistance\GSE45255\de_resistant_vs_sensitive.csv")
de["gene_symbol"] = de["gene"].map(gpl)
out = os.path.join(r"E:\01.WorkSpace\01.Breast_cancer_gene_panel\resistance_signature\results\gse_resistance\GSE45255", "top30_de_genes.csv")
de.head(30)[["gene", "gene_symbol", "log2FC", "padj"]].to_csv(out, index=False)
print("WROTE", out)
print(de.head(30)[["gene", "gene_symbol", "log2FC", "padj"]].to_string(index=False))
print("\nDE total (padj<0.05):", int((de["padj"] < 0.05).sum()))
print("DE mapped to gene symbol:", int(de["gene_symbol"].notna().sum()))
