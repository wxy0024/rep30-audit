import pandas as pd, os
base = r"E:\01.WorkSpace\01.Breast_cancer_gene_panel\resistance_signature\results\gse_resistance\GSE26459"
df = pd.read_csv(os.path.join(base, "sample_labels.csv"), index_col=0)
col = [c for c in df.columns if "source_name" in c][0]
mask = df[col].str.contains("4OHT", case=False, na=False)
sub = pd.DataFrame({"sample": df.index[mask], "group": df.loc[mask, "group"].values})
out = os.path.join(base, "labels_4OHT_only.csv")
sub.to_csv(out, index=False)
print("WROTE", out, "n_samples=", len(sub))
print(sub.to_string(index=False))
