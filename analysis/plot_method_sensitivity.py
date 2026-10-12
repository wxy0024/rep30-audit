# -*- coding: utf-8 -*-
"""Draw the method-sensitivity p-value landscape from the saved matrix."""
import io
import os

import numpy as np
import pandas as pd

BASE = r"E:\01.WorkSpace\01.Breast_cancer_gene_panel\resistance_signature"
OUTD = os.path.join(BASE, "results", "method_sensitivity")

df = pd.read_csv(os.path.join(OUTD, "method_matrix_all.csv"), low_memory=False)
df["p"] = pd.to_numeric(df["p"], errors="coerce")
df["comparison"] = df["comparison"].astype(str)
df["family"] = df["family"].astype(str)
plot_df = df.dropna(subset=["p"]).copy()

summ = pd.read_csv(os.path.join(OUTD, "method_verdict_summary.csv"))
summ["comparison"] = summ["comparison"].astype(str)
order = summ.sort_values("pct_sig")["comparison"].tolist()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

cmap = {"parametric": "#c0392b", "rank": "#2c6fbb", "resampling": "#4c8c4a",
        "exact": "#8e44ad", "large-sample": "#d9a441", "model": "#16a085",
        "paired": "#e67e22", "effect": "#7f8c8d"}

ypos = {c: i for i, c in enumerate(order)}
fig, ax = plt.subplots(figsize=(10.0, max(4.0, 0.45 * len(order))))
rng = np.random.default_rng(0)
for fam in sorted(plot_df["family"].unique()):
    s = plot_df[plot_df["family"] == fam]
    ys = [ypos[c] + float(rng.normal(0, 0.07)) for c in s["comparison"]]
    ax.scatter(s["p"].values.astype(float), ys, s=28, alpha=0.85,
               color=cmap.get(fam, "#555555"), label=fam, edgecolors="none")
ax.axvline(0.05, color="black", ls="--", lw=1.1)
ax.set_xscale("log")
ax.set_xlim(1e-6, 1.3)
ax.set_xlabel("p-value (log scale); dashed line = 0.05")
ax.set_yticks(list(range(len(order))))
labels = [c if len(c) <= 58 else c[:55] + "..." for c in order]
ax.set_yticklabels(labels, fontsize=7)
ax.legend(fontsize=7, ncol=4, loc="lower left", framealpha=0.9)
ax.set_title("Same data, same question, different algorithms: p-value landscape", fontsize=10)
for i, c in enumerate(order):
    sub = plot_df[plot_df["comparison"] == c]
    nsig = int((sub["p"] < 0.05).sum())
    ax.text(1.05, i, f"{nsig}/{len(sub)} sig", fontsize=6.5, va="center", color="#444444")
plt.tight_layout()
fig.savefig(os.path.join(OUTD, "method_pvalue_landscape.png"), dpi=200)
print("saved")
