#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_subtype_rep30.py -- main figure for the subtype / HR+HER2- analyses

Reads results/subtype_rep30/*.csv; outputs fig_subtype_rep30.png / .pdf
  Panel A  forest plot of REP30 prognostic effects (cohorts x subtypes)
  Panel B  C-index of each predictor within HR+/HER2- (which is stronger)
  Panel C  multivariable: is REP30 absorbed by the proliferation comparator
  Panel D  does REP30 remain effective within proliferation tertiles (decisive test)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, "results", "subtype_rep30")
OUT_PNG = os.path.join(IN, "fig_subtype_rep30.png")
OUT_PDF = os.path.join(IN, "fig_subtype_rep30.pdf")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "figure.dpi": 300,
})

GREY = "#b9b9b9"
RED = "#c0392b"
BLUE = "#2471a3"
GREEN = "#1e8449"
ORANGE = "#ca6f1e"


def rd(name):
    p = os.path.join(IN, name)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def pretty_err(ax, y, hr, lo, hi, color, marker="o", ms=3.4):
    ax.plot([lo, hi], [y, y], color=color, lw=0.9, solid_capstyle="butt", zorder=2)
    ax.plot([hr], [y], marker=marker, color=color, ms=ms, zorder=3)


fig = plt.figure(figsize=(7.09, 7.35))
# ------------------------------------------------------------------ A
axA = fig.add_axes([0.315, 0.585, 0.655, 0.355])
SX = rd("subtype_cox.csv")
rowsA = []
if len(SX):
    def add(coh, sub, tag, colour):
        r = SX[(SX["cohort"] == coh) & (SX["subtype"].astype(str) == sub)]
        if len(r) and np.isfinite(r.iloc[0].get("HR", np.nan)):
            r = r.iloc[0]
            rowsA.append(dict(tag=tag, hr=r["HR"], lo=r["lo"], hi=r["hi"], p=r["p"],
                              n=int(r["n"]), colour=colour, head=False))
    rowsA.append(dict(tag="TCGA-BRCA (PAM50)", head=True))
    for s, t in [("LumA", "LumA (HR+/HER2-)"), ("LumB", "LumB (HR+/HER2-)"),
                 ("HER2-enriched", "HER2-enriched"), ("Basal", "Basal")]:
        add("TCGA", s, t, BLUE)
    rowsA.append(dict(tag="METABRIC (PAM50 / IHC)", head=True))
    for s, t in [("LumA", "LumA"), ("LumB", "LumB"), ("Her2", "HER2-enriched"),
                 ("Basal", "Basal"), ("claudin-low", "claudin-low"),
                 ("Normal", "Normal-like"), ("HR+/HER2- (IHC)", "HR+/HER2- (IHC)")]:
        add("METABRIC", s, t, RED)
    G = rd("geo_hrpos_external.csv")
    if len(G):
        rowsA.append(dict(tag="GEO · ER+ / endocrine-treated", head=True))
        for _, r in G.iterrows():
            rowsA.append(dict(tag=str(r["cohort"]), hr=r["REP30_HR"], lo=r["REP30_lo"],
                              hi=r["REP30_hi"], p=r["REP30_p"], n=int(r["n"]),
                              colour=GREEN, head=False))

ys, ylab, yhead = [], [], []
y = 0.0
for r in rowsA:
    if r.get("head"):
        ys.append(y); ylab.append(r["tag"]); yhead.append(True)
    else:
        pretty_err(axA, y, r["hr"], max(r["lo"], 0.02), r["hi"], r["colour"])
        star = "*" if np.isfinite(r["p"]) and r["p"] < 0.05 else ""
        axA.text(6.4, y, f"{r['hr']:.2f}{star}", fontsize=5.4, va="center", ha="left",
                 color=r["colour"])
        axA.text(9.6, y, f"{int(r['n'])}", fontsize=5.2, va="center", ha="right", color="#666666")
        ys.append(y); ylab.append("   " + r["tag"]); yhead.append(False)
    y -= 1.0
axA.axvline(1, color="black", lw=0.7, ls="-", zorder=1)
axA.axvspan(0.02, 1.0, color="#f4f4f4", zorder=0)
axA.set_yticks(ys); axA.set_yticklabels(ylab, fontsize=5.6)
for t, hd in zip(axA.get_yticklabels(), yhead):
    if hd:
        t.set_fontweight("bold"); t.set_color("#111111"); t.set_fontsize(5.9)
axA.tick_params(axis="y", length=0, pad=2)
axA.set_xscale("log"); axA.set_xlim(0.6, 12)
axA.set_xticks([0.6, 0.8, 1, 1.5, 2, 3, 4, 6, 9])
axA.set_xticklabels(["0.6", "0.8", "1", "1.5", "2", "3", "4", "6", "9"], fontsize=5.4)
axA.set_ylim(min(ys) - 0.7, max(ys) + 0.6)
axA.set_xlabel("REP30 hazard ratio per 1 SD   (OS / RFS / DMFS)", labelpad=3, fontsize=6.2)
axA.text(9.6, max(ys) + 0.35, "n", fontsize=5.2, ha="right", color="#666666")
axA.set_title("A   REP30 is prognostic in HR+/HER2−, but the effect is NOT subtype-specific",
              loc="left", pad=4, fontsize=7.0)

# ------------------------------------------------------------------ B
axB = fig.add_axes([0.315, 0.315, 0.245, 0.155])
CI = rd("hrposher2neg_cindex.csv")
if len(CI):
    CI = CI.sort_values("cindex")
    yy = np.arange(len(CI))
    cols = [RED if "REP30" in p else (GREY if ("PROLIF" in p or "MKI67" in p or "PAM50" in p)
                                      else BLUE) for p in CI["predictor"]]
    axB.barh(yy, CI["cindex"], color=cols, height=0.62)
    for i, (c, p) in enumerate(zip(CI["cindex"], CI["predictor"])):
        axB.text(c + 0.006, i, f"{c:.3f}", va="center", fontsize=5.2, color="#333333")
    axB.set_yticks(yy)
    axB.set_yticklabels([p.replace(" (clinical)", "").replace(" (12-gene mean)", "")
                         for p in CI["predictor"]], fontsize=5.4)
    axB.set_xlim(0.5, 0.75)
    axB.set_xlabel("C-index (OS)", labelpad=2, fontsize=6.0)
    axB.set_title("B   beats clinical NPI and simple\n      proliferation scores", loc="left",
                  pad=3, fontsize=6.6)

# ------------------------------------------------------------------ C
axC = fig.add_axes([0.710, 0.315, 0.260, 0.155])
MV = rd("hrposher2neg_multivariable.csv")
rowsC = []
if len(MV):
    m = MV[(MV["stratum"] == "METABRIC·HR+/HER2-")]
    for _, r in m.iterrows():
        partner = str(r["partner"])
        rowsC.append((f"REP30\n(vs {partner})", r["REP30_HR"], r["REP30_lo"], r["REP30_hi"],
                      r["REP30_p"], RED))
        if np.isfinite(r.get("partner_HR", np.nan)):
            rowsC.append((f"{partner}", r["partner_HR"], np.nan, np.nan, r["partner_p"], GREY))
yy = np.arange(len(rowsC))[::-1]
for i, (lab, hr, lo, hi, p, col) in zip(yy, rowsC):
    if np.isfinite(lo) and np.isfinite(hi):
        axC.plot([lo, hi], [i, i], color=col, lw=0.9)
    axC.plot([hr], [i], "o", color=col, ms=3.2)
    axC.text(3.05, i, f"{hr:.2f}" + ("*" if np.isfinite(p) and p < 0.05 else " ns"),
             fontsize=5.0, va="center", ha="left", color=col)
axC.axvline(1, color="black", lw=0.7)
axC.set_yticks(yy); axC.set_yticklabels([r[0] for r in rowsC], fontsize=5.0)
axC.tick_params(axis="y", length=0)
axC.set_xlim(0.85, 3.6)
axC.set_xlabel("multivariable HR (per 1 SD)", labelpad=2, fontsize=6.0)
axC.set_title("C   proliferation is absorbed;\n      REP30 is not", loc="left", pad=3, fontsize=6.6)

# ------------------------------------------------------------------ D
axD = fig.add_axes([0.315, 0.065, 0.655, 0.165])
T = [("METABRIC·HR+/HER2-", rd("hrposher2neg_multivariable.csv"), "hrposher2neg_prolif_tertiles.csv",
      RED),
     ("TCGA·HR+/HER2-", None, "hrposher2neg_prolif_tertiles.csv", BLUE)]
TT = rd("hrposher2neg_prolif_tertiles.csv")
GG = rd("geo_hrpos_prolif_tertiles.csv")
blocks = []
if len(TT):
    blocks.append(("METABRIC · HR+/HER2−", TT[TT["cohort"] == "METABRIC"], RED))
    blocks.append(("TCGA · HR+/HER2−", TT[TT["cohort"] == "TCGA"], BLUE))
if len(GG):
    for gse in GG["cohort"].unique():
        blocks.append((f"{gse} (ER+)", GG[GG["cohort"] == gse], GREEN))
xpos, xlabs = [], []
x = 0.0
for name, df, col in blocks:
    for which, short in [("low", "low"), ("mid", "mid"), ("high", "high")]:
        r = df[df["stratum"].str.contains(f"{which} tertile")]
        if not len(r):
            continue
        r = r.iloc[0]
        hr, lo, hi, p = r["HR"], r["lo"], r["hi"], r["p"]
        if not np.isfinite(hr):
            continue
        axD.plot([x, x], [lo, hi], color=col, lw=0.9)
        axD.plot([x], [hr], "o", color=col, ms=3.4)
        if np.isfinite(p) and p < 0.05:
            axD.text(x, hi * 1.06, "*", ha="center", fontsize=6.0, color=col)
        xpos.append(x); xlabs.append(short)
        x += 1.0
    x += 0.6
axD.axvline(1, color="black", lw=0.7)
axD.axhline(1, color="black", lw=0.5, ls=":", zorder=0)
axD.set_yscale("log")
axD.set_yticks([0.6, 0.8, 1, 1.5, 2, 3])
axD.set_yticklabels(["0.6", "0.8", "1", "1.5", "2", "3"], fontsize=5.4)
axD.set_xticks(xpos); axD.set_xticklabels(xlabs, fontsize=5.2)
axD.set_ylim(0.5, 4.6)
axD.set_xlabel("REP30 HR within each proliferation tertile   (* p<0.05)", labelpad=3, fontsize=6.2)
axD.set_title("D   in the LOW-proliferation tertile REP30 still separates survival",
              loc="left", pad=4, fontsize=7.0)
hand = [Patch(color=RED, label="METABRIC"), Patch(color=BLUE, label="TCGA"),
        Patch(color=GREEN, label="GEO (ER+)")]
axD.legend(handles=hand, fontsize=5.0, loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=1.4)

fig.suptitle("REP30 in HR+/HER2− breast cancer: a prognostic program that is not a resistance marker",
             fontsize=7.6, x=0.012, y=0.995, ha="left")
fig.savefig(OUT_PNG, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
print("wrote", OUT_PNG)
print("wrote", OUT_PDF)
