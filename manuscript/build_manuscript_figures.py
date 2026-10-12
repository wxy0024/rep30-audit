# -*- coding: utf-8 -*-
"""
Build every manuscript figure as an AI-editable vector graphic.

Font policy (critical for editing in Adobe Illustrator):
    pdf.fonttype = 42    TrueType embedded as real, selectable text (not Type-3 outlines)
    ps.fonttype  = 42    the same for EPS
    ps.useafm    = False
    svg.fonttype = none
No alpha channel is used, so nothing is silently rasterised in EPS.

Run:
    C:/Users/mao02/.workbuddy/binaries/python/envs/default/Scripts/python.exe \
        manuscript/build_manuscript_figures.py
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
R = BASE / "results"
ARCH = BASE / "_archive" / "2026-09-21_deadends" / "results"
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "ps.useafm": False,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.titlesize": 7.4,
        "axes.labelsize": 6.9,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#2B2B2B",
        "axes.labelcolor": "#1A1A1A",
        "text.color": "#1A1A1A",
        "xtick.color": "#2B2B2B",
        "ytick.color": "#2B2B2B",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "lines.linewidth": 0.8,
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.unicode_minus": False,
    }
)

MM = 1.0 / 25.4
W = 180 * MM

C_POS, C_NEG, C_NS = "#C0392B", "#2471A3", "#A9AFB5"
C_GREY, C_BAND, C_INK, C_RULE = "#6E6E6E", "#EFEFEF", "#1A1A1A", "#8C8C8C"
FAMILY_COLOR = {
    "exact": "#B03A2E",
    "large-sample": "#E0A030",
    "parametric": "#C97B4A",
    "resampling": "#2E86AB",
    "model": "#7D5BA6",
    "rank": "#5B8C5A",
    "paired": "#4A7C6F",
    "nonparametric": "#5B8C5A",
    "other": "#888888",
}
CLASS_COLOR = {
    "molecular-resistance-genotype": "#B03A2E",
    "subtype-surrogate": "#5B8C5A",
    "tumour-state": "#7D5BA6",
    "treatment-exposure": "#E0A030",
    "confound": "#2E86AB",
    "other": "#BEBEBE",
}
CLASS_LABEL = {
    "molecular-resistance-genotype": "resistance genotype",
    "subtype-surrogate": "subtype surrogate",
    "tumour-state": "tumour state",
    "treatment-exposure": "treatment exposure",
    "confound": "confound control",
    "other": "other",
}
GROUP_ORDER = [
    "molecular-resistance-genotype",
    "subtype-surrogate",
    "tumour-state",
    "treatment-exposure",
    "confound",
]
GROUP_TITLE = {
    "molecular-resistance-genotype": "Resistance genotype",
    "subtype-surrogate": "Subtype surrogate",
    "tumour-state": "Tumour state",
    "treatment-exposure": "Treatment exposure",
    "confound": "Confound control (positive control)",
}
SHORT = [
    (r"4-OHT", "4-OHT acquired resistance"),
    (r"ESR1[_ ]mutation|ESR1 mutation", "ESR1 mutation"),
    (r"RB1[_ ]any[_ ]loss|RB1 loss", "RB1 loss"),
    (r"ER- \(intrinsic", "ER\u2212 vs ER+ (tamoxifen surrogate)"),
    (r"Metastatic vs Primary", "metastatic vs primary"),
    (r"primary-vs-met", "metastatic vs primary"),
    (r"Pre vs Meta", "Pre vs Meta (malignant cells)"),
    (r"days on metastatic therapy", "days on metastatic therapy"),
    (r"metastatic lines before biopsy", "prior metastatic lines"),
    (r"change in cumulative therapy days", "\u0394 cumulative therapy days"),
    (r"prior therapy", "prior therapy (Y vs N)"),
    (r"proliferation \(RNA module\)", "proliferation module"),
    (r"percent tumour cells", "tumour content"),
    (r"REP30 vs tc", "tumour content"),
]
COHORT_SHORT = {
    "GSE26459": "GSE26459", "GSE45255": "GSE45255", "TCGA-BRCA": "TCGA",
    "METABRIC": "METABRIC", "GSE264205": "GSE264205", "AURORA US": "AURORA",
    "MET500": "MET500", "MBCproject": "MBCproject",
}
ACCESS_SHORT = {
    "AURORA US (cBioPortal)": "cBioPortal",
    "AURORA US (GSE209998)": "GEO",
    "MBCproject (Count Me In)": "",
    "MET500": "",
    "TCGA": "",
    "METABRIC": "",
}


def short_label(e: str) -> str:
    for pat, lab in SHORT:
        if re.search(pat, e):
            return lab
    return e[:36]


def save(fig, stem: str) -> None:
    for ext in ("pdf", "eps", "png"):
        p = FIG / f"{stem}.{ext}"
        for attempt in range(4):
            try:
                fig.savefig(p, format=ext, dpi=600 if ext == "png" else None)
                break
            except OSError as exc:            # transient Windows file lock
                if attempt == 3:
                    raise
                print(f"  retry {ext} ({exc})")
                time.sleep(0.6)
        print("  wrote", p.relative_to(BASE))
    plt.close(fig)


def panel(ax, letter, dx=-0.14, dy=1.06, fs=8.5):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=fs, fontweight="bold",
            va="top", ha="left", color=C_INK)


def top_strip(ax, lw=0.6):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "n.a."
    if p < 1e-4:
        s = f"{p:.1e}"
        return s.replace("e-0", "e\u2212").replace("e+0", "e")
    if p < 0.001:
        return f"{p:.1e}".replace("e-0", "e\u2212")
    return f"{p:.3f}".rstrip("0").rstrip(".")


def stars(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
FULL = pd.read_csv(R / "evidence_table" / "evidence_table_full.csv")
FULL["effect"] = pd.to_numeric(FULL["effect"], errors="coerce")
FULL["p"] = pd.to_numeric(FULL["p"], errors="coerce")
INC = FULL[FULL["include"] == "yes"].copy()
INC["cls"] = INC["endpoint_class"].where(INC["endpoint_class"].isin(GROUP_ORDER), "other")


def parse_effect(s):
    m = re.search(r"([a-z_]+)=([-+0-9.eE]+)", str(s))
    return (float(m.group(2)), m.group(1)) if m else (np.nan, "n.a.")


HL = pd.read_csv(R / "evidence_table" / "submission" / "Table_S2_headline_evidence.csv")
HL[["effect", "metric"]] = HL["effect_str"].apply(lambda s: pd.Series(parse_effect(s)))
HL["p"] = pd.to_numeric(HL["p_str"], errors="coerce")
HL["__o"] = HL["endpoint_class"].apply(lambda x: GROUP_ORDER.index(x) if x in GROUP_ORDER else 99)
HL = HL.sort_values("__o", kind="stable").reset_index(drop=True)

CONF_MEAN = float(INC.loc[INC["cls"] == "confound", "effect"].abs().mean())
GENO_MEAN = float(INC.loc[INC["cls"] == "molecular-resistance-genotype", "effect"].abs().mean())
print(f"[check] confound mean|effect| ... {CONF_MEAN:.3f}  (text says 0.189)")
print(f"[check] resistance-genotype mean|effect| ... {GENO_MEAN:.3f}")


# ===========================================================================
def fig1():
    fig = plt.figure(figsize=(W, 236 * MM))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.92, 2.25, 1.15], hspace=0.55, wspace=0.26)
    fig.subplots_adjust(left=0.312, right=0.995, top=0.985, bottom=0.045)
    order = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "S1"]
    lab = ["GSE26459\nin vitro", "GSE45255\nprimary", "TCGA\nprimary", "METABRIC\nprimary",
           "GSE264205\nscRNA", "AURORA US\nmetastatic", "MET500\nmetastatic",
           "MBCproject\nmetastatic", "Harmony\n(derived)"]

    # ---- A
    axA = fig.add_subplot(gs[0, :])
    piv = (INC.pivot_table(index="cohort_id", columns="cls", values="endpoint",
                           aggfunc="count").reindex(order).fillna(0))
    cols = [c for c in GROUP_ORDER if c in piv.columns] + \
           [c for c in piv.columns if c not in GROUP_ORDER]
    bottom = np.zeros(len(order))
    for c in cols:
        v = piv[c].values
        axA.bar(range(len(order)), v, bottom=bottom, width=0.66,
                color=CLASS_COLOR.get(c, "#BEBEBE"), edgecolor="white", linewidth=0.4,
                label=CLASS_LABEL.get(c, c))
        bottom += v
    for i, t in enumerate(bottom):
        axA.text(i, t + 3, f"{int(t)}", ha="center", va="bottom", fontsize=6.0)
    axA.set_xticks(range(len(order)))
    axA.set_xticklabels(lab, fontsize=5.7)
    axA.set_ylabel("comparisons")
    axA.set_ylim(0, bottom.max() * 1.30)
    axA.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.34), handlelength=0.85,
               columnspacing=1.0, handletextpad=0.35)
    axA.set_title("461 non-redundant comparisons across eight independent datasets\n"
                  "(7 patient cohorts + 1 in-vitro model, plus one derived single-cell synthesis)",
                  fontsize=7.1, pad=16)
    top_strip(axA)
    panel(axA, "A", dx=-0.055, dy=1.34)

    # ---- B
    axB = fig.add_subplot(gs[1, :])
    ypos, ylab, ymid, yname, ybound = [], [], [], [], []
    y = 0.0
    for g in GROUP_ORDER:
        sub = HL[HL["endpoint_class"] == g]
        if sub.empty:
            continue
        y0 = y
        for _, r in sub.iterrows():
            ypos.append(y)
            c = COHORT_SHORT.get(r["cohort"], r["cohort"])
            a = ACCESS_SHORT.get(r["access_route"], "")
            ylab.append(f"{short_label(r['endpoint'])}   \u2013 {c}" + (f" ({a})" if a else ""))
            y += 1.0
        ymid.append((y0 + y - 1) / 2)
        yname.append(GROUP_TITLE[g])
        ybound.append(y - 0.5)
        y += 1.5
    y_max = y - 1.5
    axB.axvspan(-0.15, 0.15, color=C_BAND, zorder=0)
    axB.axvline(0, color=C_RULE, lw=0.6, zorder=1)
    for yp, (_, r) in zip(ypos, HL.iterrows()):
        e = r["effect"]
        sig = (r["p"] < 0.05) and (abs(e) >= 0.15)
        col = (C_POS if e > 0 else C_NEG) if sig else C_NS
        mk = "D" if r["metric"] == "spearman_r" else "o"
        axB.plot([e], [yp], marker=mk, ms=3.1, mfc=col, mec="#333333", mew=0.4, zorder=3)
        axB.text(0.66, yp, fmt_p(r["p"]) + stars(r["p"]), ha="left", va="center",
                 fontsize=5.6, color=C_INK)
        axB.text(0.92, yp, str(r["n_label/control"]), ha="left", va="center",
                 fontsize=5.6, color="#555555")
    axB.text(0.66, -1.75, "p", fontsize=6.0, fontweight="bold")
    axB.text(0.92, -1.75, "n (label / control)", fontsize=6.0, fontweight="bold")
    for b in ybound[:-1]:
        axB.axhline(b + 0.75, color="#DEDEDE", lw=0.5, zorder=0)
    axB.set_yticks(ypos)
    axB.set_yticklabels(ylab, fontsize=5.9)
    axB.set_yticks(ymid, minor=True)
    axB.set_yticklabels(yname, minor=True, fontsize=6.5, fontweight="bold")
    axB.tick_params(axis="y", which="both", length=0, pad=2)
    axB.set_ylim(y_max + 0.9, -2.6)
    axB.set_xlim(-0.62, 1.06)
    axB.set_xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])
    axB.set_xlabel("effect size   (Cliff's \u03b4, circle; Spearman r, diamond)")
    axB.spines[["left", "top", "right"]].set_visible(False)
    axB.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc=C_POS, mec="#333333", mew=0.4, ms=3.4,
               label="significant, effect > 0"),
        Line2D([], [], marker="o", ls="", mfc=C_NEG, mec="#333333", mew=0.4, ms=3.4,
               label="significant, effect < 0"),
        Line2D([], [], marker="o", ls="", mfc=C_NS, mec="#333333", mew=0.4, ms=3.4,
               label="n.s. or |effect| < 0.15"),
        Patch(facecolor=C_BAND, edgecolor="none", label="negligible-effect band (|effect| < 0.15)"),
    ], ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.002), handletextpad=0.35,
        columnspacing=1.2)
    panel(axB, "B", dx=-0.055, dy=1.10)

    # ---- C  (over the 23 drawn, non-redundant comparisons of panel B -- the
    #          complete 461-comparison landscape is Fig. S3)
    axC = fig.add_subplot(gs[2, 0])
    rng = np.random.default_rng(7)
    cls_show = [c for c in GROUP_ORDER if c in set(HL["endpoint_class"])]
    axC.axhline(0.15, color=C_RULE, lw=0.7, ls="--")
    for i, c in enumerate(cls_show):
        v = HL.loc[HL["endpoint_class"] == c, "effect"].abs().dropna().values
        axC.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=9,
                    color=CLASS_COLOR.get(c, "#BEBEBE"), edgecolor="#333333",
                    linewidth=0.3, zorder=3)
        m = float(v.mean())
        axC.plot([i - 0.30, i + 0.30], [m, m], color="#111111", lw=1.4, zorder=4)
        axC.text(i + 0.34, m, f"{m:.2f}", fontsize=5.8, va="center", ha="left")
        axC.text(i, 0.60, f"n={len(v)}", ha="center", fontsize=5.4, color="#666666")
    axC.set_xticks(range(len(cls_show)))
    axC.set_xticklabels([CLASS_LABEL.get(c, c).replace(" ", "\n") for c in cls_show], fontsize=5.6)
    axC.set_ylabel("|effect size|")
    axC.set_ylim(-0.02, 0.66)
    axC.set_xlim(-0.6, len(cls_show) - 0.30)
    axC.text(len(cls_show) - 0.55, 0.158, "0.15 (negligible)", fontsize=5.4, color=C_RULE,
             va="bottom", ha="right")
    axC.set_title("Among the drawn comparisons, the confound controls sit furthest "
                  "from zero", fontsize=6.7, pad=4)
    top_strip(axC)
    panel(axC, "C", dx=-0.20, dy=1.16)

    # ---- D
    axD = fig.add_subplot(gs[2, 1])
    s4 = pd.read_csv(R / "evidence_table" / "submission" / "Table_S4_prognostic_layer.csv")
    o = s4[s4["endpoint_class"] == "prognostic-OS"].copy()
    o["k"] = o["endpoint"].str.extract(r"\((full52|trim30)\)")[0]
    ci = o["detail"].str.extract(r"HR 95% CI ([\d.]+)-([\d.]+)").astype(float)
    o["lo"], o["hi"] = ci[0], ci[1]
    o = o.sort_values(["cohort", "k"]).reset_index(drop=True)
    yy = np.arange(len(o))[::-1]
    for i, (_, r) in enumerate(o.iterrows()):
        col = C_POS if r["cohort"] == "METABRIC" else "#7D5BA6"
        axD.plot([r["lo"], r["hi"]], [yy[i], yy[i]], color="#666666", lw=0.8, zorder=2)
        axD.plot([r["effect"]], [yy[i]], marker="s", ms=3.5, mfc=col, mec="#333333",
                 mew=0.4, zorder=3)
        axD.text(4.15, yy[i], f"{r['effect']:.2f}", fontsize=5.8, va="center")
        axD.text(5.05, yy[i], stars(r["p"]), fontsize=5.8, va="center")
    axD.set_yticks(yy)
    axD.set_yticklabels([f"{r['cohort']}  {r['k']}" for _, r in o.iterrows()], fontsize=5.9)
    axD.set_xscale("log")
    axD.set_xlim(1.0, 6.6)
    axD.set_xticks([1, 1.5, 2, 3, 4])
    axD.set_xticklabels(["1", "1.5", "2", "3", "4"])
    axD.set_xlabel("HR for overall survival (REP30-high vs low)")
    axD.set_ylim(-0.7, len(o) - 0.3)
    axD.set_title("Where it does work: prognosis", fontsize=6.9, pad=4)
    top_strip(axD)
    panel(axD, "D", dx=-0.44, dy=1.16)
    save(fig, "Fig1_audit")


# ===========================================================================
def fig2():
    S = R / "subtype_rep30"
    cox = pd.read_csv(S / "subtype_cox.csv")
    multi = pd.read_csv(S / "hrposher2neg_multivariable.csv")
    tert = pd.read_csv(S / "hrposher2neg_prolif_tertiles.csv")
    cidx = pd.read_csv(S / "hrposher2neg_cindex.csv")
    geo = pd.read_csv(S / "geo_hrpos_external.csv")
    corr = pd.read_csv(S / "score_correlations.csv")
    ril = pd.read_csv(S / "resistance_labels_hrposHER2neg.csv")

    fig = plt.figure(figsize=(W, 122 * MM))
    gs = GridSpec(2, 4, figure=fig, hspace=0.72, wspace=0.52, width_ratios=[1.55, 1, 1, 1])
    fig.subplots_adjust(left=0.175, right=0.995, top=0.945, bottom=0.095)

    # ---- A subtype forest
    axA = fig.add_subplot(gs[:, 0])
    d = cox.copy()
    d["name"] = d["cohort"] + "  " + d["subtype"]
    d = d.iloc[::-1].reset_index(drop=True)
    axA.axvline(1, color=C_RULE, lw=0.7)
    for i, (_, r) in enumerate(d.iterrows()):
        col = C_POS if r["lo"] > 1 else "#8E8E8E"
        axA.plot([r["lo"], r["hi"]], [i, i], color="#666666", lw=0.8)
        axA.plot([r["HR"]], [i], marker="o", ms=3.3, mfc=col, mec="#333333", mew=0.4)
        axA.text(9.9, i, f"{r['n']}/{r['events']}", fontsize=5.4, va="center", ha="left",
                 color="#555555")
    axA.text(9.9, len(d) - 0.25, "n / events", fontsize=5.6, fontweight="bold")
    axA.set_yticks(range(len(d)))
    axA.set_yticklabels(d["name"], fontsize=5.3)
    axA.set_xscale("log")
    axA.set_xlim(0.8, 15)
    axA.set_xticks([1, 1.5, 2, 3, 5])
    axA.set_xticklabels(["1", "1.5", "2", "3", "5"])
    axA.set_ylim(-0.7, len(d) + 0.4)
    axA.set_xlabel("HR per 1 SD of score")
    axA.set_title("Subtype-stratified prognosis\n(interaction p = 0.11\u20130.45: no subtype specificity)",
                  fontsize=6.8, pad=4)
    top_strip(axA)
    panel(axA, "A", dx=-0.56, dy=1.05)

    # ---- B correlations
    axB = fig.add_subplot(gs[0, 1])
    cc = corr[(corr["stratum"].str.startswith("METABRIC\u00b7HR")) & (corr["a"] == "REP30")]
    names = {"PROLIF": "proliferation\n(12-gene)", "MKI67": "MKI67",
             "ER_act": "ER target\nactivity (20-gene)"}
    yy = np.arange(len(cc))[::-1]
    for i, (_, r) in enumerate(cc.iterrows()):
        v = float(r["spearman"])
        axB.barh(yy[i], v, height=0.55, color=C_POS if v > 0 else C_NEG,
                 edgecolor="white", linewidth=0.4)
        axB.text(v + (0.015 if v >= 0 else -0.015), yy[i],
                 f"{v:+.3f}{stars(r['p'])}", ha="left" if v >= 0 else "right",
                 va="center", fontsize=5.6)
    axB.axvline(0, color=C_RULE, lw=0.7)
    axB.set_yticks(yy)
    axB.set_yticklabels([names.get(str(x), str(x)) for x in cc["b"]], fontsize=5.6)
    axB.set_xlim(-0.20, 0.58)
    axB.set_xticks([-0.2, 0, 0.2, 0.4])
    axB.set_xlabel("Spearman r with score")
    axB.set_title("Proliferation, not ER", fontsize=6.8, pad=3)
    top_strip(axB)
    panel(axB, "B", dx=-0.46, dy=1.20)

    # ---- C multivariable
    axC = fig.add_subplot(gs[0, 2])
    mm = multi[multi["stratum"] == "METABRIC\u00b7HR+/HER2-"].reset_index(drop=True)
    rows, cols, marks = [], [], []
    for _, r in mm.iterrows():
        rows.append((r["model"], "score", r["REP30_HR"], r["REP30_lo"], r["REP30_hi"], r["REP30_p"]))
        if pd.notna(r["partner_HR"]):
            rows.append((r["model"], r["partner"], r["partner_HR"], np.nan, np.nan, r["partner_p"]))
    rows.reverse()
    y = 0
    axC.axvline(1, color=C_RULE, lw=0.7)
    yt, yl = [], []
    for i, (model, who, hr, lo, hi, p) in enumerate(rows):
        col = C_POS if who == "score" else "#2E86AB"
        if np.isfinite(lo):
            axC.plot([lo, hi], [i, i], color="#777777", lw=0.7)
        axC.plot([hr], [i], marker="o" if who == "score" else "s", ms=3.1, mfc=col,
                 mec="#333333", mew=0.4)
        axC.text(2.32, i, ("ns" if p > 0.05 else stars(p)) if np.isfinite(p) else "",
                 fontsize=5.4, va="center")
        yt.append(i)
        yl.append(f"{who:>7s}" if who != "score" else "score")
    axC.set_yticks(yt)
    axC.set_yticklabels(yl, fontsize=5.4)
    axC.set_xlim(0.85, 2.6)
    axC.set_xticks([1, 1.5, 2])
    axC.set_xlabel("HR, multivariable")
    axC.set_title("Proliferation is absorbed", fontsize=6.8, pad=3)
    top_strip(axC)
    panel(axC, "C", dx=-0.42, dy=1.20)

    # ---- D tertiles
    axD = fig.add_subplot(gs[0, 3])
    t = tert[tert["cohort"] == "METABRIC"].reset_index(drop=True)
    x = np.arange(len(t))
    axD.axhline(1, color=C_RULE, lw=0.7)
    for i, (_, r) in enumerate(t.iterrows()):
        axD.plot([i, i], [r["lo"], r["hi"]], color="#666666", lw=0.9)
        axD.plot([i], [r["HR"]], marker="o", ms=3.4, mfc=C_POS, mec="#333333", mew=0.4)
        axD.text(i, r["hi"] + 0.045, stars(r["p"]), ha="center", fontsize=6.0)
    axD.set_xticks(x)
    axD.set_xticklabels(["low", "mid", "high"], fontsize=5.8)
    axD.set_ylim(1.35, 2.10)
    axD.set_yticks([1.5, 1.75, 2.0])
    axD.set_xlabel("proliferation tertile")
    axD.set_ylabel("HR per 1 SD")
    axD.set_title("Holds inside low\nproliferation", fontsize=6.8, pad=3)
    top_strip(axD)
    panel(axD, "D", dx=-0.42, dy=1.20)

    # ---- E c-index
    axE = fig.add_subplot(gs[1, 1])
    c = cidx.sort_values("cindex").reset_index(drop=True)
    yy = np.arange(len(c))
    for i, (_, r) in enumerate(c.iterrows()):
        col = C_POS if r["predictor"] == "REP30" else "#9AA0A6"
        axE.barh(i, r["cindex"], height=0.6, color=col, edgecolor="white", linewidth=0.4)
        axE.text(r["cindex"] + 0.006, i, f"{r['cindex']:.3f}", fontsize=5.5, va="center")
    axE.set_yticks(yy)
    axE.set_yticklabels([p.replace(" (12-gene mean)", "").replace(" alone", "")
                         .replace("PAM50 LumA vs LumB", "PAM50") for p in c["predictor"]],
                        fontsize=5.4)
    axE.set_xlim(0.44, 0.72)
    axE.set_xticks([0.5, 0.6, 0.7])
    axE.set_xlabel("C-index")
    axE.set_title("Ranked by discrimination", fontsize=6.8, pad=3)
    top_strip(axE)
    panel(axE, "E", dx=-0.46, dy=1.20)

    # ---- F external
    axF = fig.add_subplot(gs[1, 2])
    g = geo.sort_values("REP30_HR").reset_index(drop=True)
    axF.axvline(1, color=C_RULE, lw=0.7)
    for i, (_, r) in enumerate(g.iterrows()):
        rep = r["REP30_p"] < 0.05
        col = C_POS if rep else "#9AA0A6"
        axF.plot([r["REP30_lo"], r["REP30_hi"]], [i, i], color="#666666", lw=0.8)
        axF.plot([r["REP30_HR"]], [i], marker="o", ms=3.2, mfc=col, mec="#333333", mew=0.4)
        axF.text(3.05, i, f"{r['events']} ev", fontsize=5.2, va="center", color="#555555")
    axF.set_yticks(range(len(g)))
    axF.set_yticklabels(g["cohort"], fontsize=5.4)
    axF.set_xscale("log")
    axF.set_xlim(0.75, 4.4)
    axF.set_xticks([0.8, 1, 1.5, 2, 3])
    axF.set_xticklabels(["0.8", "1", "1.5", "2", "3"])
    axF.set_ylim(-0.7, len(g) - 0.3)
    axF.set_xlabel("HR (external GEO)")
    axF.set_title("2 of 4 external\ncohorts replicate", fontsize=6.8, pad=3)
    top_strip(axF)
    panel(axF, "F", dx=-0.40, dy=1.20)

    # ---- G resistance label inside HR+/HER2-
    axG = fig.add_subplot(gs[1, 3])
    r0 = ril.iloc[0]
    axG.bar([0], [r0["delta"]], width=0.45, color=C_POS, edgecolor="white", linewidth=0.4)
    axG.axhline(0, color=C_RULE, lw=0.7)
    axG.text(0, r0["delta"] + 0.006, f"{r0['delta']:+.3f}", ha="center", fontsize=5.8)
    axG.text(0, r0["delta"] + 0.024, f"p={fmt_p(r0['p'])}", ha="center", fontsize=5.5)
    axG.set_xticks([0])
    axG.set_xticklabels(["endocrine\nexposed"], fontsize=5.5)
    axG.set_ylim(0, 0.135)
    axG.set_ylabel("\u0394 score (exposed \u2212 naive)")
    axG.set_title("Still not resistance:\nsign reversed", fontsize=6.8, pad=3)
    top_strip(axG)
    panel(axG, "G", dx=-0.42, dy=1.20)
    save(fig, "Fig3_repositioning")


# ===========================================================================
def fig3():
    fig = plt.figure(figsize=(W, 168 * MM))
    gs = GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.34)
    fig.subplots_adjust(left=0.205, right=0.995, top=0.955, bottom=0.075)

    # ---- A redundancy audit
    axA = fig.add_subplot(gs[0, 0])
    au = json.loads((R / "evidence_table" / "submission" / "audit_overlap.json").read_text())
    ov = au["overlaps"]
    items = [
        ("AURORA GEO vs cBioPortal\n(patients)", ov["AURORA_geo_vs_cbioportal_patients"], "#B03A2E"),
        ("wagle_2017 vs MBCproject-2022\n(patients)", ov["wagle2017_vs_MBCproject2022_patients"], "#7D5BA6"),
        ("wagle_2017 vs MBCproject-2022\n(RNA samples)", ov["wagle2017_vs_MBCproject2022_samples"], "#2E86AB"),
    ]
    yy = np.arange(len(items))[::-1]
    for i, (lab, o, col) in enumerate(items):
        axA.barh(yy[i], o["jaccard"], height=0.48, color=col, edgecolor="white", linewidth=0.4)
        axA.text(o["jaccard"] + 0.015, yy[i] + 0.13, f"Jaccard {o['jaccard']:.3f}",
                 fontsize=5.7, va="center")
        axA.text(o["jaccard"] + 0.015, yy[i] - 0.22,
                 f"{o['n_shared']}/{o['n_a']} of set A = {o['pct_of_a']:.1f}%",
                 fontsize=5.5, va="center", color="#555555")
    axA.axvline(1.0, color=C_RULE, lw=0.7, ls=":")
    axA.set_yticks(yy)
    axA.set_yticklabels([x[0] for x in items], fontsize=5.6)
    axA.set_xlim(0, 1.42)
    axA.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axA.set_xlabel("patient / sample overlap (Jaccard)")
    axA.set_ylim(-0.6, len(items) - 0.4)
    axA.set_title("Failure mode 1 \u2014 two access routes, one cohort", fontsize=6.9, pad=4)
    top_strip(axA)
    panel(axA, "A", dx=-0.42, dy=1.15)

    # ---- B gene misalignment
    axB = fig.add_subplot(gs[0, 1])
    comp = [
        ("CAF\n(Pre vs Post1)", 0.752, 0.027, 9.5e-46, 0.62),
        ("Monocyte\n(Pre vs post)", 0.927, 0.141, 8.9e-263, 1.5e-7),
        ("Malignant\n(Pre vs Meta)", 0.527, 0.237, 3.6e-65, 2.1e-14),
    ]
    x = np.arange(len(comp))
    axB.bar(x - 0.19, [c[1] for c in comp], width=0.36, color="#C9C9C9",
            edgecolor="#555555", linewidth=0.5, label="before gene alignment")
    axB.bar(x + 0.19, [c[2] for c in comp], width=0.36, color=C_NEG,
            edgecolor="white", linewidth=0.4, label="after gene alignment")
    for i, c in enumerate(comp):
        fold = c[1] / c[2]
        axB.text(i, c[1] + 0.035, f"{c[1]:.3f}", ha="center", fontsize=5.6)
        axB.text(i + 0.19, c[2] + 0.035, f"{c[2]:.3f}", ha="center", fontsize=5.6)
        axB.text(i, -0.135, f"{fold:.0f}\u00d7 smaller", ha="center", fontsize=5.7,
                 color=C_POS, fontweight="bold")
        axB.text(i, -0.235, f"p {fmt_p(c[3])} \u2192 {fmt_p(c[4])}", ha="center", fontsize=5.2,
                 color="#555555")
    axB.axhline(0.15, color=C_RULE, lw=0.7, ls="--")
    axB.text(len(comp) - 0.45, 0.165, "0.15", fontsize=5.4, color=C_RULE)
    axB.set_xticks(x)
    axB.set_xticklabels([c[0] for c in comp], fontsize=5.6)
    axB.set_ylim(-0.32, 1.10)
    axB.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axB.set_ylabel("|Cliff's \u03b4|, score by timepoint")
    axB.legend(loc="upper right", handlelength=1.0)
    axB.set_title("Failure mode 2 \u2014 position-stacked gene tables", fontsize=6.9, pad=4)
    top_strip(axB)
    panel(axB, "B", dx=-0.20, dy=1.15)

    # ---- C method p landscape
    axC = fig.add_subplot(gs[1, 0])
    m = pd.read_csv(R / "method_sensitivity" / "method_matrix_all.csv")
    m["p"] = pd.to_numeric(m["p"], errors="coerce")
    m = m[m["p"].notna() & ~m["family"].isin(["effect"])].copy()
    fam_all = sorted(m["family"].dropna().unique())
    picks = [
        ("DFCI TP53 mut vs resistance", "TP53 vs phenotype\n2\u00d72 with an empty cell"),
        ("DFCI treatment days ~ CDKN2A mut", "CDKN2A vs treatment days\ncarrier arm n = 3"),
        ("AURORA REP30: metastatic vs primary (all biopsies)",
         "metastatic vs primary\nrepeat biopsies, one patient"),
        ("AURORA REP30 vs cumulative days on metastatic therapy",
         "score vs cumulative therapy\nright-skewed dose axis"),
    ]
    axC.axvline(0.05, color=C_POS, lw=0.9, ls="--", zorder=1)
    axC.text(0.052, -0.72, "p = 0.05", fontsize=5.8, color=C_POS, ha="left")
    for i, (key, lab) in enumerate(picks):
        sub = m[m["comparison"] == key]
        families = sorted(sub["family"].dropna().unique())
        for j, fam in enumerate(families):
            s = sub[sub["family"] == fam]
            yj = i + (j - (len(families) - 1) / 2) * 0.20
            axC.scatter(s["p"].clip(lower=2e-9), np.full(len(s), yj), s=9,
                        color=FAMILY_COLOR.get(fam, "#888888"), edgecolor="#333333",
                        linewidth=0.25, zorder=3)
    axC.set_yticks(range(len(picks)))
    axC.set_yticklabels([p[1] for p in picks], fontsize=5.5)
    axC.set_xscale("log")
    axC.set_xlim(2e-9, 2.2)
    axC.set_ylim(-0.9, len(picks) - 0.35)
    axC.set_xlabel("p value (log scale)")
    axC.legend(handles=[Line2D([], [], marker="o", ls="", ms=4,
                               mfc=FAMILY_COLOR[f], mec="#333333", mew=0.25, label=f)
                        for f in sorted(m["family"].dropna().unique())],
               ncol=5, loc="lower left", bbox_to_anchor=(0, 1.005), handletextpad=0.25,
               columnspacing=0.9)
    axC.set_title("Failure mode 3 \u2014 the same data, eight to seventeen estimators",
                  fontsize=6.9, pad=16)
    top_strip(axC)
    panel(axC, "C", dx=-0.28, dy=1.31)

    # ---- D critical band
    axD = fig.add_subplot(gs[1, 1])
    v = pd.read_csv(R / "method_sensitivity" / "method_verdict_summary.csv")
    v["p_median"] = pd.to_numeric(v["p_median"], errors="coerce")
    v["spread"] = pd.to_numeric(v["spread_order_of_magnitude"], errors="coerce")
    v["frac"] = v["n_sig"] / v["n_methods"]
    axD.axvspan(0.01, 0.15, color=C_BAND, zorder=0)
    axD.axvline(0.05, color=C_POS, lw=0.8, ls="--", zorder=1)
    for _, r in v.iterrows():
        col = C_POS if r["frac"] > 0 else C_NS
        axD.scatter(r["p_median"], r["spread"] + 0.02, s=12 + 26 * r["frac"], color=col,
                    edgecolor="#333333", linewidth=0.35, zorder=3)
    for key, lab, dx, dy in [
        ("DFCI TP53 mut vs resistance", "TP53, zero cell", -0.55, 0.28),
        ("DFCI treatment days ~ CDKN2A mut", "CDKN2A, n = 3", -0.55, -0.42),
        ("AURORA REP30: metastatic vs primary (all biopsies)", "metastatic vs primary",
         -0.80, 0.30),
        ("AURORA REP30 ~ PIK3CA mutation (metastatic biopsies only)", "PIK3CA", 0.10, 0.34),
        ("AURORA REP30 ~ RB1 loss (metastatic biopsies only)", "RB1 loss,\nmethod-immune",
         0.05, 0.30),
    ]:
        row = v[v["comparison"] == key]
        if row.empty:
            continue
        row = row.iloc[0]
        axD.annotate(lab, xy=(row["p_median"], row["spread"] + 0.02),
                     xytext=(row["p_median"] * (10 ** dx), row["spread"] + dy),
                     fontsize=5.5, ha="center",
                     arrowprops=dict(arrowstyle="-", lw=0.5, color="#777777"))
    axD.set_xscale("log")
    axD.set_xlim(2e-5, 1.6)
    axD.set_ylim(-0.35, 7.6)
    axD.set_xlabel("median p across estimators")
    axD.set_ylabel("disagreement across estimators\n(orders of magnitude in p)")
    axD.text(0.038, 7.0, "critical band", fontsize=5.8, color="#666666", ha="center")
    axD.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=4, mfc=C_NS, mec="#333333", mew=0.3,
               label="no estimator calls it significant"),
        Line2D([], [], marker="o", ls="", ms=6.5, mfc=C_POS, mec="#333333", mew=0.3,
               label="some estimators do (area = fraction)"),
    ], loc="lower right", handletextpad=0.3)
    axD.set_title("Decision flips only inside the critical band", fontsize=6.9, pad=4)
    top_strip(axD)
    panel(axD, "D", dx=-0.28, dy=1.15)
    save(fig, "Fig4_failure_modes")


# ===========================================================================
def fig4():
    fig = plt.figure(figsize=(W, 166 * MM))
    gs = GridSpec(2, 2, figure=fig, hspace=0.60, wspace=0.36)
    fig.subplots_adjust(left=0.115, right=0.995, top=0.945, bottom=0.085)

    # ---- A proliferation coupling, primary vs metastatic
    axA = fig.add_subplot(gs[0, 0])
    corr = pd.read_csv(R / "subtype_rep30" / "score_correlations.csv")
    met_p = float(corr[(corr["stratum"].str.startswith("METABRIC\u00b7HR"))
                       & (corr["a"] == "REP30") & (corr["b"] == "PROLIF")]["spearman"].iloc[0])
    items = [
        ("MBCproject\n(mostly primary)", 0.410, 9.5e-8, "primary"),
        ("METABRIC\n(HR+/HER2\u2212)", met_p, 1.4e-51, "primary"),
        ("AURORA US\n(primary biopsies)", 0.382, 1.1e-2, "primary"),
        ("AURORA US\n(metastatic biopsies)", 0.289, 9.8e-3, "metastatic"),
        ("MET500\n(all lineages)", 0.319, 5.0e-22, "metastatic"),
        ("MET500\n(breast only)", 0.099, 0.213, "metastatic"),
        ("MET500\n(breast, purity \u2265 0.40)", 0.061, 0.501, "metastatic"),
    ]
    x = np.arange(len(items))
    for i, (lab, r, p, kind) in enumerate(items):
        col = "#5B8C8A" if kind == "primary" else "#C97B4A"
        axA.bar(i, r, width=0.62, color=col, edgecolor="white", linewidth=0.4)
        axA.text(i, r + 0.015, f"{r:.3f}", ha="center", fontsize=5.5)
        axA.text(i, -0.055, "ns" if p > 0.05 else stars(p), ha="center", fontsize=5.5,
                 color="#555555")
    axA.axhline(0, color=C_RULE, lw=0.7)
    axA.set_xticks(x)
    axA.set_xticklabels([i[0] for i in items], fontsize=5.2)
    axA.set_ylim(-0.10, 0.50)
    axA.set_ylabel("Spearman r(score, proliferation module)")
    axA.legend(handles=[Patch(facecolor="#5B8C8A", edgecolor="none", label="primary tumour"),
                        Patch(facecolor="#C97B4A", edgecolor="none", label="metastatic biopsy")],
               ncol=2, loc="upper right", handlelength=1.0)
    axA.set_title("A \u2192 the proliferation coupling weakens in metastatic biopsies",
                  fontsize=6.9, pad=4)
    top_strip(axA)
    panel(axA, "A", dx=-0.28, dy=1.14)

    # ---- B what it reads in metastatic breast biopsies
    axB = fig.add_subplot(gs[0, 1])
    items = [
        ("tumour content", 0.371, 1.5e-6),
        ("proliferation module", 0.099, 0.213),
        ("MKI67", 0.061, 0.443),
        ("age", 0.073, 0.359),
        ("interval to biopsy", -0.025, 0.756),
    ]
    axB.axvline(0, color=C_RULE, lw=0.7)
    yy = np.arange(len(items))[::-1]
    for i, (lab, r, p) in enumerate(items):
        col = C_POS if p < 0.05 else "#9AA0A6"
        axB.barh(yy[i], r, height=0.5, color=col, edgecolor="white", linewidth=0.4)
        axB.text(r + (0.012 if r >= 0 else -0.012), yy[i], f"{r:+.3f}{stars(p)}",
                 ha="left" if r >= 0 else "right", va="center", fontsize=5.6)
    axB.set_yticks(yy)
    axB.set_yticklabels([i[0] for i in items], fontsize=5.8)
    axB.set_xlim(-0.14, 0.46)
    axB.set_xticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4])
    axB.set_xlabel("Spearman r with score (MET500, breast only)")
    axB.set_title("B \u2192 it now reads tumour content instead", fontsize=6.9, pad=4)
    top_strip(axB)
    panel(axB, "B", dx=-0.34, dy=1.14)

    # ---- C genotype acquisition
    axC = fig.add_subplot(gs[1, 0])
    cc = pd.read_csv(R / "brca_igr_2015" / "cohort_comparison.csv")
    x = np.arange(len(cc))
    axC.bar(x - 0.19, cc["TCGA_primary_pct"], width=0.36, color="#5B8C8A",
            edgecolor="white", linewidth=0.4, label="primary (TCGA-BRCA)")
    axC.bar(x + 0.19, cc["IGR_met_pct"], width=0.36, color="#C97B4A",
            edgecolor="white", linewidth=0.4, label="metastatic (n = 216)")
    for i, (_, r) in enumerate(cc.iterrows()):
        axC.text(i - 0.19, r["TCGA_primary_pct"] + 0.9, f"{r['TCGA_primary_pct']:.1f}",
                 ha="center", fontsize=5.6)
        axC.text(i + 0.19, r["IGR_met_pct"] + 0.9, f"{r['IGR_met_pct']:.1f}",
                 ha="center", fontsize=5.6)
    axC.annotate("13\u00d7", xy=(0.19, 10.2), xytext=(0.62, 24),
                 fontsize=7.5, fontweight="bold", color=C_POS,
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=C_POS))
    axC.text(0.19, 12.6, "14.0% in HR+/HER2\u2212", fontsize=5.5, ha="center", color="#555555")
    axC.set_xticks(x)
    axC.set_xticklabels([str(v).replace("_", " ").replace("(CNA<=-1)", "") for v in cc["event"]],
                        fontsize=5.6)
    axC.set_ylim(0, 56)
    axC.set_ylabel("carriers (%)")
    axC.legend(loc="upper center", ncol=1, handlelength=1.0)
    axC.set_title("C \u2192 acquired-resistance genotypes appear only after treatment",
                  fontsize=6.9, pad=4)
    top_strip(axC)
    panel(axC, "C", dx=-0.24, dy=1.14)

    # ---- D DFCI CDK4/6i
    axD = fig.add_subplot(gs[1, 1])
    ph = pd.read_csv(R / "metastatic" / "cdk46_resistance" / "dfci_gene_vs_phenotype.csv")
    ph = ph[ph["contrast"].str.startswith("resistant")].set_index("gene")
    genes = ["TP53", "ESR1", "RB1"]
    x = np.arange(len(genes))
    res = [float(ph.loc[g, "freq_case"]) * 100 for g in genes]
    sen = [float(ph.loc[g, "freq_ctrl"]) * 100 for g in genes]
    axD.bar(x - 0.19, res, width=0.36, color=C_POS, edgecolor="white", linewidth=0.4,
            label="resistant (n = 41)")
    axD.bar(x + 0.19, sen, width=0.36, color="#C9C9C9", edgecolor="#555555", linewidth=0.5,
            label="sensitive (n = 18)")
    for i, g in enumerate(genes):
        p = float(ph.loc[g, "fisher_p"])
        axD.text(i, max(res[i], sen[i]) + 2.2, fmt_p(p) + stars(p), ha="center", fontsize=5.6,
                 color=C_POS if p < 0.05 else "#555555")
        axD.text(i, max(res[i], sen[i]) + 6.4, f"{ph.loc[g,'case_pos']:.0f}/41 vs "
                 f"{ph.loc[g,'ctrl_pos']:.0f}/18", ha="center", fontsize=5.3, color="#777777")
    axD.set_xticks(x)
    axD.set_xticklabels(genes, fontsize=6.0)
    axD.set_ylim(0, 74)
    axD.set_ylabel("mutation carriers (%)")
    axD.legend(loc="upper right", handlelength=1.0)
    axDb = axD.inset_axes([0.60, 0.44, 0.36, 0.40])
    labels = ["Intrinsic", "Acquired", "Sensitive"]
    med = [576, 272, 103]
    nn = [28, 13, 18]
    axDb.bar(range(3), med, width=0.6, color=[C_POS, "#E0A030", "#9AA0A6"],
             edgecolor="white", linewidth=0.4)
    for i, (mm_, n_) in enumerate(zip(med, nn)):
        axDb.text(i, mm_ + 18, f"{mm_}", ha="center", fontsize=5.0)
        axDb.text(i, -142, f"n={n_}", ha="center", fontsize=4.6, color="#666666")
    axDb.set_xticks(range(3))
    axDb.set_xticklabels(labels, fontsize=4.6)
    axDb.set_ylim(0, 700)
    axDb.set_yticks([0, 300, 600])
    axDb.tick_params(axis="y", labelsize=4.6)
    axDb.set_title("median days on\nCDK4/6i", fontsize=5.0, pad=2)
    top_strip(axDb)
    axD.set_title("D \u2192 in CDK4/6i resistance TP53 is enriched, ESR1 is not",
                  fontsize=6.9, pad=4)
    panel(axD, "D", dx=-0.24, dy=1.14)
    save(fig, "Fig2_confounder_switch")


# ===========================================================================
def figs1():
    fig = plt.figure(figsize=(W, 152 * MM))
    gs = GridSpec(2, 2, figure=fig, hspace=0.66, wspace=0.36)
    fig.subplots_adjust(left=0.105, right=0.995, top=0.950, bottom=0.080)

    # A variance decomposition
    axA = fig.add_subplot(gs[0, 0])
    lab = ["study", "cell type", "sample"]
    val = [0.0252, 0.0363, 0.1030]
    axA.bar(range(3), val, width=0.55, color=["#9AA0A6", "#9AA0A6", C_POS],
            edgecolor="white", linewidth=0.4)
    for i, v in enumerate(val):
        axA.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=6.0)
    axA.set_xticks(range(3))
    axA.set_xticklabels(lab, fontsize=5.8)
    axA.set_ylim(0, 0.125)
    axA.set_ylabel("\u03b7\u00b2 of score (96,368 cells)")
    axA.set_title("Most variance is between samples, not cell types", fontsize=6.9, pad=4)
    top_strip(axA)
    panel(axA, "A", dx=-0.26, dy=1.14)

    # B cross-cohort rank agreement
    axB = fig.add_subplot(gs[0, 1])
    st = pd.read_csv(R / "sc" / "integrated" / "report" / "rep_by_celltype_study.csv")
    axB.scatter(st["GSE161529"], st["GSE249361"], s=13, color="#7D5BA6",
                edgecolor="#333333", linewidth=0.3)
    for _, r in st.iterrows():
        if abs(r["GSE161529"] - r["GSE249361"]) > 0.035:
            axB.annotate(str(r["cell_type_int"]), xy=(r["GSE161529"], r["GSE249361"]),
                         xytext=(3, 3), textcoords="offset points", fontsize=5.0,
                         color="#555555")
    lim = [0.02, 0.19]
    axB.plot(lim, lim, color=C_RULE, lw=0.6, ls=":")
    axB.set_xlim(lim)
    axB.set_ylim(lim)
    axB.set_xlabel("score by cell type, GSE161529")
    axB.set_ylabel("score by cell type, GSE249361")
    axB.text(0.04, 0.172, "\u03c1 = \u22120.042\np = 0.91", fontsize=6.0, color=C_POS)
    axB.set_title("Cell-type scores do not agree between cohorts", fontsize=6.9, pad=4)
    top_strip(axB)
    panel(axB, "B", dx=-0.26, dy=1.14)

    # C pre/post delta after fix
    axC = fig.add_subplot(gs[1, 0])
    rc = pd.read_csv(R / "sc" / "GSE264205" / "reload" / "rep30_comparisons.csv")
    axC.bar(range(len(rc)), rc["cliffs_delta"], width=0.55, color=C_NEG,
            edgecolor="white", linewidth=0.4)
    for i, (_, r) in enumerate(rc.iterrows()):
        axC.text(i, r["cliffs_delta"] + 0.008, f"{r['cliffs_delta']:.3f}", ha="center",
                 fontsize=5.6)
        axC.text(i, -0.035, fmt_p(r["mwu_p"]), ha="center", fontsize=5.0, color="#666666")
    axC.axhline(0.15, color=C_RULE, lw=0.7, ls="--")
    axC.text(len(rc) - 0.5, 0.158, "0.15", fontsize=5.3, color=C_RULE, ha="right")
    axC.set_xticks(range(len(rc)))
    axC.set_xticklabels([f"{r['compartment'].split('/')[0]}\n{r['group_A']} vs {r['group_B']}"
                         for _, r in rc.iterrows()], fontsize=4.9)
    axC.set_ylim(-0.06, 0.36)
    axC.set_ylabel("|Cliff's \u03b4|, score")
    axC.set_title("After alignment repair: no compartment moves the score",
                  fontsize=6.9, pad=4)
    top_strip(axC)
    panel(axC, "C", dx=-0.24, dy=1.14)

    # D compartment composition
    axD = fig.add_subplot(gs[1, 1])
    comp = pd.read_csv(R / "sc" / "GSE264205" / "reload" / "compartment_composition.csv")
    comp = comp.set_index("sample")
    order = ["Pre", "Post1", "Post2", "Meta"]
    comp = comp.reindex(order)
    pct = comp.div(comp.sum(axis=1), axis=0) * 100
    keep = [c for c in pct.columns if pct[c].max() > 1]
    bottom = np.zeros(len(order))
    cmap = ["#B03A2E", "#C97B4A", "#2E86AB", "#5B8C5A", "#7D5BA6", "#9AA0A6"]
    for j, c in enumerate(keep):
        axD.bar(range(len(order)), pct[c].values, bottom=bottom, width=0.62,
                color=cmap[j % len(cmap)], edgecolor="white", linewidth=0.4, label=c)
        bottom += pct[c].values
    axD.set_xticks(range(len(order)))
    axD.set_xticklabels(order, fontsize=5.8)
    axD.set_ylabel("cells (%)")
    axD.set_ylim(0, 100)
    axD.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.005), handlelength=0.8,
               columnspacing=0.8, handletextpad=0.3)
    axD.set_title("Malignant cells are almost absent after treatment", fontsize=6.9, pad=16)
    top_strip(axD)
    panel(axD, "D", dx=-0.24, dy=1.30)
    save(fig, "FigS3_single_cell")


# ===========================================================================
def figs2():
    fig = plt.figure(figsize=(W, 92 * MM))
    gs = GridSpec(1, 2, figure=fig, wspace=0.30)
    fig.subplots_adjust(left=0.100, right=0.995, top=0.920, bottom=0.110)

    axA = fig.add_subplot(gs[0, 0])
    val = [0.717, 0.086]
    err = None
    axA.bar([0, 1], val, width=0.45, color=[C_POS, "#9AA0A6"], edgecolor="white", linewidth=0.4)
    for i, v in enumerate(val):
        axA.text(i, v + 0.02, f"{v:+.3f}", ha="center", fontsize=6.6)
    axA.text(0, 0.60, "p = 1.8e\u221213", ha="center", fontsize=6.0)
    axA.text(1, 0.17, "p = 0.414", ha="center", fontsize=6.0, color="#555555")
    axA.set_xticks([0, 1])
    axA.set_xticklabels(["derivation\nMBCproject (n = 157)", "replication\nAURORA US (n = 123)"],
                        fontsize=5.8)
    axA.set_ylim(0, 0.88)
    axA.set_ylabel("Cliff's \u03b4, exposed vs naive")
    axA.set_title("Pre-specified replication fails", fontsize=6.9, pad=4)
    top_strip(axA)
    panel(axA, "A", dx=-0.22, dy=1.14)

    axB = fig.add_subplot(gs[0, 1])
    mv = pd.read_csv(ARCH / "resistance_axis" / "merged_de_vectors.csv")
    d1 = pd.to_numeric(mv["d_mbc"], errors="coerce")
    d2 = pd.to_numeric(mv["d_aur"], errors="coerce")
    ok = np.isfinite(d1) & np.isfinite(d2)
    # subsample the background cloud: 30k vector markers make the EPS far too
    # heavy to edit comfortably in Illustrator; the visual impression is unchanged
    rng = np.random.default_rng(11)
    idx = np.where(ok)[0]
    if len(idx) > 9000:
        idx = rng.choice(idx, 9000, replace=False)
    axB.scatter(d1.iloc[idx], d2.iloc[idx], s=1.6, color="#C9C9C9",
                edgecolor="none", zorder=1)
    w = pd.read_csv(ARCH / "resistance_axis" / "EAP_weights.csv")
    sub = mv.merge(w, on="gene", how="inner")
    axB.scatter(pd.to_numeric(sub["d_mbc"]), pd.to_numeric(sub["d_aur"]), s=26,
                color=C_POS, edgecolor="#333333", linewidth=0.4, zorder=3)
    for _, r in sub.iterrows():
        axB.annotate(r["gene"], xy=(float(r["d_mbc"]), float(r["d_aur"])),
                     xytext=(4, 3), textcoords="offset points", fontsize=5.4, color="#333333")
    axB.axhline(0, color=C_RULE, lw=0.5)
    axB.axvline(0, color=C_RULE, lw=0.5)
    axB.set_xlim(-0.85, 0.85)
    axB.set_ylim(-0.85, 0.85)
    axB.set_xlabel("per-gene \u03b4, derivation cohort")
    axB.set_ylabel("per-gene \u03b4, replication cohort")
    axB.text(-0.80, 0.66, f"r = \u22120.021 over "
             f"{int(ok.sum()):,} genes", fontsize=6.2, color=C_POS)
    axB.set_title("The whole genome-level effect vector fails to transfer",
                  fontsize=6.9, pad=4)
    top_strip(axB)
    panel(axB, "B", dx=-0.22, dy=1.14)
    save(fig, "FigS4_exposure_axis")


# ===========================================================================
def figs3():
    fig = plt.figure(figsize=(W, 108 * MM))
    gs = GridSpec(1, 2, figure=fig, wspace=0.24, width_ratios=[2.1, 1])
    fig.subplots_adjust(left=0.085, right=0.995, top=0.880, bottom=0.090)

    axA = fig.add_subplot(gs[0, 0])
    order = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
    rng = np.random.default_rng(3)
    axA.axhspan(-0.15, 0.15, color=C_BAND, zorder=0)
    axA.axhline(0, color=C_RULE, lw=0.6, zorder=1)
    for i, c in enumerate(order):
        sub = INC[INC["cohort_id"] == c]
        for cls, g in sub.groupby("cls"):
            axA.scatter(i + rng.uniform(-0.26, 0.26, len(g)), g["effect"], s=3.2,
                        color=CLASS_COLOR.get(cls, "#BEBEBE"), edgecolor="none", zorder=3)
    axA.set_xticks(range(len(order)))
    axA.set_xticklabels(order, fontsize=6.0)
    axA.set_xlim(-0.6, len(order) - 0.4)
    axA.set_ylim(-0.75, 1.02)
    axA.set_ylabel("effect size (Cliff's \u03b4 or Spearman r)")
    axA.set_xlabel("dataset")
    axA.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.6,
                               mfc=CLASS_COLOR[c], mec="none", label=CLASS_LABEL[c])
                        for c in GROUP_ORDER + ["other"]],
               ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.24), handletextpad=0.3,
               columnspacing=1.0)
    axA.set_title("All 461 comparisons", fontsize=6.9, pad=20)
    top_strip(axA)
    panel(axA, "A", dx=-0.10, dy=1.32)

    axB = fig.add_subplot(gs[0, 1])
    b = np.linspace(-0.75, 1.02, 36)
    for cls, col in [("confound", CLASS_COLOR["confound"]),
                     ("molecular-resistance-genotype", CLASS_COLOR["molecular-resistance-genotype"])]:
        v = INC.loc[INC["cls"] == cls, "effect"].dropna()
        h, e = np.histogram(v, bins=b)
        axB.step(0.5 * (e[1:] + e[:-1]), h / h.sum() / (e[1] - e[0]), where="mid",
                 color=col, lw=1.0, label=CLASS_LABEL[cls])
    axB.axvspan(-0.15, 0.15, color=C_BAND, zorder=0)
    axB.axvline(0, color=C_RULE, lw=0.6)
    axB.set_xlabel("effect size")
    axB.set_ylabel("density")
    axB.set_xlim(-0.75, 1.02)
    axB.legend(loc="upper left", handlelength=1.0)
    axB.set_title("Confounds sit further from zero", fontsize=6.9, pad=4)
    top_strip(axB)
    panel(axB, "B", dx=-0.30, dy=1.14)
    save(fig, "FigS1_full_landscape")


# ===========================================================================
def figs4():
    """Fig. S4 - Venet random-gene-set null distribution vs REP30.

    Two side-by-side panels.
      A: Null B - 1000 random 30-gene sets, in-sample Cox weights (no gene
         selection), scored and evaluated in METABRIC HR+/HER2-. The REP30
         in-sample C-index is the reference line.
      B: Null C - 964 random 30-gene sets with weights fitted on TCGA and
         applied unchanged to METABRIC HR+/HER2- (cross-cohort). The REP30
         cross-cohort C-index is the reference line; the in-sample value is
         drawn only as a secondary marker so the two regimes stay comparable.

      Each panel is therefore matched null-to-observation: the in-sample null
      (B) is read against the in-sample estimate, and the cross-cohort null
      (C) against the cross-cohort estimate.

    All annotation numbers are computed from the CSVs - nothing is hard-coded.
    Solid fills only (no alpha), Arial-safe glyphs (no U+2011 / U+221D).
    """
    sup = R / "manuscript_support"
    draws = pd.read_csv(sup / "venet_null_draws.csv")
    summ = pd.read_csv(sup / "venet_null_summary.csv")
    drawsC = pd.read_csv(sup / "venet_null_draws_C.csv")
    summC = pd.read_csv(sup / "venet_null_summary_C.csv")
    cf = pd.read_csv(sup / "crossfit_results.csv")

    # reference values pulled from the data, not typed in by hand
    rep30_in = float(summ.loc[summ["null"] == "B_in_sample_cox_weighted",
                              "rep30_cindex"].iloc[0])          # 0.6766 in-sample
    rep30_cc = float(summC["rep30cf_cindex"].iloc[0])           # 0.5823 cross-cohort
    assert abs(rep30_cc - float(cf.loc[cf["analysis"] == "REP30_crossfit",
                                       "cindex"].iloc[0])) < 1e-6

    b = draws.loc[draws["null"] == "B_in_sample_cox_weighted", "cindex"].to_numpy()
    c = drawsC.loc[drawsC["null"] == "C_crossfit_cox_weighted", "cindex"].to_numpy()

    # per-panel limits: the two nulls sit on very different C-index ranges,
    # so a shared axis would flatten each histogram into a spike
    xlo_a, xhi_a = float(b.min()) - 0.020, float(max(b.max(), rep30_in)) + 0.020
    xlo_b, xhi_b = float(c.min()) - 0.020, float(max(c.max(), rep30_cc)) + 0.020

    fig = plt.figure(figsize=(W, 100 * MM))
    gs = GridSpec(1, 2, figure=fig, wspace=0.30)
    fig.subplots_adjust(left=0.100, right=0.995, top=0.900, bottom=0.130)

    # ---- A : Null B
    axA = fig.add_subplot(gs[0, 0])
    nb, _, _ = axA.hist(b, bins=28, color="#C9C9C9", edgecolor="#6E6E6E", linewidth=0.4)
    axA.axvline(rep30_in, color=C_POS, lw=1.0, ls="--", zorder=4)
    yhi = nb.max() * 1.20
    axA.set_xlim(xlo_a, xhi_a)
    axA.set_ylim(0, yhi)
    nex = int((b >= rep30_in).sum())
    n = len(b)
    axA.text(xlo_a + 0.008, yhi * 0.97,
             f"random 30-gene sets, no selection\n"
             f"median C = {np.median(b):.3f}, max C = {b.max():.3f}\n"
             f"{nex}/{n} random sets \u2265 REP30",
             ha="left", va="top", fontsize=5.7, color="#1A1A1A")
    axA.text(rep30_in - 0.004, yhi * 0.97, "REP30\nin-sample\nC = %.3f" % rep30_in,
             ha="right", va="top", fontsize=5.6, color=C_POS, fontweight="bold")
    axA.set_xlabel("C-index")
    axA.set_ylabel("random gene sets (n = %d)" % n)
    axA.set_title("Null B - random sets, in-sample Cox weights", fontsize=6.9, pad=4)
    top_strip(axA)
    panel(axA, "A", dx=-0.22, dy=1.14)

    # ---- B : Null C (cross-cohort)
    axB = fig.add_subplot(gs[0, 1])
    nc, _, _ = axB.hist(c, bins=28, color="#C9C9C9", edgecolor="#6E6E6E", linewidth=0.4)
    yhi = nc.max() * 1.20
    axB.set_xlim(xlo_b, xhi_b)
    axB.set_ylim(0, yhi)
    nex = int((c >= rep30_cc).sum())
    n = len(c)
    axB.axvline(rep30_cc, color="#2E86AB", lw=1.0, ls="--", zorder=4)
    axB.text(xlo_b + 0.008, yhi * 0.97,
             f"random 30-gene sets, weights fitted on TCGA\n"
             f"median C = {np.median(c):.3f}, max C = {c.max():.3f}\n"
             f"{nex}/{n} random sets \u2265 REP30",
             ha="left", va="top", fontsize=5.7, color="#1A1A1A")
    axB.text(rep30_cc - 0.004, yhi * 0.97, "REP30\ncross-cohort\nC = %.3f" % rep30_cc,
             ha="right", va="top", fontsize=5.6, color="#2E86AB", fontweight="bold")
    axB.set_xlabel("C-index")
    axB.set_ylabel("random gene sets (n = %d)" % n)
    axB.set_title("Null C - random sets, cross-cohort Cox weights", fontsize=6.9, pad=4)
    top_strip(axB)
    panel(axB, "B", dx=-0.22, dy=1.14)
    save(fig, "FigS2_venet_null")


if __name__ == "__main__":
    print("Building manuscript figures (TrueType text embedding for Illustrator)...")
    fig1()
    fig2()
    fig3()
    fig4()
    figs1()
    figs2()
    figs3()
    figs4()
    print("all figures written to", FIG)
