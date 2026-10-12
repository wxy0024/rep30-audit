# -*- coding: utf-8 -*-
"""Export every manuscript figure panel as an individual file.

Output: manuscript/figures/figure_panels/<Fig>_panel<Letter>.{png,pdf,eps}
        (Fig1 panels: png always; pdf vector crop when page geometry matches)

Two pathways
------------
1. Fig2, Fig3, Fig4, FigS1-S4 : re-rendered from build_manuscript_figures.py.
   Each subplot axes is exported as its own tight-cropped panel (PNG 600 dpi +
   vector PDF + EPS).  The deployed composites for these figures ARE the
   matplotlib renders (no .ai source), so panels match the paper exactly.

2. Fig1 : the deployed composite is an Illustrator document whose panel set
   (5 panels A-E, incl. a flowchart panel A) DIFFERS from the matplotlib
   script (4 panels, no flowchart).  Therefore:
     - the deployed Fig1_audit.png is segmented by whitespace into A-E crops;
     - the deployed Fig1_audit.pdf is cropped into vector panels with pypdf,
       but only if its page geometry matches the PNG (Illustrator artboards
       may differ from matplotlib tight bboxes - guard below).

Run:
    C:/Users/mao02/.workbuddy/binaries/python/envs/default/Scripts/python.exe \
        manuscript/export_panels.py
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: F401  (panel_save closes figures via plt)
import numpy as np
from matplotlib.transforms import Bbox
from PIL import Image

HERE = Path(__file__).resolve().parent          # manuscript/
FIG = HERE / "figures"
OUT = FIG / "figure_panels"
OUT.mkdir(parents=True, exist_ok=True)

index_rows: list[dict] = []


def register(figure: str, panel: str, title: str, source: str) -> None:
    index_rows.append({"figure": figure, "panel": panel, "title": title, "source": source})


# ===========================================================================
# pathway 1 - re-render matplotlib figures, export each axes
# ===========================================================================
def export_matplotlib_panels() -> None:
    spec = importlib.util.spec_from_file_location("bmf", HERE / "build_manuscript_figures.py")
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)                   # runs the CSV loaders (module scope)

    def panel_save(fig, stem: str) -> None:      # replaces build script's save()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        dpi = float(renderer.dpi)
        FW, FH = (float(v) for v in fig.get_size_inches())
        letters = "ABCDEFGHIJ"
        panels = [ax for ax in fig.axes if ax.get_subplotspec() is not None]
        insets = [ax for ax in fig.axes if ax.get_subplotspec() is None]
        # Hide every panel before exporting one: some panel titles overflow into
        # the neighbouring column (e.g. Fig2 A), so a rectangular tight-bbox
        # crop would otherwise capture fragments of the neighbour's tick labels.
        # Hiding the other panels blanks those regions in ALL output formats.
        for ax in fig.axes:
            ax.set_visible(False)
        k = 0
        for P in panels:
            P.set_visible(True)
            for ch in getattr(P, "child_axes", []):   # keep own insets visible
                ch.set_visible(True)
            tb = P.get_tightbbox(renderer)
            # Generous pad: some rendered glyphs overshoot their measured
            # window extent by a few px (hinting/AA at different save dpi).
            # Neighbours are hidden, so extra pad is only white margin.
            pad = 0.06                               # inches of breathing room
            # NOTE: no clamping to the figure box - panel letters above an axes
            # (dy > 1) can exceed the canvas top, and savefig with bbox_inches
            # legitimately extends the output canvas beyond the figure rect.
            x0 = tb.x0 / dpi - pad
            y0 = tb.y0 / dpi - pad
            x1 = tb.x1 / dpi + pad
            y1 = tb.y1 / dpi + pad
            box = Bbox([[x0, y0], [x1, y1]])
            letter = letters[k]
            k += 1
            for ext in ("png", "pdf", "eps"):
                p = OUT / f"{stem}_panel{letter}.{ext}"
                fig.savefig(p, format=ext, dpi=600 if ext == "png" else None,
                            bbox_inches=box)
            title = P.get_title().replace("\n", " ")
            print(f"  {stem} panel {letter}: {title!r}")
            register(stem, letter, title, "matplotlib re-render")
            P.set_visible(False)
            for ch in getattr(P, "child_axes", []):
                ch.set_visible(False)
        for ax in fig.axes:
            ax.set_visible(True)
        plt.close(fig)

    B.save = panel_save                          # fig functions call save(fig, stem)
    for fn in (B.fig2, B.fig3, B.fig4, B.figs1, B.figs2, B.figs3, B.figs4):
        name = fn.__name__
        try:
            print(f"[render] {name}")
            fn()
        except Exception as exc:                 # keep going; report at the end
            print(f"  !! {name} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)


# ===========================================================================
# pathway 2 - segment the deployed Fig1 composite (Illustrator document)
# ===========================================================================
def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    bands, in_band, start = [], False, 0
    for i, m in enumerate(mask):
        if m and not in_band:
            start, in_band = i, True
        elif not m and in_band:
            bands.append((start, i))
            in_band = False
    if in_band:
        bands.append((start, len(mask)))
    return bands


def _merge(bands: list[tuple[int, int]], gap_min: int) -> list[list[int]]:
    merged: list[list[int]] = []
    for b in bands:
        if merged and b[0] - merged[-1][1] < gap_min:
            merged[-1][1] = b[1]
        else:
            merged.append([b[0], b[1]])
    return merged


def export_fig1_panels() -> None:
    png = FIG / "Fig1_audit.png"
    img = Image.open(png)
    rgb = np.asarray(img.convert("RGB"))
    gray = np.asarray(img.convert("L"))
    H, W = gray.shape
    ink = gray < 245

    # Row-band boundaries derived from a whitespace audit of THIS exact PNG
    # (2249x4308).  Auto-detection by gap size is unreliable here because
    # within-panel gaps (8-28 px) are as wide as some panel separators
    # (20-51 px).  Bands (y from top, px): A=10-1445, B=1475-2248,
    # C=2299-3489, D+E=3509-4300.  Sanity-check: the separator midpoints
    # must be blank rows.
    bands = [(10, 1445), (1475, 2248), (2299, 3489), (3509, 4300)]
    # NOTE (QC 2026-09-27): Fig1 panel C is a full-width band. Its source band
    # (y 2299-3489) carries a faint native element at x~2245-2247 (<=1.8% of
    # rows); the source column x=2248 is clean (0 ink) and the crop captures
    # through x=2247. This is intrinsic panel content at the right edge, NOT a
    # truncation - a border-ink QC may flag it, but do NOT "fix" by trimming.
    for (a0, a1), (b0, _) in zip(bands, bands[1:]):
        mid = (a1 + b0) // 2
        if ink[mid, :].sum() > 3:
            raise SystemExit(f"Fig1 separator sanity check failed at y={mid} - "
                             "the hardcoded band boundaries do not match this PNG")

    # ---- split the last band into D | E on the widest internal column gap
    last = bands[-1]
    sub = ink[last[0]:last[1], :]
    col_runs = _runs(sub.sum(axis=0) > 2)
    gaps = [(col_runs[i + 1][0] - col_runs[i][1], col_runs[i][1], col_runs[i + 1][0])
            for i in range(len(col_runs) - 1)]
    mid = [g for g in gaps if W * 0.25 < (g[1] + g[2]) / 2 < W * 0.75]
    if not mid:
        raise SystemExit("no mid-column gap found to split Fig1 D|E")
    width, gx0, gx1 = max(mid, key=lambda g: g[0])
    de_split = (gx0 + gx1) // 2
    print(f"[fig1] D|E split at x={de_split} (gap {width}px)")

    # fractional boxes: (x0, y0, x1, y1) with y measured from TOP (image coords)
    pad_x, pad_y = 0.004 * W, 0.004 * H
    boxes_px = []
    for bi, (y0, y1) in enumerate(bands):
        band_ink = ink[y0:y1, :]
        col_mask = band_ink.sum(axis=0) > 2
        xs = np.where(col_mask)[0]
        bx0, bx1 = max(xs.min() - pad_x, 0), min(xs.max() + pad_x, W - 1)
        by0, by1 = max(y0 - pad_y, 0), min(y1 + pad_y, H - 1)
        if bi < 3:
            boxes_px.append((bx0, by0, bx1, by1))
        else:
            boxes_px.append((bx0, by0, de_split - 0.002 * W, by1))   # D (left)
            boxes_px.append((de_split + 0.002 * W, by0, bx1, by1))   # E (right)
    letters = "ABCDE"
    frac = []
    for letter, (x0, y0, x1, y1) in zip(letters, boxes_px):
        x0i, y0i, x1i, y1i = (int(round(v)) for v in (x0, y0, x1, y1))
        img.convert("RGB").crop((x0i, y0i, x1i, y1i)).save(OUT / f"Fig1_panel{letter}.png")
        print(f"  Fig1 panel {letter}: px box ({x0i},{y0i})-({x1i},{y1i})")
        frac.append((x0i / W, y0i / H, x1i / W, y1i / H))
        register("Fig1", letter, "", "deployed Illustrator composite (png crop)")

    # ---- vector crops from the deployed PDF (guarded by geometry check)
    try:
        from pypdf import PdfReader, PdfWriter
        from pypdf.generic import RectangleObject
    except ImportError:
        print("[fig1] pypdf unavailable - vector Fig1 panels skipped")
        return
    pdf_path = FIG / "Fig1_audit.pdf"
    page = PdfReader(pdf_path).pages[0]
    pw, ph = float(page.mediabox.width), float(page.mediabox.height)
    # The fractional crop boxes are scale-invariant, so the PDF page only needs
    # the same content *framing* as the PNG - compare aspect ratios, not size.
    aspect_pdf, aspect_png = pw / ph, W / H
    rel = abs(aspect_pdf - aspect_png) / aspect_png
    print(f"[fig1] pdf page {pw:.1f}x{ph:.1f}pt (aspect {aspect_pdf:.4f}) vs png "
          f"aspect {aspect_png:.4f} (relative diff {rel:.3f})")
    if rel > 0.03:
        print("[fig1] !! PDF page geometry differs from PNG (Illustrator artboard) - "
              "vector Fig1 panel crops SKIPPED; PNG crops are authoritative")
        return
    for letter, (fx0, fy0, fx1, fy1) in zip(letters, frac):
        box = RectangleObject((fx0 * pw, (1 - fy1) * ph, fx1 * pw, (1 - fy0) * ph))
        page.mediabox = box
        page.cropbox = box
        w = PdfWriter()
        w.add_page(page)
        with open(OUT / f"Fig1_panel{letter}.pdf", "wb") as fh:
            w.write(fh)
        print(f"  Fig1 panel {letter}: vector pdf written")


# ===========================================================================
if __name__ == "__main__":
    print("=== pathway 1: matplotlib figures ===")
    export_matplotlib_panels()
    print("=== pathway 2: Fig1 (Illustrator composite) ===")
    try:
        export_fig1_panels()
    except SystemExit as exc:
        print(f"!! Fig1 segmentation problem: {exc}", file=sys.stderr)

    with open(OUT / "panels_index.csv", "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=["figure", "panel", "title", "source"])
        wtr.writeheader()
        wtr.writerows(index_rows)
    print(f"\ndone - {len(index_rows)} panels indexed in {OUT}")
