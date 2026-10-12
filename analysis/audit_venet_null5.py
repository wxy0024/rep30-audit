#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
audit_venet_null5.py -- the "most pessimistic" ceiling of Null D: deterministically take the 30 most significant univariable genes and fit them

Null D is "randomly draw 30 from the top-300 outcome-associated pool" (giving a distribution).
Here we add a **deterministic**, maximally favorable-to-critics extreme reference:
directly take the 30 genes with the highest univariable |z|, fit Cox weights, and evaluate the in-sample C / HR.
This is the theoretical ceiling a "random signature" can reach -- if REP30 exceeds it too, the Venet critique is fully closed.
"""
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "manuscript_support")
LOGPATH = os.path.join(OUT, "venet5_run_log.txt")

from rep52_backtest import load_panel
from audit_venet_null4 import load_metabric, univariate_cox_z
from audit_venet_null import cox_fit_beta, score_from_beta, cox_psd

LOG = []


def log(s=""):
    print(s)
    LOG.append(str(s))
    with open(LOGPATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG) + "\n")


def main():
    coef = load_panel(os.path.join(ROOT, "results/rep52_trim/rep52_trimmed_panel.csv"))
    log("=" * 78)
    log("[NULL D-max] deterministic univariable top-30 -> fit Cox -> in-sample C (most pessimistic reference)")

    mat, mb = load_metabric(coef)
    sub = mb[mb["HRposHER2neg"] == 1].dropna(subset=["REP30", "t", "e"])
    sub = sub[sub["t"] > 0]
    idx = sub["SAMPLE_ID"].astype(str).tolist()
    t = sub["t"].values.astype(float)
    e = sub["e"].values.astype(float)
    log(f"   n={len(sub)} events={int(e.sum())}")

    Zall = mat[idx]
    X = np.asarray(Zall.values, dtype=np.float32)
    X = np.where(np.isfinite(X), X, 0.0)
    gnames = np.array(Zall.index)
    zs = univariate_cox_z(X, t, e)

    rows = []
    # REP30 reference
    rep = sub["REP30"].values.astype(float)
    r0 = cox_psd(rep, t, e)
    log(f"   REP30 (published pooled coefs): C={r0['cindex']:.4f} HR={r0['HR']:.3f}")
    rows.append(dict(set="REP30_published", n_genes=30, cindex=r0["cindex"], HR=r0["HR"], p=r0["p"]))

    # deterministic top-K by |z|
    ordr = np.argsort(-np.abs(zs))
    for K in [30, 50, 100, 300]:
        sel = ordr[:K]
        gsel = list(gnames[sel])
        Zk = pd.DataFrame(X[sel, :], index=gsel, columns=idx)
        beta, _ = cox_fit_beta(Zk, t, e)
        if beta is None:
            log(f"   top-{K}: fit failed")
            continue
        sc = score_from_beta(Zk, beta).reindex(idx).values.astype(float)
        r = cox_psd(sc, t, e)
        log(f"   deterministic top-{K:3d} by |z|, Cox-fitted: "
            f"C={r['cindex']:.4f} HR={r['HR']:.3f} p={r['p']:.3g}")
        rows.append(dict(set=f"deterministic_top{K}_by_univariate_z", n_genes=K,
                         cindex=r["cindex"], HR=r["HR"], p=r["p"]))

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "venet_null_D_deterministic.csv"),
                              index=False, encoding="utf-8")
    log("\nDONE")


if __name__ == "__main__":
    sys.excepthook = lambda a, b, tb: (
        LOG.append("EXCEPTION:\n" + "".join(traceback.format_exception(a, b, tb))),
        open(LOGPATH, "w", encoding="utf-8").write("\n".join(LOG) + "\n"))
    main()
