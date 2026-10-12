# -*- coding: utf-8 -*-
"""Shared statistics toolkit for this project.

Purpose (two goals):
  1. Two-group comparison implemented across several *method families*
     (parametric / rank / resampling / model-based) so the same comparison
     can be inspected for method sensitivity instead of trusting one test.
  2. A collection of effect-size estimators with different assumptions
     (standardised mean difference vs stochastic dominance vs raw shift).

Conventions used throughout the project:
  - `a` is the focal group (e.g. resistant / mutant / metastatic).
  - Cliff's delta > 0 means a tends to be LARGER than b.
  - Odds ratio > 1 in a 2x2 [[a, b], [c, d]] means the focal group is enriched.
"""
import numpy as np
from scipy import stats

# --------------------------------------------------------------------------
# effect sizes
# --------------------------------------------------------------------------


def cliffs_delta(a, b):
    """P(a>b) - P(a<b), ties ignored. NaN-safe on empty input."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return np.nan
    A = a[:, None]
    B = b[None, :]
    return float(((A > B).sum() - (A < B).sum()) / (na * nb))


def rank_biserial(a, b):
    """Rank-biserial correlation == Cliff's delta numerically (no-tie case).

    Kept separate so the report can show that two "different" effect sizes
    are the same quantity under different names -- a common source of
    apparent disagreement in the literature.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return np.nan
    A = a[:, None]
    B = b[None, :]
    gt = (A > B).sum() + 0.5 * (A == B).sum()
    lt = (A < B).sum() + 0.5 * (A == B).sum()
    return float((gt - lt) / (na * nb))


def cohens_d(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    s = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if s == 0:
        return np.nan
    return float((a.mean() - b.mean()) / s)


def hedges_g(a, b):
    d = cohens_d(a, b)
    n1, n2 = len(np.asarray(a)), len(np.asarray(b))
    if not np.isfinite(d):
        return np.nan
    J = 1 - 3 / (4 * (n1 + n2) - 9)
    return float(d * J)


def glass_delta(a, b):
    """Uses only the second group's SD -- sensitive to variance imbalance."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(b) < 2:
        return np.nan
    s = b.std(ddof=1)
    if s == 0:
        return np.nan
    return float((a.mean() - b.mean()) / s)


def hodges_lehmann(a, b):
    """Median of all pairwise differences a-b: a raw (unstandardised) shift."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(np.median(a[:, None] - b[None, :]))


def wmw_odds(a, b):
    """P(a>b)/P(a<b) with ties split 50/50 -- the Mann-Whitney odds."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    A = a[:, None]
    B = b[None, :]
    gt = (A > B).sum() + 0.5 * (A == B).sum()
    lt = (A < B).sum() + 0.5 * (A == B).sum()
    if lt == 0:
        return np.inf if gt > 0 else np.nan
    return float(gt / lt)


# --------------------------------------------------------------------------
# two-group tests, continuous outcome
# --------------------------------------------------------------------------


def t_student(a, b):
    s, p = stats.ttest_ind(a, b, equal_var=True)
    return {"stat": float(s), "p": float(p)}


def t_welch(a, b):
    s, p = stats.ttest_ind(a, b, equal_var=False)
    return {"stat": float(s), "p": float(p)}


def mannwhitney(a, b):
    s, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"stat": float(s), "p": float(p)}


def mannwhitney_exact(a, b):
    """Exact/Monte-Carlo MWU -- differs from the normal approximation with ties."""
    try:
        s, p = stats.mannwhitneyu(a, b, alternative="two-sided", method="exact")
        return {"stat": float(s), "p": float(p)}
    except Exception:
        try:
            s, p = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
            return {"stat": float(s), "p": float(p), "note": "exact unavailable->asymptotic"}
        except Exception as e:
            return {"stat": np.nan, "p": np.nan, "note": str(e)}


def rank_transform_welch(a, b):
    """Conover rank-transform test (Welch t on pooled ranks).

    Stands in for Brunner-Munzel in this scipy build; BM-type tests and this
    one share the same null (stochastic equality) and give near-identical
    answers, so the point of the comparison is preserved.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    r = stats.rankdata(np.concatenate([a, b]))
    r1, r2 = r[:len(a)], r[len(a):]
    s, p = stats.ttest_ind(r1, r2, equal_var=False)
    return {"stat": float(s), "p": float(p)}


def perm_test(a, b, stat="mean", nperm=20000, seed=0):
    """Two-sided permutation test on the difference of `stat`."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    f = np.mean if stat == "mean" else np.median
    obs = float(f(a) - f(b))
    pool = np.concatenate([a, b])
    n1 = len(a)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(nperm):
        p = rng.permutation(pool)
        if abs(f(p[:n1]) - f(p[n1:])) >= abs(obs) - 1e-12:
            cnt += 1
    return {"stat": obs, "p": float((cnt + 1) / (nperm + 1))}


def bootstrap_diff_ci(a, b, stat="mean", nboot=10000, seed=0, alpha=0.05):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    f = np.mean if stat == "mean" else np.median
    rng = np.random.default_rng(seed)
    d = np.empty(nboot)
    for i in range(nboot):
        d[i] = f(rng.choice(a, len(a), replace=True)) - f(rng.choice(b, len(b), replace=True))
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"stat": float(f(a) - f(b)), "lo": float(lo), "hi": float(hi)}


def compare_continuous(a, b, nperm=20000, seed=0):
    """Run every continuous-outcome method family on one comparison.

    Returns a list of dicts: {family, method, stat, p, effect_name, effect}.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    rows = []

    def add(family, method, res, eff_name=None, eff=None):
        rows.append({
            "family": family,
            "method": method,
            "stat": res.get("stat", np.nan),
            "p": res.get("p", np.nan),
            "effect_name": eff_name,
            "effect": eff,
        })

    add("parametric", "Student t (equal var)", t_student(a, b), "Cohen's d", cohens_d(a, b))
    add("parametric", "Welch t (unequal var)", t_welch(a, b), "Hedges' g", hedges_g(a, b))
    add("parametric", "Glass delta (2nd-group SD)", {"stat": glass_delta(a, b), "p": t_student(a, b)["p"]},
        "Glass's delta", glass_delta(a, b))
    add("rank", "Mann-Whitney U (asymptotic)", mannwhitney(a, b), "Cliff's delta", cliffs_delta(a, b))
    add("rank", "Mann-Whitney U (exact/MC)", mannwhitney_exact(a, b), "rank-biserial", rank_biserial(a, b))
    add("rank", "Conover rank-transform (Welch on ranks)", rank_transform_welch(a, b),
        "WMW odds", wmw_odds(a, b))
    add("resampling", "Permutation (mean diff)", perm_test(a, b, "mean", nperm, seed),
        "Hodges-Lehmann shift", hodges_lehmann(a, b))
    add("resampling", "Permutation (median diff)", perm_test(a, b, "median", nperm, seed),
        "median diff", float(np.median(a) - np.median(b)))
    add("resampling", "Bootstrap CI (mean diff, p from CI)",
        {"stat": bootstrap_diff_ci(a, b, "mean", seed=seed)["stat"],
         "p": np.nan}, "mean diff", float(a.mean() - b.mean()))
    return rows


def ci_excludes_zero(lo, hi):
    return (lo > 0) or (hi < 0)


# --------------------------------------------------------------------------
# two-group tests, binary outcome (2x2 contingency)
# --------------------------------------------------------------------------


def _midp_fisher(table):
    (a, b), (c, d) = table
    n = a + b + c + d
    r1 = a + b
    c1 = a + c
    lo = max(0, r1 - (n - c1))
    hi = min(r1, c1)
    probs = {x: stats.hypergeom.pmf(x, n, c1, r1) for x in range(lo, hi + 1)}
    p_obs = probs[a]
    p_two = sum(p for p in probs.values() if p <= p_obs * (1 + 1e-9))
    return float(p_two - 0.5 * p_obs)


def _or_ci_woolf(a, b, c, d, alpha=0.05):
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        note = "Haldane-Anscombe 0.5 correction"
    else:
        note = ""
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = stats.norm.ppf(1 - alpha / 2)
    or_ = (a * d) / (b * c)
    return float(or_ * np.exp(-z * se)), float(or_ * np.exp(z * se)), note


def _rd_ci_newcombe(a, b, c, d, alpha=0.05):
    n1, n2 = a + b, c + d
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan
    p1, p2 = a / n1, c / n2
    rd = p1 - p2
    z = stats.norm.ppf(1 - alpha / 2)
    l1 = p1 - z * np.sqrt(p1 * (1 - p1) / n1)
    u1 = p1 + z * np.sqrt(p1 * (1 - p1) / n1)
    l2 = p2 - z * np.sqrt(p2 * (1 - p2) / n2)
    u2 = p2 + z * np.sqrt(p2 * (1 - p2) / n2)
    lo = rd - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = rd + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return float(lo), float(hi)


def firth_logit(X, y, max_iter=300, tol=1e-10):
    """Firth penalised logistic regression (Heinze & Schemper 2002).

    Needed because several contrasts here are sparse (0 cells in a 2x2),
    where ordinary MLE is biased/infinite but Firth stays finite.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        eta = X @ beta
        mu = 1 / (1 + np.exp(-eta))
        W = np.clip(mu * (1 - mu), 1e-10, None)
        I = X.T @ (X * W[:, None])
        Iinv = np.linalg.pinv(I)
        H = (X * W[:, None]) @ Iinv @ X.T
        h = np.clip(np.diag(H), 0, 1 - 1e-8)
        U = X.T @ (y - mu + h * (0.5 - mu))
        step = Iinv @ U
        beta_new = beta + step
        if np.max(np.abs(step)) < tol:
            beta = beta_new
            break
        beta = beta_new
    eta = X @ beta
    mu = 1 / (1 + np.exp(-eta))
    W = np.clip(mu * (1 - mu), 1e-10, None)
    I = X.T @ (X * W[:, None])
    se = np.sqrt(np.diag(np.linalg.pinv(I)))
    return beta, se


def compare_binary(g1, g2, nperm=20000, seed=0):
    """Compare two binary vectors (focal group g1 vs g2).

    Returns (rows, table) where rows is the method x p-value long table and
    table is the 2x2 [[both1, g1_only], [g2_only, neither]].
    """
    g1 = np.asarray(g1, float)
    g2 = np.asarray(g2, float)
    g1 = g1[np.isfinite(g1)]
    g2 = g2[np.isfinite(g2)]
    a = int((g1 == 1).sum())
    b = int((g1 == 0).sum())
    c = int((g2 == 1).sum())
    d = int((g2 == 0).sum())
    table = ((a, b), (c, d))
    rows = []

    def add(family, method, p, stat=np.nan, note=""):
        rows.append({"family": family, "method": method, "stat": stat,
                     "p": float(p) if p == p else np.nan, "note": note})

    # ---- exact / algebraic tests ----
    or_, p2 = stats.fisher_exact(table)
    _, p_less = stats.fisher_exact(table, alternative="less")
    _, p_greater = stats.fisher_exact(table, alternative="greater")
    add("exact", "Fisher exact (two-sided)", p2, or_)
    add("exact", "Fisher exact (one-sided, enrichment in focal)", p_greater, or_,
        "one-sided; reported so the direction effect is visible")
    add("exact", "Fisher mid-p (two-sided)", _midp_fisher(table), or_,
        "mid-p is anti-conservative vs Fisher")
    # ---- large-sample ----
    try:
        chi2, pchi, _, _ = stats.chi2_contingency(table, correction=False)
        add("large-sample", "Pearson chi-square (no correction)", pchi, chi2)
        chi2y, pchy, _, _ = stats.chi2_contingency(table, correction=True)
        add("large-sample", "Pearson chi-square (Yates)", pchy, chi2y,
            "Yates is conservative; differs from Fisher mostly when cells are small")
    except Exception as e:
        add("large-sample", "Pearson chi-square", np.nan, note=str(e))
    # ---- model-based ----
    lo, hi, note = _or_ci_woolf(a, b, c, d)
    rows.append({"family": "model", "method": "Woolf logit OR (CI)", "stat": or_,
                 "p": np.nan, "note": f"95% CI [{lo:.2f}, {hi:.2f}] {note}".strip()})
    # ordinary logistic via statsmodels when separable-free
    try:
        import statsmodels.api as sm
        X = sm.add_constant(np.concatenate([np.ones(len(g1)), np.zeros(len(g2))]))
        yy = np.concatenate([g1, g2])
        m = sm.Logit(yy, X).fit(disp=0)
        add("model", "Logistic regression MLE (Wald)", m.pvalues[1], float(np.exp(m.params[1])),
            "fails / inflates when a cell is 0")
    except Exception as e:
        add("model", "Logistic regression MLE (Wald)", np.nan, note=f"separation: {type(e).__name__}")
    try:
        X = np.column_stack([np.ones(len(g1) + len(g2)),
                             np.concatenate([np.ones(len(g1)), np.zeros(len(g2))])])
        yy = np.concatenate([g1, g2])
        beta, se = firth_logit(X, yy)
        z = beta[1] / se[1]
        from scipy.stats import norm
        add("model", "Firth penalised logistic (Wald)", 2 * (1 - norm.cdf(abs(z))),
            float(np.exp(beta[1])), "stable under separation")
    except Exception as e:
        add("model", "Firth penalised logistic (Wald)", np.nan, note=str(e))
    # logistic with Firth controlling a covariate is added by callers
    # ---- resampling ----
    pool = np.concatenate([g1, g2])
    n1 = len(g1)
    rng = np.random.default_rng(seed)
    obs = abs(g1.mean() - g2.mean())
    cnt = 0
    for _ in range(nperm):
        p = rng.permutation(pool)
        if abs(p[:n1].mean() - p[n1:].mean()) >= obs - 1e-12:
            cnt += 1
    add("resampling", "Permutation test (risk difference)", (cnt + 1) / (nperm + 1), obs)
    rlo, rhi = _rd_ci_newcombe(a, b, c, d)
    rows.append({"family": "resampling", "method": "Risk difference (Newcombe CI)",
                 "stat": a / (a + b) - c / (c + d) if (a + b) and (c + d) else np.nan,
                 "p": np.nan, "note": f"95% CI [{rlo:.3f}, {rhi:.3f}]"})
    # effect sizes
    rows.append({"family": "effect", "method": "Odds ratio", "stat": or_, "p": np.nan, "note": ""})
    rows.append({"family": "effect", "method": "Risk difference", "stat": (a / (a + b) - c / (c + d))
                 if (a + b) and (c + d) else np.nan, "p": np.nan, "note": ""})
    rr = ((a / (a + b)) / (c / (c + d))) if (a + b) and (c + d) and (c / (c + d)) else np.nan
    rows.append({"family": "effect", "method": "Risk ratio", "stat": rr, "p": np.nan, "note": ""})
    return rows, table


# --------------------------------------------------------------------------
# correlation families
# --------------------------------------------------------------------------


def compare_correlation(x, y, z=None):
    """Correlation of x,y under several conventions.

    z: optional covariate(s) -- Spearman partial correlation is added.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rows = []

    def add(method, stat, p, note=""):
        rows.append({"method": method, "stat": float(stat), "p": float(p), "note": note})

    pr, pp = stats.pearsonr(x, y)
    add("Pearson (linear)", pr, pp)
    sr, sp = stats.spearmanr(x, y)
    add("Spearman (rank)", sr, sp)
    kr, kp = stats.kendalltau(x, y)
    add("Kendall tau-b (rank, tie-aware)", kr, kp)
    lo, hi = _fisher_z_ci(pr, len(x))
    rows.append({"method": "Pearson (Fisher-z 95% CI)", "stat": pr, "p": np.nan,
                 "note": f"[{lo:.3f}, {hi:.3f}]"})
    if z is not None:
        Z = np.asarray(z, float)
        if Z.ndim == 1:
            Z = Z[:, None]
        ok = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
        xx, yy, ZZ = x[ok], y[ok], Z[ok]
        rx, ry = stats.rankdata(xx), stats.rankdata(yy)
        RZ = np.column_stack([stats.rankdata(ZZ[:, j]) for j in range(ZZ.shape[1])])

        def resid(v, C):
            C = np.column_stack([np.ones(len(C)), C])
            b, *_ = np.linalg.lstsq(C, v, rcond=None)
            return v - C @ b
        ex, ey = resid(rx, RZ), resid(ry, RZ)
        ppr, ppp = stats.pearsonr(ex, ey)
        add("Partial Spearman (covariate-adjusted)", ppr, ppp,
            "adjusted for: " + ",".join(f"z{j}" for j in range(ZZ.shape[1])))
    return rows


def _fisher_z_ci(r, n, alpha=0.05):
    if n < 4 or not np.isfinite(r) or abs(r) >= 1:
        return np.nan, np.nan
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    zc = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se))


def r2_from_r(r):
    return float(r) ** 2


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values (NaN-safe)."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    out = np.full(len(p), np.nan)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    tmp = np.empty(n)
    tmp[order] = q
    out[ok] = tmp
    return out
