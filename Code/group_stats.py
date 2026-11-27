# File: group_stats.py
# Version: v1.0.0
# Last Updated: Iteration 1
#
# CHANGELOG
# [v1.0.0] - Initial file: extracted from cell 0 of Pipeline_stats.ipynb
#            (newest of the 5 copies in the notebook). Statistics unchanged.
#            Added: start_log() + logging calls so each run leaves a log file.
#            Fixed: groups= now accepts any list-like (e.g. numpy array), not only lists.
#            Fixed: _fisher_rxc docstring (works for any r x c table, not only 2 x c).
"""
group_stats.py — reusable group-comparison helpers for clinical cohort tables.

Put this file in the same folder as your notebook, then:
    from group_stats import *

Every function takes:
    df     : a pandas DataFrame, one row per participant
    group  : name of the grouping column (e.g. "diagnosis")
and returns a tidy DataFrame, so results can be concatenated, exported
(df.to_csv / to_excel) or pasted into a Table 1.

Dependencies: pandas, numpy, scipy, statsmodels
    pip install pandas numpy scipy statsmodels
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

import logging

# ---------------------------------------------------------------------
# 0. Logging
#    Every main function writes a line to the log, so you can see what ran
#    and where it stopped. Call start_log() once at the top of your notebook.
# ---------------------------------------------------------------------
log = logging.getLogger("group_stats")
log.addHandler(logging.NullHandler())   # silent until start_log() is called


def start_log(path="group_stats.log", level=logging.INFO):
    """
    Send this module's messages to a log file (appends) and to the screen.
        from group_stats import *
        start_log("logs/table1_stats.log")
    """
    log.setLevel(level)
    log.handlers.clear()                 # avoid duplicate lines if called twice
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for h in (logging.FileHandler(path), logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.info("Log started -> %s", path)
    return log


def _groups(df, group, groups):
    """Return the groups to use, as a plain list (all groups, sorted, if None)."""
    return list(groups) if groups is not None else sorted(df[group].dropna().unique())



# ---------------------------------------------------------------------
# 1. Descriptive statistics
# ---------------------------------------------------------------------
def describe_by_group(df, group, variables, decimals=2):
    """n, mean, SD (plus a formatted 'M (SD)' column) for each variable per group."""
    log.info("describe_by_group: %d variable(s) by %s", len(variables), group)
    rows = []
    for var in variables:
        summary = df.groupby(group, observed=True)[var].agg(["count", "mean", "std", "min", "max"])
        summary = summary.rename(columns={"count": "n", "std": "sd"})
        summary["M (SD)"] = (summary["mean"].round(decimals).astype(str) + " ("
                             + summary["sd"].round(decimals).astype(str) + ")")
        summary.insert(0, "variable", var)
        rows.append(summary.reset_index())
    return pd.concat(rows, ignore_index=True).round(decimals)


def table1_wide(df, group, variables, decimals=2):
    """Same as above but pivoted: variables as rows, groups as columns (Table 1 style)."""
    long = describe_by_group(df, group, variables, decimals)
    return long.pivot(index="variable", columns=group, values="M (SD)").reindex(variables)


# ---------------------------------------------------------------------
# 2. Independent-samples t-test (two groups)
# ---------------------------------------------------------------------
def cohens_d(x, y):
    """Cohen's d using the pooled SD."""
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return (x.mean() - y.mean()) / pooled


def ttest_by_group(df, group, variables, g1, g2, equal_var=False):
    """
    Compare g1 vs g2 on each variable.
    equal_var=False -> Welch's t-test (default; safer when variances/n differ).
    Also reports Levene's test so you can see whether variances differ.
    """
    log.info("ttest_by_group: %s vs %s on %d variable(s)", g1, g2, len(variables))
    rows = []
    for var in variables:
        x = df.loc[df[group] == g1, var].dropna()
        y = df.loc[df[group] == g2, var].dropna()
        t, p = stats.ttest_ind(x, y, equal_var=equal_var)
        lev_p = stats.levene(x, y).pvalue
        # Welch-Satterthwaite df (or classic df if equal_var)
        if equal_var:
            dof = len(x) + len(y) - 2
        else:
            vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
            dof = (vx + vy) ** 2 / (vx**2 / (len(x) - 1) + vy**2 / (len(y) - 1))
        rows.append({
            "variable": var, "comparison": f"{g1} vs {g2}",
            f"n_{g1}": len(x), f"n_{g2}": len(y),
            "t": t, "df": dof, "p": p,
            "cohens_d": cohens_d(x, y), "levene_p": lev_p,
            "test": "Student" if equal_var else "Welch",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 3. One-way ANOVA (>2 groups) + post-hoc
#    sidak_posthoc  -> default post-hoc (Sidak-corrected pairwise t-tests)
#    tukey_posthoc  -> alternative, kept for reference
# ---------------------------------------------------------------------
def anova_by_group(df, group, variables, groups=None):
    """
    One-way ANOVA per variable. Returns F, df, p and eta-squared.
    groups: optional list to restrict/order which groups are included.
    """
    log.info("anova_by_group: %d variable(s) by %s", len(variables), group)
    groups = _groups(df, group, groups)
    rows = []
    for var in variables:
        samples = [df.loc[df[group] == g, var].dropna() for g in groups]
        F, p = stats.f_oneway(*samples)
        # eta^2 = SS_between / SS_total
        allv = pd.concat(samples)
        ss_between = sum(len(s) * (s.mean() - allv.mean()) ** 2 for s in samples)
        ss_total = ((allv - allv.mean()) ** 2).sum()
        rows.append({
            "variable": var, "groups": " / ".join(map(str, groups)),
            "F": F, "df_between": len(groups) - 1,
            "df_within": len(allv) - len(groups), "p": p,
            "eta_sq": ss_between / ss_total,
            "levene_p": stats.levene(*samples).pvalue,
        })
    return pd.DataFrame(rows)


def sidak_posthoc(df, group, var, groups=None, pooled=True, alpha=0.05, d_from="anova"):
    """
    Pairwise post-hoc comparisons with Sidak correction, for one variable.

    pooled=True  -> SPSS/GraphPad-style: each pair is a t-test using the ANOVA's
                    pooled error term (MS_within) and df_within = N - k.
    pooled=False -> Welch t-test per pair (use if Levene's test says variances differ).

    Sidak-adjusted p = 1 - (1 - p)^m, where m = number of pairwise comparisons.
    The CI for the mean difference uses the Sidak-adjusted alpha: 1 - (1 - alpha)^(1/m).

    var can be one column name or a list of names (results are stacked).
    Note: the Sidak correction is applied within each variable (across its pairs).

    d_from: standardiser for Cohen's d
        "anova" (default) -> d = mean diff / sqrt(MS_within), the ANOVA error term
                             pooled across ALL groups (as in JASP's post-hoc table;
                             consistent with the pooled Sidak test).
        "pair"            -> d = mean diff / pooled SD of the two groups only
                             (as in SPSS/JASP independent-samples t-test output).
    d is not adjusted for multiple comparisons.
    """
    if d_from not in ("anova", "pair"):
        raise ValueError("d_from must be 'anova' or 'pair'")
    if isinstance(var, (list, tuple, pd.Index)):
        return pd.concat([sidak_posthoc(df, group, v, groups, pooled, alpha, d_from) for v in var],
                         ignore_index=True)
    groups = _groups(df, group, groups)
    log.info("sidak_posthoc: %s (pooled=%s, d_from=%s)", var, pooled, d_from)
    samples = {g: df.loc[df[group] == g, var].dropna() for g in groups}
    pairs = [(a, b) for i, a in enumerate(groups) for b in groups[i + 1:]]
    m = len(pairs)
    alpha_sidak = 1 - (1 - alpha) ** (1 / m)

    # pooled error term from the one-way ANOVA
    N, k = sum(len(s) for s in samples.values()), len(groups)
    ms_within = sum(((s - s.mean()) ** 2).sum() for s in samples.values()) / (N - k)

    rows = []
    for a, b in pairs:
        x, y = samples[a], samples[b]
        diff = x.mean() - y.mean()
        if pooled:
            se = np.sqrt(ms_within * (1 / len(x) + 1 / len(y)))
            dof = N - k
        else:
            vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
            se = np.sqrt(vx + vy)
            dof = (vx + vy) ** 2 / (vx**2 / (len(x) - 1) + vy**2 / (len(y) - 1))
        t = diff / se
        p = 2 * stats.t.sf(abs(t), dof)
        p_sidak = min(1.0, 1 - (1 - p) ** m)
        half_ci = stats.t.ppf(1 - alpha_sidak / 2, dof) * se
        rows.append({
            "variable": var, "group1": a, "group2": b, "mean_diff": diff,
            "se": se, "t": t, "df": dof, "p_uncorrected": p, "p_sidak": p_sidak,
            "ci_low": diff - half_ci, "ci_high": diff + half_ci,
            "cohens_d": diff / np.sqrt(ms_within) if d_from == "anova" else cohens_d(x, y),
            "d_from": d_from, "significant": p_sidak < alpha,
        })
    return pd.DataFrame(rows)


def anova_with_posthoc(df, group, variables, groups=None, alpha=0.05,
                       method="sidak", pooled=True, d_from="anova"):
    """
    Run one-way ANOVAs, then post-hoc tests ONLY for variables with ANOVA p < alpha.

    Returns two tables:
        anova_res : one row per variable (all variables), with a 'posthoc_run' column
        posthoc   : pairwise results for the significant variables only,
                    with the ANOVA F and p attached to each row for reference

    method: 'sidak' (default) or 'tukey'.
    pooled: passed to sidak_posthoc (False = Welch per pair).
    d_from: passed to sidak_posthoc ("anova" = sqrt(MS_within), "pair" = two-group pooled SD).

    Usage:
        anova_res, posthoc = anova_with_posthoc(df, "diagnosis", variables, groups=ORDER)
    """
    log.info("anova_with_posthoc: method=%s, pooled=%s, d_from=%s", method, pooled, d_from)
    anova_res = anova_by_group(df, group, variables, groups=groups)
    anova_res["posthoc_run"] = anova_res["p"] < alpha
    sig_vars = anova_res.loc[anova_res["posthoc_run"], "variable"].tolist()

    if not sig_vars:
        print(f"No ANOVA reached p < {alpha}; no post-hoc tests run.")
        log.warning("No ANOVA reached p < %s; no post-hoc tests run.", alpha)
        return anova_res, pd.DataFrame()

    if method == "sidak":
        posthoc = sidak_posthoc(df, group, sig_vars, groups=groups, pooled=pooled, alpha=alpha,
                                d_from=d_from)
    elif method == "tukey":
        posthoc = tukey_posthoc(df, group, sig_vars, groups=groups)
    else:
        raise ValueError("method must be 'sidak' or 'tukey'")

    anova_cols = anova_res[["variable", "F", "p"]].rename(columns={"F": "anova_F", "p": "anova_p"})
    posthoc = anova_cols.merge(posthoc, on="variable", how="right")
    return anova_res, posthoc


def tukey_posthoc(df, group, var, groups=None):
    """ALTERNATIVE to sidak_posthoc: Tukey HSD pairwise comparisons.
    var can be one column name or a list of names (results are stacked)."""
    log.info("tukey_posthoc: %s", var)
    if isinstance(var, (list, tuple, pd.Index)):
        return pd.concat([tukey_posthoc(df, group, v, groups) for v in var], ignore_index=True)
    d = df[[group, var]].dropna()
    if groups is not None:
        d = d[d[group].isin(groups)]
    res = pairwise_tukeyhsd(d[var], d[group])
    out = pd.DataFrame(res.summary().data[1:], columns=res.summary().data[0])
    out.insert(0, "variable", var)
    return out


# ---------------------------------------------------------------------
# 4. Chi-square test of independence (categorical variables)
# ---------------------------------------------------------------------
def chi2_by_group(df, group, cat_vars, groups=None, min_expected=5):
    """
    Omnibus chi-square test of independence between group and each categorical variable.
    Reports Cramer's V and the smallest expected cell count.

    If any expected count is < min_expected, the chi-square approximation is
    unreliable, so an exact p is also computed ('exact_p'): Fisher's exact test
    for 2x2 tables, or a seeded Monte Carlo exact test for larger tables.
    'p_use' is the p-value to report (exact when needed, otherwise chi-square),
    and 'test' says which one it is.
    """
    log.info("chi2_by_group: %d variable(s) by %s", len(cat_vars), group)
    d = df if groups is None else df[df[group].isin(groups)]
    rows = []
    for var in cat_vars:
        table = pd.crosstab(d[group], d[var])
        table = table.loc[:, table.sum() > 0]
        chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
        n = table.values.sum()
        exact_p, test = np.nan, "Chi-square"
        if expected.min() < min_expected:
            exact_p, test = _fisher_rxc(table.values)
        rows.append({
            "variable": var, "chi2": chi2, "df": dof, "p_chi2": p, "n": n,
            "cramers_v": np.sqrt(chi2 / (n * (min(table.shape) - 1))),
            "min_expected": expected.min(), "exact_p": exact_p,
            "test": test, "p_use": p if np.isnan(exact_p) else exact_p,
        })
    return pd.DataFrame(rows)


def _fisher_rxc(table, n_sim=20000, seed=0):
    """
    Small-sample test for an r x c table.
    2 x 2  -> Fisher's exact test (exact p).
    other  -> Monte Carlo permutation test of the chi-square statistic, with a
              fixed seed so the p-value is identical on every run (equivalent to
              R's chisq.test(simulate.p.value=TRUE)). scipy's own r x c Fisher also
              simulates, but its default is unseeded, so its p changes between runs.
    """
    table = np.asarray(table)
    if table.shape == (2, 2):
        return stats.fisher_exact(table).pvalue, "Fisher exact"
    else:
        rng = np.random.default_rng(seed)
        rows = np.repeat(np.arange(table.shape[0]), table.sum(axis=1))
        cols = np.repeat(np.arange(table.shape[1]), table.sum(axis=0))

        def chi2_stat(t):
            e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / t.sum()
            return ((t - e) ** 2 / e).sum()

        obs = chi2_stat(table)
        hits = 0
        for _ in range(n_sim):
            sim = np.zeros_like(table)
            np.add.at(sim, (rows, rng.permutation(cols)), 1)
            hits += chi2_stat(sim) >= obs - 1e-12
        return (hits + 1) / (n_sim + 1), f"Monte Carlo exact ({n_sim} sims)"


def pairwise_chi2(df, group, var, groups=None, correction="holm", min_expected=5):
    """
    Post-hoc for a significant omnibus chi-square: split the table into every
    pair of groups (A vs B, A vs C, ...), run a 2 x c chi-square on each, or
    Fisher's exact test when any expected count is < min_expected.
    p-values are then corrected across the pairs.

    correction: 'holm' (default; controls familywise error, less conservative
                than Bonferroni), 'bonferroni', or 'fdr_bh' (Benjamini-Hochberg).
    """
    log.info("pairwise_chi2: %s (correction=%s)", var, correction)
    groups = _groups(df, group, groups)
    rows = []
    for i, a in enumerate(groups):
        for b in groups[i + 1:]:
            d = df[df[group].isin([a, b])]
            table = pd.crosstab(d[group], d[var])
            table = table.loc[:, table.sum() > 0]          # drop empty categories
            chi2, p_chi, dof, expected = stats.chi2_contingency(table, correction=False)
            n = table.values.sum()
            if expected.min() < min_expected:
                p, test = _fisher_rxc(table.values)
            else:
                p, test = p_chi, "Chi-square"
            rows.append({
                "variable": var, "group1": a, "group2": b, "n": n,
                "chi2": chi2, "df": dof,
                "cramers_v": np.sqrt(chi2 / (n * (min(table.shape) - 1))),
                "min_expected": expected.min(), "test": test, "p_uncorrected": p,
            })
    out = pd.DataFrame(rows)
    reject, p_adj, _, _ = multipletests(out["p_uncorrected"], method=correction)
    out[f"p_{correction}"] = p_adj
    out["significant"] = reject
    return out


def chi2_residuals(df, group, var, groups=None, alpha=0.05, correct=True):
    """
    Cell-wise post-hoc for a significant chi-square: adjusted standardized
    residuals (Haberman, 1973). Shows WHICH cells are over- or under-represented,
    using the whole table (no splitting into pairs).

        r_ij = (O_ij - E_ij) / sqrt( E_ij * (1 - R_i/N) * (1 - C_j/N) )

    Under the null each r_ij is approx. standard normal, so |r| > 1.96 ~ p < .05.
    correct=True applies a Bonferroni correction across the cells
    (Beasley & Schumacker, 1995): critical z = z_{1 - alpha/(2 * m)}.
    m = number of cells (r x c), except when the variable (or the group) has only
    2 levels: then each row's (or column's) two residuals are mirror images
    (+r / -r), so only the other dimension counts as separate tests.

    Returns a long table (one row per cell) with observed, expected, residual,
    two-sided p, Bonferroni p, and direction ('more'/'fewer' than expected).
    """
    log.info("chi2_residuals: %s", var)
    d = df if groups is None else df[df[group].isin(groups)]
    table = pd.crosstab(d[group], d[var])
    if groups is not None:
        table = table.reindex([g for g in groups if g in table.index])
    table = table.loc[:, table.sum() > 0]
    O = table.values.astype(float)
    N = O.sum()
    R = O.sum(axis=1, keepdims=True)
    C = O.sum(axis=0, keepdims=True)
    E = R * C / N
    resid = (O - E) / np.sqrt(E * (1 - R / N) * (1 - C / N))
    r_, c_ = O.shape
    m = r_ if c_ == 2 else (c_ if r_ == 2 else O.size)   # number of non-redundant tests
    if r_ == 2 and c_ == 2:
        m = 1
    p = 2 * stats.norm.sf(np.abs(resid))
    p_bonf = np.minimum(1, p * m)
    crit = stats.norm.ppf(1 - alpha / (2 * m)) if correct else stats.norm.ppf(1 - alpha / 2)

    rows = []
    for i, g in enumerate(table.index):
        for j, cat in enumerate(table.columns):
            rows.append({
                "variable": var, group: g, "category": cat,
                "observed": int(O[i, j]), "expected": E[i, j],
                "adj_residual": resid[i, j], "p": p[i, j], "p_bonferroni": p_bonf[i, j],
                "critical_z": crit, "n_tests": m,
                "significant": abs(resid[i, j]) > crit,
                "direction": "more than expected" if resid[i, j] > 0 else "fewer than expected",
            })
    return pd.DataFrame(rows)


def crosstab_with_pct(df, group, var):
    """Counts and row percentages, e.g. 'n (%)' female per group, for Table 1."""
    counts = pd.crosstab(df[group], df[var])
    pct = pd.crosstab(df[group], df[var], normalize="index") * 100
    return counts.astype(str) + " (" + pct.round(1).astype(str) + "%)"


# ---------------------------------------------------------------------
# 5. Multiple-comparison correction (optional, for any results table)
# ---------------------------------------------------------------------
def add_fdr(results, p_col="p", method="fdr_bh"):
    """Adds a corrected p column.
    method: 'fdr_bh' (Benjamini-Hochberg), 'holm', 'bonferroni', 'sidak', 'holm-sidak'."""
    log.info("add_fdr: column %s, method %s", p_col, method)
    results = results.copy()
    results[f"{p_col}_{method}"] = multipletests(results[p_col], method=method)[1]
    return results