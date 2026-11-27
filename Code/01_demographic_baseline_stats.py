# File: 01_demographic_baseline_stats.py
# Version: v1.0.1
# Last Updated: Iteration 2
#
# CHANGELOG
# [v1.0.1] - Fixed: M (SD) keeps trailing zeros; consistent 'p = .xxx'/'p < .001' in post-hoc text;
#            missing-data warning counts only participants in that block's groups.
# [v1.0.0] - Initial implementation: group comparisons of demographic and
#            baseline (cognition, speech) variables for 2 or 3 groups, using
#            the helper functions in group_stats.py. Writes manuscript-style
#            tables (CSV) and a log file.
"""
WHAT THIS SCRIPT DOES
---------------------
Reads one spreadsheet (one row per participant) and, for each "block" of
variables you list in the CONFIG section below, it:

  1. Describes each group
       continuous variables  -> M (SD) and n per group
       categorical variables -> n (%) per group
  2. Tests for group differences
       continuous, 2 groups  -> Welch's independent-samples t-test + Cohen's d
       continuous, 3+ groups -> one-way ANOVA + eta-squared; if ANOVA p < alpha,
                                Sidak-corrected pairwise post-hoc tests
       categorical           -> chi-square test of independence + Cramer's V;
                                exact p (Fisher / Monte Carlo) if any expected
                                count < 5; if significant with 3+ groups,
                                pairwise chi-square/Fisher (Holm-corrected)
                                and adjusted standardised residuals
  3. Saves everything as CSV files (one manuscript-style table per block, plus
     the full statistics tables) and writes a log file.

HOW TO RUN
----------
  1. Put this file, group_stats.py and your data file in the same project.
  2. Edit the CONFIG section (file name, group column, variable names).
  3. In a terminal:   python 01_demographic_baseline_stats.py
     or in a notebook:  %run 01_demographic_baseline_stats.py
  4. Look in OUTPUT_DIR for the tables and the .log file.
"""

import sys
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels

# group_stats.py holds the statistical functions (t-test, ANOVA, chi-square...).
# It must sit in the same folder as this script.
from group_stats import (
    start_log, crosstab_with_pct, ttest_by_group,
    anova_with_posthoc, chi2_by_group, pairwise_chi2, chi2_residuals,
)

# =====================================================================
# CONFIG — the only section you should need to edit
# =====================================================================
DATA_FILE = "data/participants.csv"   # .csv or .xlsx, one row per participant
GROUP_COL = "diagnosis"               # column holding the group labels
OUTPUT_DIR = "results/01_demographic_baseline_stats"
ALPHA = 0.05                          # significance threshold

# Each block is one table in the manuscript. A block can use 2 or 3 groups,
# so e.g. demographics can include controls while speech is patients only.
# Group labels must match the values in GROUP_COL exactly (case-sensitive).
# The order of "groups" sets the column order and the direction of mean
# differences (group1 - group2) in the post-hoc tables.
BLOCKS = [
    {
        "name": "demographics",
        "groups": ["Control", "PNFA", "LPA"],
        "continuous": ["age", "education"],
        "categorical": ["sex"],
    },
    {
        "name": "cognition",
        "groups": ["Control", "PNFA", "LPA"],
        "continuous": ["ACE_total"],          # <- replace with your measures
        "categorical": [],
    },
    {
        "name": "speech",
        "groups": ["PNFA", "LPA"],            # 2 groups -> Welch t-tests
        "continuous": ["speech_rate"],        # <- replace with your measures
        "categorical": [],
    },
]
# =====================================================================


def fmt_p(p):
    """Format a p-value the way journals usually want it: '<.001' or '.034'."""
    if pd.isna(p):
        return ""
    return "<.001" if p < 0.001 else f"{p:.3f}".lstrip("0")


def p_eq(p):
    """'= .034' or '< .001' — for writing 'p = .034' / 'p < .001' in text."""
    return "< .001" if p < 0.001 else "= " + fmt_p(p)


def m_sd(x, decimals=2):
    """'M (SD)' with a fixed number of decimals (keeps trailing zeros, e.g. 7.80)."""
    x = x.dropna()
    return f"{x.mean():.{decimals}f} ({x.std(ddof=1):.{decimals}f})"


def load_data(path):
    """Read CSV or Excel into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path.resolve()}")
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def check_block(df, block, log):
    """Stop early with a clear message if a column or group label is wrong."""
    missing_cols = [c for c in block["continuous"] + block["categorical"] if c not in df.columns]
    if missing_cols:
        raise KeyError(f"[{block['name']}] columns not in data file: {missing_cols}")
    missing_groups = [g for g in block["groups"] if g not in set(df[GROUP_COL])]
    if missing_groups:
        raise ValueError(f"[{block['name']}] group labels not found in '{GROUP_COL}': "
                         f"{missing_groups}. Labels present: {sorted(df[GROUP_COL].dropna().unique())}")
    if len(block["groups"]) < 2:
        raise ValueError(f"[{block['name']}] needs at least 2 groups")
    # Missing data: each test uses everyone in the block's groups who has that variable
    d = df[df[GROUP_COL].isin(block["groups"])]
    for var in block["continuous"] + block["categorical"]:
        n_missing = d[var].isna().sum()
        if n_missing:
            log.warning("[%s] %s: %d participant(s) missing; excluded from this variable only",
                        block["name"], var, n_missing)


def continuous_rows(d, block, log):
    """Descriptives + tests for continuous variables. Returns (summary rows, full stats tables)."""
    groups, cont = block["groups"], block["continuous"]
    if not cont:
        return [], {}

    # M (SD) per variable (rows) and group (columns, in the order given in CONFIG)
    desc = pd.DataFrame({g: [m_sd(d.loc[d[GROUP_COL] == g, v]) for v in cont] for g in groups},
                        index=cont)
    n_per = d.groupby(GROUP_COL, observed=True)[cont].count().T[groups]  # n with data, per variable x group
    full = {}
    rows = []

    if len(groups) == 2:
        # ---- 2 groups: Welch t-test ----
        res = ttest_by_group(d, GROUP_COL, cont, groups[0], groups[1])
        full["ttests"] = res
        for _, r in res.iterrows():
            rows.append({
                "variable": r["variable"],
                **{g: desc.loc[r["variable"], g] for g in groups},
                **{f"n_{g}": n_per.loc[r["variable"], g] for g in groups},
                "test": "Welch t",
                "statistic": f"t({r['df']:.1f}) = {r['t']:.2f}",
                "p": fmt_p(r["p"]),
                "effect_size": f"d = {r['cohens_d']:.2f}",
                "post_hoc": "",
            })
    else:
        # ---- 3+ groups: ANOVA, then Sidak post-hoc only where ANOVA p < ALPHA ----
        anova_res, posthoc = anova_with_posthoc(d, GROUP_COL, cont, groups=groups, alpha=ALPHA)
        full["anova"], full["posthoc_sidak"] = anova_res, posthoc
        for _, r in anova_res.iterrows():
            ph = ""
            if r["posthoc_run"]:
                sig = posthoc[(posthoc["variable"] == r["variable"]) & posthoc["significant"]]
                ph = "; ".join(f"{a} vs {b} (p_sidak {p_eq(p)})"
                               for a, b, p in zip(sig["group1"], sig["group2"], sig["p_sidak"])) or "none significant"
            rows.append({
                "variable": r["variable"],
                **{g: desc.loc[r["variable"], g] for g in groups},
                **{f"n_{g}": n_per.loc[r["variable"], g] for g in groups},
                "test": "One-way ANOVA",
                "statistic": f"F({r['df_between']}, {r['df_within']}) = {r['F']:.2f}",
                "p": fmt_p(r["p"]),
                "effect_size": f"eta2 = {r['eta_sq']:.2f}",
                "post_hoc": ph,
            })
            if r["levene_p"] < ALPHA:
                log.warning("[%s] %s: Levene p = %.3f (unequal variances); consider "
                            "pooled=False (Welch post-hocs) for this variable",
                            block["name"], r["variable"], r["levene_p"])
    return rows, full


def categorical_rows(d, block, log):
    """Descriptives + tests for categorical variables. Returns (summary rows, full stats tables)."""
    groups, cats = block["groups"], block["categorical"]
    if not cats:
        return [], {}

    res = chi2_by_group(d, GROUP_COL, cats, groups=groups)
    full = {"chi2": res}
    rows = []
    pw_all, resid_all = [], []

    for _, r in res.iterrows():
        var = r["variable"]
        counts = crosstab_with_pct(d, GROUP_COL, var).reindex(groups)   # 'n (%)' per group x level

        ph = ""
        if r["p_use"] < ALPHA and len(groups) > 2:
            pw = pairwise_chi2(d, GROUP_COL, var, groups=groups)
            pw_all.append(pw)
            resid_all.append(chi2_residuals(d, GROUP_COL, var, groups=groups, alpha=ALPHA))
            sig = pw[pw["significant"]]
            ph = "; ".join(f"{a} vs {b} (p_holm {p_eq(p)})"
                           for a, b, p in zip(sig["group1"], sig["group2"], sig["p_holm"])) or "none significant"

        # one row per category level (e.g. sex: F, sex: M); test results on the first row
        for i, level in enumerate(counts.columns):
            rows.append({
                "variable": f"{var}: {level}, n (%)",
                **{g: counts.loc[g, level] for g in groups},
                **{f"n_{g}": int((d[GROUP_COL] == g).sum() - d.loc[d[GROUP_COL] == g, var].isna().sum())
                   for g in groups},
                "test": r["test"] if i == 0 else "",
                "statistic": f"chi2({r['df']}) = {r['chi2']:.2f}" if i == 0 else "",
                "p": fmt_p(r["p_use"]) if i == 0 else "",
                "effect_size": f"V = {r['cramers_v']:.2f}" if i == 0 else "",
                "post_hoc": ph if i == 0 else "",
            })
    if pw_all:
        full["chi2_pairwise_holm"] = pd.concat(pw_all, ignore_index=True)
        full["chi2_residuals"] = pd.concat(resid_all, ignore_index=True)
    return rows, full


def main():
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    log = start_log(str(out / "01_demographic_baseline_stats.log"))

    # Record software versions: useful for the manuscript's methods/reproducibility
    log.info("Python %s | pandas %s | numpy %s | scipy %s | statsmodels %s | %s",
             platform.python_version(), pd.__version__, np.__version__,
             scipy.__version__, statsmodels.__version__, platform.platform())

    df = load_data(DATA_FILE)
    log.info("Loaded %s: %d rows, %d columns", DATA_FILE, *df.shape)
    if GROUP_COL not in df.columns:
        raise KeyError(f"Group column '{GROUP_COL}' not in data file")
    n_nogroup = df[GROUP_COL].isna().sum()
    if n_nogroup:
        log.warning("%d row(s) have no group label and are dropped", n_nogroup)
    df = df.dropna(subset=[GROUP_COL])
    log.info("Group sizes: %s", df[GROUP_COL].value_counts().to_dict())

    for block in BLOCKS:
        name = block["name"]
        log.info("===== Block '%s': groups %s =====", name, block["groups"])
        check_block(df, block, log)

        d = df[df[GROUP_COL].isin(block["groups"])].copy()
        # make the group column ordered, so tables follow the order in CONFIG
        d[GROUP_COL] = pd.Categorical(d[GROUP_COL], categories=block["groups"], ordered=True)

        c_rows, c_full = continuous_rows(d, block, log)
        k_rows, k_full = categorical_rows(d, block, log)

        # 1) manuscript-style summary table
        summary = pd.DataFrame(c_rows + k_rows)
        summary.to_csv(out / f"{name}_table.csv", index=False)
        log.info("Saved %s", out / f"{name}_table.csv")

        # 2) full statistics tables (every number, for checking / supplementary)
        for tag, tab in {**c_full, **k_full}.items():
            if tab is not None and not tab.empty:
                path = out / f"{name}_{tag}_full.csv"
                tab.to_csv(path, index=False)
                log.info("Saved %s", path)

    log.info("Finished without errors.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # write the full error to the log so you can see where it failed
        import logging
        logging.getLogger("group_stats").exception("Script stopped with an error")
        sys.exit(1)
