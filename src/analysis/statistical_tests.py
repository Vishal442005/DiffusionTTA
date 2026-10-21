"""
Paper 2: Statistical Analysis & Significance Testing Suite
=========================================================
Statistical testing engine for Generative Augmentation and Test-Time Adaptation:
- Wilcoxon Signed-Rank Test (non-parametric comparison across subjects)
- Paired Student's t-test with Shapiro-Wilk normality checking
- Effect sizes: Cohen's d (parametric) and Cliff's Delta (non-parametric)
- Multiple testing adjustment: Holm-Bonferroni & Benjamini-Hochberg (FDR)
- Bayesian Evidence Factor (JZS BF10 approximation via BIC)
- Automated publication-grade LaTeX and CSV table generation for IEEE TPAMI / NeurIPS
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import scipy.stats as stats
from dataclasses import dataclass


@dataclass
class StatisticalResult:
    test_name: str
    statistic: float
    p_value: float
    shapiro_w: float
    shapiro_p: float
    is_normal: bool
    t_statistic: float
    t_p_value: float
    wilcoxon_w: float
    wilcoxon_p: float
    cohens_d: float
    cohens_d_magnitude: str
    cliffs_delta: float
    cliffs_delta_magnitude: str
    bayes_factor_10: Optional[float]
    bf_evidence_category: str
    p_holm_bonferroni: float = 1.0
    p_benjamini_hochberg: float = 1.0
    significant_p05: bool = False
    significant_p01: bool = False
    significant_p001: bool = False

    def summary(self) -> str:
        sig = "***" if self.significant_p001 else "**" if self.significant_p01 else "*" if self.significant_p05 else "n.s."
        bf_str = f", BF10={self.bayes_factor_10:.2f} ({self.bf_evidence_category})" if self.bayes_factor_10 is not None else ""
        return (
            f"[{self.test_name}] stat={self.statistic:.4f}, p_raw={self.p_value:.4e}, "
            f"p_HB={self.p_holm_bonferroni:.4e} ({sig}), Cohen's d={self.cohens_d:.3f} ({self.cohens_d_magnitude}), "
            f"Cliff's Delta={self.cliffs_delta:.3f} ({self.cliffs_delta_magnitude}){bf_str}"
        )


def compute_cliffs_delta(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
    """
    Computes Cliff's Delta non-parametric effect size.
    Thresholds: |delta| < 0.147 negligible, < 0.330 small, < 0.474 medium, >= 0.474 large.
    """
    diff = x - y
    n_pos = np.sum(diff > 0)
    n_neg = np.sum(diff < 0)
    delta = (n_pos - n_neg) / len(diff) if len(diff) > 0 else 0.0
    abs_d = abs(delta)
    mag = "negligible" if abs_d < 0.147 else "small" if abs_d < 0.33 else "medium" if abs_d < 0.474 else "large"
    return float(delta), mag


def compute_cohens_d(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
    """
    Computes Cohen's d_z for paired samples: mean(diff) / std(diff, ddof=1).
    Thresholds: |d| < 0.2 negligible, < 0.5 small, < 0.8 medium, >= 0.8 large.
    """
    diff = x - y
    std_diff = np.std(diff, ddof=1)
    if std_diff == 0:
        return 0.0, "negligible"
    d = float(np.mean(diff) / std_diff)
    abs_d = abs(d)
    mag = "negligible" if abs_d < 0.2 else "small" if abs_d < 0.5 else "medium" if abs_d < 0.8 else "large"
    return d, mag


def approximate_bayes_factor_paired(t_stat: float, n: int) -> Tuple[float, str]:
    """
    Approximates the JZS Bayes Factor (BF10) for a paired t-test using BIC
    (Wagenmakers 2007, Rouder et al. 2009).
    Evidence categories (Jeffreys' scale):
      >100: Extreme, 30-100: Very Strong, 10-30: Strong, 3-10: Substantial, 1-3: Anecdotal, <1: Favors H0.
    """
    if n <= 1:
        return 1.0, "Anecdotal"
    df = n - 1
    r2 = (t_stat ** 2) / (t_stat ** 2 + df)
    bic_diff = n * np.log(max(1.0 - r2, 1e-8)) + np.log(n)
    bf10 = float(np.clip(np.exp(-0.5 * bic_diff), 1e-6, 1e8))

    if bf10 > 100:
        cat = "Extreme evidence"
    elif bf10 > 30:
        cat = "Very strong evidence"
    elif bf10 > 10:
        cat = "Strong evidence"
    elif bf10 > 3:
        cat = "Substantial evidence"
    elif bf10 > 1:
        cat = "Anecdotal evidence"
    else:
        cat = "Favors null hypothesis"

    return bf10, cat


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """
    Applies Holm-Bonferroni step-down procedure to control Family-Wise Error Rate (FWER).
    p_adj_(i) = min(1.0, max_{k <= i} (m - k + 1) * p_(k))
    """
    m = len(p_values)
    if m == 0:
        return []
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted_indexed = []
    current_max = 0.0

    for rank, (orig_idx, p_val) in enumerate(indexed_p):
        multiplier = m - rank
        adj_p = min(1.0, multiplier * p_val)
        current_max = max(current_max, adj_p)
        adjusted_indexed.append((orig_idx, min(1.0, current_max)))

    # Restore original ordering
    adjusted_indexed.sort(key=lambda x: x[0])
    return [p for _, p in adjusted_indexed]


def benjamini_hochberg_fdr(p_values: List[float]) -> List[float]:
    """
    Applies Benjamini-Hochberg (BH) step-up procedure to control False Discovery Rate (FDR).
    """
    m = len(p_values)
    if m == 0:
        return []
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1], reverse=True)
    adjusted_indexed = []
    current_min = 1.0

    for rank_from_end, (orig_idx, p_val) in enumerate(indexed_p):
        rank = m - rank_from_end  # 1-based rank
        adj_p = min(1.0, (m / rank) * p_val)
        current_min = min(current_min, adj_p)
        adjusted_indexed.append((orig_idx, current_min))

    # Restore original ordering
    adjusted_indexed.sort(key=lambda x: x[0])
    return [p for _, p in adjusted_indexed]


def compare_paired_distributions(
    scores_proposed: Union[List[float], np.ndarray],
    scores_baseline: Union[List[float], np.ndarray],
    alternative: str = "greater"
) -> StatisticalResult:
    """
    Performs comprehensive paired distribution analysis:
    1. Shapiro-Wilk normality test on differences.
    2. Paired Student's t-test.
    3. Wilcoxon signed-rank test.
    4. Cohen's d & Cliff's Delta effect sizes.
    5. JZS Bayes Factor (BF10) with evidence classification.
    """
    x = np.asarray(scores_proposed, dtype=np.float64)
    y = np.asarray(scores_baseline, dtype=np.float64)
    n = len(x)
    diff = x - y

    # 1. Shapiro-Wilk Normality Test
    if n >= 3:
        w_stat, p_norm = stats.shapiro(diff)
        w_stat, p_norm = float(w_stat), float(p_norm)
        is_normal = (p_norm >= 0.05)
    else:
        w_stat, p_norm = 1.0, 1.0
        is_normal = True

    # 2. Paired t-test
    if np.all(diff == 0):
        t_stat, t_pval = 0.0, 1.0
    else:
        t_res = stats.ttest_rel(x, y, alternative=alternative)
        t_stat, t_pval = float(t_res.statistic), float(t_res.pvalue)

    # 3. Wilcoxon signed-rank test
    if np.all(diff == 0):
        w_res_stat, w_pval = 0.0, 1.0
    else:
        try:
            w_res = stats.wilcoxon(x, y, alternative=alternative)
            w_res_stat, w_pval = float(w_res.statistic), float(w_res.pvalue)
        except Exception:
            w_res_stat, w_pval = 0.0, 1.0

    # 4. Effect Sizes
    cohens_d_val, cohens_mag = compute_cohens_d(x, y)
    cliffs_d_val, cliffs_mag = compute_cliffs_delta(x, y)

    # 5. Bayes Factor
    bf10, bf_cat = approximate_bayes_factor_paired(t_stat, n)

    # Select primary test based on normality & sample size
    if is_normal and n >= 10:
        primary_test = "Paired Student's t-test"
        primary_stat = t_stat
        primary_p = t_pval
    else:
        primary_test = "Wilcoxon Signed-Rank Test"
        primary_stat = w_res_stat
        primary_p = w_pval

    return StatisticalResult(
        test_name=primary_test,
        statistic=primary_stat,
        p_value=primary_p,
        shapiro_w=w_stat,
        shapiro_p=p_norm,
        is_normal=is_normal,
        t_statistic=t_stat,
        t_p_value=t_pval,
        wilcoxon_w=w_res_stat,
        wilcoxon_p=w_pval,
        cohens_d=cohens_d_val,
        cohens_d_magnitude=cohens_mag,
        cliffs_delta=cliffs_d_val,
        cliffs_delta_magnitude=cliffs_mag,
        bayes_factor_10=bf10,
        bf_evidence_category=bf_cat,
        significant_p05=(primary_p < 0.05),
        significant_p01=(primary_p < 0.01),
        significant_p001=(primary_p < 0.001)
    )


def run_comprehensive_statistical_battery(
    proposed_scores: Union[List[float], np.ndarray],
    baselines_dict: Dict[str, Union[List[float], np.ndarray]],
    category_map: Optional[Dict[str, str]] = None,
    alternative: str = "greater"
) -> Tuple[pd.DataFrame, str]:
    """
    Executes the complete statistical test suite across all baselines:
    - Calculates primary comparisons.
    - Applies Holm-Bonferroni and Benjamini-Hochberg FDR adjustments across hypotheses.
    - Returns a publication-ready Pandas DataFrame and LaTeX table code.
    """
    proposed = np.asarray(proposed_scores, dtype=np.float64)
    baseline_names = list(baselines_dict.keys())
    results: List[StatisticalResult] = []

    for name in baseline_names:
        base = np.asarray(baselines_dict[name], dtype=np.float64)
        res = compare_paired_distributions(proposed, base, alternative=alternative)
        results.append(res)

    # Multi-hypothesis corrections
    raw_p_values = [r.p_value for r in results]
    hb_p_values = holm_bonferroni_correction(raw_p_values)
    fdr_p_values = benjamini_hochberg_fdr(raw_p_values)

    records = []
    latex_rows = []

    for i, name in enumerate(baseline_names):
        base = np.asarray(baselines_dict[name], dtype=np.float64)
        res = results[i]
        res.p_holm_bonferroni = hb_p_values[i]
        res.p_benjamini_hochberg = fdr_p_values[i]
        res.significant_p05 = (res.p_holm_bonferroni < 0.05)
        res.significant_p01 = (res.p_holm_bonferroni < 0.01)
        res.significant_p001 = (res.p_holm_bonferroni < 0.001)

        diff_mean = np.mean(proposed) - np.mean(base)
        sig_str = "***" if res.significant_p001 else "**" if res.significant_p01 else "*" if res.significant_p05 else "n.s."
        sig_star = r"$^{***}$" if res.significant_p001 else r"$^{**}$" if res.significant_p01 else r"$^{*}$" if res.significant_p05 else ""

        cat = category_map.get(name, "Baseline") if category_map else "Baseline"

        records.append({
            "Category": cat,
            "Baseline Model": name,
            "Baseline Acc (%)": f"{np.mean(base)*100:.2f} ± {np.std(base)*100:.2f}",
            "Proposed Acc (%)": f"{np.mean(proposed)*100:.2f} ± {np.std(proposed)*100:.2f}",
            "Gain (Delta)": f"{diff_mean*100:+.2f}%",
            "Shapiro-Wilk p": f"{res.shapiro_p:.4f}",
            "Normality": "Normal" if res.is_normal else "Non-normal",
            "Primary Test": res.test_name,
            "Test Stat": f"{res.statistic:.3f}",
            "Raw p-value": f"{res.p_value:.4e}",
            "Holm-Bonferroni p": f"{res.p_holm_bonferroni:.4e}",
            "FDR q": f"{res.p_benjamini_hochberg:.4e}",
            "Cohen's d": f"{res.cohens_d:.3f} ({res.cohens_d_magnitude})",
            "Cliff's Delta": f"{res.cliffs_delta:.3f} ({res.cliffs_delta_magnitude})",
            "BF10": f"{res.bayes_factor_10:.2f}" if res.bayes_factor_10 is not None else "N/A",
            "Bayesian Evidence": res.bf_evidence_category,
            "Significance": sig_str
        })

        bf_fmt = f"{res.bayes_factor_10:.1f}" if res.bayes_factor_10 is not None and res.bayes_factor_10 < 1000 else ">1000" if res.bayes_factor_10 is not None else "---"
        latex_rows.append(
            f"{name} & {np.mean(base)*100:.2f} $\\pm$ {np.std(base)*100:.2f} & "
            f"\\textbf{{{np.mean(proposed)*100:.2f} $\\pm$ {np.std(proposed)*100:.2f}}} & "
            f"{diff_mean*100:+.2f}\\% & {res.p_holm_bonferroni:.4f}{sig_star} & "
            f"{res.cohens_d:.2f} ({res.cohens_d_magnitude}) & {bf_fmt} \\\\"
        )

    df = pd.DataFrame(records)

    latex_table = (
        r"\begin{table*}[t]" + "\n"
        r"\centering" + "\n"
        r"\small" + "\n"
        r"\caption{Comprehensive Statistical Benchmarking on Held-Out Test Subjects ($N=10$). "
        r"Hypothesis testing uses Wilcoxon signed-rank and paired Student's $t$-tests conditioned on Shapiro-Wilk normality, "
        r"with Holm-Bonferroni FWER multiple testing correction ($p_{\text{HB}}$), parametric Cohen's $d_z$ effect size, "
        r"and JZS Bayes Factor ($BF_{10}$). ($^{***}: p < 0.001$, $^{**}: p < 0.01$, $^{*}: p < 0.05$)}" + "\n"
        r"\label{tab:paper2_statistical_benchmarks}" + "\n"
        r"\begin{tabular}{lcccccc}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Baseline Model} & \textbf{Baseline Acc (\%)} & \textbf{Proposed Acc (\%)} & \textbf{Gain ($\Delta$)} & \textbf{$p_{\text{HB}}$} & \textbf{Cohen's $d$} & \textbf{$BF_{10}$} \\" + "\n"
        r"\midrule" + "\n"
        + "\n".join(latex_rows) + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table*}"
    )

    return df, latex_table


# Alias for cross-repo consistency
compare_models = compare_paired_distributions
