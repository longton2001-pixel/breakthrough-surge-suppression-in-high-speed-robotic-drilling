"""
Two-level damping sensitivity analysis for the breakthrough surge model.

The analysis checks whether the validation ranking remains stable when the
calibrated damping factor is perturbed globally or posture by posture.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ETA0 = 0.3845
CONSTANT_ETA_MULTIPLIERS = (0.8, 0.9, 1.0, 1.1, 1.2)
PERTURBATION_LEVELS = (0.10, 0.20, 0.30)
N_TRIALS = 1000
RANDOM_SEED = 20260513

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = BASE_DIR / "data" / "damping_sensitivity_data.csv"
DEFAULT_CSV_PATH = BASE_DIR / "results" / "damping_sensitivity_table.csv"
DEFAULT_MD_PATH = BASE_DIR / "results" / "damping_sensitivity_table.md"

REQUIRED_COLUMNS = ("theory^2", "v_feed", "v_surge_theory")
SUMMARY_COLUMNS = (
    "case",
    "eta_setting",
    "trials",
    "velocity_scale_vs_baseline",
    "spearman_mean",
    "spearman_std",
    "spearman_min",
    "spearman_p05",
    "spearman_p95",
    "spearman_max",
    "mean_predicted_v_surge_mm_s",
)


def load_validation_data(path=DEFAULT_DATA_PATH):
    """Load the damping sensitivity dataset and validate required columns."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df


def predict_surge_velocity(theory_sq, v_feed, eta):
    """Compute v_surge = eta * sqrt(theory_sq + v_feed^2)."""
    theory_sq = np.asarray(theory_sq, dtype=float)
    v_feed = np.asarray(v_feed, dtype=float)
    eta = np.asarray(eta, dtype=float)
    return eta * np.sqrt(theory_sq + v_feed**2)


def _rankdata_average(values):
    """Return average ranks for a 1-D array, matching Spearman tie handling."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)

    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def spearman_rank_correlation(reference, candidate):
    """Compute Spearman rank correlation without requiring scipy at runtime."""
    reference_ranks = _rankdata_average(reference)
    candidate_ranks = _rankdata_average(candidate)

    reference_centered = reference_ranks - reference_ranks.mean()
    candidate_centered = candidate_ranks - candidate_ranks.mean()
    denominator = np.linalg.norm(reference_centered) * np.linalg.norm(candidate_centered)
    if denominator == 0.0:
        return np.nan
    return float(np.dot(reference_centered, candidate_centered) / denominator)


def _prediction_base(df):
    return np.sqrt(df["theory^2"].to_numpy(dtype=float) + df["v_feed"].to_numpy(dtype=float) ** 2)


def _summary_row(case, eta_setting, trials, correlations, mean_velocity, velocity_scale):
    correlations = np.asarray(correlations, dtype=float)
    return {
        "case": case,
        "eta_setting": eta_setting,
        "trials": int(trials),
        "velocity_scale_vs_baseline": float(velocity_scale),
        "spearman_mean": float(np.mean(correlations)),
        "spearman_std": float(np.std(correlations, ddof=0)),
        "spearman_min": float(np.min(correlations)),
        "spearman_p05": float(np.percentile(correlations, 5)),
        "spearman_p95": float(np.percentile(correlations, 95)),
        "spearman_max": float(np.max(correlations)),
        "mean_predicted_v_surge_mm_s": float(mean_velocity),
    }


def run_constant_eta_sensitivity(df, eta0=ETA0, multipliers=CONSTANT_ETA_MULTIPLIERS):
    """Evaluate global constant eta perturbations around the calibrated eta0."""
    base = _prediction_base(df)
    baseline_velocity = eta0 * base
    rows = []

    for multiplier in multipliers:
        predicted = eta0 * multiplier * base
        correlation = spearman_rank_correlation(baseline_velocity, predicted)
        rows.append(
            _summary_row(
                case="constant_eta",
                eta_setting=f"{multiplier:.1f}eta0",
                trials=1,
                correlations=[correlation],
                mean_velocity=np.mean(predicted),
                velocity_scale=multiplier,
            )
        )
    return rows


def run_random_eta_sensitivity(
    df,
    eta0=ETA0,
    perturbation_levels=PERTURBATION_LEVELS,
    n_trials=N_TRIALS,
    seed=RANDOM_SEED,
):
    """Run posture-wise random eta perturbations for each perturbation level."""
    base = _prediction_base(df)
    baseline_velocity = eta0 * base
    rng = np.random.default_rng(seed)
    rows = []

    for level in perturbation_levels:
        correlations = np.empty(n_trials, dtype=float)
        trial_mean_velocities = np.empty(n_trials, dtype=float)

        for trial in range(n_trials):
            eps = rng.uniform(-level, level, size=len(base))
            predicted = eta0 * (1.0 + eps) * base
            correlations[trial] = spearman_rank_correlation(baseline_velocity, predicted)
            trial_mean_velocities[trial] = np.mean(predicted)

        rows.append(
            _summary_row(
                case="posture_wise_random_eta",
                eta_setting=f"+/-{int(level * 100)}%",
                trials=n_trials,
                correlations=correlations,
                mean_velocity=np.mean(trial_mean_velocities),
                velocity_scale=1.0,
            )
        )
    return rows


def build_sensitivity_table(df):
    """Build the complete two-level damping sensitivity table."""
    rows = []
    rows.extend(run_constant_eta_sensitivity(df))
    rows.extend(run_random_eta_sensitivity(df))
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _format_markdown_value(value):
    if isinstance(value, (float, np.floating)):
        return f"{value:.6f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def dataframe_to_markdown(table):
    """Render a compact GitHub-flavored Markdown table without extra packages."""
    columns = list(table.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in table.iterrows():
        rows.append("| " + " | ".join(_format_markdown_value(row[column]) for column in columns) + " |")
    return "\n".join([header, separator, *rows]) + "\n"


def save_outputs(table, csv_path=DEFAULT_CSV_PATH, md_path=DEFAULT_MD_PATH):
    """Save CSV and Markdown versions of the sensitivity table."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False, float_format="%.6f")
    md_path.write_text(dataframe_to_markdown(table), encoding="utf-8")
    return csv_path, md_path


def format_console_summary(table):
    """Create a compact console summary for manuscript table preparation."""
    display_columns = [
        "case",
        "eta_setting",
        "trials",
        "spearman_mean",
        "spearman_std",
        "spearman_min",
        "spearman_p05",
        "spearman_p95",
        "spearman_max",
        "mean_predicted_v_surge_mm_s",
    ]
    return table[display_columns].to_string(index=False, float_format=lambda value: f"{value:.3f}")


def main():
    df = load_validation_data()
    table = build_sensitivity_table(df)
    csv_path, md_path = save_outputs(table)
    print(format_console_summary(table))
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
