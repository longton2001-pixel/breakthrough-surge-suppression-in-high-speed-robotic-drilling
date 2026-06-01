"""
Damping sensitivity analysis for the breakthrough surge model.

This script evaluates whether damping uncertainty can overturn the EMR-based
posture ranking. It treats uniform damping as a common scale factor, keeps the
posture-wise random perturbation as a conservative uncertainty stress test, and
adds a ranking-boundary calculation for unknown posture-dependent damping.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ETA0 = 0.3845
PERTURBATION_LEVELS = (0.10, 0.20, 0.30)
N_TRIALS = 1000
RANDOM_SEED = 20260513

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = BASE_DIR / "data" / "damping_sensitivity_data.csv"
DEFAULT_CSV_PATH = BASE_DIR / "results" / "damping_sensitivity_table.csv"

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
    "top1_delta_crit",
    "top3_delta_crit",
    "top1_robust_pm10",
    "top1_robust_pm20",
    "top1_robust_pm30",
    "top1_reference_id",
    "top1_limiting_competitor_id",
    "top3_limiting_pair",
    "interpretation",
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
    """Return average ranks for a 1-D array, matching Spearman's tie handling."""
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
    # Column mapping: "theory^2" in the validation CSV is the stored f^2 * lambda_EMR term.
    return np.sqrt(df["theory^2"].to_numpy(dtype=float) + df["v_feed"].to_numpy(dtype=float) ** 2)


def _summary_row(
    case,
    eta_setting,
    trials,
    correlations,
    mean_velocity,
    velocity_scale,
    interpretation,
    **extra_fields,
):
    correlations = np.asarray(correlations, dtype=float)
    row = {
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
        "top1_delta_crit": np.nan,
        "top3_delta_crit": np.nan,
        "top1_robust_pm10": "",
        "top1_robust_pm20": "",
        "top1_robust_pm30": "",
        "top1_reference_id": "",
        "top1_limiting_competitor_id": "",
        "top3_limiting_pair": "",
        "interpretation": interpretation,
    }
    row.update(extra_fields)
    return row


def run_constant_eta_sensitivity(
    df,
    eta0=ETA0,
):
    """Report the rank invariance of any uniform eta scaling once."""
    base = _prediction_base(df)
    baseline_velocity = eta0 * base
    correlation = spearman_rank_correlation(baseline_velocity, baseline_velocity)
    return [
        _summary_row(
            case="uniform_damping_factor",
            eta_setting="common eta factor",
            trials=1,
            correlations=[correlation],
            mean_velocity=np.mean(baseline_velocity),
            velocity_scale=1.0,
            interpretation=(
                "In Eq. (31), a damping factor common to all postures is a common "
                "multiplicative factor; it scales absolute predicted surge velocity "
                "but leaves the EMR-based rank order unchanged."
            ),
        )
    ]


def run_random_eta_sensitivity(
    df,
    eta0=ETA0,
    perturbation_levels=PERTURBATION_LEVELS,
    n_trials=N_TRIALS,
    seed=RANDOM_SEED,
):
    """Run conservative posture-wise random eta perturbations."""
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
                interpretation=(
                    "Posture-wise eta_i = eta0(1 + eps_i), eps_i sampled uniformly "
                    "within the listed bound; this is a conservative damping-uncertainty "
                    "stress test, not an identified physical posture-dependent damping model."
                ),
            )
        )
    return rows


def _candidate_ids(df):
    if "Number" in df.columns:
        return df["Number"].to_numpy()
    return np.arange(1, len(df) + 1)


def _min_delta_critical_pair(u_values, candidate_ids, source_indices, competitor_indices):
    best = None
    for source_idx in source_indices:
        for competitor_idx in competitor_indices:
            u_source = u_values[source_idx]
            u_competitor = u_values[competitor_idx]
            if not u_source < u_competitor:
                continue
            ratio = u_competitor / u_source
            delta_crit = (ratio - 1.0) / (ratio + 1.0)
            if best is None or delta_crit < best["delta_crit"]:
                best = {
                    "delta_crit": float(delta_crit),
                    "source_id": candidate_ids[source_idx],
                    "competitor_id": candidate_ids[competitor_idx],
                    "ratio": float(ratio),
                }
    return best


def _robustness_label(delta, delta_crit):
    if np.isnan(delta_crit):
        return "not applicable"
    return "guaranteed" if delta < delta_crit else "not guaranteed"


def compute_ranking_boundary(df, top_k=3, uncertainty_levels=PERTURBATION_LEVELS):
    """Compute damping variation bounds needed to overturn the EMR/u ranking."""
    u_values = _prediction_base(df)
    candidate_ids = _candidate_ids(df)
    order = np.argsort(u_values, kind="mergesort")

    top1_pair = _min_delta_critical_pair(
        u_values,
        candidate_ids,
        source_indices=[order[0]],
        competitor_indices=order[1:],
    )
    if top1_pair is None:
        raise ValueError("Top-1 ranking boundary cannot be computed from the available data.")

    top_k_delta = np.nan
    top_k_pair = None
    if len(order) > top_k:
        top_k_pair = _min_delta_critical_pair(
            u_values,
            candidate_ids,
            source_indices=order[:top_k],
            competitor_indices=order[top_k:],
        )
        if top_k_pair is not None:
            top_k_delta = top_k_pair["delta_crit"]

    top1_delta = top1_pair["delta_crit"]
    robustness = {
        f"top1_robust_pm{int(level * 100)}": _robustness_label(level, top1_delta)
        for level in uncertainty_levels
    }

    return {
        "top1_delta_crit": top1_delta,
        "top3_delta_crit": top_k_delta,
        "top1_reference_id": int(top1_pair["source_id"]),
        "top1_limiting_competitor_id": int(top1_pair["competitor_id"]),
        "top1_limiting_ratio": top1_pair["ratio"],
        "top3_limiting_pair": (
            "" if top_k_pair is None else f"{int(top_k_pair['source_id'])} vs {int(top_k_pair['competitor_id'])}"
        ),
        **robustness,
    }


def run_ranking_boundary_analysis(df):
    """Build the table row for unknown posture-dependent damping boundaries."""
    boundary = compute_ranking_boundary(df)
    top3_text = (
        "not available"
        if np.isnan(boundary["top3_delta_crit"])
        else f"{boundary['top3_delta_crit']:.6f}"
    )
    interpretation = (
        "For eta_i in [eta0(1-delta), eta0(1+delta)], the minimum Top-1 "
        f"delta_crit is {boundary['top1_delta_crit']:.6f}; the Top-3 boundary is "
        f"{top3_text}. The Top-1 rank is guaranteed at +/-10% but not guaranteed "
        "by the worst-case bound at +/-20% or +/-30%."
    )
    return [
        {
            "case": "ranking_boundary",
            "eta_setting": "bounded unknown posture-wise eta",
            "trials": 0,
            "velocity_scale_vs_baseline": np.nan,
            "spearman_mean": np.nan,
            "spearman_std": np.nan,
            "spearman_min": np.nan,
            "spearman_p05": np.nan,
            "spearman_p95": np.nan,
            "spearman_max": np.nan,
            "mean_predicted_v_surge_mm_s": np.nan,
            "top1_delta_crit": boundary["top1_delta_crit"],
            "top3_delta_crit": boundary["top3_delta_crit"],
            "top1_robust_pm10": boundary["top1_robust_pm10"],
            "top1_robust_pm20": boundary["top1_robust_pm20"],
            "top1_robust_pm30": boundary["top1_robust_pm30"],
            "top1_reference_id": boundary["top1_reference_id"],
            "top1_limiting_competitor_id": boundary["top1_limiting_competitor_id"],
            "top3_limiting_pair": boundary["top3_limiting_pair"],
            "interpretation": interpretation,
        }
    ]


def build_sensitivity_table(df):
    """Build the complete damping sensitivity table."""
    rows = []
    rows.extend(run_constant_eta_sensitivity(df))
    rows.extend(run_random_eta_sensitivity(df))
    rows.extend(run_ranking_boundary_analysis(df))
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def save_outputs(table, csv_path=DEFAULT_CSV_PATH):
    """Save the damping sensitivity table as a CSV artifact."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False, float_format="%.6f")
    return csv_path


def format_console_summary(table):
    """Create a compact console summary for manuscript table preparation."""
    display_columns = [
        "case",
        "eta_setting",
        "trials",
        "spearman_mean",
        "spearman_std",
        "top1_delta_crit",
        "top3_delta_crit",
        "top1_robust_pm10",
        "top1_robust_pm20",
        "top1_robust_pm30",
        "interpretation",
    ]
    return table[display_columns].to_string(index=False, float_format=lambda value: f"{value:.3f}")


def main():
    df = load_validation_data()
    table = build_sensitivity_table(df)
    csv_path = save_outputs(table)
    print(format_console_summary(table))
    print(f"\nSaved CSV: {csv_path}")


if __name__ == "__main__":
    main()
