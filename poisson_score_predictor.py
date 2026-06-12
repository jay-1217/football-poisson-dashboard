"""
Poisson score prediction helpers for football matches.

The functions in this module accept Pandas DataFrames produced by
fetch_worldcup_football_data.py and estimate:
    - attack and defense strength factors
    - goal-count probabilities for 0, 1, 2, 3, 4+ goals
    - a score probability matrix
    - top 3 exact scorelines
    - home win, draw, away win probabilities

Example:
    py outputs\\poisson_score_predictor.py ^
        --home-recent outputs\\recent_form_argentina.csv ^
        --away-recent outputs\\recent_form_france.csv ^
        --h2h outputs\\h2h_argentina_france.csv ^
        --home-team Argentina ^
        --away-team France
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    print(
        f"Missing Python package: {exc.name}. "
        "Install dependencies with: py -m pip install -r outputs\\requirements.txt",
        file=sys.stderr,
    )
    sys.exit(1)


GOAL_BUCKETS = ["0", "1", "2", "3", "4+"]
FINAL_RESULTS = {"W", "D", "L"}


def _safe_divide(numerator: float, denominator: float, default: float = 1.0) -> float:
    if denominator == 0 or math.isnan(denominator):
        return default
    return numerator / denominator


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _poisson_pmf(lam: float, goals: int) -> float:
    return math.exp(-lam) * (lam**goals) / math.factorial(goals)


def goal_probabilities(expected_goals: float, bucket_max: int = 4) -> pd.Series:
    """
    Return probabilities for scoring 0, 1, 2, 3, and 4+ goals.

    bucket_max=4 means the final bucket aggregates P(4 goals or more).
    """
    if expected_goals < 0:
        raise ValueError("expected_goals must be non-negative.")
    if bucket_max < 1:
        raise ValueError("bucket_max must be at least 1.")

    exact_probs = [_poisson_pmf(expected_goals, goals) for goals in range(bucket_max)]
    tail_probability = max(0.0, 1.0 - sum(exact_probs))
    labels = [str(goals) for goals in range(bucket_max)] + [f"{bucket_max}+"]
    return pd.Series(exact_probs + [tail_probability], index=labels, name="probability")


def clean_recent_form(df: pd.DataFrame) -> pd.DataFrame:
    """Keep completed matches and coerce model columns to numeric values."""
    required_columns = {"goals_for", "goals_against", "result"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"recent form DataFrame is missing columns: {sorted(missing)}")

    cleaned = df.copy()
    cleaned["goals_for"] = pd.to_numeric(cleaned["goals_for"], errors="coerce")
    cleaned["goals_against"] = pd.to_numeric(cleaned["goals_against"], errors="coerce")
    cleaned["result"] = cleaned["result"].fillna("")
    cleaned = cleaned[cleaned["result"].isin(FINAL_RESULTS)]
    cleaned = cleaned.dropna(subset=["goals_for", "goals_against"])
    return cleaned


def team_recent_stats(df: pd.DataFrame, team_label: str) -> dict[str, Any]:
    """Summarize recent attacking and defensive performance."""
    cleaned = clean_recent_form(df)
    matches = len(cleaned)
    if matches == 0:
        raise ValueError(f"No completed recent matches available for {team_label}.")

    goals_for = float(cleaned["goals_for"].sum())
    goals_against = float(cleaned["goals_against"].sum())
    wins = int((cleaned["result"] == "W").sum())
    draws = int((cleaned["result"] == "D").sum())
    losses = int((cleaned["result"] == "L").sum())

    return {
        "team": team_label,
        "matches": matches,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "avg_goals_for": goals_for / matches,
        "avg_goals_against": goals_against / matches,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / matches,
    }


def h2h_expected_goals(
    h2h_df: pd.DataFrame | None,
    home_team: str,
    away_team: str,
    fallback_home_lambda: float,
    fallback_away_lambda: float,
) -> tuple[float, float]:
    """
    Estimate team-specific expected goals from historical head-to-head matches.

    If no usable H2H rows exist, return the fallback lambdas.
    """
    if h2h_df is None or h2h_df.empty:
        return fallback_home_lambda, fallback_away_lambda

    required_columns = {"home_team", "away_team", "home_goals", "away_goals"}
    missing = required_columns - set(h2h_df.columns)
    if missing:
        return fallback_home_lambda, fallback_away_lambda

    h2h = h2h_df.copy()
    h2h["home_goals"] = pd.to_numeric(h2h["home_goals"], errors="coerce")
    h2h["away_goals"] = pd.to_numeric(h2h["away_goals"], errors="coerce")
    h2h = h2h.dropna(subset=["home_goals", "away_goals"])

    home_name = home_team.lower()
    away_name = away_team.lower()
    home_goals = []
    away_goals = []

    for _, row in h2h.iterrows():
        row_home = str(row["home_team"]).lower()
        row_away = str(row["away_team"]).lower()
        if row_home == home_name and row_away == away_name:
            home_goals.append(float(row["home_goals"]))
            away_goals.append(float(row["away_goals"]))
        elif row_home == away_name and row_away == home_name:
            home_goals.append(float(row["away_goals"]))
            away_goals.append(float(row["home_goals"]))

    if not home_goals or not away_goals:
        return fallback_home_lambda, fallback_away_lambda

    return sum(home_goals) / len(home_goals), sum(away_goals) / len(away_goals)


def calculate_strength_factors(
    home_recent_df: pd.DataFrame,
    away_recent_df: pd.DataFrame,
    home_team: str = "Home",
    away_team: str = "Away",
) -> dict[str, Any]:
    """
    Calculate attack and defense strength factors from recent-form DataFrames.

    attack_strength > 1 means the team scores more than the two-team baseline.
    defense_strength > 1 means the team concedes fewer goals than the baseline.
    """
    home_stats = team_recent_stats(home_recent_df, home_team)
    away_stats = team_recent_stats(away_recent_df, away_team)

    baseline_goals_for = (
        home_stats["avg_goals_for"] + away_stats["avg_goals_for"]
    ) / 2
    baseline_goals_against = (
        home_stats["avg_goals_against"] + away_stats["avg_goals_against"]
    ) / 2

    baseline_goals_for = max(baseline_goals_for, 0.05)
    baseline_goals_against = max(baseline_goals_against, 0.05)

    home_attack = _safe_divide(home_stats["avg_goals_for"], baseline_goals_for)
    away_attack = _safe_divide(away_stats["avg_goals_for"], baseline_goals_for)
    home_defense = _safe_divide(baseline_goals_against, home_stats["avg_goals_against"], default=1.0)
    away_defense = _safe_divide(baseline_goals_against, away_stats["avg_goals_against"], default=1.0)

    return {
        "baseline_goals_for": baseline_goals_for,
        "baseline_goals_against": baseline_goals_against,
        "home": {
            **home_stats,
            "attack_strength": home_attack,
            "defense_strength": home_defense,
        },
        "away": {
            **away_stats,
            "attack_strength": away_attack,
            "defense_strength": away_defense,
        },
    }


def estimate_expected_goals(
    home_recent_df: pd.DataFrame,
    away_recent_df: pd.DataFrame,
    h2h_df: pd.DataFrame | None = None,
    home_team: str = "Home",
    away_team: str = "Away",
    home_advantage: float = 1.05,
    h2h_weight: float = 0.20,
    min_lambda: float = 0.05,
    max_lambda: float = 5.0,
) -> dict[str, Any]:
    """
    Estimate expected goals for both teams.

    Recent form drives the model. H2H data can apply a light blend to avoid
    overfitting to older historical fixtures.
    """
    strengths = calculate_strength_factors(
        home_recent_df=home_recent_df,
        away_recent_df=away_recent_df,
        home_team=home_team,
        away_team=away_team,
    )
    baseline = strengths["baseline_goals_for"]

    home_lambda = (
        baseline
        * strengths["home"]["attack_strength"]
        / max(strengths["away"]["defense_strength"], 0.05)
        * home_advantage
    )
    away_lambda = (
        baseline
        * strengths["away"]["attack_strength"]
        / max(strengths["home"]["defense_strength"], 0.05)
    )

    h2h_home_lambda, h2h_away_lambda = h2h_expected_goals(
        h2h_df,
        home_team=home_team,
        away_team=away_team,
        fallback_home_lambda=home_lambda,
        fallback_away_lambda=away_lambda,
    )

    h2h_weight = _clamp(h2h_weight, 0.0, 1.0)
    blended_home_lambda = ((1 - h2h_weight) * home_lambda) + (h2h_weight * h2h_home_lambda)
    blended_away_lambda = ((1 - h2h_weight) * away_lambda) + (h2h_weight * h2h_away_lambda)

    return {
        "home_expected_goals": _clamp(blended_home_lambda, min_lambda, max_lambda),
        "away_expected_goals": _clamp(blended_away_lambda, min_lambda, max_lambda),
        "recent_home_expected_goals": home_lambda,
        "recent_away_expected_goals": away_lambda,
        "h2h_home_expected_goals": h2h_home_lambda,
        "h2h_away_expected_goals": h2h_away_lambda,
        "h2h_weight": h2h_weight,
        "home_advantage": home_advantage,
        "strengths": strengths,
    }


def score_probability_matrix(
    home_expected_goals: float,
    away_expected_goals: float,
    bucket_max: int = 4,
) -> pd.DataFrame:
    """Return a bucketed score matrix with rows as home goals and columns as away goals."""
    home_probs = goal_probabilities(home_expected_goals, bucket_max=bucket_max)
    away_probs = goal_probabilities(away_expected_goals, bucket_max=bucket_max)
    matrix = pd.DataFrame(index=home_probs.index, columns=away_probs.index, dtype=float)

    for home_goals, home_prob in home_probs.items():
        for away_goals, away_prob in away_probs.items():
            matrix.loc[home_goals, away_goals] = home_prob * away_prob

    matrix.index.name = "home_goals"
    matrix.columns.name = "away_goals"
    return matrix


def exact_score_probabilities(
    home_expected_goals: float,
    away_expected_goals: float,
    max_goals: int = 10,
) -> pd.DataFrame:
    """Return exact scoreline probabilities up to max_goals for each team."""
    rows = []
    for home_goals in range(max_goals + 1):
        home_prob = _poisson_pmf(home_expected_goals, home_goals)
        for away_goals in range(max_goals + 1):
            away_prob = _poisson_pmf(away_expected_goals, away_goals)
            rows.append(
                {
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "score": f"{home_goals}-{away_goals}",
                    "probability": home_prob * away_prob,
                }
            )
    return pd.DataFrame(rows)


def top_scorelines(score_probs_df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Return the most likely exact scorelines."""
    return (
        score_probs_df.sort_values("probability", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def match_outcome_probabilities(score_probs_df: pd.DataFrame) -> pd.Series:
    """
    Return home win, draw, and away win probabilities from exact scores.

    Use exact scores rather than the 4+ bucket so 4-4, 5-4, 4-5, etc. are not
    collapsed into an ambiguous 4+ vs 4+ cell.
    """
    home_win = score_probs_df.loc[
        score_probs_df["home_goals"] > score_probs_df["away_goals"], "probability"
    ].sum()
    draw = score_probs_df.loc[
        score_probs_df["home_goals"] == score_probs_df["away_goals"], "probability"
    ].sum()
    away_win = score_probs_df.loc[
        score_probs_df["home_goals"] < score_probs_df["away_goals"], "probability"
    ].sum()
    total = home_win + draw + away_win

    return pd.Series(
        {
            "home_win": _safe_divide(home_win, total, default=0.0),
            "draw": _safe_divide(draw, total, default=0.0),
            "away_win": _safe_divide(away_win, total, default=0.0),
        },
        name="probability",
    )


def predict_match_score(
    home_recent_df: pd.DataFrame,
    away_recent_df: pd.DataFrame,
    h2h_df: pd.DataFrame | None = None,
    home_team: str = "Home",
    away_team: str = "Away",
    home_advantage: float = 1.05,
    h2h_weight: float = 0.20,
    bucket_max: int = 4,
    exact_max_goals: int = 10,
) -> dict[str, Any]:
    """Run the full Poisson prediction workflow."""
    expected = estimate_expected_goals(
        home_recent_df=home_recent_df,
        away_recent_df=away_recent_df,
        h2h_df=h2h_df,
        home_team=home_team,
        away_team=away_team,
        home_advantage=home_advantage,
        h2h_weight=h2h_weight,
    )
    home_lambda = expected["home_expected_goals"]
    away_lambda = expected["away_expected_goals"]
    score_matrix = score_probability_matrix(home_lambda, away_lambda, bucket_max=bucket_max)
    exact_scores = exact_score_probabilities(home_lambda, away_lambda, max_goals=exact_max_goals)

    return {
        "expected_goals": expected,
        "home_goal_probabilities": goal_probabilities(home_lambda, bucket_max=bucket_max),
        "away_goal_probabilities": goal_probabilities(away_lambda, bucket_max=bucket_max),
        "score_matrix": score_matrix,
        "exact_scores": exact_scores,
        "top_3_scores": top_scorelines(exact_scores, top_n=3),
        "outcome_probabilities": match_outcome_probabilities(exact_scores),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict football score probabilities using Poisson models.")
    parser.add_argument("--home-recent", required=True, help="CSV containing home-team recent form.")
    parser.add_argument("--away-recent", required=True, help="CSV containing away-team recent form.")
    parser.add_argument("--h2h", default=None, help="Optional CSV containing historical head-to-head matches.")
    parser.add_argument("--home-team", default="Home", help="Home team display name.")
    parser.add_argument("--away-team", default="Away", help="Away team display name.")
    parser.add_argument("--home-advantage", type=float, default=1.05, help="Home advantage multiplier.")
    parser.add_argument("--h2h-weight", type=float, default=0.20, help="Blend weight for H2H expected goals.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for prediction CSV outputs.")
    return parser.parse_args()


def format_percent_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.applymap(lambda value: f"{value:.2%}")


def main() -> int:
    args = parse_args()
    home_recent_df = pd.read_csv(args.home_recent)
    away_recent_df = pd.read_csv(args.away_recent)
    h2h_df = pd.read_csv(args.h2h) if args.h2h else None

    prediction = predict_match_score(
        home_recent_df=home_recent_df,
        away_recent_df=away_recent_df,
        h2h_df=h2h_df,
        home_team=args.home_team,
        away_team=args.away_team,
        home_advantage=args.home_advantage,
        h2h_weight=args.h2h_weight,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.home_team.lower().replace(' ', '_')}_{args.away_team.lower().replace(' ', '_')}"

    prediction["score_matrix"].to_csv(output_dir / f"poisson_score_matrix_{suffix}.csv", encoding="utf-8-sig")
    prediction["top_3_scores"].to_csv(output_dir / f"poisson_top_3_scores_{suffix}.csv", index=False, encoding="utf-8-sig")
    prediction["outcome_probabilities"].to_frame().to_csv(
        output_dir / f"poisson_outcome_probabilities_{suffix}.csv",
        encoding="utf-8-sig",
    )

    expected = prediction["expected_goals"]
    print(f"\nExpected goals: {args.home_team} {expected['home_expected_goals']:.3f}, {args.away_team} {expected['away_expected_goals']:.3f}")
    print("\nHome goal probabilities:")
    print(prediction["home_goal_probabilities"].map(lambda value: f"{value:.2%}").to_string())
    print("\nAway goal probabilities:")
    print(prediction["away_goal_probabilities"].map(lambda value: f"{value:.2%}").to_string())
    print("\nScore probability matrix:")
    print(format_percent_table(prediction["score_matrix"]).to_string())
    print("\nTop 3 exact scores:")
    print(prediction["top_3_scores"].assign(probability=lambda df: df["probability"].map(lambda value: f"{value:.2%}")).to_string(index=False))
    print("\nOutcome probabilities:")
    print(prediction["outcome_probabilities"].map(lambda value: f"{value:.2%}").to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
