from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fetch_worldcup_football_data import (
    ApiFootballError,
    find_team_id,
    get_head_to_head,
    get_recent_matches,
    slugify,
)
from poisson_score_predictor import predict_match_score


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DEFAULT_TEAMS = [
    "Argentina",
    "France",
    "Brazil",
    "England",
    "Spain",
    "Germany",
    "Portugal",
    "Netherlands",
    "Italy",
    "Uruguay",
    "Belgium",
    "Croatia",
    "Morocco",
    "Japan",
    "United States",
    "Mexico",
]


st.set_page_config(
    page_title="World Cup Score Predictor",
    page_icon="",
    layout="wide",
)


def pct(value: float) -> str:
    return f"{value:.1%}"


def number_or_na(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def csv_path(prefix: str, team_or_suffix: str) -> Path:
    return DATA_DIR / f"{prefix}_{slugify(team_or_suffix)}.csv"


@st.cache_data(ttl=3600, show_spinner=False)
def load_match_data(home_team: str, away_team: str, last: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    api_key = os.getenv("API_FOOTBALL_KEY", "").strip()
    if api_key:
        home_id, home_name, _ = find_team_id(home_team)
        away_id, away_name, _ = find_team_id(away_team)
        h2h_df = get_head_to_head(home_id, away_id)
        home_recent_df = get_recent_matches(home_id, home_name, last=last)
        away_recent_df = get_recent_matches(away_id, away_name, last=last)
        return home_recent_df, away_recent_df, h2h_df, "API-FOOTBALL live data"

    suffix = f"{slugify(home_team)}_{slugify(away_team)}"
    h2h_file = csv_path("h2h", suffix)
    home_file = csv_path("recent_form", home_team)
    away_file = csv_path("recent_form", away_team)
    missing = [path.name for path in [h2h_file, home_file, away_file] if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing demo CSV files and API_FOOTBALL_KEY is not configured: "
            + ", ".join(missing)
        )

    return (
        pd.read_csv(home_file),
        pd.read_csv(away_file),
        pd.read_csv(h2h_file),
        "bundled demo CSV files",
    )


def metric_from_recent(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def team_comparison_frame(home_df: pd.DataFrame, away_df: pd.DataFrame, home_team: str, away_team: str) -> pd.DataFrame:
    metrics = [
        ("Avg goals", "goals_for"),
        ("Avg conceded", "goals_against"),
        ("Possession %", "possession_pct"),
        ("Conversion %", "conversion_rate"),
    ]
    rows = []
    for label, column in metrics:
        rows.append({"team": home_team, "metric": label, "value": metric_from_recent(home_df, column)})
        rows.append({"team": away_team, "metric": label, "value": metric_from_recent(away_df, column)})
    return pd.DataFrame(rows)


def build_comparison_chart(comparison_df: pd.DataFrame) -> go.Figure:
    available = comparison_df.dropna(subset=["value"]).copy()
    if available.empty:
        fig = go.Figure()
        fig.update_layout(
            height=360,
            annotations=[
                {
                    "text": "No numeric comparison metrics available",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 18},
                }
            ],
        )
        return fig

    fig = px.bar(
        available,
        x="metric",
        y="value",
        color="team",
        barmode="group",
        text_auto=".2f",
        color_discrete_sequence=["#2563eb", "#dc2626"],
    )
    fig.update_layout(
        height=360,
        margin=dict(l=12, r=12, t=24, b=12),
        xaxis_title="",
        yaxis_title="",
        legend_title="",
    )
    return fig


def build_heatmap(score_matrix: pd.DataFrame) -> go.Figure:
    heatmap_values = score_matrix.astype(float) * 100
    fig = px.imshow(
        heatmap_values,
        labels=dict(x="Away goals", y="Home goals", color="Probability (%)"),
        x=score_matrix.columns,
        y=score_matrix.index,
        text_auto=".2f",
        color_continuous_scale="YlOrRd",
        aspect="auto",
    )
    fig.update_layout(height=430, margin=dict(l=12, r=12, t=24, b=12))
    return fig


def render_strength_table(prediction: dict[str, Any], home_team: str, away_team: str) -> None:
    strengths = prediction["expected_goals"]["strengths"]
    df = pd.DataFrame(
        [
            {
                "Team": home_team,
                "Avg goals": strengths["home"]["avg_goals_for"],
                "Avg conceded": strengths["home"]["avg_goals_against"],
                "Attack factor": strengths["home"]["attack_strength"],
                "Defense factor": strengths["home"]["defense_strength"],
                "Recent win rate": strengths["home"]["win_rate"],
            },
            {
                "Team": away_team,
                "Avg goals": strengths["away"]["avg_goals_for"],
                "Avg conceded": strengths["away"]["avg_goals_against"],
                "Attack factor": strengths["away"]["attack_strength"],
                "Defense factor": strengths["away"]["defense_strength"],
                "Recent win rate": strengths["away"]["win_rate"],
            },
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def main() -> None:
    st.title("World Cup Score Predictor")
    st.caption("Poisson-based score probabilities from recent form and head-to-head data.")

    with st.sidebar:
        st.header("Match setup")
        home_team = st.selectbox("Home team", DEFAULT_TEAMS, index=DEFAULT_TEAMS.index("Argentina"))
        away_options = [team for team in DEFAULT_TEAMS if team != home_team]
        default_away_index = away_options.index("France") if "France" in away_options else 0
        away_team = st.selectbox("Away team", away_options, index=default_away_index)
        recent_matches = st.slider("Recent matches", 5, 20, 10, 1)
        home_advantage = st.slider("Home advantage", 1.00, 1.20, 1.05, 0.01)
        h2h_weight = st.slider("H2H weight", 0.00, 0.50, 0.20, 0.05)
        if st.button("Refresh data", type="primary"):
            st.cache_data.clear()

    try:
        with st.spinner("Loading data and calculating probabilities..."):
            home_recent_df, away_recent_df, h2h_df, source = load_match_data(home_team, away_team, recent_matches)
            prediction = predict_match_score(
                home_recent_df=home_recent_df,
                away_recent_df=away_recent_df,
                h2h_df=h2h_df,
                home_team=home_team,
                away_team=away_team,
                home_advantage=home_advantage,
                h2h_weight=h2h_weight,
            )
    except (ApiFootballError, FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.info("Add API_FOOTBALL_KEY in Streamlit Secrets, or keep Argentina vs France to use bundled demo data.")
        st.stop()

    top_score = prediction["top_3_scores"].iloc[0]
    outcomes = prediction["outcome_probabilities"]
    expected = prediction["expected_goals"]

    st.caption(f"Data source: {source}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Most likely score", top_score["score"], pct(float(top_score["probability"])))
    col2.metric("Home win", pct(float(outcomes["home_win"])))
    col3.metric("Draw", pct(float(outcomes["draw"])))
    col4.metric("Away win", pct(float(outcomes["away_win"])))

    eg_col1, eg_col2 = st.columns(2)
    eg_col1.metric(f"{home_team} expected goals", number_or_na(expected["home_expected_goals"], 3))
    eg_col2.metric(f"{away_team} expected goals", number_or_na(expected["away_expected_goals"], 3))

    chart_col, heatmap_col = st.columns([1, 1.25])
    with chart_col:
        st.subheader("Recent Team Comparison")
        comparison_df = team_comparison_frame(home_recent_df, away_recent_df, home_team, away_team)
        st.plotly_chart(build_comparison_chart(comparison_df), use_container_width=True)

    with heatmap_col:
        st.subheader("Score Probability Heatmap")
        st.plotly_chart(build_heatmap(prediction["score_matrix"]), use_container_width=True)

    table_col1, table_col2 = st.columns(2)
    with table_col1:
        st.subheader("Top 3 Scorelines")
        top_scores = prediction["top_3_scores"].copy()
        top_scores["probability"] = top_scores["probability"].map(pct)
        st.dataframe(top_scores, use_container_width=True, hide_index=True)

    with table_col2:
        st.subheader("Model Strength Factors")
        render_strength_table(prediction, home_team, away_team)

    with st.expander("Recent match data"):
        st.write(f"{home_team} recent matches")
        st.dataframe(home_recent_df, use_container_width=True)
        st.write(f"{away_team} recent matches")
        st.dataframe(away_recent_df, use_container_width=True)


if __name__ == "__main__":
    main()
