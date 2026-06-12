from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from poisson_score_predictor import (
    exact_score_probabilities,
    goal_probabilities,
    match_outcome_probabilities,
    score_probability_matrix,
    top_scorelines,
)


WORLD_CUP_TEAMS: dict[str, dict[str, Any]] = {
    "Argentina": {"group": "J", "attack": 1.55, "defense": 0.78, "form": 0.82, "possession": 58, "conversion": 15.2},
    "France": {"group": "I", "attack": 1.48, "defense": 0.82, "form": 0.76, "possession": 56, "conversion": 14.4},
    "Spain": {"group": "H", "attack": 1.44, "defense": 0.80, "form": 0.78, "possession": 64, "conversion": 13.6},
    "England": {"group": "L", "attack": 1.40, "defense": 0.84, "form": 0.74, "possession": 57, "conversion": 13.2},
    "Brazil": {"group": "C", "attack": 1.39, "defense": 0.88, "form": 0.68, "possession": 55, "conversion": 13.8},
    "Portugal": {"group": "K", "attack": 1.37, "defense": 0.86, "form": 0.72, "possession": 59, "conversion": 13.7},
    "Netherlands": {"group": "F", "attack": 1.28, "defense": 0.90, "form": 0.70, "possession": 54, "conversion": 12.9},
    "Germany": {"group": "E", "attack": 1.30, "defense": 0.94, "form": 0.69, "possession": 60, "conversion": 12.7},
    "Croatia": {"group": "L", "attack": 1.12, "defense": 0.92, "form": 0.66, "possession": 55, "conversion": 11.3},
    "Belgium": {"group": "G", "attack": 1.18, "defense": 0.98, "form": 0.64, "possession": 53, "conversion": 12.0},
    "Uruguay": {"group": "H", "attack": 1.20, "defense": 0.91, "form": 0.69, "possession": 50, "conversion": 12.2},
    "Morocco": {"group": "C", "attack": 1.10, "defense": 0.88, "form": 0.70, "possession": 49, "conversion": 11.8},
    "Japan": {"group": "F", "attack": 1.13, "defense": 0.98, "form": 0.71, "possession": 52, "conversion": 12.1},
    "United States": {"group": "D", "attack": 1.08, "defense": 1.02, "form": 0.62, "possession": 51, "conversion": 10.9},
    "Mexico": {"group": "A", "attack": 1.02, "defense": 1.04, "form": 0.58, "possession": 50, "conversion": 10.4},
    "Czechia": {"group": "A", "attack": 0.95, "defense": 1.05, "form": 0.57, "possession": 49, "conversion": 9.8},
    "Canada": {"group": "B", "attack": 1.00, "defense": 1.08, "form": 0.56, "possession": 49, "conversion": 10.2},
    "Bosnia and Herzegovina": {"group": "B", "attack": 0.93, "defense": 1.07, "form": 0.55, "possession": 48, "conversion": 9.7},
    "Switzerland": {"group": "B", "attack": 1.03, "defense": 0.99, "form": 0.60, "possession": 51, "conversion": 10.7},
    "Austria": {"group": "L", "attack": 1.05, "defense": 1.00, "form": 0.63, "possession": 52, "conversion": 10.9},
    "Senegal": {"group": "I", "attack": 1.02, "defense": 0.96, "form": 0.65, "possession": 49, "conversion": 10.8},
    "Colombia": {"group": "G", "attack": 1.09, "defense": 0.99, "form": 0.67, "possession": 51, "conversion": 11.2},
    "Ecuador": {"group": "E", "attack": 0.98, "defense": 0.95, "form": 0.62, "possession": 48, "conversion": 10.1},
    "Paraguay": {"group": "D", "attack": 0.89, "defense": 1.02, "form": 0.54, "possession": 46, "conversion": 9.2},
    "Australia": {"group": "D", "attack": 0.94, "defense": 1.05, "form": 0.58, "possession": 47, "conversion": 9.7},
    "Turkey": {"group": "D", "attack": 1.06, "defense": 1.03, "form": 0.61, "possession": 51, "conversion": 10.9},
    "South Korea": {"group": "F", "attack": 1.01, "defense": 1.04, "form": 0.60, "possession": 51, "conversion": 10.5},
    "Iran": {"group": "G", "attack": 0.96, "defense": 1.03, "form": 0.59, "possession": 47, "conversion": 10.1},
    "Saudi Arabia": {"group": "H", "attack": 0.84, "defense": 1.13, "form": 0.49, "possession": 45, "conversion": 8.7},
    "Qatar": {"group": "I", "attack": 0.80, "defense": 1.16, "form": 0.48, "possession": 46, "conversion": 8.4},
    "Iraq": {"group": "I", "attack": 0.82, "defense": 1.14, "form": 0.50, "possession": 45, "conversion": 8.6},
    "Uzbekistan": {"group": "J", "attack": 0.83, "defense": 1.12, "form": 0.52, "possession": 45, "conversion": 8.8},
    "Jordan": {"group": "J", "attack": 0.78, "defense": 1.18, "form": 0.47, "possession": 43, "conversion": 8.0},
    "New Zealand": {"group": "F", "attack": 0.72, "defense": 1.22, "form": 0.44, "possession": 42, "conversion": 7.6},
    "Ghana": {"group": "I", "attack": 0.93, "defense": 1.09, "form": 0.54, "possession": 48, "conversion": 9.6},
    "South Africa": {"group": "A", "attack": 0.88, "defense": 1.08, "form": 0.55, "possession": 47, "conversion": 9.3},
    "Egypt": {"group": "B", "attack": 0.97, "defense": 1.02, "form": 0.60, "possession": 49, "conversion": 10.3},
    "Algeria": {"group": "J", "attack": 0.98, "defense": 1.04, "form": 0.58, "possession": 50, "conversion": 10.5},
    "Tunisia": {"group": "F", "attack": 0.84, "defense": 1.06, "form": 0.53, "possession": 46, "conversion": 8.9},
    "Ivory Coast": {"group": "E", "attack": 0.99, "defense": 1.05, "form": 0.59, "possession": 49, "conversion": 10.4},
    "Norway": {"group": "G", "attack": 1.12, "defense": 1.03, "form": 0.64, "possession": 50, "conversion": 11.6},
    "Scotland": {"group": "C", "attack": 0.92, "defense": 1.08, "form": 0.55, "possession": 47, "conversion": 9.5},
    "Sweden": {"group": "F", "attack": 1.03, "defense": 1.02, "form": 0.60, "possession": 50, "conversion": 10.6},
    "DR Congo": {"group": "K", "attack": 0.90, "defense": 1.08, "form": 0.54, "possession": 46, "conversion": 9.4},
    "Panama": {"group": "L", "attack": 0.77, "defense": 1.20, "form": 0.46, "possession": 43, "conversion": 8.1},
    "Haiti": {"group": "D", "attack": 0.70, "defense": 1.25, "form": 0.42, "possession": 41, "conversion": 7.3},
    "Curacao": {"group": "G", "attack": 0.68, "defense": 1.28, "form": 0.40, "possession": 40, "conversion": 7.0},
    "Cape Verde": {"group": "A", "attack": 0.82, "defense": 1.14, "form": 0.51, "possession": 45, "conversion": 8.6},
}

FEATURED_FIXTURES = [
    ("Mexico", "South Africa"),
    ("Canada", "Bosnia and Herzegovina"),
    ("Spain", "Uruguay"),
    ("Netherlands", "Japan"),
    ("Argentina", "Algeria"),
    ("France", "Senegal"),
    ("England", "Croatia"),
    ("Brazil", "Morocco"),
    ("United States", "Paraguay"),
    ("Portugal", "DR Congo"),
]


st.set_page_config(
    page_title="2026 World Cup Predictor",
    page_icon="",
    layout="wide",
)


def pct(value: float) -> str:
    return f"{value:.1%}"


def number_or_na(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def team_names() -> list[str]:
    return sorted(WORLD_CUP_TEAMS.keys())


def expected_goals(home_team: str, away_team: str, home_advantage: float = 1.05) -> dict[str, float]:
    home = WORLD_CUP_TEAMS[home_team]
    away = WORLD_CUP_TEAMS[away_team]
    tournament_avg_goals = 1.28

    home_lambda = tournament_avg_goals * home["attack"] / away["defense"] * home_advantage
    away_lambda = tournament_avg_goals * away["attack"] / home["defense"]

    form_gap = home["form"] - away["form"]
    home_lambda *= 1 + (form_gap * 0.10)
    away_lambda *= 1 - (form_gap * 0.10)

    return {
        "home_expected_goals": max(0.05, min(4.5, home_lambda)),
        "away_expected_goals": max(0.05, min(4.5, away_lambda)),
    }


def run_prediction(home_team: str, away_team: str, home_advantage: float) -> dict[str, Any]:
    lambdas = expected_goals(home_team, away_team, home_advantage)
    home_lambda = lambdas["home_expected_goals"]
    away_lambda = lambdas["away_expected_goals"]
    matrix = score_probability_matrix(home_lambda, away_lambda, bucket_max=4)
    exact_scores = exact_score_probabilities(home_lambda, away_lambda, max_goals=10)

    return {
        "expected_goals": lambdas,
        "home_goal_probabilities": goal_probabilities(home_lambda, bucket_max=4),
        "away_goal_probabilities": goal_probabilities(away_lambda, bucket_max=4),
        "score_matrix": matrix,
        "exact_scores": exact_scores,
        "top_3_scores": top_scorelines(exact_scores, top_n=3),
        "outcome_probabilities": match_outcome_probabilities(exact_scores),
    }


def team_comparison_frame(home_team: str, away_team: str) -> pd.DataFrame:
    rows = []
    metrics = [
        ("Attack rating", "attack"),
        ("Defense rating", "defense"),
        ("Recent form", "form"),
        ("Possession %", "possession"),
        ("Conversion %", "conversion"),
    ]
    for label, key in metrics:
        rows.append({"team": home_team, "metric": label, "value": WORLD_CUP_TEAMS[home_team][key]})
        rows.append({"team": away_team, "metric": label, "value": WORLD_CUP_TEAMS[away_team][key]})
    return pd.DataFrame(rows)


def build_comparison_chart(comparison_df: pd.DataFrame):
    fig = px.bar(
        comparison_df,
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


def build_heatmap(score_matrix: pd.DataFrame):
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


def strength_table(home_team: str, away_team: str) -> pd.DataFrame:
    rows = []
    for team in [home_team, away_team]:
        data = WORLD_CUP_TEAMS[team]
        rows.append(
            {
                "Team": team,
                "Group": data["group"],
                "Attack rating": data["attack"],
                "Defense rating": data["defense"],
                "Recent form": data["form"],
                "Possession %": data["possession"],
                "Conversion %": data["conversion"],
            }
        )
    return pd.DataFrame(rows)


def fixture_label(fixture: tuple[str, str]) -> str:
    return f"{fixture[0]} vs {fixture[1]}"


def main() -> None:
    st.title("2026 World Cup Score Predictor")
    st.caption("No API key needed. This site uses built-in 2026 World Cup team ratings and a Poisson score model.")

    teams = team_names()
    with st.sidebar:
        st.header("Match setup")
        mode = st.radio("Choose match", ["Featured 2026 fixture", "Custom teams"], horizontal=False)

        if mode == "Featured 2026 fixture":
            labels = [fixture_label(fixture) for fixture in FEATURED_FIXTURES]
            selected = st.selectbox("Fixture", labels, index=3)
            home_team, away_team = FEATURED_FIXTURES[labels.index(selected)]
        else:
            home_team = st.selectbox("Home team", teams, index=teams.index("Japan"))
            away_options = [team for team in teams if team != home_team]
            default_away = "Netherlands" if "Netherlands" in away_options else away_options[0]
            away_team = st.selectbox("Away team", away_options, index=away_options.index(default_away))

        home_advantage = st.slider("Home advantage", 1.00, 1.20, 1.05, 0.01)

    prediction = run_prediction(home_team, away_team, home_advantage)
    top_score = prediction["top_3_scores"].iloc[0]
    outcomes = prediction["outcome_probabilities"]
    expected = prediction["expected_goals"]

    st.subheader(f"{home_team} vs {away_team}")

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
        st.subheader("Team Comparison")
        st.plotly_chart(build_comparison_chart(team_comparison_frame(home_team, away_team)), use_container_width=True)

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
        st.subheader("Team Ratings")
        st.dataframe(strength_table(home_team, away_team), use_container_width=True, hide_index=True)

    with st.expander("Model notes"):
        st.write(
            "This is a lightweight forecasting model for exploration. Expected goals are estimated from built-in "
            "team attack ratings, defensive ratings, recent-form assumptions, and a small home-advantage factor. "
            "Exact score probabilities are then calculated with independent Poisson distributions."
        )
        st.write(
            "Because no paid API is used, ratings are static and should be updated manually as tournament news, "
            "injuries, lineups, and results change."
        )


if __name__ == "__main__":
    main()
