"""
Fetch football data for two national teams from API-FOOTBALL.

Default example:
    python fetch_worldcup_football_data.py --team1 Argentina --team2 France

Required environment variable:
    API_FOOTBALL_KEY

Output CSV files are written to ./outputs by default:
    h2h_argentina_france.csv
    recent_form_argentina.csv
    recent_form_france.csv
    injuries_argentina_france.csv
    recent_form_summary_argentina_france.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    import pandas as pd
    import requests
except ModuleNotFoundError as exc:
    missing_package = exc.name
    print(
        f"Missing Python package: {missing_package}. "
        "Install dependencies with: py -m pip install -r outputs\\requirements.txt",
        file=sys.stderr,
    )
    sys.exit(1)


BASE_URL = "https://v3.football.api-sports.io"
API_KEY_ENV = "API_FOOTBALL_KEY"
TERMINAL_STATUSES = {"FT", "AET", "PEN"}
RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504}
FIXTURE_COLUMNS = [
    "fixture_id",
    "date",
    "timestamp",
    "status_short",
    "status_long",
    "league_id",
    "league_name",
    "league_country",
    "league_season",
    "round",
    "home_team_id",
    "home_team",
    "away_team_id",
    "away_team",
    "home_goals",
    "away_goals",
    "halftime_home_goals",
    "halftime_away_goals",
    "fulltime_home_goals",
    "fulltime_away_goals",
    "extra_home_goals",
    "extra_away_goals",
    "penalty_home_goals",
    "penalty_away_goals",
]
RECENT_FORM_COLUMNS = [
    "fixture_id",
    "date",
    "status_short",
    "status_long",
    "league_name",
    "league_country",
    "league_season",
    "round",
    "team_id",
    "team_name",
    "venue",
    "opponent_id",
    "opponent",
    "goals_for",
    "goals_against",
    "goal_difference",
    "result",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
]
INJURY_COLUMNS = [
    "fixture_id",
    "date",
    "team_id",
    "team_name",
    "player_id",
    "player_name",
    "injury_type",
    "reason",
    "league_name",
    "league_season",
]


class ApiFootballError(RuntimeError):
    """Raised when API-FOOTBALL returns an unusable response."""


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def slugify(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def get_api_key() -> str:
    api_key = os.getenv(API_KEY_ENV, "").strip()
    if not api_key:
        raise ApiFootballError(
            f"Missing API key. Please set environment variable {API_KEY_ENV}."
        )
    return api_key


def api_get(
    endpoint: str,
    params: dict[str, Any] | None = None,
    retries: int = 3,
    timeout: int = 20,
) -> list[dict[str, Any]]:
    """Call API-FOOTBALL with retries and return the response list."""
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    headers = {"x-apisports-key": get_api_key()}
    params = params or {}
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            logging.debug("GET %s params=%s attempt=%s", url, params, attempt)
            response = requests.get(url, headers=headers, params=params, timeout=timeout)

            if response.status_code in RETRYABLE_STATUSES and attempt < retries:
                retry_after = response.headers.get("Retry-After")
                sleep_seconds = int(retry_after) if retry_after and retry_after.isdigit() else attempt * 2
                logging.warning(
                    "Retryable API status %s for %s. Retrying in %s seconds.",
                    response.status_code,
                    endpoint,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
                continue

            response.raise_for_status()
            payload = response.json()

            api_errors = payload.get("errors")
            if api_errors:
                raise ApiFootballError(
                    f"API-FOOTBALL returned errors for {endpoint}: {api_errors}"
                )

            data = payload.get("response")
            if not isinstance(data, list):
                raise ApiFootballError(
                    f"Unexpected API response shape for {endpoint}: response is not a list."
                )

            return data

        except (requests.RequestException, ValueError, ApiFootballError) as exc:
            last_error = exc
            if attempt < retries:
                sleep_seconds = attempt * 2
                logging.warning(
                    "API request failed for %s: %s. Retrying in %s seconds.",
                    endpoint,
                    exc,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
                continue
            break

    raise ApiFootballError(f"API request failed for {endpoint}: {last_error}") from last_error


def find_team_id(team_name: str) -> tuple[int, str, str]:
    """Find the best national-team match for a team name."""
    teams = api_get("teams", {"search": team_name})
    if not teams:
        raise ApiFootballError(f"Team not found: {team_name}")

    exact_matches = [
        item
        for item in teams
        if item.get("team", {}).get("name", "").lower() == team_name.lower()
    ]
    candidates = exact_matches or teams

    for item in candidates:
        team = item.get("team", {})
        if team.get("national"):
            return int(team["id"]), str(team["name"]), str(team.get("country", ""))

    team = candidates[0].get("team", {})
    if "id" not in team:
        raise ApiFootballError(f"Malformed team lookup response for: {team_name}")
    return int(team["id"]), str(team.get("name", team_name)), str(team.get("country", ""))


def extract_fixture_row(match: dict[str, Any]) -> dict[str, Any]:
    fixture = match.get("fixture", {})
    league = match.get("league", {})
    teams = match.get("teams", {})
    goals = match.get("goals", {})
    score = match.get("score", {})
    status = fixture.get("status", {})

    return {
        "fixture_id": fixture.get("id"),
        "date": fixture.get("date"),
        "timestamp": fixture.get("timestamp"),
        "status_short": status.get("short", ""),
        "status_long": status.get("long", ""),
        "league_id": league.get("id"),
        "league_name": league.get("name", ""),
        "league_country": league.get("country", ""),
        "league_season": league.get("season"),
        "round": league.get("round", ""),
        "home_team_id": teams.get("home", {}).get("id"),
        "home_team": teams.get("home", {}).get("name", ""),
        "away_team_id": teams.get("away", {}).get("id"),
        "away_team": teams.get("away", {}).get("name", ""),
        "home_goals": goals.get("home"),
        "away_goals": goals.get("away"),
        "halftime_home_goals": score.get("halftime", {}).get("home"),
        "halftime_away_goals": score.get("halftime", {}).get("away"),
        "fulltime_home_goals": score.get("fulltime", {}).get("home"),
        "fulltime_away_goals": score.get("fulltime", {}).get("away"),
        "extra_home_goals": score.get("extratime", {}).get("home"),
        "extra_away_goals": score.get("extratime", {}).get("away"),
        "penalty_home_goals": score.get("penalty", {}).get("home"),
        "penalty_away_goals": score.get("penalty", {}).get("away"),
    }


def clean_fixture_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=FIXTURE_COLUMNS)

    df = df.drop_duplicates(subset=["fixture_id"]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

    text_columns = df.select_dtypes(include=["object"]).columns
    df[text_columns] = df[text_columns].fillna("")

    goal_columns = [column for column in df.columns if column.endswith("_goals")]
    for column in goal_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.sort_values(["date", "fixture_id"], ascending=[False, False])


def get_head_to_head(team1_id: int, team2_id: int) -> pd.DataFrame:
    matches = api_get("fixtures/headtohead", {"h2h": f"{team1_id}-{team2_id}"})
    rows = [extract_fixture_row(match) for match in matches]
    return clean_fixture_dataframe(pd.DataFrame(rows))


def result_for_team(row: pd.Series, team_id: int) -> str:
    if row.get("status_short") not in TERMINAL_STATUSES:
        return "NOT_FINAL"

    home_goals = row.get("home_goals")
    away_goals = row.get("away_goals")
    if pd.isna(home_goals) or pd.isna(away_goals):
        return "UNKNOWN"

    is_home = int(row["home_team_id"]) == team_id
    team_goals = home_goals if is_home else away_goals
    opponent_goals = away_goals if is_home else home_goals

    if team_goals > opponent_goals:
        return "W"
    if team_goals < opponent_goals:
        return "L"
    return "D"


def add_team_perspective(df: pd.DataFrame, team_id: int, team_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=RECENT_FORM_COLUMNS)

    df = df.copy()
    df["team_id"] = team_id
    df["team_name"] = team_name
    df["venue"] = df["home_team_id"].apply(lambda value: "home" if value == team_id else "away")
    df["opponent_id"] = df.apply(
        lambda row: row["away_team_id"] if int(row["home_team_id"]) == team_id else row["home_team_id"],
        axis=1,
    )
    df["opponent"] = df.apply(
        lambda row: row["away_team"] if int(row["home_team_id"]) == team_id else row["home_team"],
        axis=1,
    )
    df["goals_for"] = df.apply(
        lambda row: row["home_goals"] if int(row["home_team_id"]) == team_id else row["away_goals"],
        axis=1,
    )
    df["goals_against"] = df.apply(
        lambda row: row["away_goals"] if int(row["home_team_id"]) == team_id else row["home_goals"],
        axis=1,
    )
    df["goal_difference"] = df["goals_for"] - df["goals_against"]
    df["result"] = df.apply(lambda row: result_for_team(row, team_id), axis=1)

    ordered_columns = [
        "fixture_id",
        "date",
        "status_short",
        "status_long",
        "league_name",
        "league_country",
        "league_season",
        "round",
        "team_id",
        "team_name",
        "venue",
        "opponent_id",
        "opponent",
        "goals_for",
        "goals_against",
        "goal_difference",
        "result",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    ]
    return df[ordered_columns]


def get_recent_matches(team_id: int, team_name: str, last: int = 10) -> pd.DataFrame:
    matches = api_get("fixtures", {"team": team_id, "last": last})
    rows = [extract_fixture_row(match) for match in matches]
    fixtures = clean_fixture_dataframe(pd.DataFrame(rows))
    return add_team_perspective(fixtures, team_id, team_name)


def summarize_recent_form(df: pd.DataFrame, team_name: str) -> dict[str, Any]:
    final_matches = df[df["result"].isin(["W", "D", "L"])].copy() if not df.empty else df
    games_played = int(len(final_matches))

    if games_played == 0:
        return {
            "team_name": team_name,
            "games_played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "win_rate": 0.0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_difference": 0,
            "avg_goals_for": 0.0,
            "avg_goals_against": 0.0,
        }

    wins = int((final_matches["result"] == "W").sum())
    draws = int((final_matches["result"] == "D").sum())
    losses = int((final_matches["result"] == "L").sum())
    goals_for = int(final_matches["goals_for"].fillna(0).sum())
    goals_against = int(final_matches["goals_against"].fillna(0).sum())

    return {
        "team_name": team_name,
        "games_played": games_played,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": round(wins / games_played, 4),
        "goals_for": goals_for,
        "goals_against": goals_against,
        "goal_difference": goals_for - goals_against,
        "avg_goals_for": round(goals_for / games_played, 4),
        "avg_goals_against": round(goals_against / games_played, 4),
    }


def get_next_fixture_id(team1_id: int, team2_id: int) -> int | None:
    try:
        upcoming = api_get("fixtures", {"team": team1_id, "next": 10})
    except ApiFootballError as exc:
        logging.warning("Could not fetch upcoming fixtures: %s", exc)
        return None

    for match in upcoming:
        teams = match.get("teams", {})
        home_id = teams.get("home", {}).get("id")
        away_id = teams.get("away", {}).get("id")
        if {home_id, away_id} == {team1_id, team2_id}:
            return match.get("fixture", {}).get("id")
    return None


def get_injuries(team_id: int, team_name: str, fixture_id: int | None = None) -> pd.DataFrame:
    params: dict[str, Any] = {"team": team_id}
    if fixture_id:
        params["fixture"] = fixture_id

    try:
        injuries = api_get("injuries", params)
    except ApiFootballError as exc:
        logging.warning("Could not fetch injuries for %s: %s", team_name, exc)
        return pd.DataFrame(columns=INJURY_COLUMNS)

    rows = []
    for item in injuries:
        fixture = item.get("fixture", {})
        team = item.get("team", {})
        player = item.get("player", {})
        league = item.get("league", {})
        rows.append(
            {
                "fixture_id": fixture.get("id"),
                "date": fixture.get("date"),
                "team_id": team.get("id", team_id),
                "team_name": team.get("name", team_name),
                "player_id": player.get("id"),
                "player_name": player.get("name", ""),
                "injury_type": player.get("type", ""),
                "reason": player.get("reason", ""),
                "league_name": league.get("name", ""),
                "league_season": league.get("season"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=INJURY_COLUMNS)

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    text_columns = df.select_dtypes(include=["object"]).columns
    df[text_columns] = df[text_columns].fillna("")
    return df.drop_duplicates().sort_values(["date", "team_name", "player_name"], ascending=[False, True, True])


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logging.info("Saved %s rows to %s", len(df), path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch head-to-head, recent form, and injury data from API-FOOTBALL."
    )
    parser.add_argument("--team1", default="Argentina", help="First team name.")
    parser.add_argument("--team2", default="France", help="Second team name.")
    parser.add_argument("--last", type=int, default=10, help="Number of recent matches per team.")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where CSV files will be written.",
    )
    parser.add_argument(
        "--fixture-id",
        type=int,
        default=None,
        help="Optional upcoming fixture id for injury lookup.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    try:
        team1_id, team1_name, team1_country = find_team_id(args.team1)
        team2_id, team2_name, team2_country = find_team_id(args.team2)
        logging.info("Team 1: %s (%s), id=%s", team1_name, team1_country, team1_id)
        logging.info("Team 2: %s (%s), id=%s", team2_name, team2_country, team2_id)

        h2h_df = get_head_to_head(team1_id, team2_id)
        recent_team1_df = get_recent_matches(team1_id, team1_name, args.last)
        recent_team2_df = get_recent_matches(team2_id, team2_name, args.last)

        fixture_id = args.fixture_id or get_next_fixture_id(team1_id, team2_id)
        injuries_df = pd.concat(
            [
                get_injuries(team1_id, team1_name, fixture_id),
                get_injuries(team2_id, team2_name, fixture_id),
            ],
            ignore_index=True,
        )

        summary_df = pd.DataFrame(
            [
                summarize_recent_form(recent_team1_df, team1_name),
                summarize_recent_form(recent_team2_df, team2_name),
            ]
        )

        output_dir = Path(args.output_dir)
        suffix = f"{slugify(team1_name)}_{slugify(team2_name)}"
        save_csv(h2h_df, output_dir / f"h2h_{suffix}.csv")
        save_csv(recent_team1_df, output_dir / f"recent_form_{slugify(team1_name)}.csv")
        save_csv(recent_team2_df, output_dir / f"recent_form_{slugify(team2_name)}.csv")
        save_csv(injuries_df, output_dir / f"injuries_{suffix}.csv")
        save_csv(summary_df, output_dir / f"recent_form_summary_{suffix}.csv")

        print("\nRecent form summary:")
        print(summary_df.to_string(index=False))
        if fixture_id:
            print(f"\nInjury lookup fixture_id: {fixture_id}")
        else:
            print("\nNo upcoming fixture id found; injury lookup used team-level search.")
        return 0

    except ApiFootballError as exc:
        logging.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logging.error("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
