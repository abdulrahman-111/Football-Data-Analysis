"""
data_cleaning.py

Fetches raw StatsBomb open data from:
  https://github.com/statsbomb/open-data

Cleans the data and saves two CSV files:
  - data/shots_clean.csv   → used by classification_models.py
  - data/matches_clean.csv → used by regression_models.py

Run this file first before running the model files:
    python data_cleaning.py

"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Output CSV paths
DATA_DIR        = os.path.join(os.path.dirname(__file__), "data")
SHOTS_CSV       = os.path.join(DATA_DIR, "shots_clean.csv")
MATCHES_CSV     = os.path.join(DATA_DIR, "matches_clean.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# StatsBomb free competitions
# None = fetch ALL available competitions from the GitHub repo
# Or use a list like [(2, 44), (11, 90)] to limit scope for testing
FREE_COMPS = None

# Known team ratings for feature engineering
_KNOWN_TEAMS = {
    "manchester city":    {"attack": 88, "defense": 85, "midfield": 87, "form": 78, "avg_goals": 2.4, "avg_conceded": 0.9},
    "arsenal":            {"attack": 84, "defense": 82, "midfield": 83, "form": 75, "avg_goals": 2.2, "avg_conceded": 1.1},
    "liverpool":          {"attack": 86, "defense": 80, "midfield": 85, "form": 77, "avg_goals": 2.3, "avg_conceded": 1.0},
    "chelsea":            {"attack": 81, "defense": 79, "midfield": 80, "form": 70, "avg_goals": 1.9, "avg_conceded": 1.2},
    "real madrid":        {"attack": 91, "defense": 86, "midfield": 89, "form": 82, "avg_goals": 2.6, "avg_conceded": 0.8},
    "barcelona":          {"attack": 89, "defense": 83, "midfield": 88, "form": 80, "avg_goals": 2.5, "avg_conceded": 0.9},
    "atletico madrid":    {"attack": 82, "defense": 88, "midfield": 81, "form": 74, "avg_goals": 1.9, "avg_conceded": 0.8},
    "atlético madrid":    {"attack": 82, "defense": 88, "midfield": 81, "form": 74, "avg_goals": 1.9, "avg_conceded": 0.8},
    "bayern munich":      {"attack": 90, "defense": 84, "midfield": 87, "form": 81, "avg_goals": 2.7, "avg_conceded": 0.9},
    "borussia dortmund":  {"attack": 83, "defense": 78, "midfield": 82, "form": 72, "avg_goals": 2.1, "avg_conceded": 1.2},
    "paris saint-germain":{"attack": 88, "defense": 81, "midfield": 85, "form": 79, "avg_goals": 2.5, "avg_conceded": 1.0},
    "psg":                {"attack": 88, "defense": 81, "midfield": 85, "form": 79, "avg_goals": 2.5, "avg_conceded": 1.0},
    "inter milan":        {"attack": 85, "defense": 84, "midfield": 83, "form": 76, "avg_goals": 2.2, "avg_conceded": 0.9},
    "internazionale":     {"attack": 85, "defense": 84, "midfield": 83, "form": 76, "avg_goals": 2.2, "avg_conceded": 0.9},
    "ac milan":           {"attack": 82, "defense": 81, "midfield": 80, "form": 73, "avg_goals": 2.0, "avg_conceded": 1.1},
    "juventus":           {"attack": 80, "defense": 82, "midfield": 79, "form": 71, "avg_goals": 1.8, "avg_conceded": 1.0},
    "napoli":             {"attack": 83, "defense": 79, "midfield": 81, "form": 74, "avg_goals": 2.1, "avg_conceded": 1.1},
    "tottenham hotspur":  {"attack": 80, "defense": 75, "midfield": 78, "form": 65, "avg_goals": 1.8, "avg_conceded": 1.4},
    "manchester united":  {"attack": 79, "defense": 74, "midfield": 76, "form": 62, "avg_goals": 1.7, "avg_conceded": 1.5},
    "newcastle united":   {"attack": 78, "defense": 77, "midfield": 76, "form": 68, "avg_goals": 1.8, "avg_conceded": 1.3},
    "aston villa":        {"attack": 77, "defense": 76, "midfield": 75, "form": 67, "avg_goals": 1.7, "avg_conceded": 1.3},
    "brighton":           {"attack": 76, "defense": 73, "midfield": 77, "form": 66, "avg_goals": 1.6, "avg_conceded": 1.4},
    "west ham united":    {"attack": 74, "defense": 72, "midfield": 73, "form": 60, "avg_goals": 1.5, "avg_conceded": 1.6},
    "brazil":             {"attack": 87, "defense": 83, "midfield": 86, "form": 79, "avg_goals": 2.4, "avg_conceded": 0.9},
    "france":             {"attack": 88, "defense": 85, "midfield": 84, "form": 81, "avg_goals": 2.3, "avg_conceded": 0.8},
    "argentina":          {"attack": 87, "defense": 82, "midfield": 85, "form": 82, "avg_goals": 2.2, "avg_conceded": 0.9},
    "england":            {"attack": 82, "defense": 80, "midfield": 81, "form": 75, "avg_goals": 2.0, "avg_conceded": 1.0},
    "germany":            {"attack": 83, "defense": 79, "midfield": 82, "form": 74, "avg_goals": 2.1, "avg_conceded": 1.1},
    "spain":              {"attack": 84, "defense": 83, "midfield": 86, "form": 77, "avg_goals": 2.0, "avg_conceded": 0.8},
}
_DEFAULT_RATING = {"attack": 75, "defense": 74, "midfield": 75, "form": 65, "avg_goals": 1.5, "avg_conceded": 1.5}


def get_team_rating(name: str) -> dict:
    """Look up known team ratings; return defaults for unknown teams."""
    if not name:
        return _DEFAULT_RATING.copy()
    key = str(name).lower().strip()
    for k, v in _KNOWN_TEAMS.items():
        if k in key or key in k:
            return v.copy()
    return _DEFAULT_RATING.copy()


# STEP 1: FETCH RAW DATA FROM STATSBOMB GITHUB

def _fetch_raw(max_matches: int = None) -> tuple:
    """
    Downloads ALL raw StatsBomb events and match records from the GitHub repo.
    If max_matches is set, limits matches per competition (useful for testing).
    Returns (raw_shots: list[dict], raw_matches: list[dict])
    """
    from statsbombpy import sb

    raw_shots   = []
    raw_matches = []

    # Discover all available competitions
    if FREE_COMPS is None:
        print("  Discovering all available StatsBomb competitions...")
        all_comps = sb.competitions()
        comp_list = list(zip(
            all_comps["competition_id"].tolist(),
            all_comps["season_id"].tolist()
        ))
        print(f"  Found {len(comp_list)} competition/season combinations")
    else:
        comp_list = FREE_COMPS

    for comp_id, season_id in comp_list:
        print(f"  → Fetching competition {comp_id} / season {season_id} ...")
        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
            if matches is None or len(matches) == 0:
                continue

            comp_name   = matches["competition"].iloc[0] if "competition" in matches.columns else str(comp_id)
            season_name = matches["season"].iloc[0]      if "season"      in matches.columns else str(season_id)

            # Load all matches or sample if max_matches is set
            if max_matches is not None:
                n_sample = max(1, max_matches // len(comp_list))
                sample = matches.sample(min(n_sample, len(matches)), random_state=42)
            else:
                sample = matches  # load every single match

            for _, match in sample.iterrows():
                mid = match["match_id"]
                try:
                    evs = sb.events(match_id=mid)
                    if evs is None or len(evs) == 0:
                        continue

                    # Collect raw shot rows
                    for _, row in evs[evs["type"] == "Shot"].iterrows():
                        loc = row.get("location", [None, None])
                        raw_shots.append({
                            "match_id":           mid,
                            "competition":        comp_name,
                            "season":             season_name,
                            "home_team":          match.get("home_team", "Unknown"),
                            "away_team":          match.get("away_team", "Unknown"),
                            "minute":             row.get("minute"),
                            "x":                  loc[0] if isinstance(loc, list) and len(loc) > 0 else None,
                            "y":                  loc[1] if isinstance(loc, list) and len(loc) > 1 else None,
                            "statsbomb_xg":       row.get("shot_statsbomb_xg"),
                            "shot_outcome":       row.get("shot_outcome"),
                            "body_part_raw":      row.get("shot_body_part"),
                            "under_pressure_raw": row.get("under_pressure"),
                        })

                    # Collect raw match rows
                    home_passes = len(evs[(evs["type"] == "Pass") & (evs["team"] == match.get("home_team", ""))])
                    away_passes = len(evs[(evs["type"] == "Pass") & (evs["team"] == match.get("away_team", ""))])
                    home_shots  = len(evs[(evs["type"] == "Shot") & (evs["team"] == match.get("home_team", ""))])
                    away_shots  = len(evs[(evs["type"] == "Shot") & (evs["team"] == match.get("away_team", ""))])
                    shot_evs    = evs[evs["type"] == "Shot"]
                    home_xg_raw = shot_evs[shot_evs["team"] == match.get("home_team", "")].get("shot_statsbomb_xg", pd.Series()).sum()
                    away_xg_raw = shot_evs[shot_evs["team"] == match.get("away_team", "")].get("shot_statsbomb_xg", pd.Series()).sum()

                    raw_matches.append({
                        "match_id":    mid,
                        "competition": comp_name,
                        "season":      season_name,
                        "home_team":   match.get("home_team", "Unknown"),
                        "away_team":   match.get("away_team", "Unknown"),
                        "home_score":  match.get("home_score"),
                        "away_score":  match.get("away_score"),
                        "home_shots":  home_shots,
                        "away_shots":  away_shots,
                        "home_passes": home_passes,
                        "away_passes": away_passes,
                        "home_xg_raw": float(home_xg_raw),
                        "away_xg_raw": float(away_xg_raw),
                    })

                    print(f"     ✓ {match.get('home_team','?')} vs {match.get('away_team','?')} "
                          f"({match.get('home_score')}-{match.get('away_score')})")

                except Exception as e:
                    print(f"     ✗ match {mid} skipped: {e}")

        except Exception as e:
            print(f"  ✗ Competition {comp_id}/{season_id} skipped: {e}")

    if not raw_matches:
        raise ValueError("No data fetched from StatsBomb")

    return raw_shots, raw_matches



# STEP 2: CLEAN SHOTS DATA


def _clean_shots(raw_shots: list) -> pd.DataFrame:
    """
    Cleaning steps applied to shot-level data:
      1. Drop rows missing x/y coordinates
      2. Clip x/y to valid StatsBomb pitch bounds (0–120, 0–80)
      3. Engineer distance and angle from raw coordinates
      4. Encode body_part as binary (1 = head, 0 = foot/other)
      5. Encode under_pressure as binary (1 = yes, 0 = no/missing)
      6. Encode goal as binary from shot_outcome string/dict
      7. Fill missing xG using logistic fallback formula
      8. Clip xG to [0.01, 0.98]
      9. Drop duplicates
     10. Reset index
    """
    df = pd.DataFrame(raw_shots)
    print(f"  Raw shots: {len(df)} rows")

    # 1. Drop rows with missing coordinates
    df = df.dropna(subset=["x", "y"])
    print(f"  After dropping missing coordinates: {len(df)} rows")

    # 2. Clip to valid pitch bounds
    df["x"] = df["x"].clip(0, 120)
    df["y"] = df["y"].clip(0, 80)

    # 3. Engineer distance and angle
    df["distance"] = df.apply(
        lambda r: round(np.sqrt((120 - r["x"]) ** 2 + (40 - r["y"]) ** 2), 2), axis=1
    )

    def _angle(row):
        post1 = np.array([120, 36])
        post2 = np.array([120, 44])
        pos   = np.array([row["x"], row["y"]])
        v1, v2 = post1 - pos, post2 - pos
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        return round(np.degrees(np.arccos(np.clip(cos_a, -1, 1))), 2)

    df["angle"] = df.apply(_angle, axis=1)

    # 4. Encode body_part
    df["body_part"] = df["body_part_raw"].apply(
        lambda b: 1 if (isinstance(b, dict) and "Head" in str(b)) or "Head" in str(b) else 0
    )

    # 5. Encode under_pressure
    df["under_pressure"] = df["under_pressure_raw"].apply(
        lambda v: 0 if pd.isna(v) or v is None or v is False else 1
    )

    # 6. Encode goal
    df["goal"] = df["shot_outcome"].apply(
        lambda o: 1 if (isinstance(o, dict) and o.get("name") == "Goal") or o == "Goal" else 0
    )

    # 7. Fill missing xG with fallback formula
    def _fallback_xg(dist, angle):
        return round(float(1 / (1 + np.exp(0.18 * dist - 0.045 * angle - 1.8))), 4)

    df["xg"] = df.apply(
        lambda r: r["statsbomb_xg"] if pd.notna(r["statsbomb_xg"])
                  else _fallback_xg(r["distance"], r["angle"]),
        axis=1
    )

    # 8. Clip xG
    df["xg"] = df["xg"].clip(0.01, 0.98)

    # 9. Select final columns
    shots_clean = df[[
        "match_id", "competition", "season",
        "home_team", "away_team", "minute",
        "x", "y", "distance", "angle",
        "body_part", "under_pressure", "xg", "goal"
    ]].copy()

    # 10. Drop duplicates and reset index
    shots_clean = shots_clean.drop_duplicates().reset_index(drop=True)

    print(f"  Cleaned shots: {len(shots_clean)} rows")
    print(f"  Goal rate: {shots_clean['goal'].mean()*100:.1f}%")
    print(f"  Avg xG: {shots_clean['xg'].mean():.3f}")

    return shots_clean


# STEP 3: CLEAN MATCHES DATA

def _clean_matches(raw_matches: list) -> pd.DataFrame:
    """
    Cleaning steps applied to match-level data:
      1. Drop rows missing home_score or away_score
      2. Cast scores to int
      3. Clip scores to [0, 15] (remove corrupt outliers)
      4. Engineer total_goals and outcome label
      5. Engineer possession % from pass counts
      6. Join team ratings as engineered features
      7. Engineer expected goals ratio features
      8. Drop duplicates
      9. Reset index
    """
    df = pd.DataFrame(raw_matches)
    print(f"  Raw matches: {len(df)} rows")

    # 1. Drop rows missing scores
    df = df.dropna(subset=["home_score", "away_score"])

    # 2. Cast to int
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # 3. Clip scores to sensible range
    df["home_score"] = df["home_score"].clip(0, 15)
    df["away_score"] = df["away_score"].clip(0, 15)

    # 4. Engineer labels
    df["total_goals"] = df["home_score"] + df["away_score"]
    df["outcome"] = df.apply(
        lambda r: 0 if r["home_score"] > r["away_score"]
                  else (2 if r["away_score"] > r["home_score"] else 1),
        axis=1
    )  # 0 = home win, 1 = draw, 2 = away win

    # 5. Engineer possession
    total_passes    = df["home_passes"] + df["away_passes"]
    df["possession"] = (df["home_passes"] / total_passes.replace(0, 1) * 100).round(1)

    # 6. Join team ratings
    def _add_ratings(team_col, prefix):
        ratings = df[team_col].apply(get_team_rating)
        for key in ["attack", "defense", "midfield", "form", "avg_goals", "avg_conceded"]:
            df[f"{prefix}_{key}"] = ratings.apply(lambda r: r[key])

    _add_ratings("home_team", "home")
    _add_ratings("away_team", "away")

    # 7. Engineer expected goals features
    df["exp_home_goals"] = ((df["home_avg_goals"] + df["away_avg_conceded"]) / 2).round(3)
    df["exp_away_goals"] = ((df["away_avg_goals"] + df["home_avg_conceded"]) / 2).round(3)

    # 8 & 9. Drop duplicates and reset index
    matches_clean = df.drop_duplicates(subset=["match_id"]).reset_index(drop=True)

    print(f"  Cleaned matches: {len(matches_clean)} rows")
    print(f"  Outcome dist — Home wins: {(matches_clean['outcome']==0).sum()}, "
          f"Draws: {(matches_clean['outcome']==1).sum()}, "
          f"Away wins: {(matches_clean['outcome']==2).sum()}")

    return matches_clean

# MAIN: fetch, clean, and save to CSV

def run(max_matches: int = None, force_refresh: bool = False):
    """
    Fetch, clean, and save data to:
      data/shots_clean.csv
      data/matches_clean.csv

    Parameters
    ----------
    max_matches   : None = load everything; int = cap per competition (for testing)
    force_refresh : True = re-download even if CSVs already exist
    """
    # Skip if CSVs already exist
    if not force_refresh and os.path.exists(SHOTS_CSV) and os.path.exists(MATCHES_CSV):
        print("✓ CSV files already exist — skipping download.")
        print(f"  {SHOTS_CSV}")
        print(f"  {MATCHES_CSV}")
        print("  Run with force_refresh=True to re-download.\n")
        return

    scope = "ALL competitions" if max_matches is None else f"up to {max_matches} matches/competition"
    print("\n" + "=" * 55)
    print("  DATA CLEANING PIPELINE")
    print(f"  Source : github.com/statsbomb/open-data ({scope})")
    print(f"  Output : data/shots_clean.csv")
    print(f"           data/matches_clean.csv")
    print("=" * 55)

    # Step 1: Fetch
    try:
        print("\n[Step 1] Fetching raw data from StatsBomb GitHub...")
        raw_shots, raw_matches = _fetch_raw(max_matches=max_matches)

        #Step 2: Clean shots
        print("\n[Step 2] Cleaning shot data...")
        shots_clean = _clean_shots(raw_shots)

        # Step 3: Clean matches
        print("\n[Step 3] Cleaning match data...")
        matches_clean = _clean_matches(raw_matches)

    except Exception as e:
        print(f"\n⚠  GitHub fetch failed: {e}")
        print("   Falling back to synthetic StatsBomb-schema data...\n")
        shots_clean, matches_clean = _generate_synthetic()

    # Step 4: Save to CSV
    print("\n[Step 4] Saving cleaned data to CSV files...")
    shots_clean.to_csv(SHOTS_CSV, index=False)
    matches_clean.to_csv(MATCHES_CSV, index=False)

    print(f"\n✓ Saved {len(shots_clean):,} shot rows   → {SHOTS_CSV}")
    print(f"✓ Saved {len(matches_clean):,} match rows  → {MATCHES_CSV}")
    print("\nYou can now run regression_models.py and classification_models.py\n")


if __name__ == "__main__":
    run(force_refresh=True)
