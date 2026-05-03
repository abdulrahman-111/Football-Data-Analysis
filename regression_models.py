"""
regression_models.py
====================
One regression model trained on StatsBomb open data:

  Model — Linear Regression
    Predicts: home goals scored AND away goals scored (separately),
              then derives the predicted winner from the goal difference.
    Features: exp_home_goals, exp_away_goals

Data is imported and cleaned via data_cleaning.py.

Run standalone:
    python regression_models.py
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ── CSV paths (written by data_cleaning.py) ────────────────────────────────────
DATA_DIR     = os.path.join(os.path.dirname(__file__), "data")
MATCHES_CSV  = os.path.join(DATA_DIR, "matches_clean.csv")

# ── Feature set ───────────────────────────────────────────────────────────────
MATCH_FEATURES = ["exp_home_goals", "exp_away_goals"]

OUTCOME_LABELS = {0: "Home Win", 1: "Draw", 2: "Away Win"}


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL — LINEAR REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════

class LinearRegressionGoals:
    """
    Linear Regression model that:
      1. Predicts home goals scored
      2. Predicts away goals scored
      3. Derives the predicted winner from the goal difference

    Two separate LinearRegression models are trained — one per target.
    Features used: exp_home_goals, exp_away_goals
    """

    def __init__(self):
        self.home_model  = None   # predicts home goals
        self.away_model  = None   # predicts away goals
        self.metrics     = {}
        self.is_fitted   = False

    def fit(self, matches_df: pd.DataFrame = None, verbose: bool = True) -> "LinearRegressionGoals":
        """Train both goal models. Reads from data/matches_clean.csv if no DataFrame passed."""
        if matches_df is None:
            if not os.path.exists(MATCHES_CSV):
                raise FileNotFoundError(
                    f"CSV not found: {MATCHES_CSV}\n"
                    "Run data_cleaning.py first to generate the CSV files."
                )
            matches_df = pd.read_csv(MATCHES_CSV)
            if verbose:
                print(f"  Loaded {len(matches_df):,} rows from {MATCHES_CSV}")
        if verbose:
            print("\n" + "─" * 50)
            print("  MODEL — Linear Regression")
            print("  Targets: home goals, away goals → winner")
            print("  Features:", MATCH_FEATURES)
            print("─" * 50)

        df = matches_df.dropna(subset=MATCH_FEATURES + ["home_score", "away_score"]).copy()

        X = df[MATCH_FEATURES]

        # ── Train/test split (same split for both targets) ────────────────────
        X_train, X_test, idx_train, idx_test = train_test_split(
            X, X.index, test_size=0.2, random_state=42
        )
        y_home_train = df.loc[idx_train, "home_score"]
        y_home_test  = df.loc[idx_test,  "home_score"]
        y_away_train = df.loc[idx_train, "away_score"]
        y_away_test  = df.loc[idx_test,  "away_score"]

        # ── Fit home goals model ───────────────────────────────────────────────
        self.home_model = LinearRegression()
        self.home_model.fit(X_train, y_home_train)
        home_pred  = self.home_model.predict(X_test)
        home_r2    = r2_score(y_home_test, home_pred)
        home_rmse  = mean_squared_error(y_home_test, home_pred) ** 0.5

        # ── Fit away goals model ───────────────────────────────────────────────
        self.away_model = LinearRegression()
        self.away_model.fit(X_train, y_away_train)
        away_pred  = self.away_model.predict(X_test)
        away_r2    = r2_score(y_away_test, away_pred)
        away_rmse  = mean_squared_error(y_away_test, away_pred) ** 0.5

        # ── Winner accuracy from predicted scores ──────────────────────────────
        pred_outcome  = self._scores_to_outcome(home_pred, away_pred)
        true_outcome  = df.loc[idx_test, "outcome"].values
        winner_acc    = (pred_outcome == true_outcome).mean()

        self.metrics = {
            "model":       "Linear Regression",
            "features":    MATCH_FEATURES,
            "train_size":  len(X_train),
            "test_size":   len(X_test),
            "home_goals": {
                "r2":          round(home_r2, 4),
                "rmse":        round(home_rmse, 4),
                "coefficients": dict(zip(MATCH_FEATURES, self.home_model.coef_.tolist())),
                "intercept":   float(self.home_model.intercept_),
            },
            "away_goals": {
                "r2":          round(away_r2, 4),
                "rmse":        round(away_rmse, 4),
                "coefficients": dict(zip(MATCH_FEATURES, self.away_model.coef_.tolist())),
                "intercept":   float(self.away_model.intercept_),
            },
            "winner_prediction_accuracy": round(winner_acc, 4),
        }
        self.is_fitted = True

        if verbose:
            print(f"\n  Home goals model (test data):")
            print(f"    R²   = {home_r2:.4f}")
            print(f"    RMSE = {home_rmse:.4f}")
            print(f"    Coefficients:")
            for feat, coef in zip(MATCH_FEATURES, self.home_model.coef_):
                print(f"      {feat:<22} {coef:+.4f}")
            print(f"      {'intercept':<22} {float(self.home_model.intercept_):+.4f}")

            print(f"\n  Away goals model (test data):")
            print(f"    R²   = {away_r2:.4f}")
            print(f"    RMSE = {away_rmse:.4f}")
            print(f"    Coefficients:")
            for feat, coef in zip(MATCH_FEATURES, self.away_model.coef_):
                print(f"      {feat:<22} {coef:+.4f}")
            print(f"      {'intercept':<22} {float(self.away_model.intercept_):+.4f}")

            print(f"\n  Winner prediction accuracy (test data): {winner_acc*100:.1f}%")
            print(f"  Train samples: {len(X_train)}  |  Test samples: {len(X_test)}")

        return self

    def predict(self, home_stats: dict, away_stats: dict) -> dict:
        """
        Predict goals and winner for a given matchup.

        Parameters
        ----------
        home_stats : dict with keys: avg_goals, avg_conceded
        away_stats : same structure

        Returns
        -------
        dict with keys:
          predicted_home_goals : float
          predicted_away_goals : float
          goal_difference      : float  (home - away)
          predicted_winner     : str    ("Home Win", "Draw", "Away Win")
          outcome_code         : int    (0, 1, or 2)
        """
        features = self._build_features(home_stats, away_stats)
        X = pd.DataFrame([features])[MATCH_FEATURES]

        home_goals = max(0.0, float(self.home_model.predict(X)[0]))
        away_goals = max(0.0, float(self.away_model.predict(X)[0]))
        diff       = home_goals - away_goals

        # Classify winner: draw band ±0.15 goals
        if diff > 0.15:
            outcome = 0
        elif diff < -0.15:
            outcome = 2
        else:
            outcome = 1

        return {
            "predicted_home_goals": round(home_goals, 2),
            "predicted_away_goals": round(away_goals, 2),
            "goal_difference":      round(diff, 2),
            "predicted_winner":     OUTCOME_LABELS[outcome],
            "outcome_code":         outcome,
        }

    @staticmethod
    def _build_features(home: dict, away: dict) -> dict:
        return {
            "exp_home_goals": (home["avg_goals"] + away["avg_conceded"]) / 2,
            "exp_away_goals": (away["avg_goals"] + home["avg_conceded"]) / 2,
        }

    @staticmethod
    def _scores_to_outcome(home_preds, away_preds):
        """Convert arrays of predicted scores to outcome codes."""
        diff = home_preds - away_preds
        return np.where(diff > 0.15, 0, np.where(diff < -0.15, 2, 1))


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class RegressionModels:
    """
    Holds the Linear Regression model.
    Reads training data from data/matches_clean.csv (written by data_cleaning.py).
    Imported by app.py and other modules that need prediction.
    """

    def __init__(self):
        self.lr = LinearRegressionGoals()

    def fit(self, shots_df: pd.DataFrame = None, matches_df: pd.DataFrame = None,
            verbose: bool = True) -> "RegressionModels":
        print("\n" + "=" * 50)
        print("  REGRESSION MODEL")
        print("  Data source: data/matches_clean.csv")
        print("=" * 50)
        self.lr.fit(matches_df, verbose=verbose)
        print("\n✓ Regression model trained successfully")
        return self

    def predict(self, home_stats: dict, away_stats: dict) -> dict:
        """Predict goals and winner using Linear Regression."""
        return self.lr.predict(home_stats, away_stats)

    def predict_goals(self, home_stats: dict, away_stats: dict) -> float:
        """Returns predicted home goals (for backwards compatibility with app.py)."""
        return self.lr.predict(home_stats, away_stats)["predicted_home_goals"]

    @property
    def metrics(self) -> dict:
        return {"linear_regression": self.lr.metrics}


# ── Standalone run ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(MATCHES_CSV):
        print("ERROR: data/matches_clean.csv not found.")
        print("Please run data_cleaning.py first:\n  python data_cleaning.py")
        exit(1)

    print(f"Loading matches from {MATCHES_CSV}...")
    reg = RegressionModels()
    reg.fit()

    print("\n" + "=" * 50)
    print("  PREDICTION EXAMPLES")
    print("=" * 50)

    matchups = [
        (
            {"name": "Real Madrid", "avg_goals": 2.6, "avg_conceded": 0.8},
            {"name": "Barcelona",   "avg_goals": 2.5, "avg_conceded": 0.9},
        ),
        (
            {"name": "Manchester City", "avg_goals": 2.4, "avg_conceded": 0.9},
            {"name": "West Ham United", "avg_goals": 1.5, "avg_conceded": 1.6},
        ),
        (
            {"name": "Brighton",   "avg_goals": 1.6, "avg_conceded": 1.4},
            {"name": "Juventus",   "avg_goals": 1.8, "avg_conceded": 1.0},
        ),
    ]

    for home, away in matchups:
        result = reg.predict(home, away)
        print(f"\n  {home['name']} vs {away['name']}")
        print(f"    Predicted score:  {result['predicted_home_goals']} – {result['predicted_away_goals']}")
        print(f"    Goal difference:  {result['goal_difference']:+.2f}")
        print(f"    Predicted winner: {result['predicted_winner']}")
