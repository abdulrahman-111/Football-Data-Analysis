"""
classification_models.py
========================
One classification model trained on StatsBomb open data:

  Model: Decision Tree Classifier
    Predicts: Goal (1) or No Goal (0) for a given shot
    Features: distance, angle, body_part, under_pressure

Data is imported and cleaned via data_cleaning.py.

Run standalone:
    python classification_models.py
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# CSV paths (written by data_cleaning.py)
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
SHOTS_CSV  = os.path.join(DATA_DIR, "shots_clean.csv")

# Feature set
SHOT_FEATURES = ["distance", "angle", "body_part", "under_pressure"]


# MODEL: DECISION TREE CLASSIFIER (predict goal / no goal)

class DecisionTreeGoal:
    """
    Decision Tree Classifier that predicts whether a shot results
    in a Goal (1) or No Goal (0), based on shot characteristics.
    """

    def __init__(self, max_depth: int = 7, min_samples_split: int = 20):
        self.model             = None
        self.metrics           = {}
        self.importances       = {}
        self.is_fitted         = False
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, shots_df: pd.DataFrame = None, verbose: bool = True) -> "DecisionTreeGoal":
        """Train the model. Reads from data/shots_clean.csv if no DataFrame passed."""
        if shots_df is None:
            if not os.path.exists(SHOTS_CSV):
                raise FileNotFoundError(
                    f"CSV not found: {SHOTS_CSV}\n"
                    "Run data_cleaning.py first to generate the CSV files."
                )
            shots_df = pd.read_csv(SHOTS_CSV)
            if verbose:
                print(f"  Loaded {len(shots_df):,} rows from {SHOTS_CSV}")
        if verbose:
            print("\n" + "─" * 50)
            print("  MODEL — Decision Tree Classifier")
            print("  Target : Goal (1) / No Goal (0) per shot")
            print("  Features:", SHOT_FEATURES)
            print("─" * 50)

        valid_feats = [f for f in SHOT_FEATURES if f in shots_df.columns]
        df = shots_df.dropna(subset=valid_feats + ["goal"]).copy()

        # Fallback if not enough real shot data
        if len(df) < 100:
            if verbose:
                print("  ⚠ Not enough shot data — generating fallback data")
            df = self._generate_fallback_shots(5000)
            valid_feats = SHOT_FEATURES

        X = df[valid_feats]
        y = df["goal"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=42,
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        cm     = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(
            y_test, y_pred,
            target_names=["No Goal", "Goal"],
            output_dict=True,
            zero_division=0,
        )

        self.importances = dict(zip(valid_feats, self.model.feature_importances_.tolist()))
        self.metrics = {
            "model":                 "Decision Tree Classifier",
            "target":                "Goal / No Goal",
            "features":              valid_feats,
            "accuracy":              round(acc, 4),
            "train_size":            len(X_train),
            "test_size":             len(X_test),
            "classification_report": report,
            "confusion_matrix":      cm,
            "feature_importances":   self.importances,
            "max_depth":             self.max_depth,
            "min_samples_split":     self.min_samples_split,
        }
        self.is_fitted = True

        if verbose:
            print(f"\n  Accuracy (test data) = {acc:.4f}")
            print(f"  Train samples: {len(X_train)}  |  Test samples: {len(X_test)}")
            print(f"\n  Classification report (test data):")
            for label in ["No Goal", "Goal"]:
                r = report[label]
                print(f"    {label:<10} precision={r['precision']:.2f}  "
                      f"recall={r['recall']:.2f}  f1={r['f1-score']:.2f}")
            print(f"\n  Confusion matrix:")
            print(f"    {'':10} Pred No  Pred Goal")
            for i, row_label in enumerate(["Act No   ", "Act Goal "]):
                print(f"    {row_label}  {cm[i]}")
            print(f"\n  Feature importances:")
            for feat, imp in sorted(self.importances.items(), key=lambda x: -x[1]):
                bar = "█" * int(imp * 40)
                print(f"    {feat:<18} {imp:.4f}  {bar}")

        return self

    def predict(self, distance: float, angle: float,
                body_part: int = 0, under_pressure: int = 0) -> int:
        """
        Predict Goal (1) or No Goal (0) for a shot.

        Parameters
        ----------
        distance       : distance to goal in metres (5–35)
        angle          : angle to goal in degrees (5–85)
        body_part      : 1 = head, 0 = foot
        under_pressure : 1 = under pressure, 0 = not

        Returns
        -------
        1 (Goal) or 0 (No Goal)
        """
        X = pd.DataFrame([{
            "distance":       distance,
            "angle":          angle,
            "body_part":      body_part,
            "under_pressure": under_pressure,
        }])[SHOT_FEATURES]
        return int(self.model.predict(X)[0])

    def predict_proba(self, distance: float, angle: float,
                      body_part: int = 0, under_pressure: int = 0) -> float:
        """
        Predict probability of goal for a shot.

        Returns
        -------
        Goal probability (float, 0–1)
        """
        X = pd.DataFrame([{
            "distance":       distance,
            "angle":          angle,
            "body_part":      body_part,
            "under_pressure": under_pressure,
        }])[SHOT_FEATURES]
        proba   = self.model.predict_proba(X)[0]
        classes = self.model.classes_
        d = dict(zip(classes.tolist(), proba.tolist()))
        return float(d.get(1, 0.0))

    @staticmethod
    def _generate_fallback_shots(n: int = 5000, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(n):
            dist  = rng.uniform(5, 35)
            angle = rng.uniform(5, 85)
            body  = int(rng.choice([0, 1], p=[0.72, 0.28]))
            pres  = int(rng.choice([0, 1], p=[0.38, 0.62]))
            xg    = float(1 / (1 + np.exp(0.18 * dist - 0.045 * angle + 0.4 * (1 - body) - 1.8)))
            goal  = int(rng.random() < xg)
            rows.append({"distance": dist, "angle": angle,
                         "body_part": body, "under_pressure": pres,
                         "xg": xg, "goal": goal})
        return pd.DataFrame(rows)


# CONVENIENCE WRAPPER

class ClassificationModels:
    """
    Holds the Decision Tree model.
    Reads training data from data/shots_clean.csv (written by data_cleaning.py).
    Imported by app.py and other modules that need prediction.
    """

    def __init__(self):
        self.dt = DecisionTreeGoal()

    def fit(self, shots_df: pd.DataFrame = None, matches_df: pd.DataFrame = None,
            verbose: bool = True) -> "ClassificationModels":
        print("\n" + "=" * 50)
        print("  CLASSIFICATION MODEL")
        print("  Data source: data/shots_clean.csv")
        print("=" * 50)
        self.dt.fit(shots_df, verbose=verbose)
        print("\n✓ Classification model trained successfully")
        return self

    def classify_goal(self, distance: float, angle: float,
                      body_part: int = 0, under_pressure: int = 0) -> int:
        """Classify shot as Goal (1) or No Goal (0)."""
        return self.dt.predict(distance, angle, body_part, under_pressure)

    def classify_goal_proba(self, distance: float, angle: float,
                             body_part: int = 0, under_pressure: int = 0) -> float:
        """Goal probability for a shot."""
        return self.dt.predict_proba(distance, angle, body_part, under_pressure)

    @property
    def metrics(self) -> dict:
        return {"decision_tree": self.dt.metrics}

    @property
    def feature_importances(self) -> dict:
        return {"decision_tree": self.dt.importances}


# Standalone run
if __name__ == "__main__":
    if not os.path.exists(SHOTS_CSV):
        print("ERROR: data/shots_clean.csv not found.")
        print("Please run data_cleaning.py first:\n  python data_cleaning.py")
        exit(1)

    print(f"Loading shots from {SHOTS_CSV}...")
    clf = ClassificationModels()
    clf.fit()

    print("\n" + "=" * 50)
    print("  PREDICTION EXAMPLES")
    print("=" * 50)

    shots = [
        (5,  80, 0, 0, "Close range, central, foot, no pressure"),
        (18, 35, 0, 0, "Mid range, central, foot, no pressure"),
        (18, 35, 0, 1, "Mid range, central, foot, under pressure"),
        (18, 35, 1, 0, "Mid range, central, head, no pressure"),
        (28, 10, 0, 1, "Long range, narrow, foot, under pressure"),
    ]

    for dist, angle, body, pres, desc in shots:
        pred  = clf.classify_goal(dist, angle, body, pres)
        proba = clf.classify_goal_proba(dist, angle, body, pres)
        print(f"\n  {desc}")
        print(f"    Prediction:  {'⚽ Goal' if pred else 'No Goal'}")
        print(f"    Goal prob:   {proba*100:.1f}%")
