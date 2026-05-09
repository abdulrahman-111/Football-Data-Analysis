"""
rating_predictor.py
Wraps:
  - overall_rating_prediction.pkl   (Ridge regression)
  - position_classification.pkl     (KNN classifier)
  - player_recommender.pkl          (NearestNeighbors dict)
  - cleaned_male_fc_24_players.csv  (player database)
"""
import os
import numpy as np
import pandas as pd

try:
    import joblib
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")

# ── Feature columns (must match training set) ────────────────────
RATING_FEATURES = [
    "height_cm","weight_kg","weak_foot","skill_moves",
    "crossing","finishing","heading_accuracy","short_passing","volleys",
    "dribbling","curve","fk_accuracy","long_passing","ball_control",
    "acceleration","sprint_speed","agility","reactions","balance",
    "shot_power","jumping","stamina","strength","long_shots",
    "aggression","interceptions","positioning","vision","penalties",
    "composure","defensive_awareness","standing_tackle","sliding_tackle",
    "gk_diving","gk_handling","gk_kicking","gk_positioning","gk_reflexes",
    "preferred_foot_encoded",
]

POSITION_LABELS = [
    "GK","CB","RB","LB","CDM","CM","CAM","RM","LM","RW","LW","ST","CF","SS","FW"
]


def _load(fname):
    if not JOBLIB_OK:
        return None
    path = os.path.join(MODELS_DIR, fname)
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


def _load_player_db():
    path = os.path.join(DATA_DIR, "cleaned_male_fc_24_players.csv")
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return None


class RatingPredictor:
    """Predict overall rating, best position, and similar players."""

    def __init__(self):
        self._rating_model  = _load("overall_rating_prediction.pkl")
        self._pos_model     = _load("position_classification.pkl")
        rec_bundle          = _load("player_recommender.pkl")
        if isinstance(rec_bundle, dict):
            self._rec_model    = rec_bundle.get("model")
            self._rec_scaler   = rec_bundle.get("scaler")
        else:
            self._rec_model    = None
            self._rec_scaler   = None
        self._player_db     = _load_player_db()

    def predict_rating(self, attrs: dict) -> dict:
        """
        attrs: dict mapping each RATING_FEATURES key to numeric value.
        Returns: rating (int), position (str), similar_players (list of dicts),
                 improvements (list of str)
        """
        row = np.array([float(attrs.get(f, 50)) for f in RATING_FEATURES]).reshape(1, -1)

        # ── Overall rating ──────────────────────────────────────
        if self._rating_model:
            try:
                rating = int(round(float(self._rating_model.predict(row)[0])))
            except Exception:
                rating = self._heuristic_rating(attrs)
        else:
            rating = self._heuristic_rating(attrs)
        rating = max(40, min(99, rating))

        # ── Position ────────────────────────────────────────────
        if self._pos_model:
            try:
                pos = str(self._pos_model.predict(row)[0])
            except Exception:
                pos = self._heuristic_position(attrs)
        else:
            pos = self._heuristic_position(attrs)

        # ── Similar players ─────────────────────────────────────
        similar = self._find_similar(row)

        # ── Improvement tips ────────────────────────────────────
        improvements = self._improvements(attrs, rating)

        return {
            "rating": rating,
            "position": pos,
            "similar_players": similar,
            "improvements": improvements,
        }

    def _heuristic_rating(self, attrs):
        key_attrs = [
            "finishing","dribbling","short_passing","vision","reactions",
            "ball_control","acceleration","sprint_speed","composure","positioning"
        ]
        vals = [float(attrs.get(k, 50)) for k in key_attrs]
        return int(round(np.mean(vals) * 0.95 + 5))

    def _heuristic_position(self, attrs):
        gk = float(attrs.get("gk_reflexes", 0))
        if gk > 60:
            return "GK"
        fin = float(attrs.get("finishing", 50))
        def_a = float(attrs.get("defensive_awareness", 50))
        if fin > def_a + 15:
            return "ST"
        if def_a > fin + 15:
            return "CB"
        return "CM"

    def _find_similar(self, row):
        if self._rec_model and self._rec_scaler and self._player_db is not None:
            try:
                db_feats = self._player_db.drop(
                    columns=["name","best_position","overall_rating"], errors="ignore"
                )
                # match columns
                missing = [c for c in db_feats.columns if c not in RATING_FEATURES]
                row_df = pd.DataFrame(row, columns=RATING_FEATURES)
                for m in missing:
                    row_df[m] = 0
                row_aligned = row_df[db_feats.columns].values
                row_scaled  = self._rec_scaler.transform(row_aligned)
                dists, idxs = self._rec_model.kneighbors(row_scaled, n_neighbors=6)
                result = []
                for i in range(min(5, len(idxs[0]))):
                    idx = idxs[0][i]
                    p   = self._player_db.iloc[idx]
                    result.append({
                        "name":     str(p.get("name","Unknown")),
                        "rating":   int(p.get("overall_rating", 75)),
                        "position": str(p.get("best_position","—")),
                        "distance": round(float(dists[0][i]), 2),
                    })
                return result
            except Exception:
                pass
        # Fallback: static list
        return [
            {"name":"Kevin De Bruyne","rating":91,"position":"CAM","distance":8.2},
            {"name":"Luka Modrić",    "rating":87,"position":"CM", "distance":9.1},
            {"name":"Bruno Fernandes","rating":88,"position":"CAM","distance":9.7},
            {"name":"Toni Kroos",     "rating":88,"position":"CM", "distance":10.3},
            {"name":"Casemiro",       "rating":87,"position":"CDM","distance":11.5},
        ]

    def _improvements(self, attrs, rating):
        tips = []
        if float(attrs.get("finishing", 50)) < 70:
            tips.append("Improve Finishing to reach striker potential")
        if float(attrs.get("short_passing", 50)) < 70:
            tips.append("Work on Short Passing for better buildup")
        if float(attrs.get("composure", 50)) < 70:
            tips.append("Composure training needed under pressure")
        if float(attrs.get("stamina", 50)) < 70:
            tips.append("Increase Stamina for full 90-minute performance")
        if float(attrs.get("reactions", 50)) < 70:
            tips.append("Sharpen Reactions for faster decision making")
        if rating < 75:
            tips.append("Focus on core technical attributes to boost overall rating")
        return tips or ["Already a high-quality player — maintain fitness and consistency"]
