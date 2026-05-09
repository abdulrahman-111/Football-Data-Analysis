"""
player_evaluator.py
Wraps the real trained .pkl models:
  - performance_prediction_goals_next_model.pkl
  - performance_prediction_assists_next_model.pkl
  - Injury_classifier_model.pkl
  - transfer_value_prediction_model.pkl

Falls back to heuristics when models are absent.
"""
import os
import numpy as np
import pandas as pd

try:
    import joblib
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

GOALS_FEATURES = [
    'Expected_xG','Expected_npxG','Standard_Gls','Performance_Gls',
    'Standard_SoT','Expected_npxG+xAG','Touches_Att Pen','Performance_G-PK',
    'Performance_G+A','Standard_Sh','Per 90 Minutes_xG',
    'Per 90 Minutes_xG+xAG','Per 90 Minutes_npxG+xAG','Per 90 Minutes_npxG',
    'assists_next','Per 90 Minutes_G+A','Per 90 Minutes_Gls',
    'Standard_SoT/90','Performance_Off','Standard_Sh/90','SCA Types_Sh',
    'Per 90 Minutes_G+A-PK','Carries_CPA','Progression_PrgR'
]
ASSISTS_FEATURES = [
    'Expected_xA','xAG_','Expected_xAG','Touches_Att 3rd','SCA_SCA',
    'goals_next','KP_','GCA_GCA','PPA_','SCA Types_PassLive',
    'Expected_npxG+xAG','Progression_PrgR','Receiving_PrgR',
    'GCA Types_PassLive','Performance_Ast','Ast_','Performance_G+A',
    'Progression_PrgC','Carries_PrgC','SCA_SCA90','Carries_1/3',
    'Carries_CPA','Touches_Att Pen','Standard_Sh'
]
INJURY_FEATURES = [
    'Age','Height_cm','Weight_kg','Training_Hours_Per_Week',
    'Matches_Played_Past_Season','Previous_Injury_Count','Knee_Strength_Score',
    'Hamstring_Flexibility','Reaction_Time_ms','Balance_Test_Score',
    'Sprint_Speed_10m_s','Agility_Score','Sleep_Hours_Per_Night',
    'Stress_Level_Score','Nutrition_Quality_Score','Warmup_Routine_Adherence','BMI'
]
TRANSFER_FEATURES = [
    'age','appearance','Goals_per90_min','Assists_per90_min',
    'Goal Conceded_per90_min','minutes played','days_injured','games_injured',
    'award','highest_value','position_encoded','team_encoded'
]


def _load_model(fname: str):
    if not JOBLIB_OK:
        return None
    path = os.path.join(MODELS_DIR, fname)
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


def _build_df(features, data_dict):
    row = [float(data_dict.get(f, 0)) for f in features]
    return pd.DataFrame([row], columns=features)


class PlayerEvaluator:
    """Evaluate player performance, injury risk, and market value."""

    def __init__(self):
        self._model_goals    = _load_model("performance_prediction_goals_next_model.pkl")
        self._model_assists  = _load_model("performance_prediction_assists_next_model.pkl")
        self._model_injury   = _load_model("Injury_classifier_model.pkl")
        self._model_transfer = _load_model("transfer_value_prediction_model.pkl")

    def evaluate(self, data: dict) -> dict:
        """
        data must contain keys:
          player_info, goals_model_features, assists_model_features,
          injury_model_features, transfer_model_features
        Returns dict with: goals_pred, assists_pred, injury_prob, market_value,
          strengths, weaknesses, career_growth, trends
        """
        gf = data.get("goals_model_features", {})
        af = data.get("assists_model_features", {})
        inj = data.get("injury_model_features", {})
        tf = data.get("transfer_model_features", {})

        # ── Goals ──────────────────────────────────────────────
        if self._model_goals:
            try:
                X = _build_df(GOALS_FEATURES, gf)
                goals_pred = float(self._model_goals.predict(X)[0])
            except Exception:
                goals_pred = float(gf.get("Performance_Gls", 10)) * 1.05
        else:
            goals_pred = float(gf.get("Performance_Gls", 10)) * 1.05

        # ── Assists ─────────────────────────────────────────────
        if self._model_assists:
            try:
                X = _build_df(ASSISTS_FEATURES, af)
                assists_pred = float(self._model_assists.predict(X)[0])
            except Exception:
                assists_pred = float(af.get("Performance_Ast", 5)) * 1.05
        else:
            assists_pred = float(af.get("Performance_Ast", 5)) * 1.05

        # ── Injury ──────────────────────────────────────────────
        if self._model_injury:
            try:
                X = _build_df(INJURY_FEATURES, inj)
                if hasattr(self._model_injury, "predict_proba"):
                    injury_prob = float(self._model_injury.predict_proba(X)[0][1])
                else:
                    injury_prob = float(self._model_injury.predict(X)[0])
            except Exception:
                prev = float(inj.get("Previous_Injury_Count", 1))
                injury_prob = min(0.15 + prev * 0.08, 0.9)
        else:
            prev = float(inj.get("Previous_Injury_Count", 1))
            injury_prob = min(0.15 + prev * 0.08, 0.9)

        # ── Transfer ────────────────────────────────────────────
        if self._model_transfer:
            try:
                X = _build_df(TRANSFER_FEATURES, tf)
                market_value = float(self._model_transfer.predict(X)[0])
            except Exception:
                market_value = float(tf.get("highest_value", 20e6)) * 0.9
        else:
            market_value = float(tf.get("highest_value", 20e6)) * 0.9

        market_value = max(100_000, market_value)

        # ── Derived analytics ────────────────────────────────────
        strengths, weaknesses = self._sw(gf, af, inj)
        career_growth = self._career_growth(goals_pred, assists_pred)
        trends = self._trends(gf, af)

        return {
            "goals_pred": goals_pred,
            "assists_pred": assists_pred,
            "injury_prob": injury_prob,
            "market_value": market_value,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "career_growth": career_growth,
            "trends": trends,
        }

    def _sw(self, gf, af, inj):
        strengths, weaknesses = [], []
        if float(gf.get("Per 90 Minutes_xG", 0)) > 0.45:
            strengths.append("High xG rate — clinical finisher")
        else:
            weaknesses.append("Low xG rate — needs to improve finishing")
        if float(gf.get("Standard_SoT/90", 0)) > 1.2:
            strengths.append("High shots on target per 90")
        if float(af.get("SCA_SCA90", 0)) > 3.0:
            strengths.append("Excellent shot-creating actions")
        if float(inj.get("Previous_Injury_Count", 0)) >= 3:
            weaknesses.append("Significant injury history")
        if float(inj.get("Sleep_Hours_Per_Night", 8)) < 7:
            weaknesses.append("Below-average sleep recovery")
        if float(inj.get("Nutrition_Quality_Score", 70)) > 80:
            strengths.append("Excellent physical conditioning")
        if float(gf.get("Progression_PrgR", 0)) > 80:
            strengths.append("Strong progressive running")
        if float(af.get("KP_", 0)) > 40:
            strengths.append("Key pass creator")
        return strengths or ["Good all-round performer"], weaknesses or ["No major weaknesses identified"]

    def _career_growth(self, goals, assists):
        base_goals   = [round(max(0, goals   * (0.7 + 0.1 * i)), 1) for i in range(5)]
        base_assists = [round(max(0, assists * (0.7 + 0.1 * i)), 1) for i in range(5)]
        seasons = [f"S{i+1}" for i in range(5)]
        return {"seasons": seasons, "goals": base_goals, "assists": base_assists}

    def _trends(self, gf, af):
        xg  = float(gf.get("Per 90 Minutes_xG", 0.4))
        sot = float(gf.get("Standard_SoT/90", 1.0))
        sca = float(af.get("SCA_SCA90", 2.5))
        prgr = float(gf.get("Progression_PrgR", 70))
        return {
            "labels": ["xG/90", "SoT/90", "SCA/90", "PrgR"],
            "values": [
                min(100, int(xg * 100)),
                min(100, int(sot * 40)),
                min(100, int(sca * 15)),
                min(100, int(prgr))
            ]
        }
