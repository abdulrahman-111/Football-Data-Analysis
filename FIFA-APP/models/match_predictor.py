"""
match_predictor.py
Simulates match prediction logic.
Placeholder — swap predict() internals with a real model when ready.
"""
import random
import math


class MatchPredictor:
    """Predicts match outcome, score, and scorer probability."""

    TEAM_STRENGTH = {
        "Real Madrid": 92, "Manchester City": 91, "Bayern Munich": 90,
        "Barcelona": 89, "Arsenal": 87, "Liverpool": 88, "Chelsea": 85,
        "PSG": 88, "Atletico Madrid": 86, "Juventus": 84,
        "Inter Milan": 86, "AC Milan": 85, "Borussia Dortmund": 85,
        "Napoli": 84, "Tottenham": 82,
    }

    def __init__(self):
        self._rng = random.Random()

    def _get_strength(self, team: str) -> float:
        return self.TEAM_STRENGTH.get(team, 75 + self._rng.randint(-5, 5))

    def predict(self, team1: str, team2: str,
                goals_t1: int, shots_t1: int, poss_t1: float,
                goals_t2: int, shots_t2: int, poss_t2: float,
                player_name: str = "") -> dict:
        """
        Returns a prediction dict:
          predicted_home / predicted_away  — integer goals
          win1 / draw / win2              — float probabilities (sum=1)
          scorer_prob                     — float 0-1 for player_name
          confidence                      — float 0-1
          analysis                        — short text explanation
        """
        s1 = self._get_strength(team1)
        s2 = self._get_strength(team2)

        # Blend stated stats with known strength
        eff1 = 0.5 * s1 + 0.3 * (goals_t1 * 10) + 0.2 * (poss_t1 * 100)
        eff2 = 0.5 * s2 + 0.3 * (goals_t2 * 10) + 0.2 * (poss_t2 * 100)

        total = eff1 + eff2
        p1_ratio = eff1 / total
        p2_ratio = eff2 / total

        # Simulate expected goals
        xg1 = round(max(0, p1_ratio * 3.2 + random.gauss(0, 0.3)), 2)
        xg2 = round(max(0, p2_ratio * 3.2 + random.gauss(0, 0.3)), 2)

        pred_home = min(int(round(xg1)), 9)
        pred_away = min(int(round(xg2)), 9)

        # Win probabilities via softmax-ish
        raw1  = math.exp(xg1 * 0.7)
        raw2  = math.exp(xg2 * 0.7)
        raw_d = math.exp(0.5)
        denom = raw1 + raw2 + raw_d
        win1  = round(raw1 / denom, 3)
        win2  = round(raw2 / denom, 3)
        draw  = round(1 - win1 - win2, 3)

        # Scorer probability
        scorer_prob = 0.0
        if player_name:
            # crude proxy: if team1 is predicted to score, assign some prob
            scorer_prob = round(min(0.95, xg1 * 0.2 + 0.1), 2)

        confidence = round(min(0.95, abs(xg1 - xg2) * 0.2 + 0.55), 2)

        # Analysis text
        winner = team1 if win1 > win2 else (team2 if win2 > win1 else "Draw")
        analysis = (
            f"Based on team strength and recent statistics, {team1} carries "
            f"{win1*100:.0f}% win probability vs {team2}'s {win2*100:.0f}%. "
            f"Expected goals: {xg1:.1f} – {xg2:.1f}. "
            f"{'A draw is likely.' if draw > 0.35 else f'{winner} is favoured to win.'}"
        )

        return {
            "team1": team1, "team2": team2,
            "predicted_home": pred_home, "predicted_away": pred_away,
            "xg1": xg1, "xg2": xg2,
            "win1": win1, "draw": draw, "win2": win2,
            "scorer_prob": scorer_prob,
            "confidence": confidence,
            "analysis": analysis,
        }
