# ⚽ FIFA Career Mode AI Decision-Support System

An AI-powered assistant for FIFA Career Mode that helps users make smarter club management decisions using machine learning — covering match simulation, player evaluation, squad planning, and transfer analysis.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Modules](#modules)
- [Datasets](#datasets)
- [Model Performance Summary](#model-performance-summary)
- [Installation & Dependencies](#installation--dependencies)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Outputs](#outputs)
- [Limitations](#limitations)
- [Team Contributions](#team-contributions)

---

## Project Overview

Traditional FIFA Career Mode decisions rely on visible player ratings, market value, and personal judgment. This system replaces guesswork with data-driven predictions across three major decision areas:

- **Match Simulation** — predict scorelines, match winners, and shot outcomes
- **Player Development & Squad Planning** — predict best positions, estimate overall ratings, and find similar players
- **Transfer Decisions** — assess injury risk, forecast future performance, and estimate market value

The final output is an automated **player scouting report** that combines all predictions into a single transfer recommendation.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│            FIFA Career Mode AI Assistant                │
├───────────────────┬─────────────────┬───────────────────┤
│  Match Simulation │ Player Dev &    │ Transfer Decision │
│  Engine           │ Squad Planning  │ Engine            │
│                   │ Engine          │                   │
│  • Shot outcome   │ • Position      │ • Injury risk     │
│  • Scoreline      │   prediction    │ • Future goals    │
│  • Winner         │ • Overall       │ • Future assists  │
│    prediction     │   rating        │ • Market value    │
│                   │ • Similar       │                   │
│                   │   player finder │                   │
└───────────────────┴─────────────────┴───────────────────┘
                            │
                ┌───────────▼───────────┐
                │  Automated Scouting   │
                │  Report (.docx)       │
                └───────────────────────┘
```

---

## Modules

### 1. Match Simulation Engine
Predicts match-day outcomes using StatsBomb real match-event data.

| Sub-module | Model | Task |
|---|---|---|
| Shot Outcome | Decision Tree | Classifies each shot as goal / no goal |
| Scoreline | Linear Regression (×2) | Predicts home goals and away goals separately |
| Winner | Derived from goal difference | Home Win / Draw / Away Win |

---

### 2. Player Development & Squad Planning Engine
Evaluates players using EA Sports FC 24 attribute data.

| Sub-module | Model | Task |
|---|---|---|
| Position Classification | Logistic Regression | Predicts a player's best position |
| Overall Rating | Polynomial Regression (Degree 2) | Estimates quality from attributes |
| Player Recommender | Nearest Neighbors | Finds the 5 most similar players |
| Explainability | Random Forest | Identifies the most influential attributes |

---

### 3. Transfer Decision Engine
Supports transfer decisions with three predictive models.

| Sub-module | Model | Task |
|---|---|---|
| Injury Risk | Logistic Regression (Pipeline) | Classifies injury risk for next season |
| Future Performance | Random Forest (×2) | Predicts next-season goals and assists |
| Market Value | Random Forest | Estimates current transfer market value |

---

### 4. Automated Scouting Report
Combines all model outputs into a formatted `.docx` report with:
- Player overview
- Predicted goals & assists
- Injury risk assessment
- Predicted market value
- Score summary table
- Final recommendation

**Recommendation thresholds:**

| Overall Score | Recommendation |
|---|---|
| ≥ 8.0 | Strongly Recommended |
| ≥ 6.5 | Recommended |
| ≥ 5.0 | Conditional |
| < 5.0 | Not Recommended |

---

## Datasets

| Dataset | Source | Used For |
|---|---|---|
| StatsBomb Open Data | statsbombpy library | Match simulation, shot classification |
| EA Sports FC 24 Players | Kaggle | Position & rating prediction, player recommender |
| University Football Injury | Kaggle | Injury risk classification |
| Top 5 League Player Stats 2017–2025 | Kaggle | Future goals & assists prediction |
| Football Transfer Fee Prediction | Kaggle | Market value estimation |

> **Note:** Run `data_cleaning.py` first. It produces `shots_clean.csv` and `matches_clean.csv`, which are required by the match simulation models.

---

## Model Performance Summary

### Player Development Engine

| Model | Task | Metric | Result |
|---|---|---|---|
| Logistic Regression | Position classification | Accuracy | **82.82%** |
| K-Nearest Neighbors | Position classification | Accuracy | 70.25% |
| Gaussian Naive Bayes | Position classification | Accuracy | 53.71% |
| Polynomial Regression (Deg 2) | Overall rating | R² / MSE | **0.9801 / 0.962** |
| Ridge Regression | Overall rating | R² / MSE | 0.8941 / 5.127 |

### Transfer Decision Engine

| Model | Task | Metric | Result |
|---|---|---|---|
| Logistic Regression | Injury risk | Accuracy | **95.0%** |
| Logistic Regression | Injury risk | Recall | **96.25%** |
| Random Forest | Future goals | Test R² | See dashboard |
| Random Forest | Market value | R² | See dashboard |

### Match Simulation Engine

| Model | Task | Metric | Result |
|---|---|---|---|
| Decision Tree | Shot outcome | Accuracy | **89.7%** |
| Linear Regression | Winner prediction | Overall accuracy | 51.7% |
| Linear Regression | Home wins | Class accuracy | 88.6% |

---

## Installation & Dependencies

### Requirements

```
Python 3.8+
```

### Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn statsbombpy joblib python-docx
```

### Full dependency list

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, and transformation |
| `numpy` | Numerical operations and feature engineering |
| `scikit-learn` | ML models, pipelines, cross-validation, metrics |
| `matplotlib` & `seaborn` | Visualizations, heatmaps, confusion matrices |
| `statsbombpy` | Accessing StatsBomb open football data |
| `joblib` | Saving and loading trained models |
| `python-docx` | Generating the automated scouting report |

---

## Project Structure

```
fifa-career-mode-ai/
│
├── data/
│   ├── raw/                    # Original downloaded datasets
│   └── cleaned/
│       ├── shots_clean.csv     # Cleaned shot data (generated)
│       └── matches_clean.csv   # Cleaned match data (generated)
│
├── models/                     # Saved trained models (.joblib)
│
├── outputs/
│   └── scouting_report.docx    # Generated player report
│
├── dashboards/                 # Interactive visual dashboards
│
├── data_cleaning.py            # ⚠️ Run this first
├── match_simulation.py         # Shot & match prediction models
├── player_position.py          # Position classification models
├── player_rating.py            # Overall rating regression models
├── player_recommender.py       # Nearest Neighbors recommender
├── explainability.py           # Feature importance analysis
├── injury_risk.py              # Injury classifier pipeline
├── performance_prediction.py   # Goals & assists forecasting
├── transfer_value.py           # Market value estimation
├── generate_player_report.py   # Automated scouting report generator
│
└── README.md
```

---

## How to Run

### Step 1 — Clean the data
```bash
python data_cleaning.py
```
This generates `shots_clean.csv` and `matches_clean.csv` in the `data/cleaned/` folder.

### Step 2 — Run individual engines

```bash
# Match Simulation
python match_simulation.py

# Player Development & Squad Planning
python player_position.py
python player_rating.py
python player_recommender.py
python explainability.py

# Transfer Decision Engine
python injury_risk.py
python performance_prediction.py
python transfer_value.py
```

### Step 3 — Generate a scouting report
```bash
python generate_player_report.py --input player_data.json
```

The report is saved as a `.docx` file in the `outputs/` folder.

---

## Outputs

| Output | Format | Description |
|---|---|---|
| Scouting report | `.docx` | Full player evaluation with final recommendation |
| Confusion matrices | PNG / dashboard | Model classification performance |
| Feature importance charts | PNG / dashboard | Most influential attributes per model |
| Radar charts | PNG / dashboard | Player physical & health profiles |
| Correlation heatmaps | PNG / dashboard | Feature relationships with target variables |
| Live prediction chart | Dashboard | Interactive scoreline sensitivity analysis |

---

## Limitations

- Datasets come from different sources and are not perfectly aligned — full integration requires additional standardization.
- The overall rating model's very high R² (0.9801) partly reflects the structured nature of FIFA player ratings rather than real-world data.
- Transfer value estimation does not account for contract length, release clauses, wages, player potential, or negotiation behavior.
- The match winner model achieves only 51.7% overall accuracy, with very weak draw prediction (3.3%). Use it as a simulation guide, not a reliable predictor.
- Injury and performance models are trained on historical patterns and may not capture in-game Career Mode events such as training boosts, morale, or user-controlled playtime.

---

## Team Contributions

| Members | Module |
|---|---|
| Zaytoun & Hesham | Match Simulation Engine — data prep, shot/match models, dashboards |
| Abdelrahman & Ziad | Player Development & Squad Planning Engine — data cleaning, position & rating models, recommender, explainability, report writing |
| Gomaa & Seif | Transfer Decision Engine — injury classifier, performance prediction, market value, automated scouting report, dashboards |

---

## Acknowledgements

- [StatsBomb](https://github.com/statsbomb/statsbombpy) for open football match-event data
- [Kaggle](https://www.kaggle.com) for player, injury, performance, and transfer datasets
- EA Sports FC 24 player dataset contributors
- scikit-learn, pandas, matplotlib, seaborn, and the broader Python data science ecosystem
