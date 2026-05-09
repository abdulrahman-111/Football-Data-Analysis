# ⚽ Football Player Performance & Transfer Prediction

## 📌 Overview
This project builds a machine learning system to support football club decisions by predicting player performance, injury risk, and transfer suitability.

---

## 🎯 Objectives

- Predict next season:
  - Goals  
  - Assists  
  - Matches / Minutes played  
- Estimate:
  - Injury risk / number of injuries  
- Classify:
  - Whether a player should be signed or not  

---

## 🧠 Models Used

### Regression Models
- Predict:
  - `goals_next_season`
  - `assists_next_season`
  - `matches_next_season`
  - `injuries_next_season`
- Algorithm:
  - Random Forest Regressor

### Classification Model
- Predict:
  - `sign_player` (1 = Yes, 0 = No)
- Algorithm:
  - Random Forest Classifier

---

## 📊 Dataset Structure

Each row represents a player in a specific season.

Targets are shifted to represent the next season:

| player | season | goals | assists | goals_next |
|--------|--------|-------|---------|------------|
| A      | 2021   | 10    | 5       | 12         |
| A      | 2022   | 12    | 6       | 15         |

---

## 🔧 Features

### Player Info
- Age  
- Position  

### Performance Metrics
- Goals per 90  
- Assists per 90  
- xG (Expected Goals)  
- xA (Expected Assists)  

### Playing Time
- Minutes played  
- Matches played  

### Team Context
- Team strength  
- League strength  

### Injury Features
- Previous injuries  
- Days missed  
- Workload metrics  

---

## ⚙️ Data Processing

- Handling missing values  
- Data type conversion  
- Feature engineering (per90 stats, trends)  
- Encoding categorical features  
- Target shifting:

```python
df['goals_next'] = df.groupby('player')['goals'].shift(-1)
