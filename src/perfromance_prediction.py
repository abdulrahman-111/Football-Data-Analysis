import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt


import joblib
import os

from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "data", "raw", "performance", "Top5_League_Players_2017to2024_dataset.csv")

df = pd.read_csv(file_path, sep=';',decimal=',') # delimeter is ; 



cols = ['player', 'season', 'age_',"pos_","Performance_Gls","Team Success_PPM","Take-Ons_Succ%","Touches_Att Pen","Performance_Ast","Expected_xG","Expected_xA","Per 90 Minutes_Gls","Per 90 Minutes_Ast","Per 90 Minutes_xG","KP_","Playing Time_90s","Progression_PrgP","Standard_Sh/90"]

selected_df = df[cols]


selected_df= selected_df.sort_values(['player', 'season'])

selected_df['goals_next'] = selected_df.groupby('player')['Performance_Gls'].shift(-1)
selected_df['assists_next'] = selected_df.groupby('player')['Performance_Ast'].shift(-1)
selected_df['XG_per90_next'] = selected_df.groupby('player')['Per 90 Minutes_xG'].shift(-1)
selected_df['XA_per90_next'] = selected_df.groupby('player')['Per 90 Minutes_Ast'].shift(-1)


selected_df = selected_df.dropna(axis=0)

position_map = {
    'GK': 1,
    'DF': 2,
    'MF': 3,
    'FW': 4,
    'DF,FW': 4 ,
    'FW,MF':4 ,
    'MF,FW': 4 ,
    'FW,DF': 4 ,
    'MF,DF': 3 ,
    'DF,MF':3
}

selected_df['position_encoded'] = selected_df['pos_'].map(position_map)



final_df = selected_df.drop(columns=['player','season','pos_'])

Y = final_df[['goals_next']]
X = final_df.drop( columns = ['goals_next'] )

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size= .2, random_state=42)


pipe = Pipeline([
    ("regressor", RandomForestRegressor(n_estimators=150))
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
r2_squared_score = r2_score(y_test, y_pred)
print(f"Root squared score with RandomForestRegressor is: {r2_squared_score}")

model_path = os.path.join(BASE_DIR, "models","performance_prediction_model.pkl")
joblib.dump(pipe, model_path)


encoding_path = os.path.join(BASE_DIR, "models","position_encoding.pkl")
joblib.dump(position_map, encoding_path)


