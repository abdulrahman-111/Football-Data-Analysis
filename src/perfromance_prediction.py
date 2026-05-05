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

from sklearn.decomposition import PCA



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "data", "raw", "performance", "Top5_League_Players_2017to2024_dataset.csv")

df = pd.read_csv(file_path, sep=';',decimal=',') # delimeter is ; 



# cols = ['player', 'season', 'age_',"pos_","Performance_Gls","Team Success_PPM","Take-Ons_Succ%","Touches_Att Pen","Performance_Ast","Expected_xG","Expected_xA","Per 90 Minutes_Gls","Per 90 Minutes_Ast","Per 90 Minutes_xG","KP_","Playing Time_90s","Progression_PrgP","Standard_Sh/90"]


selected_df = df


selected_df= selected_df.sort_values(['player', 'season'])


## feature enginnering 
selected_df['goals_next'] = selected_df.groupby('player')['Performance_Gls'].shift(-1)
selected_df['assists_next'] = selected_df.groupby('player')['Performance_Ast'].shift(-1)

# selected_df['XG_per90_next'] = selected_df.groupby('player')['Per 90 Minutes_xG'].shift(-1)
# selected_df['XA_per90_next'] = selected_df.groupby('player')['Per 90 Minutes_Ast'].shift(-1)

selected_df['goals_trend'] = selected_df['Performance_Gls'] - df.groupby('player')['Performance_Gls'].shift(1)


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


modified_df = selected_df.drop(columns=['player','pos_','league','team','nation_'])


targets_index = ['goals_next','assists_next']

for target in targets_index :

    corr = modified_df.corr()[target].abs().sort_values(ascending=False)


    ## feature Selection  
    top_features = corr.abs().sort_values(ascending=False).head(25).index
    

    corr_with_top = modified_df[top_features].corr()[target].abs().sort_values(ascending=False)


    plt.figure(figsize=(6, 10))

    sns.heatmap(corr_with_top.to_frame(), annot=True, cmap='coolwarm')



    plt.title(f"Correlation with {target}")

    image_path= os.path.join(BASE_DIR, "outputs",  f"heatmap_perform_{target}.png" ) 

    plt.savefig(image_path, dpi=300, bbox_inches = 'tight')


    final_df = modified_df[ list(top_features) + ['season']]


    train_mask = final_df['season'] < 2324   # train on older seasons
    test_mask  = final_df['season'] >= 2324  # test on the most recent season

    X_train = final_df[train_mask].drop(columns=[target,'season'])

    X_test  = final_df[test_mask].drop(columns=[target,'season'])

    y_train = final_df[train_mask][target]
    y_test  = final_df[test_mask][target]




    pipe = Pipeline([
        ("RandomForestregressor",RandomForestRegressor(n_estimators=300 ))
    ])



    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    train_pred = pipe.predict(X_train)


    r2_squared_score_test = r2_score(y_test, y_pred)
    r2_squared_score_train = r2_score(y_train, train_pred)


    print(f"Root squared test _ score with RANDOM Regressor for [[ {target}  ]] is: {r2_squared_score_test}")
    print(f"Root squared train _ score  with RANDOM Regressor for [[ {target}  ]] is: {r2_squared_score_train}")


    model_path = os.path.join(BASE_DIR, "models",f"performance_prediction_{ target }_model.pkl")
    joblib.dump(pipe, model_path)

    out_dataset_path = os.path.join(BASE_DIR, "data","processed",f"performance_pred_{target}_dataset.csv")
    final_df.to_csv(out_dataset_path, index=False)


        




encoding_path = os.path.join(BASE_DIR, "models","position_encoding.pkl")
joblib.dump(position_map, encoding_path)






























