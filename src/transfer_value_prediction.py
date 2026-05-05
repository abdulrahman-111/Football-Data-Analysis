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
file_path = os.path.join(BASE_DIR, "data", "raw", "Transfer_value", "transfer_value_prediction_dataset.csv")

df = pd.read_csv(file_path)

# df.info()
# df.isna().sum()
# df.duplicated().sum()


df = df.drop_duplicates(subset=['player']) ## to get unique players


if ( df.isnull().sum().sum()) :  ## check for missing values 
    print("Handle Missing Values")



if df.isnull().values.any():
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print("Missing values found:")
    print(missing)
else:
    print("No missing values")



# Feature Engineering -> Goal, Assists, Yellow Cards, Red Cards , Goal Conceded per 90 min

cols = [
    'Goals',
    'Assists',
    'Yellow Cards',
    'Second Yellow Card',
    'Red Card',
    'Goal Conceded',
    'Clean Sheets'
]

for col in cols:
    df[col] = (df[col] * 90) / df['minutes played']
    df.rename(columns={col: f"{col}_per90_min"}, inplace=True)






# Target encoding by usinf mean of team players 
team_mean = df.groupby('team')['current_value'].mean()
df['team_encoded'] = df['team'].map(team_mean)


print(team_mean)


df_cleaned = df.drop(columns=["name",'player','team','position'], axis=1)


# standarize featuers  

features = df_cleaned.drop(columns=['current_value'])

target = df_cleaned['current_value']



## standarize features for correlation 
scaler = StandardScaler()

features_scaled = pd.DataFrame( scaler.fit_transform(features), columns=features.columns)



modified_df = pd.concat([features_scaled,target], axis=1)



corr = modified_df.corr()['current_value'].abs().sort_values(ascending=False)

plt.figure(figsize=(6, 10))

sns.heatmap(corr.to_frame(), annot=True, cmap='coolwarm')


plt.title("Correlation with Current Value")

image_path= os.path.join(BASE_DIR, "outputs","heatmap.png")

plt.savefig(image_path, dpi=300, bbox_inches='tight')

plt.show()






threshold = .005
corr = modified_df.corr()['current_value']




features_less_than_threshold = corr[abs(corr)<threshold].index

print(f"Removing these features: {features_less_than_threshold}")


df_final = df_cleaned.drop( columns=features_less_than_threshold, axis=1)



X_train, X_test, y_train, y_test = train_test_split(df_final.drop('current_value', axis=1), df_final['current_value'], test_size= .2, random_state=42)

# model_rfr= RandomForestRegressor()

# model_rfr.fit(X_train,y_train)

# y_pred = model_rfr.predict(X_test)

# r2 = r2_score(y_test, y_pred) 

# print(r2)




# pipe = Pipeline([
#     ("scaler", StandardScaler()), 
#     ("regressor", RandomForestRegressor(n_estimators=150))
# ])
# pipe.fit(X_train, y_train)
# y_pred = pipe.predict(X_test)
# r2_squared_score = r2_score(y_test, y_pred)
# print(f"Root squared score with RandomForestRegressor is: {r2_squared_score}")



pipe = Pipeline([
    ('model', RandomForestRegressor())
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
r2_squared_score = r2_score(y_test, y_pred)
print(f"Root squared score with RandomForestRegressor is: {r2_squared_score}")





model_path = os.path.join(BASE_DIR, "models","transfer_value_prediction_model.pkl")
joblib.dump(pipe, model_path)


encoding_path = os.path.join(BASE_DIR, "models","team_encoding.pkl")
joblib.dump(team_mean, encoding_path)




out_dataset_path = os.path.join(BASE_DIR, "data","processed"," transfer_value_prediction_processed_dataset.csv")
df_final.to_csv(out_dataset_path, index=False)
