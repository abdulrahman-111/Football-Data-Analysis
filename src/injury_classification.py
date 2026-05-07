import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt


import joblib
import os

from sklearn.preprocessing import StandardScaler ,MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline 
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import  accuracy_score,classification_report, confusion_matrix, ConfusionMatrixDisplay



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "data", "raw", "injury", "data_injury.csv")

df= pd.read_csv(file_path)

df= df.drop_duplicates()


# print(df.dtypes)
# print(df.info())

df.describe()


df =df.drop(columns=['Position'] )

#detect missing data and fill numerical data
if df.isnull().values.any():
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    
    print("Missing values found:")
    print(missing)
    
    # Fill missing values with column mean (only numeric columns)
    for col in missing.index:
        if df[col].dtype != 'object':  # avoid categorical columns
            df[col] = df[col].fillna(df[col].mean())




# features = df.drop(columns=['Injury_Next_Season'])

# target = df['Injury_Next_Season']

# scaler = StandardScaler()

# features_scaled = pd.DataFrame( scaler.fit_transform(features), columns=features.columns)



# modified_df = pd.concat([features_scaled,target], axis=1)

#heatmap correlation 
corr = df.corr()['Injury_Next_Season'].abs().sort_values(ascending=False)


plt.figure(figsize=(6, 10))

sns.heatmap(corr.to_frame(), annot=True, cmap='coolwarm')


plt.title("Correlation with injury_next_season")

image_path= os.path.join(BASE_DIR, "outputs","heatmap_injury.png")

plt.savefig(image_path, dpi=300, bbox_inches='tight')
plt.close()



stats_cols = [
    'Age', 'Height_cm', 'Weight_kg', 'Training_Hours_Per_Week',
    'Matches_Played_Past_Season', 'Previous_Injury_Count',
    'Knee_Strength_Score', 'Hamstring_Flexibility', 'Reaction_Time_ms',
    'Balance_Test_Score', 'Sprint_Speed_10m_s', 'Agility_Score',
    'Sleep_Hours_Per_Night', 'Stress_Level_Score', 'Nutrition_Quality_Score',
    'Warmup_Routine_Adherence', 'BMI'
]

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[stats_cols]), columns=stats_cols)

df['Overall_Score'] = df_scaled.mean(axis=1)
best_player_idx = df['Overall_Score'].idxmax()
best_player = df.loc[best_player_idx]

print("Best player details:")
print(best_player)

player_stats = df_scaled.loc[best_player_idx]

labels = stats_cols
values = player_stats.values
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
values = np.concatenate((values, [values[0]]))
angles += angles[:1]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.plot(angles, values, linewidth=2, linestyle='solid')
ax.fill(angles, values, alpha=0.3)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=9)
ax.set_title("Best Player Overall Stats (Radar Chart)", fontsize=14, pad=20)
plt.show()

#model 

X_train, X_test, y_train, y_test = train_test_split(df.drop('Injury_Next_Season', axis=1), df['Injury_Next_Season'], test_size= .2, random_state=42)



#pipline
pipeline = Pipeline([
    ('scaler', StandardScaler()),

    ('model', LogisticRegression(
        max_iter=10000,
        class_weight='balanced',  # important for injuries
        C= 0.1
    ))
])

pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)

train_acc = pipeline.score(X_train, y_train)
test_acc = pipeline.score(X_test, y_test)

cm = confusion_matrix(y_test, preds)



labels = ['No Injury', 'Injury']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Heatmap ---
sns.heatmap(cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[0])

axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
axes[0].set_title("Confusion Matrix (Heatmap)")



# --- Sklearn ConfusionMatrixDisplay ---


disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', ax=axes[1], colorbar=False)

axes[1].set_title("Confusion Matrix (Sklearn)")

plt.tight_layout()
plt.show()

# --- Metrics ---
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

print("\n--- Classification Report ---")
print(classification_report(y_test, preds , target_names=labels))





out_dataset_path = os.path.join(BASE_DIR, "data","processed","Injury_prediction_processed_dataset.csv")
df.to_csv(out_dataset_path, index=False)

model_path = os.path.join(BASE_DIR, "models","Injury_classifier_model.pkl")
joblib.dump(pipeline, model_path)


