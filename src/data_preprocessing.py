import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# 1. Load the new dataset
filename = os.path.join(project_root, 'data', 'raw', 'new-players-data-full.csv')
print(f"*************** Loading data from {filename} ***************")
df = pd.read_csv(filename)

# 2. Remove duplicate entries
print('*************** Handling Duplicates *******************')
print(f"Total rows before removing duplicates: {len(df)}")
df = df.drop_duplicates()
print(f"Total rows after removing duplicates: {len(df)}")
print()

# 3. Select only the columns we actually need for Machine Learning
# This automatically drops URLs, images, descriptions, and leakage positions
print('*************** Selecting Relevant Features *******************')
columns_to_keep = [
    'name', 'best_position', 'overall_rating', 'preferred_foot', 'height_cm', 'weight_kg', 
    'weak_foot', 'skill_moves', 'crossing', 'finishing', 'heading_accuracy', 
    'short_passing', 'volleys', 'dribbling', 'curve', 'fk_accuracy', 'long_passing', 
    'ball_control', 'acceleration', 'sprint_speed', 'agility', 'reactions', 
    'balance', 'shot_power', 'jumping', 'stamina', 'strength', 'long_shots', 
    'aggression', 'interceptions', 'positioning', 'vision', 'penalties', 
    'composure', 'defensive_awareness', 'standing_tackle', 'sliding_tackle', 
    'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning', 'gk_reflexes'
]
df = df[columns_to_keep].copy()
print(f"Selected {len(columns_to_keep)} core attribute columns.")
print()

# 4. Handle Missing Values
print('*************** Handling Missing Values *******************')
# This dataset is extremely clean, but we drop any stray missing values just in case
missing_before = df.isnull().sum().sum()
df = df.dropna()
print(f"Dropped {missing_before} missing values.")
print(f"Total rows ready for modeling: {len(df)}")
print()

# 5. Encode Categorical Features
print('*************** Encoding Categorical Features *******************')
# Map 'preferred_foot' to 0 (Left) and 1 (Right)
df['preferred_foot_encoded'] = df['preferred_foot'].map({'Left': 0, 'Right': 1})
df.drop('preferred_foot', axis=1, inplace=True)

print("First 2 rows of final dataset:")
pd.set_option('display.max_columns', None)
print(df.head(2))
print()

# 6. Save the cleaned dataset
print('*************** Saving Cleaned Data *******************')
output_filename = os.path.join(project_root, 'data', 'processed', 'cleaned_male_fc_24_players.csv')
df.to_csv(output_filename, index=False)
print(f"Data successfully written to {output_filename}")
print()
