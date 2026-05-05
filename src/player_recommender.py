import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

filename = os.path.join(project_root, 'data', 'processed', 'cleaned_male_fc_24_players.csv')
print(f"*************** Loading data from {filename} ***************")
df = pd.read_csv(filename)

print('*************** Building Recommender System *******************')
# Features to use for similarity (drop strings and targets)
features = df.drop(columns=['name', 'best_position', 'overall_rating'])

# Scale features so Pace doesn't overpower Passing, etc.
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Nearest Neighbors with Euclidean Distance
# Euclidean distance measures the absolute straight-line distance between player profiles
nn_model = NearestNeighbors(n_neighbors=6, metric='euclidean', algorithm='brute')
nn_model.fit(features_scaled)

# Save the model and scaler together so they can be loaded in the future
model_path = os.path.join(project_root, 'models', 'player_recommender.pkl')
joblib.dump({'model': nn_model, 'scaler': scaler}, model_path)
print(f"Saved Recommender Model and Scaler to {model_path}")

def recommend_player(player_name, top_n=5):
    # Find the player in the dataset
    player_idx = df.index[df['name'].str.lower() == player_name.lower()].tolist()
    
    if not player_idx:
        print(f"Player '{player_name}' not found in the database.")
        return
        
    idx = player_idx[0] # Take the first match
    target_player = df.iloc[idx]
    
    safe_target_name = target_player['name'].encode('ascii', 'ignore').decode('ascii')
    print(f"\nTarget Player: {safe_target_name} (Overall: {target_player['overall_rating']}, Position: {target_player['best_position']})")
    print(f"Top {top_n} Recommended Alternatives:")
    
    # Find neighbors
    distances, indices = nn_model.kneighbors([features_scaled[idx]], n_neighbors=top_n+1)
    
    # Skip the first one because it's the player themselves (distance = 0)
    for i in range(1, len(distances[0])):
        match_idx = indices[0][i]
        match_dist = distances[0][i]
        match_player = df.iloc[match_idx]
        
        safe_match_name = match_player['name'].encode('ascii', 'ignore').decode('ascii')
        # For Euclidean distance, a smaller number means they are more similar (0 is identical)
        print(f"{i}. {safe_match_name} (Distance Score: {match_dist:.2f} | Overall: {match_player['overall_rating']} | Position: {match_player['best_position']})")

if __name__ == "__main__":
    # Testing the model
    recommend_player("Kevin De Bruyne")
    print("-" * 50)
    recommend_player("Rodri")
