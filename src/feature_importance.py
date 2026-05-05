import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

filename = os.path.join(project_root, 'data', 'processed', 'cleaned_male_fc_24_players.csv')
print(f"*************** Loading data from {filename} ***************")
df = pd.read_csv(filename)

print('*************** Training Random Forest for Explainability *******************')
# Target is 'overall_rating'
X = df.drop(columns=['name', 'best_position', 'overall_rating'])
y = df['overall_rating']

# We don't need train/test split because we just want the mathematical weights of the whole dataset
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X, y)

print('*************** Extracting Feature Importances *******************')
# Get importance scores
importances = rf_model.feature_importances_
feature_names = X.columns

# Create a DataFrame for easy sorting and plotting
feature_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort by importance and get top 15
feature_df = feature_df.sort_values(by='Importance', ascending=False).head(15)

print("\nTop 5 Most Important Stats for Overall Rating:")
for i, row in feature_df.head(5).iterrows():
    print(f"- {row['Feature']}: {row['Importance']:.4f}")

print('\n*************** Generating Premium Visualization *******************')
# Set a sleek dark mode theme
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#1e1e24",
    "figure.facecolor": "#1e1e24",
    "axes.edgecolor": "#1e1e24",
    "grid.color": "#2b2b36",
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.weight": "bold"
})

plt.figure(figsize=(12, 8))

# Create a beautiful gradient barplot
ax = sns.barplot(
    x='Importance', 
    y='Feature', 
    data=feature_df, 
    palette='magma',
    edgecolor='black',
    linewidth=1.5
)

# Add data labels to the end of each bar
for i, v in enumerate(feature_df['Importance']):
    ax.text(v + 0.005, i + 0.15, f"{v:.3f}", color='white', fontweight='bold', fontsize=10)

plt.title('THE DNA OF A FIFA STAR: Top 15 Attributes for Overall Rating', fontsize=18, pad=20, fontweight='heavy', color='#00ffcc')
plt.xlabel('Relative Importance Score', fontsize=14, fontweight='bold', labelpad=15)
plt.ylabel('Player Attribute', fontsize=14, fontweight='bold', labelpad=15)

# Clean up the borders
sns.despine(left=True, bottom=True)
plt.tight_layout()

output_image = os.path.join(project_root, 'outputs', 'feature_importance_premium.png')
plt.savefig(output_image, dpi=300, bbox_inches='tight')
plt.close()

print(f"SUCCESS! Premium static graph saved to: {output_image}")
