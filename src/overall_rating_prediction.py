import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Loading cleaned data
filename = os.path.join(project_root, 'data', 'processed', 'cleaned_male_fc_24_players.csv')
print(f"*************** Loading data from {filename} ***************")
df_model = pd.read_csv(filename)

print('\n*************** Regression Model Comparison ***************')
X_reg = df_model.drop(columns=['overall_rating', 'best_position'])
y_reg = df_model['overall_rating']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

scaler_r = StandardScaler()
X_train_r_scaled = scaler_r.fit_transform(X_train_r)
X_test_r_scaled = scaler_r.transform(X_test_r)

regression_models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=10.0),
    "Polynomial Regression (Degree 2)": make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
}

best_reg_r2 = -float('inf')
best_reg_name = ""
best_reg_model_obj = None

for name, model in regression_models.items():
    model.fit(X_train_r_scaled, y_train_r)
    y_pred = model.predict(X_test_r_scaled)
    r2 = r2_score(y_test_r, y_pred)
    mse = mean_squared_error(y_test_r, y_pred)
    print(f"{name} -> MSE: {mse:.4f} | R-squared: {r2:.4f}")
    
    if r2 > best_reg_r2:
        best_reg_r2 = r2
        best_reg_name = name
        best_reg_model_obj = model

print(f"\nWINNER FOR REGRESSION: {best_reg_name} (R-squared: {best_reg_r2:.4f})")
joblib.dump(best_reg_model_obj, os.path.join(project_root, 'models', 'overall_rating_prediction.pkl'))
print(f"Saved {best_reg_name} to models/overall_rating_prediction.pkl")
