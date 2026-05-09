import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
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
X_reg = df_model.drop(columns=['overall_rating', 'best_position', 'name'])
y_reg = df_model['overall_rating']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

scaler_r = StandardScaler()
X_train_r_scaled = scaler_r.fit_transform(X_train_r)
X_test_r_scaled = scaler_r.transform(X_test_r)

print('\n*************** Ridge Regression Alpha Tuning (5-Fold CV) ***************')
ridge_params = {'alpha': [0.1, 1.0, 10.0]}
ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=5, scoring='r2')
ridge_grid.fit(X_train_r_scaled, y_train_r)
best_alpha = ridge_grid.best_params_['alpha']

print("CV Results for Ridge Regression:")
for i in range(len(ridge_params['alpha'])):
    print(f"Alpha: {ridge_grid.cv_results_['param_alpha'][i]} -> Mean CV R-squared: {ridge_grid.cv_results_['mean_test_score'][i]:.4f}")
print(f"Selected best Alpha: {best_alpha}")

print('\n*************** Final Regression Model Comparison ***************')
regression_models = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    f"Ridge Regression (alpha={best_alpha})": make_pipeline(StandardScaler(), Ridge(alpha=best_alpha)),
    "Polynomial Regression (Degree 2)": make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), LinearRegression())
}

best_reg_r2 = -float('inf')
best_reg_name = ""
best_reg_model_obj = None

for name, model in regression_models.items():
    # FIT ON RAW X_train_r! The pipeline handles the scaling automatically!
    model.fit(X_train_r, y_train_r)
    y_pred = model.predict(X_test_r)
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
