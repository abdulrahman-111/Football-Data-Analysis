import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Loading cleaned data
filename = os.path.join(project_root, 'data', 'processed', 'cleaned_male_fc_24_players.csv')
print(f"*************** Loading data from {filename} ***************")
df_model = pd.read_csv(filename)

print('\n*************** Classification Model Comparison ***************')
X_class = df_model.drop(columns=['best_position', 'overall_rating', 'name'])
y_class = df_model['best_position']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

classification_models = {
    "K-Nearest Neighbors (KNN)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
    "Naive Bayes (Gaussian)": make_pipeline(StandardScaler(), GaussianNB())
}

best_class_acc = 0
best_class_name = ""
best_class_model_obj = None

for name, model in classification_models.items():
    # Fit directly on the raw X_train_c!
    model.fit(X_train_c, y_train_c)
    y_pred = model.predict(X_test_c)
    acc = accuracy_score(y_test_c, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    
    if acc > best_class_acc:
        best_class_acc = acc
        best_class_name = name
        best_class_model_obj = model

print(f"\nWINNER FOR CLASSIFICATION: {best_class_name} ({best_class_acc:.4f})")
joblib.dump(best_class_model_obj, os.path.join(project_root, 'models', 'position_classification.pkl'))
print(f"Saved {best_class_name} to models/position_classification.pkl")
