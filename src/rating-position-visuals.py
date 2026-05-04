import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error, accuracy_score

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# ── Load cleaned data ─────────────────────────────────────────────────────────
filename = os.path.join(project_root, 'data', 'processed', 'cleaned_male_fc_24_players.csv')
print(f"*************** Loading data from {filename} ***************")
df_model = pd.read_csv(filename)

# ── Load saved models ─────────────────────────────────────────────────────────
best_reg_model   = joblib.load(os.path.join(project_root, 'models', 'overall_rating_prediction.pkl'))
best_class_model = joblib.load(os.path.join(project_root, 'models', 'position_classification.pkl'))

best_reg_name   = type(best_reg_model).__name__
best_class_name = type(best_class_model).__name__

# ── Recreate splits and scalers (must match training scripts) ─────────────────
X_reg   = df_model.drop(columns=['overall_rating', 'best_position'])
y_reg   = df_model['overall_rating']
X_class = df_model.drop(columns=['best_position', 'overall_rating'])
y_class = df_model['best_position']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg,   y_reg,   test_size=0.2, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

scaler_r = StandardScaler()
X_train_r_scaled = scaler_r.fit_transform(X_train_r)
X_test_r_scaled  = scaler_r.transform(X_test_r)

scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled  = scaler_c.transform(X_test_c)

# ── Get predictions ───────────────────────────────────────────────────────────
reg_preds   = best_reg_model.predict(X_test_r_scaled)
class_preds = best_class_model.predict(X_test_c_scaled)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.style.use("dark_background")
BG   = "#0d1117"
CARD = "#161b22"
GRID = "#21262d"
CYAN = "#00e5ff"
PINK = "#ff4081"
AMBER= "#f39c12"

def _style(ax, title=""):
    ax.set_facecolor(CARD)
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.grid(color=GRID, linestyle="--", linewidth=0.5, alpha=0.7)
    if title:
        ax.set_title(title, color=CYAN, fontsize=10, fontweight="bold", pad=8)

fig = plt.figure(figsize=(20, 15), facecolor=BG)
fig.suptitle("FC 24 — ML Results Dashboard", color="white",
             fontsize=20, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.6)

# ── (A) Classification Accuracy ───────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
class_names = ["KNN", "Logistic Reg.", "Naive Bayes"]
class_models_temp = {
    "KNN": __import__('sklearn.neighbors', fromlist=['KNeighborsClassifier']).KNeighborsClassifier(n_neighbors=5),
    "Logistic Reg.": __import__('sklearn.linear_model', fromlist=['LogisticRegression']).LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": __import__('sklearn.naive_bayes', fromlist=['GaussianNB']).GaussianNB()
}
accs = []
for n, m in class_models_temp.items():
    m.fit(X_train_c_scaled, y_train_c)
    accs.append(accuracy_score(y_test_c, m.predict(X_test_c_scaled)))

colors_c = [CYAN if a == max(accs) else "#444d56" for a in accs]
bars_a = ax_a.barh(class_names, accs, color=colors_c, edgecolor="none", height=0.45)
ax_a.set_xlim(0, 1.1)
for b, v in zip(bars_a, accs):
    ax_a.text(v + 0.01, b.get_y() + b.get_height()/2, f"{v:.3f}", va="center", color="white", fontsize=8)
_style(ax_a, "Classification Accuracy")
ax_a.set_xlabel("Accuracy", color="white", fontsize=8)

# ── (B) Regression R² ────────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

reg_models_temp = {
    "Linear Reg.": LinearRegression(),
    "Ridge (α=10)": Ridge(alpha=10.0),
    "Poly (d=2)": make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
}
r2s, mses, reg_names = [], [], []
for n, m in reg_models_temp.items():
    m.fit(X_train_r_scaled, y_train_r)
    p = m.predict(X_test_r_scaled)
    r2s.append(r2_score(y_test_r, p))
    mses.append(mean_squared_error(y_test_r, p))
    reg_names.append(n)

colors_r = [PINK if r == max(r2s) else "#444d56" for r in r2s]
bars_b = ax_b.barh(reg_names, r2s, color=colors_r, edgecolor="none", height=0.45)
ax_b.set_xlim(0, 1.1)
for b, v in zip(bars_b, r2s):
    ax_b.text(v + 0.005, b.get_y() + b.get_height()/2, f"{v:.4f}", va="center", color="white", fontsize=8)
_style(ax_b, "Regression R² Score")
ax_b.set_xlabel("R²", color="white", fontsize=8)

# ── (C) Regression MSE ───────────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
bars_c = ax_c.barh(reg_names, mses, color=AMBER, edgecolor="none", height=0.45)
for b, v in zip(bars_c, mses):
    ax_c.text(v + 0.01, b.get_y() + b.get_height()/2, f"{v:.3f}", va="center", color="white", fontsize=8)
_style(ax_c, "Regression MSE (lower = better)")
ax_c.set_xlabel("MSE", color="white", fontsize=8)

# ── (D) Predicted vs Actual ───────────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 0:2])
ax_d.scatter(y_test_r, reg_preds, alpha=0.25, s=5, color=PINK, edgecolors="none")
lims = [min(y_test_r.min(), reg_preds.min()), max(y_test_r.max(), reg_preds.max())]
ax_d.plot(lims, lims, "--", color=CYAN, linewidth=1.5, label="Perfect fit")
ax_d.set_xlabel("Actual Rating", color="white", fontsize=8)
ax_d.set_ylabel("Predicted Rating", color="white", fontsize=8)
ax_d.legend(facecolor=CARD, edgecolor=GRID, labelcolor="white", fontsize=8)
_style(ax_d, f"Predicted vs Actual Overall Rating  [{best_reg_name}]")

# ── (E) Top-15 Features (Ridge |coef|) ───────────────────────────────────────
ax_g = fig.add_subplot(gs[1, 2])
ridge_model   = reg_models_temp["Ridge (α=10)"]
feature_names = X_reg.columns.tolist()
coefs   = np.abs(ridge_model.coef_)
top_idx = np.argsort(coefs)[-15:]
ax_g.barh([feature_names[i] for i in top_idx], coefs[top_idx],
          color=CYAN, edgecolor="none", height=0.8)
ax_g.set_xlabel("|Coefficient|", color="white", fontsize=8)
ax_g.tick_params(labelsize=7)
_style(ax_g, "Top 15 Features (Ridge |coef|)")

# ── Save ──────────────────────────────────────────────────────────────────────
output_path = os.path.join(project_root, 'outputs', 'position-overall rating-dashboard.png')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Dashboard saved → {output_path}")
plt.show()