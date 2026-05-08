import os
import numpy as np
import pandas as pd
import warnings
import tkinter as tk
from tkinter import ttk
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")
matplotlib.use("TkAgg")

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SHOTS_CSV = os.path.join(DATA_DIR, "shots_clean.csv")
SHOT_FEATURES = ["distance", "angle", "body_part", "under_pressure"]

class DecisionTreeGoal:
    """Predicts Goal (1) or No Goal (0) using a Decision Tree."""
    def __init__(self, max_depth=7):
        self.model = None
        self.metrics = {}
        self.is_fitted = False

    def fit(self, shots_df: pd.DataFrame = None):
        if shots_df is None:
            shots_df = pd.read_csv(SHOTS_CSV) if os.path.exists(SHOTS_CSV) else self._generate_fallback(5000)

        df = shots_df.dropna(subset=SHOT_FEATURES + ["goal"]).copy()
        X_train, X_test, y_train, y_test = train_test_split(df[SHOT_FEATURES], df["goal"], test_size=0.2, random_state=42, stratify=df["goal"])
        
        self.model = DecisionTreeClassifier(max_depth=7, min_samples_split=20, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        self.metrics = {"accuracy": accuracy_score(y_test, y_pred), "cm": confusion_matrix(y_test, y_pred),
                        "importances": dict(zip(SHOT_FEATURES, self.model.feature_importances_)),
                        "goal_f1": report.get("1", report.get("1.0", {"f1-score": 0}))["f1-score"]}
        self.is_fitted = True
        return self

    def predict_proba(self, dist, angle, body, pres):
        return float(self.model.predict_proba(pd.DataFrame([[dist, angle, body, pres]], columns=SHOT_FEATURES))[0][1])

    def _generate_fallback(self, n):
        rng = np.random.default_rng(42)
        dist, ang = rng.uniform(5, 35, n), rng.uniform(5, 85, n)
        prob = 1 / (1 + np.exp(0.15*dist - 0.05*ang - 1.5))
        return pd.DataFrame({"distance": dist, "angle": ang, "body_part": rng.choice([0,1], n), 
                             "under_pressure": rng.choice([0,1], n), "goal": (rng.random(n) < prob).astype(int), "xg": prob})

# ── GUI Dashboard ─────────────────────────────────────────────────────────────

def run_visualisations(dt: DecisionTreeGoal, shots_df: pd.DataFrame):
    BG, CARD, FG, C_ACC = "#0f1117", "#1a1d2e", "#e0e0e0", "#f1c40f"
    C_GOAL, C_NOGOAL, C_BLUE = "#27ae60", "#e74c3c", "#3498db"

    # Confusion matrix colours: TN=dark teal, FP=orange, FN=orange, TP=bright green
    CM_COLORS = {(0,0): "#1a4a5e", (0,1): "#c0392b", (1,0): "#c0392b", (1,1): "#1e8449"}

    plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": CARD, "axes.edgecolor": "#4a4d5e",
                         "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG, "text.color": FG, "grid.color": "#2a2d3e"})

    root = tk.Tk()
    root.title("Advanced Shot Classification")
    root.configure(bg=BG)
    root.state("zoomed")

    style = ttk.Style()
    style.theme_use('default')
    style.configure("TNotebook", background=BG, borderwidth=0)
    style.configure("TNotebook.Tab", background=CARD, foreground=FG, padding=[15, 5], borderwidth=0)
    style.map("TNotebook.Tab", background=[("selected", C_BLUE)], foreground=[("selected", "white")])

    # Control Panel (Left)
    left = tk.Frame(root, bg=CARD, width=300, padx=20, pady=20); left.pack(side="left", fill="y"); left.pack_propagate(False)
    vars_ctrl = {"dist": tk.DoubleVar(value=12), "angle": tk.DoubleVar(value=45), "body": tk.IntVar(value=0), "pres": tk.IntVar(value=0)}
    
    for lbl, key, lo, hi in [("Distance (m)", "dist", 2, 40), ("Angle (deg)", "angle", 0, 90)]:
        tk.Label(left, text=lbl, bg=CARD, fg=C_BLUE, font=("Arial", 10, "bold")).pack(anchor="w")
        tk.Scale(left, from_=lo, to=hi, resolution=1, orient="horizontal", variable=vars_ctrl[key], bg=CARD, fg=FG, troughcolor=BG, highlightthickness=0).pack(fill="x", pady=(0, 15))

    for lbl, key, o1, o2 in [("Body Part", "body", "Foot", "Head"), ("Pressure", "pres", "No", "Yes")]:
        tk.Label(left, text=lbl, bg=CARD, fg=C_BLUE, font=("Arial", 10, "bold")).pack(anchor="w")
        f = tk.Frame(left, bg=CARD); f.pack(fill="x", pady=(0, 15))
        tk.Radiobutton(f, text=o1, variable=vars_ctrl[key], value=0, bg=CARD, fg=FG, selectcolor=BG).pack(side="left")
        tk.Radiobutton(f, text=o2, variable=vars_ctrl[key], value=1, bg=CARD, fg=FG, selectcolor=BG).pack(side="left", padx=20)

    res_v, out_v = tk.StringVar(value="0%"), tk.StringVar(value="NO GOAL")
    res_f = tk.Frame(left, bg=BG, pady=15); res_f.pack(fill="x", pady=10)
    tk.Label(res_f, textvariable=res_v, fg=C_ACC, font=("Arial", 28, "bold"), bg=BG).pack()
    out_lbl = tk.Label(res_f, textvariable=out_v, font=("Arial", 14, "bold"), bg=BG); out_lbl.pack()

    # Notebook Tabs (Right)
    # ── 4 tabs: Confusion Matrix | Feature Importance | Historical xG | Live Sensitivity
    nb = ttk.Notebook(root); nb.pack(side="right", fill="both", expand=True, padx=10, pady=10)

    fig_cm,   ax_cm   = plt.subplots(figsize=(10, 8))
    fig_fi,   ax_fi   = plt.subplots(figsize=(8, 4))
    fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
    fig_live, (ax_ld, ax_la) = plt.subplots(1, 2, figsize=(10, 4))

    t1 = ttk.Frame(nb); nb.add(t1, text="Confusion Matrix")
    t2 = ttk.Frame(nb); nb.add(t2, text="Feature Importance")
    t3 = ttk.Frame(nb); nb.add(t3, text="Historical xG Context")
    t4 = ttk.Frame(nb); nb.add(t4, text="Live Sensitivity Analysis")

    for fig, tab in [(fig_cm, t1), (fig_fi, t2), (fig_hist, t3), (fig_live, t4)]:
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ── Confusion Matrix (permanent, labelled, colour-coded) ──────────────────
    cm = dt.metrics["cm"]
    # Draw each cell with its own colour
    cell_colours = [[CM_COLORS[(i, j)] for j in range(2)] for i in range(2)]
    colour_array = np.array([[0, 1], [1, 0]], dtype=float)  # just for imshow shape
    ax_cm.imshow([[0.3, 0.7], [0.7, 0.3]], cmap="Blues", vmin=0, vmax=1, alpha=0)  # invisible base for sizing

    for i in range(2):
        for j in range(2):
            ax_cm.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=CM_COLORS[(i, j)], zorder=1))
            count = cm[i, j]
            total = cm.sum()
            pct = count / total * 100
            label_map = {(0,0): "True Negative\n(Correct Miss)",
                         (0,1): "False Positive\n(Predicted Goal,\nActually Miss)",
                         (1,0): "False Negative\n(Predicted Miss,\nActually Goal)",
                         (1,1): "True Positive\n(Correct Goal)"}
            ax_cm.text(j, i - 0.12, f"{count:,}", ha='center', va='center',
                       color="white", fontsize=22, fontweight="bold", zorder=2)
            ax_cm.text(j, i + 0.13, f"({pct:.1f}%)", ha='center', va='center',
                       color="white", fontsize=13, zorder=2)
            ax_cm.text(j, i + 0.35, label_map[(i, j)], ha='center', va='center',
                       color="#cccccc", fontsize=10, zorder=2)

    ax_cm.set_xlim(-0.5, 1.5); ax_cm.set_ylim(-0.5, 1.5)
    ax_cm.set_xticks([0, 1]); ax_cm.set_xticklabels(["Predicted: No Goal", "Predicted: Goal"], fontsize=13)
    ax_cm.set_yticks([0, 1]); ax_cm.set_yticklabels(["Actual: No Goal", "Actual: Goal"], fontsize=13)
    ax_cm.set_xlabel("Predicted Label", fontsize=14, fontweight="bold", labelpad=12)
    ax_cm.set_ylabel("Actual Label", fontsize=14, fontweight="bold", labelpad=12)
    ax_cm.set_title("Confusion Matrix", fontsize=16, fontweight="bold", pad=18)
    acc = dt.metrics["accuracy"]
    ax_cm.text(0.5, -0.09, f"Overall Accuracy: {acc*100:.1f}%", ha='center', va='center',
               transform=ax_cm.transAxes, color=C_ACC, fontsize=13, fontweight="bold")
    fig_cm.tight_layout(rect=[0, 0.04, 1, 1])
    fig_cm.canvas.draw()

    # ── Feature Importance (permanent, own tab) ───────────────────────────────
    feats = list(dt.metrics["importances"].keys())
    vals  = list(dt.metrics["importances"].values())
    feat_labels = {"distance": "Distance (m)", "angle": "Angle (°)",
                   "body_part": "Body Part", "under_pressure": "Under Pressure"}
    bar_colors = [C_BLUE if v < max(vals) else C_ACC for v in vals]
    bars = ax_fi.barh([feat_labels.get(f, f) for f in feats], vals, color=bar_colors, edgecolor="#2a2d3e", height=0.5)
    for bar, val in zip(bars, vals):
        ax_fi.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                   f"{val*100:.1f}%", va='center', color=FG, fontsize=9)
    ax_fi.set_title("Feature Importance", fontsize=13, fontweight="bold", pad=15)
    ax_fi.set_xlabel("Relative Importance (proportion of variance explained)", fontsize=10, fontweight="bold")
    ax_fi.set_xlim(0, max(vals) * 1.25)
    ax_fi.grid(True, axis='x', alpha=0.15)
    fig_fi.tight_layout()
    fig_fi.canvas.draw()

    # ── Dynamic updates (Historical + Live Sensitivity) ───────────────────────
    hist_annotation = None
    def update_all(*args):
        nonlocal hist_annotation
        d, a, b, p = vars_ctrl["dist"].get(), vars_ctrl["angle"].get(), vars_ctrl["body"].get(), vars_ctrl["pres"].get()
        prob = dt.predict_proba(d, a, b, p)
        res_v.set(f"{prob*100:.1f}%")
        out_v.set("GOAL" if prob > 0.5 else "NO GOAL")
        out_lbl.config(fg=C_GOAL if prob > 0.5 else C_NOGOAL)

        # ── Historical xG Context ─────────────────────────────────────────────
        ax_hist.cla()
        ax_hist.hist(shots_df[shots_df["goal"]==0]["xg"], bins=40, alpha=0.45, color=C_NOGOAL, label="Misses (No Goal)")
        ax_hist.hist(shots_df[shots_df["goal"]==1]["xg"], bins=40, alpha=0.75, color=C_GOAL, label="Goals")
        ax_hist.axvline(prob, color=C_ACC, lw=3, ls="--", label=f"Your Shot  ({prob*100:.1f}% xG)")

        # Determine where in the distribution the current shot falls
        miss_xg = shots_df[shots_df["goal"]==0]["xg"]
        goal_xg = shots_df[shots_df["goal"]==1]["xg"]
        pct_above_misses = (miss_xg < prob).mean() * 100
        pct_above_goals  = (goal_xg < prob).mean() * 100

        if prob < 0.15:
            quality = "Low-quality chance"
            quality_color = C_NOGOAL
        elif prob < 0.35:
            quality = "Moderate-quality chance"
            quality_color = C_ACC
        else:
            quality = "High-quality chance"
            quality_color = C_GOAL

        # Annotation box
        annotation_text = (
            f"How to read this chart:\n"
            f"  • Each bar shows how many historical shots had a given xG score.\n"
            f"  • Red bars = shots that were missed.  Green bars = shots that became goals.\n"
            f"  • Most shots cluster near 0 — low probability — because scoring is rare.\n"
            f"  • The yellow line is YOUR current shot selection.\n\n"
            f"Your shot  →  xG = {prob*100:.1f}%\n"
            f"  • Higher than {pct_above_misses:.0f}% of all misses in the dataset.\n"
            f"  • Higher than {pct_above_goals:.0f}% of all goals in the dataset.\n"
            f"  • Verdict: {quality}"
        )


        ax_hist.set_title("Historical Shot Distribution — Where Does Your Shot Sit?",
                          fontsize=12, fontweight="bold", pad=15)
        ax_hist.set_xlabel("xG Probability Score  (0 = very unlikely to score, 1 = almost certain)", fontsize=10, fontweight="bold")
        ax_hist.set_ylabel("Number of Shots in Dataset", fontsize=10, fontweight="bold")
        ax_hist.spines[['left', 'bottom']].set_linewidth(2)
        ax_hist.spines[['left', 'bottom']].set_color(FG)
        ax_hist.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax_hist.grid(True, alpha=0.1)

        # ── Live Sensitivity ──────────────────────────────────────────────────
        ax_ld.cla(); ax_la.cla()
        sweep_d = np.linspace(2, 40, 40); sweep_a = np.linspace(0, 90, 40)
        ax_ld.plot(sweep_d, [dt.predict_proba(x, a, b, p) for x in sweep_d], color=C_GOAL, lw=2)
        ax_ld.axvline(d, color=C_ACC, ls="--", label=f"Current: {d:.0f}m")
        ax_la.plot(sweep_a, [dt.predict_proba(d, x, b, p) for x in sweep_a], color=C_BLUE, lw=2)
        ax_la.axvline(a, color=C_ACC, ls="--", label=f"Current: {a:.0f}°")
        for ax, title, xlabel in [
            (ax_ld, "Distance Sensitivity", "Distance (m)"),
            (ax_la, "Angle Sensitivity",    "Angle (°)")
        ]:
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel("Goal Probability", fontsize=9)
            ax.set_ylim(0, 1); ax.grid(True, alpha=0.1); ax.legend(fontsize=8)

        for fig in [fig_hist, fig_live]:
            fig.canvas.draw_idle()

    for v in vars_ctrl.values():
        v.trace_add("write", update_all)
    update_all()
    root.mainloop()

if __name__ == "__main__":
    model = DecisionTreeGoal().fit()
    data = pd.read_csv(SHOTS_CSV) if os.path.exists(SHOTS_CSV) else model._generate_fallback(5000)
    run_visualisations(model, data)