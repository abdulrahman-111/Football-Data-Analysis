"""
regression_models.py
====================
One regression model trained on StatsBomb open data:

  Model — Linear Regression
    Predicts: home goals scored AND away goals scored (separately),
              then derives the predicted winner from the goal difference.
    Features: exp_home_goals, exp_away_goals

Data is imported and cleaned via data_cleaning.py.

Run standalone:
    python regression_models.py
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ── CSV paths ──────────────────────────────────────────────────────────────────
DATA_DIR     = os.path.join(os.path.dirname(__file__), "data")
MATCHES_CSV  = os.path.join(DATA_DIR, "matches_clean.csv")

MATCH_FEATURES = ["exp_home_goals", "exp_away_goals"]
OUTCOME_LABELS = {0: "Home Win", 1: "Draw", 2: "Away Win"}


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL — LINEAR REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════

class LinearRegressionGoals:
    def __init__(self):
        self.home_model = None
        self.away_model = None
        self.metrics    = {}
        self.is_fitted  = False

    def fit(self, matches_df: pd.DataFrame = None, verbose: bool = True):
        if matches_df is None:
            if not os.path.exists(MATCHES_CSV):
                raise FileNotFoundError(
                    f"CSV not found: {MATCHES_CSV}\n"
                    "Run data_cleaning.py first.")
            matches_df = pd.read_csv(MATCHES_CSV)
            if verbose:
                print(f"  Loaded {len(matches_df):,} rows from {MATCHES_CSV}")

        if verbose:
            print("\n" + "─" * 50)
            print("  MODEL — Linear Regression")
            print("  Targets: home goals, away goals → winner")
            print("─" * 50)

        df = matches_df.dropna(subset=MATCH_FEATURES + ["home_score", "away_score"]).copy()
        X  = df[MATCH_FEATURES]

        X_train, X_test, idx_train, idx_test = train_test_split(
            X, X.index, test_size=0.2, random_state=42)

        y_home_train = df.loc[idx_train, "home_score"]
        y_home_test  = df.loc[idx_test,  "home_score"]
        y_away_train = df.loc[idx_train, "away_score"]
        y_away_test  = df.loc[idx_test,  "away_score"]

        self.home_model = LinearRegression()
        self.home_model.fit(X_train, y_home_train)
        home_pred = self.home_model.predict(X_test)
        home_r2   = r2_score(y_home_test, home_pred)
        home_rmse = mean_squared_error(y_home_test, home_pred) ** 0.5

        self.away_model = LinearRegression()
        self.away_model.fit(X_train, y_away_train)
        away_pred = self.away_model.predict(X_test)
        away_r2   = r2_score(y_away_test, away_pred)
        away_rmse = mean_squared_error(y_away_test, away_pred) ** 0.5

        pred_outcomes = self._scores_to_outcome(home_pred, away_pred)
        true_outcomes = df.loc[idx_test, "outcome"].values
        winner_acc    = (pred_outcomes == true_outcomes).mean()

        self.metrics = {
            "model": "Linear Regression", "features": MATCH_FEATURES,
            "train_size": len(X_train), "test_size": len(X_test),
            "home_goals": {"r2": round(home_r2, 4), "rmse": round(home_rmse, 4),
                           "coefficients": dict(zip(MATCH_FEATURES, self.home_model.coef_.tolist())),
                           "intercept": float(self.home_model.intercept_)},
            "away_goals": {"r2": round(away_r2, 4), "rmse": round(away_rmse, 4),
                           "coefficients": dict(zip(MATCH_FEATURES, self.away_model.coef_.tolist())),
                           "intercept": float(self.away_model.intercept_)},
            "winner_prediction_accuracy": round(winner_acc, 4),
        }
        self.is_fitted = True

        if verbose:
            print(f"\n  Home R²={home_r2:.4f}  RMSE={home_rmse:.4f}")
            print(f"  Away R²={away_r2:.4f}  RMSE={away_rmse:.4f}")
            print(f"  Winner accuracy={winner_acc*100:.1f}%")
        return self

    def predict(self, home_stats: dict, away_stats: dict) -> dict:
        features = self._build_features(home_stats, away_stats)
        X = pd.DataFrame([features])[MATCH_FEATURES]
        home_goals = max(0.0, float(self.home_model.predict(X)[0]))
        away_goals = max(0.0, float(self.away_model.predict(X)[0]))
        diff = home_goals - away_goals
        outcome = 0 if diff > 0.15 else (2 if diff < -0.15 else 1)
        return {"predicted_home_goals": round(home_goals, 2),
                "predicted_away_goals": round(away_goals, 2),
                "goal_difference": round(diff, 2),
                "predicted_winner": OUTCOME_LABELS[outcome],
                "outcome_code": outcome}

    @staticmethod
    def _build_features(home, away):
        return {"exp_home_goals": (home["avg_goals"] + away["avg_conceded"]) / 2,
                "exp_away_goals": (away["avg_goals"] + home["avg_conceded"]) / 2}

    @staticmethod
    def _scores_to_outcome(home_preds, away_preds):
        diff = home_preds - away_preds
        return np.where(diff > 0.15, 0, np.where(diff < -0.15, 2, 1))


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class RegressionModels:
    def __init__(self):
        self.lr = LinearRegressionGoals()

    def fit(self, shots_df=None, matches_df=None, verbose=True):
        print("\n" + "=" * 50)
        print("  REGRESSION MODEL")
        print("  Data source: data/matches_clean.csv")
        print("=" * 50)
        self.lr.fit(matches_df, verbose=verbose)
        print("\n✓ Regression model trained successfully")
        return self

    def predict(self, home_stats, away_stats):
        return self.lr.predict(home_stats, away_stats)

    def predict_goals(self, home_stats, away_stats):
        return self.lr.predict(home_stats, away_stats)["predicted_home_goals"]

    @property
    def metrics(self):
        return {"linear_regression": self.lr.metrics}


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL GUI DASHBOARD  (tkinter + matplotlib — no browser needed)
# ═══════════════════════════════════════════════════════════════════════════════

def run_visualisations(reg: RegressionModels, matches_df: pd.DataFrame):
    """
    Single-window GUI dashboard for Linear Regression.
    Left panel  — interactive controls (sliders)
    Right panel — tabbed charts that update live when sliders move
    """
    import tkinter as tk
    from tkinter import ttk
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    lr = reg.lr
    m  = lr.metrics
    df = matches_df.dropna(subset=MATCH_FEATURES + ["home_score", "away_score"]).copy()
    X  = df[MATCH_FEATURES]

    home_pred_all = lr.home_model.predict(X)
    away_pred_all = lr.away_model.predict(X)
    home_resid    = df["home_score"].values - home_pred_all
    away_resid    = df["away_score"].values - away_pred_all
    pred_outs     = lr._scores_to_outcome(home_pred_all, away_pred_all)
    true_outs     = df["outcome"].values

    # ── Theme ──────────────────────────────────────────────────────────────────
    BG      = "#0f1117"
    CARD    = "#1a1d2e"
    FG      = "#e0e0e0"
    C_HOME  = "#1a6bb5"
    C_AWAY  = "#c0392b"
    C_ACC   = "#f39c12"
    C_GRAY  = "#7f8c8d"
    C_GREEN = "#27ae60"
    OUT_COLORS = [C_HOME, C_GRAY, C_AWAY]
    OUT_NAMES  = ["Home Win", "Draw", "Away Win"]

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": CARD,
        "axes.edgecolor": "#2a2d3e", "axes.labelcolor": FG,
        "xtick.color": FG, "ytick.color": FG, "text.color": FG,
        "grid.color": "#2a2d3e", "grid.linestyle": "--", "grid.alpha": 0.5,
        "legend.facecolor": CARD, "legend.edgecolor": CARD,
        "font.family": "sans-serif",
    })

    # ── Root window ────────────────────────────────────────────────────────────
    root = tk.Tk()
    root.title("Linear Regression — Interactive Dashboard")
    root.configure(bg=BG)
    root.state("zoomed")

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TNotebook",     background=BG, borderwidth=0)
    style.configure("TNotebook.Tab", background=CARD, foreground=FG,
                    padding=[12, 5], font=("Segoe UI", 9))
    style.map("TNotebook.Tab",       background=[("selected", C_HOME)],
                                     foreground=[("selected", "white")])
    style.configure("TFrame",        background=BG)
    style.configure("TSeparator",    background="#2a2d3e")

    # ── Title bar ──────────────────────────────────────────────────────────────
    title_bar = tk.Frame(root, bg=BG)
    title_bar.pack(fill="x", padx=16, pady=(10, 0))
    tk.Label(title_bar, text="Linear Regression — Interactive Dashboard",
             bg=BG, fg=FG, font=("Segoe UI", 15, "bold")).pack(side="left")
    tk.Label(title_bar,
             text=(f"Home R²={m['home_goals']['r2']:.4f}  "
                   f"Away R²={m['away_goals']['r2']:.4f}  "
                   f"RMSE={m['home_goals']['rmse']:.4f}  "
                   f"Winner Acc={m['winner_prediction_accuracy']*100:.1f}%"),
             bg=BG, fg=C_ACC, font=("Segoe UI", 9)).pack(side="right")

    # ── Main layout: left controls | right charts ──────────────────────────────
    main = tk.Frame(root, bg=BG)
    main.pack(fill="both", expand=True, padx=10, pady=8)

    # LEFT PANEL
    left = tk.Frame(main, bg=CARD, width=270)
    left.pack(side="left", fill="y", padx=(0, 8), pady=0)
    left.pack_propagate(False)

    tk.Label(left, text="Match Predictor", bg=CARD, fg=FG,
             font=("Segoe UI", 12, "bold")).pack(pady=(14, 2), padx=12)
    tk.Label(left, text="Adjust sliders to predict live",
             bg=CARD, fg=C_GRAY, font=("Segoe UI", 8)).pack(pady=(0, 10))
    ttk.Separator(left, orient="horizontal").pack(fill="x", padx=12, pady=(0, 10))

    sliders = {}
    slider_cfg = [
        ("Home avg goals",    "hg", 0.5, 3.5, 2.0,  C_HOME),
        ("Home avg conceded", "hc", 0.3, 2.5, 1.1,  C_HOME),
        ("Away avg goals",    "ag", 0.5, 3.5, 1.8,  C_AWAY),
        ("Away avg conceded", "ac", 0.3, 2.5, 1.3,  C_AWAY),
    ]
    for lbl, key, lo, hi, default, color in slider_cfg:
        tk.Label(left, text=lbl, bg=CARD, fg=color,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=14)
        var = tk.DoubleVar(value=default)
        tk.Scale(left, from_=lo, to=hi, resolution=0.1, orient="horizontal",
                 variable=var, length=240, bg=CARD, fg=FG, troughcolor=BG,
                 highlightthickness=0, activebackground=color,
                 font=("Segoe UI", 8)).pack(padx=10, pady=(0, 8))
        sliders[key] = var

    ttk.Separator(left, orient="horizontal").pack(fill="x", padx=12, pady=8)

    # Result boxes in left panel
    result_frame = tk.Frame(left, bg=CARD)
    result_frame.pack(fill="x", padx=12, pady=4)

    def _metric_box(parent, title, textvariable, color):
        f = tk.Frame(parent, bg=BG, padx=6, pady=6)
        tk.Label(f, text=title, bg=BG, fg=C_GRAY, font=("Segoe UI", 8)).pack()
        tk.Label(f, textvariable=textvariable, bg=BG, fg=color,
                 font=("Segoe UI", 18, "bold")).pack()
        return f

    var_hg_out   = tk.StringVar(value="—")
    var_ag_out   = tk.StringVar(value="—")
    var_diff_out = tk.StringVar(value="—")
    var_win_out  = tk.StringVar(value="—")
    var_win_col  = tk.StringVar(value=C_GRAY)

    b1 = _metric_box(result_frame, "Home Goals", var_hg_out,  C_HOME)
    b2 = _metric_box(result_frame, "Away Goals", var_ag_out,  C_AWAY)
    b3 = _metric_box(result_frame, "Difference", var_diff_out, C_ACC)
    b1.grid(row=0, column=0, padx=4, pady=4, sticky="ew")
    b2.grid(row=0, column=1, padx=4, pady=4, sticky="ew")
    b3.grid(row=1, column=0, columnspan=2, padx=4, pady=4, sticky="ew")

    winner_label = tk.Label(result_frame, textvariable=var_win_out,
                            bg=BG, fg=C_GRAY, font=("Segoe UI", 14, "bold"),
                            pady=8, relief="flat")
    winner_label.grid(row=2, column=0, columnspan=2, padx=4, pady=4, sticky="ew")
    result_frame.columnconfigure(0, weight=1)
    result_frame.columnconfigure(1, weight=1)

    # Probability bar
    ttk.Separator(left, orient="horizontal").pack(fill="x", padx=12, pady=6)
    bar_canvas = tk.Canvas(left, width=246, height=16, bg=BG, highlightthickness=0)
    bar_canvas.pack(padx=12, pady=(0, 12))

    # RIGHT PANEL — tabbed charts
    right = tk.Frame(main, bg=BG)
    right.pack(side="left", fill="both", expand=True)

    notebook = ttk.Notebook(right)
    notebook.pack(fill="both", expand=True)

    def make_tab(title):
        f = ttk.Frame(notebook)
        notebook.add(f, text=title)
        return f

    def embed_fig(fig, parent, toolbar=True):
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        if toolbar:
            tb = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
            tb.config(background=BG)
            tb.pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return canvas

    max_g = max(int(df["home_score"].max()), int(df["away_score"].max()), 6)

    # ── TAB 1: Predicted vs Actual ─────────────────────────────────────────────
    tab1 = make_tab("Predicted vs Actual")
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(11, 5))
    fig1.suptitle("Predicted vs Actual Goals", fontsize=13, fontweight="bold")
    for ax, pred, actual, color, lbl in [
        (ax1a, home_pred_all, df["home_score"].values, C_HOME, "Home Goals"),
        (ax1b, away_pred_all, df["away_score"].values, C_AWAY, "Away Goals"),
    ]:
        sc = ax.scatter(actual, pred, c=color, alpha=0.45, s=22, label=lbl)
        ax.plot([0, max_g], [0, max_g], "w--", lw=1.2, label="Perfect fit")
        ax.set_xlabel("Actual Goals"); ax.set_ylabel("Predicted Goals")
        ax.set_title(lbl, color=color, fontweight="bold"); ax.legend(); ax.grid(True)
    fig1.tight_layout()
    c1 = embed_fig(fig1, tab1)

    # ── TAB 2: Residuals ───────────────────────────────────────────────────────
    tab2 = make_tab("Residuals")
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(11, 5))
    fig2.suptitle("Residuals Distribution  (Actual − Predicted)", fontsize=13, fontweight="bold")
    for ax, resid, color, lbl in [
        (ax2a, home_resid, C_HOME, "Home Residuals"),
        (ax2b, away_resid, C_AWAY, "Away Residuals"),
    ]:
        ax.hist(resid, bins=22, color=color, alpha=0.8, edgecolor="white", lw=0.4)
        ax.axvline(0, color="white", ls="--", lw=1.5)
        ax.set_xlabel("Residual (goals)"); ax.set_ylabel("Count")
        ax.set_title(lbl, color=color, fontweight="bold"); ax.grid(True)
    fig2.tight_layout()
    embed_fig(fig2, tab2)

    # ── TAB 3: Coefficients ────────────────────────────────────────────────────
    tab3 = make_tab("Coefficients")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    fig3.suptitle("Model Coefficients", fontsize=13, fontweight="bold")
    all_feats  = MATCH_FEATURES + ["intercept"]
    home_coefs = list(lr.home_model.coef_) + [float(lr.home_model.intercept_)]
    away_coefs = list(lr.away_model.coef_) + [float(lr.away_model.intercept_)]
    xp = np.arange(len(all_feats))
    ax3.bar(xp - 0.2, home_coefs, 0.4, label="Home model", color=C_HOME, alpha=0.85)
    ax3.bar(xp + 0.2, away_coefs, 0.4, label="Away model", color=C_AWAY, alpha=0.85)
    ax3.axhline(0, color="white", ls="--", lw=1)
    ax3.set_xticks(xp); ax3.set_xticklabels(all_feats, rotation=12, ha="right")
    ax3.set_ylabel("Coefficient value"); ax3.legend(); ax3.grid(True, axis="y")
    fig3.tight_layout()
    embed_fig(fig3, tab3)

    # ── TAB 4: Winner Accuracy ─────────────────────────────────────────────────
    tab4 = make_tab("Winner Accuracy")
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    fig4.suptitle("Winner Prediction Accuracy by Outcome Class", fontsize=13, fontweight="bold")
    correct_by, total_by = [], []
    for code in [0, 1, 2]:
        mask = true_outs == code
        correct_by.append((pred_outs[mask] == code).sum())
        total_by.append(mask.sum())
    pct_by = [c / t * 100 if t > 0 else 0 for c, t in zip(correct_by, total_by)]
    bars4 = ax4.bar(OUT_NAMES, pct_by, color=OUT_COLORS, alpha=0.87, edgecolor="white", lw=0.5)
    ax4.axhline(m["winner_prediction_accuracy"] * 100, color=C_ACC, ls="--", lw=2,
                label=f"Overall {m['winner_prediction_accuracy']*100:.1f}%")
    for bar, pct in zip(bars4, pct_by):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{pct:.1f}%", ha="center", va="bottom", fontsize=11)
    ax4.set_ylabel("Accuracy (%)"); ax4.set_ylim(0, 115); ax4.legend(); ax4.grid(True, axis="y")
    fig4.tight_layout()
    embed_fig(fig4, tab4)

    # ── TAB 5: Exp vs Actual scatter ───────────────────────────────────────────
    tab5 = make_tab("Exp vs Actual")
    fig5, ax5 = plt.subplots(figsize=(9, 5))
    fig5.suptitle("Expected Home Goals vs Actual Home Goals (coloured by outcome)",
                  fontsize=13, fontweight="bold")
    for code, name, color in zip([0, 1, 2], OUT_NAMES, OUT_COLORS):
        mask = df["outcome"] == code
        ax5.scatter(df.loc[mask, "exp_home_goals"], df.loc[mask, "home_score"],
                    color=color, alpha=0.55, s=30, label=name)
    ax5.set_xlabel("Expected Home Goals (feature)"); ax5.set_ylabel("Actual Home Goals")
    ax5.legend(); ax5.grid(True)
    fig5.tight_layout()
    embed_fig(fig5, tab5)

    # ── TAB 6: Live Prediction Chart (updates with sliders) ────────────────────
    tab6 = make_tab("Live Prediction Chart")
    fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(11, 5))
    fig6.suptitle("Live: Predicted Goals vs Avg Goals  (drag sliders to update)",
                  fontsize=12, fontweight="bold")
    canvas6 = FigureCanvasTkAgg(fig6, master=tab6)
    canvas6.draw()
    NavigationToolbar2Tk(canvas6, tab6, pack_toolbar=False).pack(side="bottom", fill="x")
    canvas6.get_tk_widget().pack(fill="both", expand=True)

    sweep = np.linspace(0.5, 3.5, 60)

    def _redraw_live_chart(hg, hc, ag, ac):
        ax6a.cla(); ax6b.cla()
        # Home goals as home avg_goals sweeps
        h_goals = [max(0, float(lr.home_model.predict(
            pd.DataFrame([{"exp_home_goals": (g + ac) / 2,
                           "exp_away_goals": (ag + hc) / 2}])[MATCH_FEATURES])[0]))
                   for g in sweep]
        a_goals = [max(0, float(lr.away_model.predict(
            pd.DataFrame([{"exp_home_goals": (hg + ac) / 2,
                           "exp_away_goals": (g + hc) / 2}])[MATCH_FEATURES])[0]))
                   for g in sweep]

        ax6a.plot(sweep, h_goals, color=C_HOME, lw=2.5, label="Pred home goals")
        ax6a.axvline(hg, color="white", ls="--", lw=1.2, label=f"Current: {hg:.1f}")
        ax6a.set_xlabel("Home avg goals"); ax6a.set_ylabel("Predicted home goals")
        ax6a.set_title("Home Goals Sensitivity", color=C_HOME, fontweight="bold")
        ax6a.legend(); ax6a.grid(True)

        ax6b.plot(sweep, a_goals, color=C_AWAY, lw=2.5, label="Pred away goals")
        ax6b.axvline(ag, color="white", ls="--", lw=1.2, label=f"Current: {ag:.1f}")
        ax6b.set_xlabel("Away avg goals"); ax6b.set_ylabel("Predicted away goals")
        ax6b.set_title("Away Goals Sensitivity", color=C_AWAY, fontweight="bold")
        ax6b.legend(); ax6b.grid(True)

        fig6.tight_layout()
        canvas6.draw()

    # ── Slider callback — updates result boxes + live chart ────────────────────
    def _update(*_):
        hg = sliders["hg"].get()
        hc = sliders["hc"].get()
        ag = sliders["ag"].get()
        ac = sliders["ac"].get()

        res = reg.predict({"avg_goals": hg, "avg_conceded": hc},
                          {"avg_goals": ag, "avg_conceded": ac})

        var_hg_out.set(str(res["predicted_home_goals"]))
        var_ag_out.set(str(res["predicted_away_goals"]))
        var_diff_out.set(f"{res['goal_difference']:+.2f}")
        var_win_out.set(res["predicted_winner"])

        win_col = C_HOME if res["outcome_code"] == 0 else \
                  C_AWAY if res["outcome_code"] == 2 else C_GRAY
        winner_label.config(fg=win_col)

        # Probability bar (home xG as proxy for win probability)
        h_prob = max(0.05, min(0.95, 0.5 + res["goal_difference"] * 0.12))
        bar_canvas.delete("all")
        bar_canvas.create_rectangle(0, 0, 246, 16, fill="#2a2d3e", outline="")
        bar_canvas.create_rectangle(0, 0, int(246 * h_prob), 16, fill=C_HOME, outline="")
        bar_canvas.create_rectangle(int(246 * h_prob), 0, 246, 16, fill=C_AWAY, outline="")
        bar_canvas.create_text(123, 8, text=f"H {h_prob*100:.0f}%  |  A {(1-h_prob)*100:.0f}%",
                               fill=FG, font=("Segoe UI", 8, "bold"))

        if notebook.index(notebook.select()) == 5:
            _redraw_live_chart(hg, hc, ag, ac)

    notebook.bind("<<NotebookTabChanged>>",
                  lambda e: _redraw_live_chart(
                      sliders["hg"].get(), sliders["hc"].get(),
                      sliders["ag"].get(), sliders["ac"].get())
                  if notebook.index(notebook.select()) == 5 else None)

    for var in sliders.values():
        var.trace_add("write", _update)
    _update()

    root.mainloop()


# ── Standalone run ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(MATCHES_CSV):
        print("ERROR: data/matches_clean.csv not found.")
        print("Run data_cleaning.py first:\n  python data_cleaning.py")
        exit(1)

    matches_df = pd.read_csv(MATCHES_CSV)
    reg = RegressionModels()
    reg.fit(matches_df=matches_df)

    print("\n" + "=" * 50)
    for home, away in [
        ({"name": "Real Madrid",      "avg_goals": 2.6, "avg_conceded": 0.8},
         {"name": "Barcelona",        "avg_goals": 2.5, "avg_conceded": 0.9}),
        ({"name": "Manchester City",  "avg_goals": 2.4, "avg_conceded": 0.9},
         {"name": "West Ham United",  "avg_goals": 1.5, "avg_conceded": 1.6}),
    ]:
        r = reg.predict(home, away)
        print(f"\n  {home['name']} vs {away['name']}")
        print(f"    Score:  {r['predicted_home_goals']} – {r['predicted_away_goals']}")
        print(f"    Winner: {r['predicted_winner']}")

    run_visualisations(reg, matches_df)
