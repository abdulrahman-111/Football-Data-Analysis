"""
⚽ Football Analytics Dashboard
Interactive local GUI — tkinter + matplotlib
Auto-loads:
  • data_injury.csv
  • Top5_League_Players_2017to2024_dataset.csv
  • transfer_value_prediction_dataset.csv

Features
--------
• Hover tooltips on every plot (bar, scatter, line, pie, box)
• Filter controls per chart (dropdowns, sliders, bin-size)
• 5 colour themes switchable at any time
• Zoom / Pan / Save toolbar
• Friendly card-style sidebar
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import seaborn as sns
import os, warnings, textwrap
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# AUTO-SEARCH
# ══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_NAMES = {
    "injury":       "data_injury.csv",
    "performance":  "Top5_League_Players_2017to2024_dataset.csv",
    "transfer":     "transfer_value_prediction_dataset.csv",
}

def find_dataset(name):
    target = DATASET_NAMES[name]
    d = SCRIPT_DIR
    roots = [d]
    for _ in range(4):
        d = os.path.dirname(d)
        roots.append(d)
    for root in roots:
        for dp, _, files in os.walk(root):
            if target in files:
                return os.path.join(dp, target)
    return None

# ══════════════════════════════════════════════════════════════════════════════
# COLOUR THEMES
# ══════════════════════════════════════════════════════════════════════════════
THEMES = {
    "🌑 Dark Ocean": dict(
        BG="#0b0e1a", PANEL="#131626", CARD="#1a1f35", BORDER="#2a3155",
        TEXT="#dde3f0", SUBTEXT="#7b88a8", BTN_HL="#1e2a4a",
        A1="#00d4ff", A2="#ff5f40", A3="#9d7fea", A4="#2ecc8f",
        PAL=["#00d4ff","#ff5f40","#9d7fea","#2ecc8f","#f472b6","#fbbf24","#60a5fa","#a3e635"],
        SNS="dark",
    ),
    "🌿 Forest": dict(
        BG="#0d1a0f", PANEL="#122016", CARD="#1a2e1d", BORDER="#264d2b",
        TEXT="#d6f0da", SUBTEXT="#7aab80", BTN_HL="#1e3d22",
        A1="#4ade80", A2="#fb923c", A3="#a78bfa", A4="#38bdf8",
        PAL=["#4ade80","#fb923c","#a78bfa","#38bdf8","#f472b6","#fbbf24","#34d399","#e879f9"],
        SNS="dark",
    ),
    "🔥 Ember": dict(
        BG="#1a0800", PANEL="#251200", CARD="#331a00", BORDER="#4d2a00",
        TEXT="#ffe8cc", SUBTEXT="#b07040", BTN_HL="#3d2200",
        A1="#ff9f43", A2="#ff4757", A3="#ffd32a", A4="#2ed573",
        PAL=["#ff9f43","#ff4757","#ffd32a","#2ed573","#eccc68","#ff6b81","#70a1ff","#a29bfe"],
        SNS="dark",
    ),
    "🌸 Rose": dict(
        BG="#fdf4f7", PANEL="#f5e6ed", CARD="#ecdaf0", BORDER="#d4a8c7",
        TEXT="#2d1a2e", SUBTEXT="#7a4f6a", BTN_HL="#e0c8d8",
        A1="#c0446c", A2="#e07b39", A3="#7c3aed", A4="#059669",
        PAL=["#c0446c","#e07b39","#7c3aed","#059669","#db2777","#d97706","#2563eb","#16a34a"],
        SNS="whitegrid",
    ),
    "🧊 Arctic": dict(
        BG="#f0f7ff", PANEL="#e0eeff", CARD="#d0e5ff", BORDER="#a0c4f0",
        TEXT="#0a1f40", SUBTEXT="#3a6090", BTN_HL="#c0d8f0",
        A1="#0077cc", A2="#e63950", A3="#7c3aed", A4="#059669",
        PAL=["#0077cc","#e63950","#7c3aed","#059669","#0891b2","#d97706","#4f46e5","#16a34a"],
        SNS="whitegrid",
    ),
}

# Active theme globals (read by all plot functions)
T = THEMES["🌑 Dark Ocean"].copy()

def apply_theme(name):
    global T
    T = THEMES[name].copy()
    sns.set_theme(style=T["SNS"], rc={
        "axes.facecolor":   T["CARD"],  "figure.facecolor": T["PANEL"],
        "axes.edgecolor":   T["BORDER"],"axes.labelcolor":  T["TEXT"],
        "xtick.color":      T["SUBTEXT"],"ytick.color":     T["SUBTEXT"],
        "text.color":       T["TEXT"],  "grid.color":       T["BORDER"],
        "grid.linestyle":   "--",       "grid.alpha":       0.35,
    })

apply_theme("🌑 Dark Ocean")

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════
def load_transfer(path):
    df = pd.read_csv(path).drop_duplicates(subset=['player'])
    for col in ['Goals','Assists','Yellow Cards','Second Yellow Card',
                'Red Card','Goal Conceded','Clean Sheets']:
        if col in df.columns and 'minutes played' in df.columns:
            df[col] = (df[col] * 90) / df['minutes played'].replace(0, np.nan)
            df.rename(columns={col: f"{col}_per90"}, inplace=True)
    if 'team' in df.columns and 'current_value' in df.columns:
        df['team_encoded'] = df['team'].map(df.groupby('team')['current_value'].mean())
    return df

def load_performance(path):
    df = pd.read_csv(path, sep=';', decimal=',')
    df = df.sort_values(['player','season'])
    df['goals_next']   = df.groupby('player')['Performance_Gls'].shift(-1)
    df['assists_next'] = df.groupby('player')['Performance_Ast'].shift(-1)
    df['goals_trend']  = df['Performance_Gls'] - df.groupby('player')['Performance_Gls'].shift(1)
    df = df.dropna(subset=['goals_next','assists_next','goals_trend'])
    pos_map = {'GK':1,'DF':2,'MF':3,'FW':4,'DF,FW':4,'FW,MF':4,
               'MF,FW':4,'FW,DF':4,'MF,DF':3,'DF,MF':3}
    df['position_encoded'] = df['pos_'].map(pos_map)
    return df

def load_injury(path):
    df = pd.read_csv(path).drop_duplicates()
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].mean())
    return df

# ══════════════════════════════════════════════════════════════════════════════
# SHARED PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def style_ax(ax, title="", xlabel="", ylabel="", xrot=0):
    ax.set_facecolor(T["CARD"])
    ax.set_title(title, color=T["TEXT"], fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, color=T["SUBTEXT"], fontsize=9)
    ax.set_ylabel(ylabel, color=T["SUBTEXT"], fontsize=9)
    ax.tick_params(colors=T["SUBTEXT"], labelsize=8)
    if xrot: ax.tick_params(axis='x', rotation=xrot)
    for sp in ax.spines.values(): sp.set_edgecolor(T["BORDER"])
    ax.grid(True, color=T["BORDER"], linestyle='--', alpha=0.3)

def add_tooltip(fig, canvas, artists, labels):
    """Universal hover tooltip for bars, scatter dots, lines."""
    annot = None

    def on_move(event):
        nonlocal annot
        if event.inaxes is None:
            if annot: annot.set_visible(False); canvas.draw_idle()
            return
        hit = False
        for art, lbl in zip(artists, labels):
            cont = False
            try:
                if hasattr(art, 'contains'):
                    cont, _ = art.contains(event)
            except Exception:
                pass
            if cont:
                hit = True
                ax = art.axes
                if annot: annot.remove()
                annot = ax.annotate(
                    str(lbl),
                    xy=(event.xdata, event.ydata),
                    xytext=(12, 12), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.4", fc=T["PANEL"],
                              ec=T["A1"], lw=1.2, alpha=0.92),
                    color=T["TEXT"], fontsize=8.5,
                    arrowprops=dict(arrowstyle="-", color=T["A1"], lw=0.8),
                    zorder=999,
                )
                canvas.draw_idle()
                break
        if not hit:
            if annot: annot.set_visible(False); canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)


def add_bar_tooltips(fig, canvas, ax, bars, values, fmt="{:.2f}"):
    arts, labels = [], []
    for bar, val in zip(bars, values):
        arts.append(bar)
        labels.append(fmt.format(val))
    add_tooltip(fig, canvas, arts, labels)

def add_scatter_tooltips(fig, canvas, ax, sc, names):
    """Works with PathCollection (scatter). Shows name on hover."""
    annot = ax.annotate("", xy=(0,0), xytext=(14,14),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.4", fc=T["PANEL"],
                                  ec=T["A1"], lw=1.2, alpha=0.92),
                        color=T["TEXT"], fontsize=8.5,
                        arrowprops=dict(arrowstyle="-", color=T["A1"], lw=0.8),
                        zorder=999)
    annot.set_visible(False)

    def on_move(event):
        vis = annot.get_visible()
        if event.inaxes != ax:
            if vis: annot.set_visible(False); canvas.draw_idle()
            return
        cont, ind = sc.contains(event)
        if cont:
            idx = ind["ind"][0]
            pos = sc.get_offsets()[idx]
            annot.xy = pos
            annot.set_text(str(names[idx]))
            annot.set_visible(True)
            canvas.draw_idle()
        else:
            if vis: annot.set_visible(False); canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)

# ══════════════════════════════════════════════════════════════════════════════
# ── TRANSFER VALUE PLOTS ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def tv_dist(df, fig, canvas, controls):
    bins  = int(controls.get("Bins", 30))
    v = df['current_value'].dropna()

    ax  = fig.add_subplot(1,2,1)
    n, edges, patches = ax.hist(v, bins=bins, color=T["A1"], edgecolor=T["BG"], alpha=0.85)
    ax.axvline(v.mean(),  color=T["A2"], lw=2, ls='--', label=f'Mean  {v.mean():.1f} M€')
    ax.axvline(v.median(),color=T["A3"], lw=2, ls=':',  label=f'Median {v.median():.1f} M€')
    ax.legend(fontsize=8, labelcolor=T["TEXT"], framealpha=0.3)
    style_ax(ax, "Transfer Value Distribution", "Value (M€)", "Count")
    # tooltip: bin centres + counts
    centres = (edges[:-1] + edges[1:]) / 2
    add_bar_tooltips(fig, canvas, ax, patches, centres,
                     fmt="Range ≈ {:.1f} M€")

    ax2 = fig.add_subplot(1,2,2)
    n2, edges2, patches2 = ax2.hist(np.log1p(v), bins=bins,
                                     color=T["A3"], edgecolor=T["BG"], alpha=0.85)
    style_ax(ax2, "Log-Scale Distribution", "log(Value + 1)", "Count")
    add_bar_tooltips(fig, canvas, ax2, patches2,
                     (edges2[:-1]+edges2[1:])/2, fmt="log ≈ {:.2f}")

def tv_pos(df, fig, canvas, controls):
    if 'position' not in df.columns: return
    positions = sorted(df['position'].dropna().unique().tolist())
    sel = controls.get("Position", "All")
    dff = df if sel == "All" else df[df['position'] == sel]

    ax  = fig.add_subplot(1,2,1)
    pm  = dff.groupby('position')['current_value'].median().sort_values(ascending=False)
    bars = ax.bar(pm.index, pm.values, color=T["PAL"][:len(pm)],
                  edgecolor=T["BG"], linewidth=0.5)
    for b, v in zip(bars, pm.values):
        ax.text(b.get_x()+b.get_width()/2, v+0.3,
                f'{v:.1f}', ha='center', va='bottom', fontsize=7.5, color=T["SUBTEXT"])
    add_bar_tooltips(fig, canvas, ax, bars, pm.values, fmt="{:.1f} M€ median")
    style_ax(ax, "Median Value by Position", "Position", "M€", xrot=25)

    ax2 = fig.add_subplot(1,2,2)
    data_bp = [df[df['position']==p]['current_value'].dropna().values for p in positions]
    bp = ax2.boxplot(data_bp, patch_artist=True, labels=positions,
                     medianprops=dict(color=T["TEXT"], lw=1.5))
    for patch, c in zip(bp['boxes'], T["PAL"]):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    for el in ['whiskers','caps','fliers']:
        for item in bp[el]: item.set_color(T["SUBTEXT"])
    style_ax(ax2, "Value Spread by Position", "Position", "Value (M€)", xrot=25)
    # tooltip on box labels
    for patch, pos in zip(bp['boxes'], positions):
        med = df[df['position']==pos]['current_value'].median()
        arts = [patch]
        add_tooltip(fig, canvas, arts, [f"{pos}  median {med:.1f} M€"])

def tv_age(df, fig, canvas, controls):
    min_age = int(controls.get("Min Age", df['age'].min()))
    max_age = int(controls.get("Max Age", df['age'].max()))
    dff = df[(df['age'] >= min_age) & (df['age'] <= max_age)]

    ax = fig.add_subplot(1,2,1)
    sc = ax.scatter(dff['age'], dff['current_value'],
                    alpha=0.45, s=18, c=T["A1"], edgecolors='none')
    d = dff[['age','current_value']].dropna()
    if len(d) > 3:
        z  = np.polyfit(d['age'], d['current_value'], 2)
        xs = np.linspace(d['age'].min(), d['age'].max(), 200)
        ax.plot(xs, np.poly1d(z)(xs), color=T["A2"], lw=2.2, label='Trend')
        ax.legend(fontsize=8, labelcolor=T["TEXT"], framealpha=0.3)
    style_ax(ax, "Age vs Transfer Value", "Age", "Value (M€)")
    names = dff.get('player', dff.get('name', pd.Series(dff.index))).values
    add_scatter_tooltips(fig, canvas, ax, sc,
        [f"{n}  age {a}  {v:.1f}M€"
         for n,a,v in zip(names, dff['age'], dff['current_value'])])

    ax2 = fig.add_subplot(1,2,2)
    sc2 = ax2.scatter(dff['highest_value'], dff['current_value'],
                      alpha=0.45, s=18, c=T["A3"], edgecolors='none')
    style_ax(ax2, "Highest vs Current Value", "Highest (M€)", "Current (M€)")
    add_scatter_tooltips(fig, canvas, ax2, sc2,
        [f"{n}  peak {h:.1f}  curr {c:.1f} M€"
         for n,h,c in zip(names, dff['highest_value'], dff['current_value'])])

def tv_teams(df, fig, canvas, controls):
    n_top = int(controls.get("Top N Teams", 15))
    ax = fig.add_subplot(1,1,1)
    top = df.groupby('team')['current_value'].sum().nlargest(n_top).sort_values()
    colors = [T["A1"] if v == top.max() else T["A3"] for v in top.values]
    bars = ax.barh(top.index, top.values, color=colors, edgecolor=T["BG"], height=0.65)
    for b in bars:
        ax.text(b.get_width()+0.4, b.get_y()+b.get_height()/2,
                f'{b.get_width():.0f} M€', va='center', fontsize=8, color=T["SUBTEXT"])
    add_bar_tooltips(fig, canvas, ax, bars, top.values, fmt="{:.0f} M€ total")
    style_ax(ax, f"Top {n_top} Teams by Total Squad Value", "Total Value (M€)", "")

def tv_corr(df, fig, canvas, controls):
    num = df.select_dtypes(include=np.number)
    if 'current_value' not in num.columns: return
    corr = num.corr()['current_value'].drop('current_value').sort_values()
    ax = fig.add_subplot(1,1,1)
    colors = [T["A4"] if v > 0 else T["A2"] for v in corr.values]
    bars = ax.barh(corr.index, corr.values, color=colors, edgecolor=T["BG"], height=0.65)
    ax.axvline(0, color=T["SUBTEXT"], lw=0.9)
    add_bar_tooltips(fig, canvas, ax, bars, corr.values, fmt="r = {:.3f}")
    style_ax(ax, "Feature Correlation with Transfer Value", "Pearson r", "")

def tv_goals(df, fig, canvas, controls):
    gc = next((c for c in df.columns if 'Goals_per90' in c or 'Goals_per' in c), None)
    ac = next((c for c in df.columns if 'Assists_per90' in c or 'Assists_per' in c), None)
    names = df.get('player', df.get('name', pd.Series(df.index))).values

    ax = fig.add_subplot(1,2,1)
    if gc:
        sc = ax.scatter(df[gc], df['current_value'],
                        alpha=0.4, s=18, c=T["A4"], edgecolors='none')
        style_ax(ax, "Goals / 90 min  vs  Transfer Value", "Goals per 90", "Value (M€)")
        add_scatter_tooltips(fig, canvas, ax, sc,
            [f"{n}  {g:.2f} g/90  {v:.1f}M€"
             for n,g,v in zip(names, df[gc], df['current_value'])])

    ax2 = fig.add_subplot(1,2,2)
    if ac:
        sc2 = ax2.scatter(df[ac], df['current_value'],
                          alpha=0.4, s=18, c=T["A2"], edgecolors='none')
        style_ax(ax2, "Assists / 90 min  vs  Transfer Value", "Assists per 90", "Value (M€)")
        add_scatter_tooltips(fig, canvas, ax2, sc2,
            [f"{n}  {a:.2f} a/90  {v:.1f}M€"
             for n,a,v in zip(names, df[ac], df['current_value'])])

# ══════════════════════════════════════════════════════════════════════════════
# ── PERFORMANCE PLOTS ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def pf_trend(df, fig, canvas, controls):
    league_opts = ["All"] + sorted(df['league'].dropna().unique().tolist())
    sel = controls.get("League", "All")
    dff = df if sel == "All" else df[df['league'] == sel]

    sg = dff.groupby('season')['Performance_Gls'].mean()
    sa = dff.groupby('season')['Performance_Ast'].mean()
    xs = range(len(sg))

    ax = fig.add_subplot(1,2,1)
    line, = ax.plot(xs, sg.values, marker='o', color=T["A1"], lw=2.5, markersize=7)
    ax.fill_between(xs, sg.values, alpha=0.12, color=T["A1"])
    ax.set_xticks(list(xs)); ax.set_xticklabels(sg.index.astype(str), rotation=38, fontsize=7)
    style_ax(ax, f"Avg Goals per Season  [{sel}]", "Season", "Avg Goals")
    # scatter overlay for hover
    sc = ax.scatter(xs, sg.values, s=60, c=T["A1"], zorder=5, edgecolors=T["BG"])
    add_scatter_tooltips(fig, canvas, ax, sc,
        [f"Season {s}  avg {v:.2f} goals" for s,v in zip(sg.index, sg.values)])

    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(xs, sa.values, marker='s', color=T["A2"], lw=2.5, markersize=7)
    ax2.fill_between(xs, sa.values, alpha=0.12, color=T["A2"])
    ax2.set_xticks(list(xs)); ax2.set_xticklabels(sa.index.astype(str), rotation=38, fontsize=7)
    style_ax(ax2, f"Avg Assists per Season  [{sel}]", "Season", "Avg Assists")
    sc2 = ax2.scatter(xs, sa.values, s=60, c=T["A2"], zorder=5, edgecolors=T["BG"])
    add_scatter_tooltips(fig, canvas, ax2, sc2,
        [f"Season {s}  avg {v:.2f} assists" for s,v in zip(sa.index, sa.values)])

def pf_league(df, fig, canvas, controls):
    metric = controls.get("Metric", "Goals")
    col_map = {"Goals":"Performance_Gls","Assists":"Performance_Ast","xG":"Expected_xG","xAG":"Expected_xAG"}
    col = col_map.get(metric, "Performance_Gls")
    if col not in df.columns: col = "Performance_Gls"

    ax = fig.add_subplot(1,2,1)
    lg = df.groupby('league')[col].mean().sort_values(ascending=False)
    bars = ax.bar(lg.index, lg.values, color=T["PAL"][:len(lg)], edgecolor=T["BG"])
    add_bar_tooltips(fig, canvas, ax, bars, lg.values, fmt=f"{{:.2f}} avg {metric}")
    style_ax(ax, f"Avg {metric} by League", "League", f"Avg {metric}", xrot=25)

    ax2 = fig.add_subplot(1,2,2)
    xg = df.groupby('league')['Expected_xG'].mean().sort_values(ascending=False)
    bars2 = ax2.bar(xg.index, xg.values, color=T["PAL"][:len(xg)], edgecolor=T["BG"])
    add_bar_tooltips(fig, canvas, ax2, bars2, xg.values, fmt="{:.2f} avg xG")
    style_ax(ax2, "Avg xG by League", "League", "Avg xG", xrot=25)

def pf_position(df, fig, canvas, controls):
    season_opts = ["All"] + sorted(df['season'].dropna().unique().astype(str).tolist())
    sel = controls.get("Season", "All")
    dff = df if sel == "All" else df[df['season'].astype(str) == sel]

    ax = fig.add_subplot(1,2,1)
    pg = dff.groupby('pos_')['Performance_Gls'].mean().sort_values(ascending=False).head(10)
    bars = ax.bar(pg.index, pg.values, color=T["A3"], edgecolor=T["BG"], alpha=0.85)
    add_bar_tooltips(fig, canvas, ax, bars, pg.values, fmt="{:.2f} avg goals")
    style_ax(ax, "Avg Goals by Position", "Position", "Goals", xrot=30)

    ax2 = fig.add_subplot(1,2,2)
    pa = dff.groupby('pos_')['Performance_Ast'].mean().sort_values(ascending=False).head(10)
    bars2 = ax2.bar(pa.index, pa.values, color=T["A4"], edgecolor=T["BG"], alpha=0.85)
    add_bar_tooltips(fig, canvas, ax2, bars2, pa.values, fmt="{:.2f} avg assists")
    style_ax(ax2, "Avg Assists by Position", "Position", "Assists", xrot=30)

def pf_xg(df, fig, canvas, controls):
    pos_opts = ["All"] + sorted(df['pos_'].dropna().unique().tolist())
    sel = controls.get("Position", "All")
    dff = df if sel == "All" else df[df['pos_'] == sel]
    sample = dff.sample(min(2000, len(dff)), random_state=42)

    ax = fig.add_subplot(1,2,1)
    sc = ax.scatter(sample['Expected_xG'], sample['Performance_Gls'],
                    alpha=0.35, s=14, c=T["A1"], edgecolors='none')
    mn = min(sample['Expected_xG'].min(), sample['Performance_Gls'].min())
    mx = max(sample['Expected_xG'].max(), sample['Performance_Gls'].max())
    ax.plot([mn,mx],[mn,mx], color=T["A2"], lw=1.8, ls='--', label='xG = Goals')
    ax.legend(fontsize=8, labelcolor=T["TEXT"], framealpha=0.3)
    style_ax(ax, "xG vs Actual Goals", "Expected Goals (xG)", "Actual Goals")
    names = sample['player'].values if 'player' in sample.columns else sample.index.astype(str)
    add_scatter_tooltips(fig, canvas, ax, sc,
        [f"{n}  xG {x:.2f}  Goals {g:.0f}"
         for n,x,g in zip(names, sample['Expected_xG'], sample['Performance_Gls'])])

    ax2 = fig.add_subplot(1,2,2)
    sc2 = ax2.scatter(sample['Expected_xAG'], sample['Performance_Ast'],
                      alpha=0.35, s=14, c=T["A3"], edgecolors='none')
    mn2 = min(sample['Expected_xAG'].min(), sample['Performance_Ast'].min())
    mx2 = max(sample['Expected_xAG'].max(), sample['Performance_Ast'].max())
    ax2.plot([mn2,mx2],[mn2,mx2], color=T["A2"], lw=1.8, ls='--')
    style_ax(ax2, "xAG vs Actual Assists", "Expected Ast. Goals (xAG)", "Actual Assists")
    add_scatter_tooltips(fig, canvas, ax2, sc2,
        [f"{n}  xAG {x:.2f}  Ast {a:.0f}"
         for n,x,a in zip(names, sample['Expected_xAG'], sample['Performance_Ast'])])

def pf_time(df, fig, canvas, controls):
    ax = fig.add_subplot(1,2,1)
    sc = ax.scatter(df['Playing Time_Min'], df['Performance_Gls'],
                    alpha=0.28, s=12, c=T["A4"], edgecolors='none')
    style_ax(ax, "Minutes Played vs Goals", "Minutes Played", "Goals")
    names = df['player'].values if 'player' in df.columns else df.index.astype(str)
    add_scatter_tooltips(fig, canvas, ax, sc,
        [f"{n}  {m:.0f} min  {g:.0f} goals"
         for n,m,g in zip(names, df['Playing Time_Min'], df['Performance_Gls'])])

    ax2 = fig.add_subplot(1,2,2)
    if 'Per 90 Minutes_Gls' in df.columns:
        v = df['Per 90 Minutes_Gls'].dropna()
        v = v[v < v.quantile(0.99)]
        n, edges, patches = ax2.hist(v, bins=35, color=T["A2"], edgecolor=T["BG"], alpha=0.85)
        add_bar_tooltips(fig, canvas, ax2, patches,
                         (edges[:-1]+edges[1:])/2, fmt="{:.2f} goals/90")
        style_ax(ax2, "Goals per 90 Distribution", "Goals / 90 min", "Count")

def pf_corr(df, fig, canvas, controls):
    cols = ['Performance_Gls','Performance_Ast','Expected_xG','Expected_xAG',
            'Playing Time_Min','Playing Time_90s','Standard_Sh','Standard_SoT',
            'Progression_PrgC','Progression_PrgP','goals_next','assists_next']
    avail = [c for c in cols if c in df.columns]
    ax = fig.add_subplot(1,1,1)
    corr = df[avail].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, ax=ax, annot=True, fmt='.2f',
                cmap='coolwarm', annot_kws={'size':7.5},
                linewidths=0.4, linecolor=T["BG"],
                cbar_kws={'shrink':0.75})
    style_ax(ax, "Correlation Matrix — Performance Features", "", "")

# ══════════════════════════════════════════════════════════════════════════════
# ── INJURY PLOTS ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def inj_balance(df, fig, canvas, controls):
    ax = fig.add_subplot(1,2,1)
    counts = df['Injury_Next_Season'].value_counts().sort_index()
    labels_pie = ['No Injury','Injury']
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=labels_pie,
        autopct='%1.1f%%', colors=[T["A4"],T["A2"]],
        startangle=90, pctdistance=0.72,
        wedgeprops={'edgecolor':T["BG"],'linewidth':2})
    for t in texts: t.set_color(T["TEXT"]); t.set_fontsize(9)
    for at in autotexts: at.set_color(T["BG"]); at.set_fontweight('bold')
    ax.set_facecolor(T["CARD"])
    ax.set_title("Class Distribution", color=T["TEXT"], fontsize=10, fontweight='bold')
    # tooltip on wedges
    add_tooltip(fig, canvas, list(wedges),
                [f"{l}: {v} players ({v/counts.sum()*100:.1f}%)"
                 for l,v in zip(labels_pie, counts.values)])

    ax2 = fig.add_subplot(1,2,2)
    bars = ax2.bar(labels_pie, counts.values,
                   color=[T["A4"],T["A2"]], edgecolor=T["BG"], width=0.45)
    for i,v in enumerate(counts.values):
        ax2.text(i, v+4, str(v), ha='center', fontsize=10,
                 color=T["TEXT"], fontweight='bold')
    add_bar_tooltips(fig, canvas, ax2, bars, counts.values, fmt="{:.0f} players")
    style_ax(ax2, "Injury Count", "Class", "Count")

def inj_age(df, fig, canvas, controls):
    ax = fig.add_subplot(1,2,1)
    for cls, c, lbl in [(0,T["A4"],'No Injury'),(1,T["A2"],'Injury')]:
        vals = df[df['Injury_Next_Season']==cls]['Age'].dropna()
        n, edges, patches = ax.hist(vals, bins=25, alpha=0.68,
                                     color=c, edgecolor=T["BG"], label=lbl)
    ax.legend(fontsize=8, labelcolor=T["TEXT"], framealpha=0.3)
    style_ax(ax, "Age Distribution by Injury Status", "Age", "Count")

    ax2 = fig.add_subplot(1,2,2)
    for cls, c, lbl in [(0,T["A4"],'No Injury'),(1,T["A2"],'Injury')]:
        ax2.hist(df[df['Injury_Next_Season']==cls]['BMI'].dropna(),
                 bins=25, alpha=0.68, color=c, edgecolor=T["BG"], label=lbl)
    ax2.legend(fontsize=8, labelcolor=T["TEXT"], framealpha=0.3)
    style_ax(ax2, "BMI Distribution by Injury Status", "BMI", "Count")

def inj_phys(df, fig, canvas, controls):
    cols = ['Knee_Strength_Score','Hamstring_Flexibility','Balance_Test_Score',
            'Agility_Score','Sprint_Speed_10m_s','Reaction_Time_ms']
    cols = [c for c in cols if c in df.columns]
    for i, col in enumerate(cols):
        ax = fig.add_subplot(2, 3, i+1)
        d0 = df[df['Injury_Next_Season']==0][col].dropna()
        d1 = df[df['Injury_Next_Season']==1][col].dropna()
        bp = ax.boxplot([d0,d1], patch_artist=True,
                        labels=['No Inj','Inj'],
                        medianprops=dict(color=T["TEXT"], lw=1.5))
        bp['boxes'][0].set_facecolor(T["A4"]); bp['boxes'][0].set_alpha(0.75)
        bp['boxes'][1].set_facecolor(T["A2"]); bp['boxes'][1].set_alpha(0.75)
        for el in ['whiskers','caps','fliers']:
            for item in bp[el]: item.set_color(T["SUBTEXT"])
        style_ax(ax, col.replace('_',' '), "", "")
        # tooltip
        add_tooltip(fig, canvas,
                    [bp['boxes'][0], bp['boxes'][1]],
                    [f"No Injury  median={d0.median():.2f}",
                     f"Injury  median={d1.median():.2f}"])

def inj_corr(df, fig, canvas, controls):
    # Use ONLY numeric columns — avoids "cannot convert string" error
    num_df = df.select_dtypes(include=np.number)
    if 'Injury_Next_Season' not in num_df.columns:
        ax = fig.add_subplot(1,1,1)
        ax.text(0.5,0.5,"Injury_Next_Season not numeric", ha='center',
                va='center', color=T["TEXT"]); return
    corr = num_df.corr()['Injury_Next_Season'].drop('Injury_Next_Season').sort_values()
    ax = fig.add_subplot(1,1,1)
    colors = [T["A2"] if v > 0 else T["A4"] for v in corr.values]
    bars = ax.barh(corr.index, corr.values, color=colors,
                   edgecolor=T["BG"], height=0.65)
    ax.axvline(0, color=T["SUBTEXT"], lw=0.9)
    add_bar_tooltips(fig, canvas, ax, bars, corr.values, fmt="r = {:.3f}")
    style_ax(ax, "Feature Correlation with Injury Next Season", "Pearson r", "")

def inj_life(df, fig, canvas, controls):
    ax = fig.add_subplot(1,2,1)
    for cls, c, lbl in [(0,T["A4"],'No Injury'),(1,T["A2"],'Injury')]:
        ax.hist(df[df['Injury_Next_Season']==cls]['Sleep_Hours_Per_Night'].dropna(),
                bins=20, alpha=0.68, color=c, edgecolor=T["BG"], label=lbl)
    ax.legend(fontsize=8, labelcolor=T["TEXT"], framealpha=0.3)
    style_ax(ax, "Sleep Hours by Injury Status", "Sleep Hrs / Night", "Count")

    ax2 = fig.add_subplot(1,2,2)
    for cls, c, lbl in [(0,T["A4"],'No Injury'),(1,T["A2"],'Injury')]:
        ax2.hist(df[df['Injury_Next_Season']==cls]['Stress_Level_Score'].dropna(),
                 bins=20, alpha=0.68, color=c, edgecolor=T["BG"], label=lbl)
    ax2.legend(fontsize=8, labelcolor=T["TEXT"], framealpha=0.3)
    style_ax(ax2, "Stress Level by Injury Status", "Stress Score", "Count")

def inj_train(df, fig, canvas, controls):
    ax = fig.add_subplot(1,2,1)
    sc = ax.scatter(df['Training_Hours_Per_Week'],
                    df['Injury_Next_Season'] + np.random.normal(0, 0.04, len(df)),
                    alpha=0.3, s=12,
                    c=df['Injury_Next_Season'].map({0:T["A4"], 1:T["A2"]}),
                    edgecolors='none')
    style_ax(ax, "Training Hours vs Injury", "Hrs / Week", "Injured (jittered)")
    add_scatter_tooltips(fig, canvas, ax, sc,
        [f"{h:.1f} hrs/wk  {'Injured' if inj else 'Healthy'}"
         for h, inj in zip(df['Training_Hours_Per_Week'], df['Injury_Next_Season'])])

    ax2 = fig.add_subplot(1,2,2)
    prev = df.groupby('Previous_Injury_Count')['Injury_Next_Season'].mean()
    bars = ax2.bar(prev.index.astype(str), prev.values,
                   color=T["A3"], edgecolor=T["BG"], width=0.6)
    add_bar_tooltips(fig, canvas, ax2, bars, prev.values,
                     fmt="{:.1%} injury rate")
    style_ax(ax2, "Injury Rate by Prior Injury Count",
             "Previous Injuries", "Injury Rate")

def inj_radar(df, fig, canvas, controls):
    from sklearn.preprocessing import MinMaxScaler
    stats_cols = ['Age','Height_cm','Weight_kg','Training_Hours_Per_Week',
                  'Matches_Played_Past_Season','Previous_Injury_Count',
                  'Knee_Strength_Score','Hamstring_Flexibility','Reaction_Time_ms',
                  'Balance_Test_Score','Sprint_Speed_10m_s','Agility_Score',
                  'Sleep_Hours_Per_Night','Stress_Level_Score',
                  'Nutrition_Quality_Score','Warmup_Routine_Adherence','BMI']
    sc = [c for c in stats_cols if c in df.columns]
    scaled = pd.DataFrame(
        MinMaxScaler().fit_transform(df[sc]), columns=sc)
    mi  = scaled[df['Injury_Next_Season'].values == 1].mean()
    mni = scaled[df['Injury_Next_Season'].values == 0].mean()
    N      = len(sc)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    ax = fig.add_subplot(1,1,1, polar=True)
    ax.set_facecolor(T["CARD"])
    for means, color, label in [(mi,T["A2"],'Injured'),(mni,T["A4"],'Not Injured')]:
        vals = means.values.tolist() + [means.values[0]]
        ax.plot(angles, vals, color=color, lw=2.2)
        ax.fill(angles, vals, color=color, alpha=0.18)
    ax.set_thetagrids(np.degrees(angles[:-1]), sc, fontsize=6.5, color=T["SUBTEXT"])
    ax.tick_params(colors=T["SUBTEXT"], labelsize=5.5)
    ax.spines['polar'].set_color(T["BORDER"])
    ax.set_title("Player Profile: Injured vs Healthy",
                 color=T["TEXT"], fontsize=10, fontweight='bold', pad=18)
    p1 = mpatches.Patch(color=T["A2"], label='Injured')
    p2 = mpatches.Patch(color=T["A4"], label='Not Injured')
    ax.legend(handles=[p1,p2], loc='upper right',
              bbox_to_anchor=(1.28,1.12), labelcolor=T["TEXT"],
              fontsize=9, framealpha=0)

# ══════════════════════════════════════════════════════════════════════════════
# PLOT REGISTRY
# Each entry: (label, function, {control_name: [options|default]})
# controls dict values:
#   list  → OptionMenu (first item = default)
#   int   → Spinbox (value)
# ══════════════════════════════════════════════════════════════════════════════
PLOTS = {
    "transfer": [
        ("Value Distribution",      tv_dist,    {"Bins": 30}),
        ("Value by Position",       tv_pos,     {"Position": ["All"]}),   # filled at runtime
        ("Age & Peak Value",        tv_age,     {"Min Age": 16, "Max Age": 45}),
        ("Top Teams by Value",      tv_teams,   {"Top N Teams": 15}),
        ("Feature Correlations",    tv_corr,    {}),
        ("Goals & Assists vs Val",  tv_goals,   {}),
    ],
    "performance": [
        ("Goals & Assists Trend",   pf_trend,    {"League": ["All"]}),
        ("Stats by League",         pf_league,   {"Metric": ["Goals","Assists","xG","xAG"]}),
        ("Stats by Position",       pf_position, {"Season": ["All"]}),
        ("xG vs Actual Goals",      pf_xg,       {"Position": ["All"]}),
        ("Playing Time Analysis",   pf_time,     {}),
        ("Correlation Matrix",      pf_corr,     {}),
    ],
    "injury": [
        ("Class Balance",           inj_balance, {}),
        ("Age & BMI",               inj_age,     {}),
        ("Physical Attributes",     inj_phys,    {}),
        ("Feature Correlations",    inj_corr,    {}),
        ("Sleep & Stress",          inj_life,    {}),
        ("Training & History",      inj_train,   {}),
        ("Radar: Inj vs Healthy",   inj_radar,   {}),
    ],
}

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
class Dashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("⚽  Football Analytics Dashboard")
        self.configure(bg=T["BG"])

        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        W = min(1320, int(sw * 0.90))
        H = min(840,  int(sh * 0.88))
        self.geometry(f"{W}x{H}+{(sw-W)//2}+{(sh-H)//2}")
        self.minsize(960, 600)

        self.df             = {"transfer": None, "performance": None, "injury": None}
        self.current_fig    = None
        self.selected_plot  = None   # (ds_key, fn, spec_dict, name)
        self._all_btns      = []
        self._ctrl_vars     = {}     # name → tk variable
        self._ctrl_widgets  = []     # list of widgets to destroy on plot change
        self._current_theme = "🌑 Dark Ocean"

        self._build()
        self.after(120, self._auto_load)

    # ── Auto-load ─────────────────────────────────────────────────────────────
    def _auto_load(self):
        loaders = {"transfer": load_transfer,
                   "performance": load_performance,
                   "injury": load_injury}
        for key, loader in loaders.items():
            path = find_dataset(key)
            if path:
                try:
                    self.df[key] = loader(path)
                    self._set_status(key, True, os.path.basename(path), len(self.df[key]))
                except Exception as e:
                    self._set_status(key, False, f"Error: {e}", 0)
            else:
                self._set_status(key, False, "Not found — click Browse", 0)

    # ══════════════════════════════════════════════════════════════════════════
    # UI BUILD
    # ══════════════════════════════════════════════════════════════════════════
    def _build(self):
        self._build_header()
        body = tk.Frame(self, bg=T["BG"])
        body.pack(fill='both', expand=True)
        self._build_sidebar(body)
        self._build_main(body)

    # ── Header ────────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self, bg=T["PANEL"], height=56)
        hdr.pack(fill='x', side='top')
        hdr.pack_propagate(False)

        # Logo + title
        tk.Label(hdr, text="⚽", bg=T["PANEL"], fg=T["A1"],
                 font=("Segoe UI Emoji", 22)).pack(side='left', padx=(18,6), pady=8)
        tk.Label(hdr, text="Football Analytics Dashboard",
                 bg=T["PANEL"], fg=T["TEXT"],
                 font=("Segoe UI", 15, "bold")).pack(side='left', pady=8)
        tk.Label(hdr, text="  ·  Transfer · Performance · Injury",
                 bg=T["PANEL"], fg=T["SUBTEXT"],
                 font=("Segoe UI", 10)).pack(side='left')

        # Theme picker (right side)
        tk.Label(hdr, text="Theme:", bg=T["PANEL"], fg=T["SUBTEXT"],
                 font=("Segoe UI", 9)).pack(side='right', padx=(0,6))
        self._theme_var = tk.StringVar(value="🌑 Dark Ocean")
        theme_om = ttk.OptionMenu(hdr, self._theme_var,
                                  "🌑 Dark Ocean",
                                  *THEMES.keys(),
                                  command=self._change_theme)
        theme_om.pack(side='right', padx=(0,18), pady=12)
        self._style_optionmenu(theme_om)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    def _build_sidebar(self, body):
        outer = tk.Frame(body, bg=T["PANEL"], width=220)
        outer.pack(fill='y', side='left')
        outer.pack_propagate(False)

        sb_cv = tk.Canvas(outer, bg=T["PANEL"], highlightthickness=0, width=220)
        sb_sc = ttk.Scrollbar(outer, orient='vertical', command=sb_cv.yview)
        sb_cv.configure(yscrollcommand=sb_sc.set)
        sb_sc.pack(side='right', fill='y')
        sb_cv.pack(side='left', fill='both', expand=True)

        self.sb_inner = tk.Frame(sb_cv, bg=T["PANEL"])
        sb_cv.create_window((0,0), window=self.sb_inner, anchor='nw', width=215)
        self.sb_inner.bind("<Configure>",
            lambda e: sb_cv.configure(scrollregion=sb_cv.bbox("all")))
        sb_cv.bind_all("<MouseWheel>",
            lambda e: sb_cv.yview_scroll(int(-1*(e.delta/120)), "units"))

        self._fill_sidebar()

    def _fill_sidebar(self):
        p = self.sb_inner

        # ── Dataset cards ──
        self._sec_lbl(p, "📂  DATASETS")
        self.ds_labels = {}
        ds_defs = [
            ("transfer",    "💰", "Transfer Value",
             "transfer_value_prediction_dataset.csv"),
            ("performance", "📈", "Performance",
             "Top5_League_Players_2017to2024_dataset.csv"),
            ("injury",      "🩺", "Injury Risk",
             "data_injury.csv"),
        ]
        for key, icon, title, fname in ds_defs:
            card = tk.Frame(p, bg=T["CARD"], padx=8, pady=6,
                            relief='flat', bd=0)
            card.pack(fill='x', padx=10, pady=4)

            top_row = tk.Frame(card, bg=T["CARD"])
            top_row.pack(fill='x')
            tk.Label(top_row, text=icon, bg=T["CARD"], fg=T["A1"],
                     font=("Segoe UI Emoji",13)).pack(side='left')
            tk.Label(top_row, text=f" {title}", bg=T["CARD"], fg=T["TEXT"],
                     font=("Segoe UI",9,"bold")).pack(side='left')
            tk.Button(top_row, text="Browse", bg=T["BORDER"], fg=T["TEXT"],
                      font=("Segoe UI",8), relief='flat', cursor='hand2',
                      padx=6, pady=1,
                      command=lambda k=key: self._browse(k)).pack(side='right')

            lbl = tk.Label(card, text="Searching…", bg=T["CARD"],
                           fg=T["SUBTEXT"], font=("Segoe UI", 8),
                           wraplength=185, justify='left')
            lbl.pack(anchor='w', pady=(3,0))
            self.ds_labels[key] = lbl

        # Divider
        self._divider(p)

        # ── Visualization list ──
        self._sec_lbl(p, "📊  VISUALIZATIONS")
        groups = [
            ("💰 Transfer Value", "transfer"),
            ("📈 Performance",    "performance"),
            ("🩺 Injury Risk",    "injury"),
        ]
        self._all_btns.clear()
        for g_title, ds_key in groups:
            tk.Label(p, text=g_title, bg=T["PANEL"], fg=T["A3"],
                     font=("Segoe UI",9,"bold")).pack(anchor='w', padx=12, pady=(10,2))
            for name, fn, spec in PLOTS[ds_key]:
                btn = tk.Button(
                    p, text=f"   {name}", anchor='w',
                    bg=T["PANEL"], fg=T["TEXT"],
                    font=("Segoe UI",9), relief='flat', cursor='hand2',
                    padx=8, pady=5,
                    activebackground=T["CARD"], activeforeground=T["A1"],
                    command=lambda d=ds_key, f=fn, s=spec, n=name:
                        self._select(d, f, s, n))
                btn.pack(fill='x', padx=8, pady=1)
                self._all_btns.append((btn, name))

        self._divider(p)

        # Draw button
        self.draw_btn = tk.Button(
            p, text="▶   Draw Plot", command=self._draw,
            bg=T["A1"], fg=T["BG"], font=("Segoe UI",11,"bold"),
            relief='flat', cursor='hand2', padx=8, pady=10,
            activebackground=T["A3"], activeforeground=T["BG"])
        self.draw_btn.pack(fill='x', padx=12, pady=(0,12))

    # ── Main (plot area + control bar) ────────────────────────────────────────
    def _build_main(self, body):
        right = tk.Frame(body, bg=T["BG"])
        right.pack(fill='both', expand=True, side='left')

        # Control bar (top of right area)
        self.ctrl_bar = tk.Frame(right, bg=T["PANEL"], height=42)
        self.ctrl_bar.pack(fill='x', side='top')
        self.ctrl_bar.pack_propagate(False)
        tk.Label(self.ctrl_bar, text="  Select a chart and click  ▶ Draw Plot",
                 bg=T["PANEL"], fg=T["SUBTEXT"],
                 font=("Segoe UI",9)).pack(side='left', padx=10)

        # Plot canvas area
        self.plot_frame = tk.Frame(right, bg=T["BG"])
        self.plot_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Status bar (bottom)
        bot = tk.Frame(right, bg=T["PANEL"], height=34)
        bot.pack(fill='x', side='bottom')
        bot.pack_propagate(False)
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(bot, textvariable=self.status_var, bg=T["PANEL"],
                 fg=T["SUBTEXT"], font=("Segoe UI",8)).pack(side='left', padx=12)
        tk.Button(bot, text="💾  Save Figure", command=self._save,
                  bg=T["CARD"], fg=T["TEXT"], font=("Segoe UI",8),
                  relief='flat', cursor='hand2',
                  padx=10, pady=3).pack(side='right', padx=10, pady=4)

        self._show_placeholder()

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _sec_lbl(self, parent, text):
        tk.Label(parent, text=text, bg=T["PANEL"], fg=T["SUBTEXT"],
                 font=("Segoe UI",8,"bold")).pack(anchor='w', padx=12, pady=(12,3))

    def _divider(self, parent):
        tk.Frame(parent, bg=T["BORDER"], height=1).pack(fill='x', padx=10, pady=8)

    def _style_optionmenu(self, om):
        om.config(direction='below')

    def _set_status(self, key, ok, fname, rows):
        txt = (f"✓  {fname}\n     {rows:,} rows loaded") if ok else fname
        self.ds_labels[key].config(text=txt, fg=T["A4"] if ok else T["A2"])

    # ── Browse ────────────────────────────────────────────────────────────────
    def _browse(self, key):
        loaders = {"transfer": load_transfer,
                   "performance": load_performance,
                   "injury": load_injury}
        path = filedialog.askopenfilename(
            filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if path:
            try:
                self.df[key] = loaders[key](path)
                self._set_status(key, True, os.path.basename(path), len(self.df[key]))
            except Exception as e:
                messagebox.showerror("Load Error", str(e))

    # ── Selection + control bar ───────────────────────────────────────────────
    def _select(self, ds_key, fn, spec, name):
        self.selected_plot = (ds_key, fn, spec, name)
        for btn, bname in self._all_btns:
            if bname == name:
                btn.config(bg=T["BTN_HL"], fg=T["A1"],
                           font=("Segoe UI",9,"bold"))
            else:
                btn.config(bg=T["PANEL"], fg=T["TEXT"],
                           font=("Segoe UI",9,"normal"))
        self._build_controls(ds_key, spec)
        self.status_var.set(f"Selected: {name}  —  click ▶ Draw Plot")

    def _build_controls(self, ds_key, spec):
        # Clear old controls
        for w in self._ctrl_widgets:
            try: w.destroy()
            except: pass
        self._ctrl_widgets.clear()
        self._ctrl_vars.clear()

        if not spec:
            lbl = tk.Label(self.ctrl_bar,
                           text="  No filters for this chart",
                           bg=T["PANEL"], fg=T["SUBTEXT"],
                           font=("Segoe UI",8,"italic"))
            lbl.pack(side='left', padx=12)
            self._ctrl_widgets.append(lbl)
            return

        lbl0 = tk.Label(self.ctrl_bar, text="  Filters:",
                        bg=T["PANEL"], fg=T["A3"],
                        font=("Segoe UI",9,"bold"))
        lbl0.pack(side='left', padx=(10,4))
        self._ctrl_widgets.append(lbl0)

        for ctrl_name, default in spec.items():
            lbl = tk.Label(self.ctrl_bar, text=ctrl_name+":",
                           bg=T["PANEL"], fg=T["TEXT"],
                           font=("Segoe UI",8))
            lbl.pack(side='left', padx=(10,2))
            self._ctrl_widgets.append(lbl)

            if isinstance(default, list):
                # OptionMenu — populate with dataset values at draw time
                var = tk.StringVar(value=default[0])
                self._ctrl_vars[ctrl_name] = var
                # Build options from dataset if available
                opts = self._get_opts(ds_key, ctrl_name, default)
                var.set(opts[0])
                om = ttk.OptionMenu(self.ctrl_bar, var, opts[0], *opts)
                om.pack(side='left')
                self._ctrl_widgets.append(om)

            elif isinstance(default, int):
                var = tk.IntVar(value=default)
                self._ctrl_vars[ctrl_name] = var
                sb = tk.Spinbox(self.ctrl_bar, from_=1, to=999,
                                textvariable=var, width=5,
                                bg=T["CARD"], fg=T["TEXT"],
                                buttonbackground=T["BORDER"],
                                font=("Segoe UI",9), relief='flat')
                sb.pack(side='left')
                self._ctrl_widgets.append(sb)

        # Redraw button in control bar
        rd = tk.Button(self.ctrl_bar, text="↻ Apply",
                       command=self._draw,
                       bg=T["A3"], fg=T["BG"],
                       font=("Segoe UI",8,"bold"),
                       relief='flat', cursor='hand2', padx=8, pady=3)
        rd.pack(side='left', padx=(12,0))
        self._ctrl_widgets.append(rd)

    def _get_opts(self, ds_key, ctrl_name, default):
        """Populate OptionMenu from actual dataset values."""
        df = self.df.get(ds_key)
        if df is None:
            return default if default else ["All"]
        col_map = {
            "Position": "position",
            "League":   "league",
            "Season":   "season",
        }
        col = col_map.get(ctrl_name)
        if col and col in df.columns:
            vals = sorted(df[col].dropna().unique().astype(str).tolist())
            return ["All"] + vals
        return default if default else ["All"]

    def _get_controls(self):
        out = {}
        for name, var in self._ctrl_vars.items():
            try: out[name] = var.get()
            except: pass
        return out

    # ── Draw ──────────────────────────────────────────────────────────────────
    def _show_placeholder(self):
        for w in self.plot_frame.winfo_children(): w.destroy()
        f = tk.Frame(self.plot_frame, bg=T["BG"])
        f.pack(fill='both', expand=True)
        tk.Label(f, text="⚽", bg=T["BG"], fg=T["BORDER"],
                 font=("Segoe UI Emoji",60)).pack(expand=True)
        tk.Label(f, text="Pick a chart from the sidebar, then click  ▶ Draw Plot",
                 bg=T["BG"], fg=T["SUBTEXT"],
                 font=("Segoe UI",12)).pack()
        tk.Label(f, text="Hover over chart elements to see details",
                 bg=T["BG"], fg=T["BORDER"],
                 font=("Segoe UI",9)).pack(pady=4)

    def _draw(self):
        if not self.selected_plot:
            messagebox.showinfo("Hint",
                "Click a chart name in the sidebar first, then ▶ Draw Plot.")
            return
        ds_key, fn, spec, name = self.selected_plot
        if self.df[ds_key] is None:
            messagebox.showwarning("Dataset Not Loaded",
                f"Please load the {ds_key} dataset first.\n"
                f"Expected file: {DATASET_NAMES[ds_key]}")
            return

        for w in self.plot_frame.winfo_children(): w.destroy()
        if self.current_fig: plt.close(self.current_fig)

        self.update_idletasks()
        aw = max(self.plot_frame.winfo_width(),  650)
        ah = max(self.plot_frame.winfo_height(), 420)
        fw = aw / 96
        fh = (ah - 42) / 96

        fig = Figure(facecolor=T["PANEL"])
        fig.set_size_inches(fw, fh)
        fig.subplots_adjust(hspace=0.50, wspace=0.36,
                            left=0.10, right=0.97,
                            top=0.91, bottom=0.13)
        self.current_fig = fig

        # Embed canvas BEFORE calling plot fn (so tooltips can connect)
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.get_tk_widget().pack(fill='both', expand=True)

        # Toolbar
        tb_frame = tk.Frame(self.plot_frame, bg=T["PANEL"], height=36)
        tb_frame.pack(fill='x', side='bottom')
        tb_frame.pack_propagate(False)
        tb = NavigationToolbar2Tk(canvas, tb_frame)
        tb.config(bg=T["PANEL"])
        for child in tb.winfo_children():
            try: child.config(bg=T["PANEL"], fg=T["TEXT"],
                              highlightbackground=T["PANEL"])
            except: pass
        tb.update()

        controls = self._get_controls()
        try:
            fn(self.df[ds_key], fig, canvas, controls)
        except Exception as e:
            messagebox.showerror("Plot Error",
                f"Could not render '{name}':\n\n{e}")
            self._show_placeholder()
            return

        canvas.draw()
        self.status_var.set(
            f"Showing: {name}  │  {len(self.df[ds_key]):,} rows  │  "
            f"Hover over bars/points for details")

    # ── Theme switch ──────────────────────────────────────────────────────────
    def _change_theme(self, name):
        apply_theme(name)
        self.configure(bg=T["BG"])
        # Rebuild sidebar and header (quickest full refresh)
        for w in self.winfo_children(): w.destroy()
        self._all_btns.clear()
        self._ctrl_widgets.clear()
        self._ctrl_vars.clear()
        self._build()
        # Reload dataset statuses
        for key in ("transfer","performance","injury"):
            df = self.df[key]
            if df is not None:
                self._set_status(key, True, DATASET_NAMES[key], len(df))
            else:
                self._set_status(key, False, "Not loaded", 0)

    # ── Save ──────────────────────────────────────────────────────────────────
    def _save(self):
        if not self.current_fig:
            messagebox.showinfo("Nothing to save", "Draw a plot first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("PDF","*.pdf"),("SVG","*.svg")])
        if path:
            self.current_fig.savefig(path, dpi=180,
                                     bbox_inches='tight', facecolor=T["PANEL"])
            messagebox.showinfo("Saved ✓", f"Figure saved to:\n{path}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = Dashboard()
    app.mainloop()