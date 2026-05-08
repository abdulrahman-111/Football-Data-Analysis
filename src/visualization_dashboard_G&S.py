"""
Football Analytics Visualization Dashboard
Local GUI — tkinter + matplotlib
Auto-loads: data_injury.csv | Top5_League_Players_2017to2024_dataset.csv | transfer_value_prediction_dataset.csv
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
import os, warnings
warnings.filterwarnings("ignore")

# ── Auto-search for datasets near this script ──────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_NAMES = {
    "injury":       "data_injury.csv",
    "performance":  "Top5_League_Players_2017to2024_dataset.csv",
    "transfer":     "transfer_value_prediction_dataset.csv",
}

def find_dataset(name):
    search_name = DATASET_NAMES[name]
    d = SCRIPT_DIR
    dirs = [d]
    for _ in range(4):
        d = os.path.dirname(d)
        dirs.append(d)
    for root_dir in dirs:
        for dirpath, _, files in os.walk(root_dir):
            if search_name in files:
                return os.path.join(dirpath, search_name)
    return None

# ── Palette ────────────────────────────────────────────────────────────────────
BG      = "#0b0e1a"
PANEL   = "#131626"
CARD    = "#1a1f35"
BORDER  = "#252c48"
ACCENT1 = "#00d4ff"
ACCENT2 = "#ff5f40"
ACCENT3 = "#9d7fea"
ACCENT4 = "#2ecc8f"
TEXT    = "#dde3f0"
SUBTEXT = "#7b88a8"
PALETTE = [ACCENT1, ACCENT2, ACCENT3, ACCENT4, "#f472b6", "#fbbf24", "#60a5fa", "#a3e635"]

sns.set_theme(style="dark", rc={
    "axes.facecolor":   CARD,
    "figure.facecolor": PANEL,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  TEXT,
    "xtick.color":      SUBTEXT,
    "ytick.color":      SUBTEXT,
    "text.color":       TEXT,
    "grid.color":       BORDER,
    "grid.linestyle":   "--",
    "grid.alpha":       0.35,
})

# ── Data loaders ───────────────────────────────────────────────────────────────
def load_transfer(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=['player'])
    for col in ['Goals','Assists','Yellow Cards','Second Yellow Card','Red Card','Goal Conceded','Clean Sheets']:
        if col in df.columns and 'minutes played' in df.columns:
            df[col] = (df[col] * 90) / df['minutes played'].replace(0, np.nan)
            df.rename(columns={col: f"{col}_per90"}, inplace=True)
    if 'team' in df.columns and 'current_value' in df.columns:
        team_mean = df.groupby('team')['current_value'].mean()
        df['team_encoded'] = df['team'].map(team_mean)
    return df

def load_performance(path):
    df = pd.read_csv(path, sep=';', decimal=',')
    df = df.sort_values(['player','season'])
    df['goals_next']   = df.groupby('player')['Performance_Gls'].shift(-1)
    df['assists_next'] = df.groupby('player')['Performance_Ast'].shift(-1)
    df['goals_trend']  = df['Performance_Gls'] - df.groupby('player')['Performance_Gls'].shift(1)
    df = df.dropna(subset=['goals_next','assists_next','goals_trend'])
    pos_map = {'GK':1,'DF':2,'MF':3,'FW':4,'DF,FW':4,'FW,MF':4,'MF,FW':4,'FW,DF':4,'MF,DF':3,'DF,MF':3}
    df['position_encoded'] = df['pos_'].map(pos_map)
    return df

def load_injury(path):
    df = pd.read_csv(path).drop_duplicates()
    for col in df.columns:
        if df[col].dtype != 'object' and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())
    return df

# ── Plot helpers ───────────────────────────────────────────────────────────────
def style_ax(ax, title="", xlabel="", ylabel="", xrot=0):
    ax.set_facecolor(CARD)
    ax.set_title(title, color=TEXT, fontsize=9, fontweight='bold', pad=6)
    ax.set_xlabel(xlabel, color=SUBTEXT, fontsize=8)
    ax.set_ylabel(ylabel, color=SUBTEXT, fontsize=8)
    ax.tick_params(colors=SUBTEXT, labelsize=7)
    if xrot:
        ax.tick_params(axis='x', rotation=xrot)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, linestyle='--', alpha=0.3)

# ══════════════════════════════════════════════════════════════════════════════
# TRANSFER VALUE PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def tv_dist(df, fig):
    ax = fig.add_subplot(1,2,1)
    v = df['current_value'].dropna()
    ax.hist(v, bins=40, color=ACCENT1, edgecolor=BG, alpha=0.85)
    ax.axvline(v.mean(), color=ACCENT2, lw=1.5, ls='--', label=f'Mean {v.mean():.1f}')
    ax.legend(fontsize=7, labelcolor=TEXT)
    style_ax(ax, "Transfer Value Distribution", "Value (M€)", "Count")
    ax2 = fig.add_subplot(1,2,2)
    ax2.hist(np.log1p(v), bins=40, color=ACCENT3, edgecolor=BG, alpha=0.85)
    style_ax(ax2, "Log-scale Distribution", "log(Value+1)", "Count")

def tv_pos(df, fig):
    if 'position' not in df.columns:
        return
    ax = fig.add_subplot(1,2,1)
    pm = df.groupby('position')['current_value'].median().sort_values(ascending=False)
    bars = ax.bar(pm.index, pm.values, color=PALETTE[:len(pm)], edgecolor=BG, linewidth=0.4)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2,
                f'{b.get_height():.1f}', ha='center', va='bottom', fontsize=6, color=SUBTEXT)
    style_ax(ax, "Median Value by Position", "Position", "M€", xrot=30)
    ax2 = fig.add_subplot(1,2,2)
    positions = df['position'].dropna().unique()
    data_bp = [df[df['position']==p]['current_value'].dropna().values for p in positions]
    bp = ax2.boxplot(data_bp, patch_artist=True, labels=positions)
    for patch, c in zip(bp['boxes'], PALETTE):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    for el in ['whiskers','caps','medians','fliers']:
        for item in bp[el]: item.set_color(SUBTEXT)
    style_ax(ax2, "Box Plot by Position", "Position", "Value (M€)", xrot=30)

def tv_age(df, fig):
    ax = fig.add_subplot(1,2,1)
    ax.scatter(df['age'], df['current_value'], alpha=0.35, s=10, color=ACCENT1, edgecolors='none')
    d = df[['age','current_value']].dropna()
    if len(d) > 2:
        z = np.polyfit(d['age'], d['current_value'], 2)
        xs = np.linspace(d['age'].min(), d['age'].max(), 200)
        ax.plot(xs, np.poly1d(z)(xs), color=ACCENT2, lw=2)
    style_ax(ax, "Age vs Transfer Value", "Age", "Value (M€)")
    ax2 = fig.add_subplot(1,2,2)
    ax2.scatter(df['highest_value'], df['current_value'], alpha=0.35, s=10, color=ACCENT3, edgecolors='none')
    style_ax(ax2, "Highest Value vs Current", "Highest (M€)", "Current (M€)")

def tv_teams(df, fig):
    ax = fig.add_subplot(1,1,1)
    top = df.groupby('team')['current_value'].sum().nlargest(15).sort_values()
    colors = [ACCENT1 if v == top.max() else ACCENT3 for v in top.values]
    bars = ax.barh(top.index, top.values, color=colors, edgecolor=BG, height=0.6)
    for b in bars:
        ax.text(b.get_width()+0.3, b.get_y()+b.get_height()/2,
                f'{b.get_width():.0f}M', va='center', fontsize=7, color=SUBTEXT)
    style_ax(ax, "Top 15 Teams by Total Squad Value", "Total Value (M€)", "")

def tv_corr(df, fig):
    num = df.select_dtypes(include=np.number)
    if 'current_value' not in num.columns:
        return
    corr = num.corr()['current_value'].drop('current_value').sort_values()
    ax = fig.add_subplot(1,1,1)
    ax.barh(corr.index, corr.values,
            color=[ACCENT4 if v > 0 else ACCENT2 for v in corr.values],
            edgecolor=BG, height=0.6)
    ax.axvline(0, color=SUBTEXT, lw=0.8)
    style_ax(ax, "Feature Correlation with Current Value", "Pearson r", "")

def tv_goals(df, fig):
    ax = fig.add_subplot(1,2,1)
    gc = next((c for c in df.columns if 'Goals_per90' in c or 'Goals_per' in c), None)
    if gc:
        ax.scatter(df[gc], df['current_value'], alpha=0.35, s=10, color=ACCENT4, edgecolors='none')
        style_ax(ax, "Goals/90 vs Transfer Value", "Goals per 90", "Value (M€)")
    ax2 = fig.add_subplot(1,2,2)
    ac = next((c for c in df.columns if 'Assists_per90' in c or 'Assists_per' in c), None)
    if ac:
        ax2.scatter(df[ac], df['current_value'], alpha=0.35, s=10, color=ACCENT2, edgecolors='none')
        style_ax(ax2, "Assists/90 vs Transfer Value", "Assists per 90", "Value (M€)")

# ══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def pf_trend(df, fig):
    ax = fig.add_subplot(1,2,1)
    sg = df.groupby('season')['Performance_Gls'].mean()
    ax.plot(range(len(sg)), sg.values, marker='o', color=ACCENT1, lw=2, markersize=5)
    ax.fill_between(range(len(sg)), sg.values, alpha=0.12, color=ACCENT1)
    ax.set_xticks(range(len(sg))); ax.set_xticklabels(sg.index.astype(str), rotation=35, fontsize=6)
    style_ax(ax, "Avg Goals per Season", "Season", "Goals")
    ax2 = fig.add_subplot(1,2,2)
    sa = df.groupby('season')['Performance_Ast'].mean()
    ax2.plot(range(len(sa)), sa.values, marker='s', color=ACCENT2, lw=2, markersize=5)
    ax2.fill_between(range(len(sa)), sa.values, alpha=0.12, color=ACCENT2)
    ax2.set_xticks(range(len(sa))); ax2.set_xticklabels(sa.index.astype(str), rotation=35, fontsize=6)
    style_ax(ax2, "Avg Assists per Season", "Season", "Assists")

def pf_league(df, fig):
    ax = fig.add_subplot(1,2,1)
    lg = df.groupby('league')['Performance_Gls'].mean().sort_values(ascending=False)
    ax.bar(lg.index, lg.values, color=PALETTE[:len(lg)], edgecolor=BG)
    style_ax(ax, "Avg Goals by League", "League", "Goals", xrot=25)
    ax2 = fig.add_subplot(1,2,2)
    xg = df.groupby('league')['Expected_xG'].mean().sort_values(ascending=False)
    ax2.bar(xg.index, xg.values, color=PALETTE[:len(xg)], edgecolor=BG)
    style_ax(ax2, "Avg xG by League", "League", "xG", xrot=25)

def pf_position(df, fig):
    ax = fig.add_subplot(1,2,1)
    pg = df.groupby('pos_')['Performance_Gls'].mean().sort_values(ascending=False).head(8)
    ax.bar(pg.index, pg.values, color=ACCENT3, edgecolor=BG, alpha=0.85)
    style_ax(ax, "Avg Goals by Position", "Position", "Goals", xrot=30)
    ax2 = fig.add_subplot(1,2,2)
    pa = df.groupby('pos_')['Performance_Ast'].mean().sort_values(ascending=False).head(8)
    ax2.bar(pa.index, pa.values, color=ACCENT4, edgecolor=BG, alpha=0.85)
    style_ax(ax2, "Avg Assists by Position", "Position", "Assists", xrot=30)

def pf_xg(df, fig):
    sample = df.sample(min(1500, len(df)), random_state=42)
    ax = fig.add_subplot(1,2,1)
    ax.scatter(sample['Expected_xG'], sample['Performance_Gls'],
               alpha=0.25, s=8, color=ACCENT1, edgecolors='none')
    mn = min(sample['Expected_xG'].min(), sample['Performance_Gls'].min())
    mx = max(sample['Expected_xG'].max(), sample['Performance_Gls'].max())
    ax.plot([mn,mx],[mn,mx], color=ACCENT2, lw=1.5, ls='--', label='xG=Goals')
    ax.legend(fontsize=7, labelcolor=TEXT)
    style_ax(ax, "xG vs Actual Goals", "xG", "Goals")
    ax2 = fig.add_subplot(1,2,2)
    ax2.scatter(sample['Expected_xAG'], sample['Performance_Ast'],
                alpha=0.25, s=8, color=ACCENT3, edgecolors='none')
    mn2 = min(sample['Expected_xAG'].min(), sample['Performance_Ast'].min())
    mx2 = max(sample['Expected_xAG'].max(), sample['Performance_Ast'].max())
    ax2.plot([mn2,mx2],[mn2,mx2], color=ACCENT2, lw=1.5, ls='--')
    style_ax(ax2, "xAG vs Actual Assists", "xAG", "Assists")

def pf_time(df, fig):
    ax = fig.add_subplot(1,2,1)
    ax.scatter(df['Playing Time_Min'], df['Performance_Gls'],
               alpha=0.2, s=8, color=ACCENT4, edgecolors='none')
    style_ax(ax, "Minutes Played vs Goals", "Minutes", "Goals")
    ax2 = fig.add_subplot(1,2,2)
    if 'Per 90 Minutes_Gls' in df.columns:
        v = df['Per 90 Minutes_Gls'].dropna()
        v = v[v < v.quantile(0.99)]
        ax2.hist(v, bins=40, color=ACCENT2, edgecolor=BG, alpha=0.85)
        style_ax(ax2, "Goals per 90 Distribution", "Goals/90", "Count")

def pf_corr(df, fig):
    cols = ['Performance_Gls','Performance_Ast','Expected_xG','Expected_xAG',
            'Playing Time_Min','Playing Time_90s','Standard_Sh','Standard_SoT',
            'Progression_PrgC','Progression_PrgP','goals_next','assists_next']
    avail = [c for c in cols if c in df.columns]
    ax = fig.add_subplot(1,1,1)
    corr = df[avail].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, ax=ax, annot=True, fmt='.2f', cmap='coolwarm',
                annot_kws={'size':6}, linewidths=0.3, linecolor=BG,
                cbar_kws={'shrink':0.7})
    style_ax(ax, "Correlation Matrix — Performance Features", "", "")

# ══════════════════════════════════════════════════════════════════════════════
# INJURY PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def inj_balance(df, fig):
    ax = fig.add_subplot(1,2,1)
    counts = df['Injury_Next_Season'].value_counts()
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=['No Injury','Injury'],
        autopct='%1.1f%%', colors=[ACCENT4,ACCENT2],
        startangle=90, pctdistance=0.75,
        wedgeprops={'edgecolor':BG,'linewidth':1.5})
    for t in texts: t.set_color(TEXT)
    for at in autotexts: at.set_color(BG); at.set_fontweight('bold')
    ax.set_facecolor(CARD)
    ax.set_title("Class Distribution", color=TEXT, fontsize=9, fontweight='bold')
    ax2 = fig.add_subplot(1,2,2)
    ax2.bar(['No Injury','Injury'], counts.values,
            color=[ACCENT4,ACCENT2], edgecolor=BG, width=0.45)
    for i,v in enumerate(counts.values):
        ax2.text(i, v+3, str(v), ha='center', fontsize=9, color=TEXT, fontweight='bold')
    style_ax(ax2, "Injury Count", "Class", "Count")

def inj_age(df, fig):
    ax = fig.add_subplot(1,2,1)
    for cls,c,lbl in [(0,ACCENT4,'No Injury'),(1,ACCENT2,'Injury')]:
        ax.hist(df[df['Injury_Next_Season']==cls]['Age'].dropna(),
                bins=25, alpha=0.65, color=c, edgecolor=BG, label=lbl)
    ax.legend(fontsize=7, labelcolor=TEXT)
    style_ax(ax, "Age Distribution by Injury Status", "Age", "Count")
    ax2 = fig.add_subplot(1,2,2)
    for cls,c,lbl in [(0,ACCENT4,'No Injury'),(1,ACCENT2,'Injury')]:
        ax2.hist(df[df['Injury_Next_Season']==cls]['BMI'].dropna(),
                 bins=25, alpha=0.65, color=c, edgecolor=BG, label=lbl)
    ax2.legend(fontsize=7, labelcolor=TEXT)
    style_ax(ax2, "BMI Distribution by Injury Status", "BMI", "Count")

def inj_phys(df, fig):
    cols = ['Knee_Strength_Score','Hamstring_Flexibility','Balance_Test_Score',
            'Agility_Score','Sprint_Speed_10m_s','Reaction_Time_ms']
    cols = [c for c in cols if c in df.columns]
    for i, col in enumerate(cols):
        ax = fig.add_subplot(2, 3, i+1)
        d0 = df[df['Injury_Next_Season']==0][col].dropna()
        d1 = df[df['Injury_Next_Season']==1][col].dropna()
        bp = ax.boxplot([d0,d1], patch_artist=True, labels=['No Inj','Inj'])
        bp['boxes'][0].set_facecolor(ACCENT4); bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor(ACCENT2); bp['boxes'][1].set_alpha(0.7)
        for el in ['whiskers','caps','medians','fliers']:
            for item in bp[el]: item.set_color(SUBTEXT)
        style_ax(ax, col.replace('_',' '), "", "")

def inj_corr(df, fig):
    # Keep only numeric columns — drops 'Position' (string) and any other non-numeric
    num_df = df.select_dtypes(include=np.number)
    if 'Injury_Next_Season' not in num_df.columns:
        ax = fig.add_subplot(1,1,1)
        ax.text(0.5, 0.5, "'Injury_Next_Season' not found in numeric columns",
                ha='center', va='center', color=TEXT)
        return
    corr = num_df.corr()['Injury_Next_Season'].drop('Injury_Next_Season').sort_values()
    ax = fig.add_subplot(1,1,1)
    ax.barh(corr.index, corr.values,
            color=[ACCENT2 if v > 0 else ACCENT4 for v in corr.values],
            edgecolor=BG, height=0.6)
    ax.axvline(0, color=SUBTEXT, lw=0.8)
    style_ax(ax, "Feature Correlation with Injury Next Season", "Pearson r", "")

def inj_life(df, fig):
    ax = fig.add_subplot(1,2,1)
    for cls,c,lbl in [(0,ACCENT4,'No Injury'),(1,ACCENT2,'Injury')]:
        ax.hist(df[df['Injury_Next_Season']==cls]['Sleep_Hours_Per_Night'].dropna(),
                bins=20, alpha=0.65, color=c, edgecolor=BG, label=lbl)
    ax.legend(fontsize=7, labelcolor=TEXT)
    style_ax(ax, "Sleep Hours by Injury Status", "Sleep Hrs/Night", "Count")
    ax2 = fig.add_subplot(1,2,2)
    for cls,c,lbl in [(0,ACCENT4,'No Injury'),(1,ACCENT2,'Injury')]:
        ax2.hist(df[df['Injury_Next_Season']==cls]['Stress_Level_Score'].dropna(),
                 bins=20, alpha=0.65, color=c, edgecolor=BG, label=lbl)
    ax2.legend(fontsize=7, labelcolor=TEXT)
    style_ax(ax2, "Stress Level by Injury Status", "Stress Score", "Count")

def inj_train(df, fig):
    ax = fig.add_subplot(1,2,1)
    ax.scatter(df['Training_Hours_Per_Week'],
               df['Injury_Next_Season'] + np.random.normal(0, 0.04, len(df)),
               alpha=0.25, s=10,
               c=df['Injury_Next_Season'].map({0:ACCENT4, 1:ACCENT2}),
               edgecolors='none')
    style_ax(ax, "Training Hours vs Injury", "Hrs/Week", "Injured (jittered)")
    ax2 = fig.add_subplot(1,2,2)
    prev = df.groupby('Previous_Injury_Count')['Injury_Next_Season'].mean()
    ax2.bar(prev.index, prev.values, color=ACCENT3, edgecolor=BG, width=0.6)
    style_ax(ax2, "Injury Rate by Prior Injury Count", "Previous Injuries", "Injury Rate")

def inj_radar(df, fig):
    from sklearn.preprocessing import MinMaxScaler
    stats_cols = ['Age','Height_cm','Weight_kg','Training_Hours_Per_Week',
                  'Matches_Played_Past_Season','Previous_Injury_Count',
                  'Knee_Strength_Score','Hamstring_Flexibility','Reaction_Time_ms',
                  'Balance_Test_Score','Sprint_Speed_10m_s','Agility_Score',
                  'Sleep_Hours_Per_Night','Stress_Level_Score','Nutrition_Quality_Score',
                  'Warmup_Routine_Adherence','BMI']
    sc = [c for c in stats_cols if c in df.columns]
    scaler = MinMaxScaler()
    scaled = pd.DataFrame(scaler.fit_transform(df[sc]), columns=sc)
    mi  = scaled[df['Injury_Next_Season'].values == 1].mean()
    mni = scaled[df['Injury_Next_Season'].values == 0].mean()
    N = len(sc)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    ax = fig.add_subplot(1,1,1, polar=True)
    ax.set_facecolor(CARD)
    for means, color, label in [(mi, ACCENT2, 'Injured'),(mni, ACCENT4, 'Not Injured')]:
        vals = means.values.tolist() + [means.values[0]]
        ax.plot(angles, vals, color=color, lw=2)
        ax.fill(angles, vals, color=color, alpha=0.18)
    ax.set_thetagrids(np.degrees(angles[:-1]), sc, fontsize=6, color=SUBTEXT)
    ax.tick_params(colors=SUBTEXT, labelsize=5)
    ax.spines['polar'].set_color(BORDER)
    ax.set_title("Player Profile: Injured vs Healthy", color=TEXT,
                 fontsize=9, fontweight='bold', pad=16)
    p1 = mpatches.Patch(color=ACCENT2, label='Injured')
    p2 = mpatches.Patch(color=ACCENT4, label='Not Injured')
    ax.legend(handles=[p1,p2], loc='upper right', bbox_to_anchor=(1.25,1.1),
              labelcolor=TEXT, fontsize=8, framealpha=0)

# ── Plot registry: (label, function, (rows_hint, fig_w, fig_h)) ───────────────
PLOTS = {
    "transfer": [
        ("Value Distribution",     tv_dist,     (2, 10, 4.2)),
        ("Value by Position",      tv_pos,      (2, 10, 4.2)),
        ("Age & Peak Value",       tv_age,      (2, 10, 4.2)),
        ("Top Teams by Value",     tv_teams,    (1,  9, 4.5)),
        ("Feature Correlations",   tv_corr,     (1,  8, 5.0)),
        ("Goals & Assists vs Val", tv_goals,    (2, 10, 4.2)),
    ],
    "performance": [
        ("Goals & Assists Trend",  pf_trend,    (2, 10, 4.2)),
        ("Stats by League",        pf_league,   (2, 10, 4.2)),
        ("Stats by Position",      pf_position, (2, 10, 4.2)),
        ("xG vs Actual Goals",     pf_xg,       (2, 10, 4.2)),
        ("Playing Time Analysis",  pf_time,     (2, 10, 4.2)),
        ("Correlation Matrix",     pf_corr,     (1,  9, 5.5)),
    ],
    "injury": [
        ("Class Balance",          inj_balance, (2, 10, 4.2)),
        ("Age & BMI",              inj_age,     (2, 10, 4.2)),
        ("Physical Attributes",    inj_phys,    (6, 10, 5.5)),
        ("Feature Correlations",   inj_corr,    (1,  8, 5.0)),
        ("Sleep & Stress",         inj_life,    (2, 10, 4.2)),
        ("Training & History",     inj_train,   (2, 10, 4.2)),
        ("Radar: Inj vs Healthy",  inj_radar,   (1,  7, 5.0)),
    ],
}

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
class Dashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("⚽  Football Analytics Dashboard")
        self.configure(bg=BG)

        # Responsive: 88% of screen, centred
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        W = min(1280, int(sw * 0.88))
        H = min(820,  int(sh * 0.88))
        self.geometry(f"{W}x{H}+{(sw-W)//2}+{(sh-H)//2}")
        self.minsize(900, 580)

        self.df = {"transfer": None, "performance": None, "injury": None}
        self.current_fig = None
        self.selected_plot = None   # (ds_key, fn, spec, name)
        self._all_btns = []         # for highlight reset

        self._build()
        self.after(100, self._auto_load)   # auto-load after window appears

    # ── Auto-load ─────────────────────────────────────────────────────────────
    def _auto_load(self):
        loaders = {
            "transfer":    load_transfer,
            "performance": load_performance,
            "injury":      load_injury,
        }
        for key, loader in loaders.items():
            path = find_dataset(key)
            if path:
                try:
                    self.df[key] = loader(path)
                    self._set_ds_status(key, True,
                                        os.path.basename(path), len(self.df[key]))
                except Exception as e:
                    self._set_ds_status(key, False, f"Error: {e}", 0)
            else:
                self._set_ds_status(key, False, "Not found — click Browse", 0)

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=PANEL, height=48)
        hdr.pack(fill='x', side='top')
        hdr.pack_propagate(False)
        tk.Label(hdr, text="⚽  Football Analytics Dashboard",
                 bg=PANEL, fg=ACCENT1, font=("Georgia",15,"bold")).pack(side='left', padx=18, pady=10)
        tk.Label(hdr, text="Transfer · Performance · Injury",
                 bg=PANEL, fg=SUBTEXT, font=("Courier",9)).pack(side='left')

        # Body
        body = tk.Frame(self, bg=BG)
        body.pack(fill='both', expand=True)

        # ── Sidebar ──
        sidebar_outer = tk.Frame(body, bg=PANEL, width=205)
        sidebar_outer.pack(fill='y', side='left')
        sidebar_outer.pack_propagate(False)

        # Scrollable inner
        sb_canvas = tk.Canvas(sidebar_outer, bg=PANEL, highlightthickness=0, width=205)
        sb_scroll  = ttk.Scrollbar(sidebar_outer, orient='vertical', command=sb_canvas.yview)
        sb_canvas.configure(yscrollcommand=sb_scroll.set)
        sb_scroll.pack(side='right', fill='y')
        sb_canvas.pack(side='left', fill='both', expand=True)
        self.sb_inner = tk.Frame(sb_canvas, bg=PANEL)
        sb_canvas.create_window((0,0), window=self.sb_inner, anchor='nw', width=200)
        self.sb_inner.bind("<Configure>",
            lambda e: sb_canvas.configure(scrollregion=sb_canvas.bbox("all")))
        # Mouse-wheel scroll
        def _on_scroll(e):
            sb_canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        sb_canvas.bind_all("<MouseWheel>", _on_scroll)

        # ── Plot area ──
        right = tk.Frame(body, bg=BG)
        right.pack(fill='both', expand=True, side='left')

        self.plot_frame = tk.Frame(right, bg=BG)
        self.plot_frame.pack(fill='both', expand=True, padx=4, pady=4)

        # Status bar
        bot = tk.Frame(right, bg=PANEL, height=32)
        bot.pack(fill='x', side='bottom')
        bot.pack_propagate(False)
        self.status_var = tk.StringVar(value="Select a visualization and click  ▶ Draw")
        tk.Label(bot, textvariable=self.status_var,
                 bg=PANEL, fg=SUBTEXT, font=("Courier",8)).pack(side='left', padx=10)
        tk.Button(bot, text="💾 Save PNG", command=self._save,
                  bg=CARD, fg=TEXT, font=("Courier",8), relief='flat',
                  cursor='hand2', padx=8, pady=2).pack(side='right', padx=8, pady=4)

        self._build_sidebar()
        self._show_placeholder()

    def _build_sidebar(self):
        p = self.sb_inner

        # ── Dataset cards ──
        self._sec(p, "📂 DATASETS")
        self.ds_labels = {}
        ds_defs = [
            ("transfer",    "💰 Transfer Value",
             "transfer_value_prediction_dataset.csv"),
            ("performance", "📈 Performance",
             "Top5_League_Players_2017to2024_dataset.csv"),
            ("injury",      "🩺 Injury Risk",
             "data_injury.csv"),
        ]
        for key, title, fname in ds_defs:
            card = tk.Frame(p, bg=CARD, padx=6, pady=4)
            card.pack(fill='x', padx=8, pady=3)
            tk.Label(card, text=title, bg=CARD, fg=TEXT,
                     font=("Courier",8,"bold")).pack(anchor='w')
            tk.Label(card, text=fname, bg=CARD, fg=SUBTEXT,
                     font=("Courier",6), wraplength=172, justify='left').pack(anchor='w')
            lbl = tk.Label(card, text="Searching…", bg=CARD, fg=SUBTEXT,
                           font=("Courier",7), wraplength=172, justify='left')
            lbl.pack(anchor='w', pady=(2,0))
            tk.Button(card, text="📂 Browse", bg=BORDER, fg=TEXT,
                      font=("Courier",7), relief='flat', cursor='hand2',
                      padx=4, pady=2,
                      command=lambda k=key: self._browse(k)).pack(anchor='e', pady=(3,0))
            self.ds_labels[key] = lbl

        ttk.Separator(p, orient='horizontal').pack(fill='x', padx=8, pady=8)

        # ── Visualization list ──
        self._sec(p, "📊 VISUALIZATIONS")

        groups = [
            ("💰 Transfer Value", "transfer"),
            ("📈 Performance",    "performance"),
            ("🩺 Injury Risk",    "injury"),
        ]
        for group_title, ds_key in groups:
            tk.Label(p, text=group_title, bg=PANEL, fg=ACCENT3,
                     font=("Courier",8,"bold")).pack(anchor='w', padx=10, pady=(8,1))
            for name, fn, spec in PLOTS[ds_key]:
                btn = tk.Button(
                    p, text=f"  {name}", anchor='w',
                    bg=PANEL, fg=TEXT, font=("Courier",8),
                    relief='flat', cursor='hand2', padx=6, pady=4,
                    activebackground=CARD, activeforeground=ACCENT1,
                    command=lambda d=ds_key, f=fn, s=spec, n=name:
                        self._select(d, f, s, n))
                btn.pack(fill='x', padx=6, pady=1)
                self._all_btns.append((btn, name))

        ttk.Separator(p, orient='horizontal').pack(fill='x', padx=8, pady=8)

        tk.Button(p, text="▶  Draw Plot", command=self._draw,
                  bg=ACCENT1, fg=BG, font=("Courier",10,"bold"),
                  relief='flat', cursor='hand2', padx=6, pady=8,
                  activebackground=ACCENT3).pack(fill='x', padx=10, pady=(0,10))

    def _sec(self, parent, text):
        tk.Label(parent, text=text, bg=PANEL, fg=SUBTEXT,
                 font=("Courier",8,"bold")).pack(anchor='w', padx=10, pady=(10,2))

    # ── Dataset status ─────────────────────────────────────────────────────────
    def _set_ds_status(self, key, ok, fname, rows):
        txt = f"✓ {fname}  ({rows:,} rows)" if ok else fname
        self.ds_labels[key].config(text=txt, fg=ACCENT4 if ok else ACCENT2)

    def _browse(self, key):
        loaders = {"transfer": load_transfer,
                   "performance": load_performance,
                   "injury": load_injury}
        path = filedialog.askopenfilename(filetypes=[("CSV","*.csv"),("All","*.*")])
        if path:
            try:
                self.df[key] = loaders[key](path)
                self._set_ds_status(key, True, os.path.basename(path), len(self.df[key]))
            except Exception as e:
                messagebox.showerror("Load Error", str(e))

    # ── Selection ─────────────────────────────────────────────────────────────
    def _select(self, ds_key, fn, spec, name):
        self.selected_plot = (ds_key, fn, spec, name)
        for btn, bname in self._all_btns:
            if bname == name:
                btn.config(bg=CARD, fg=ACCENT1)
            else:
                btn.config(bg=PANEL, fg=TEXT)
        self.status_var.set(f"Selected: {name}  —  click ▶ Draw Plot")

    # ── Draw ───────────────────────────────────────────────────────────────────
    def _show_placeholder(self):
        for w in self.plot_frame.winfo_children():
            w.destroy()
        f = tk.Frame(self.plot_frame, bg=BG)
        f.pack(fill='both', expand=True)
        tk.Label(f, text="⚽", bg=BG, fg=BORDER,
                 font=("Segoe UI Emoji",52)).pack(expand=True)
        tk.Label(f, text="Select a visualization → click  ▶ Draw Plot",
                 bg=BG, fg=SUBTEXT, font=("Courier",11)).pack()

    def _draw(self):
        if not self.selected_plot:
            messagebox.showinfo("Hint", "Click a visualization name first, then ▶ Draw Plot.")
            return
        ds_key, fn, spec, name = self.selected_plot
        if self.df[ds_key] is None:
            messagebox.showwarning("No Data",
                f"The {ds_key} dataset isn't loaded.\n"
                f"Place '{DATASET_NAMES[ds_key]}' near this script or use Browse.")
            return

        # Clear previous
        for w in self.plot_frame.winfo_children():
            w.destroy()
        if self.current_fig:
            plt.close(self.current_fig)

        # Measure available pixels
        self.update_idletasks()
        aw = max(self.plot_frame.winfo_width(),  600)
        ah = max(self.plot_frame.winfo_height(), 400)
        DPI = 96
        fw = aw / DPI
        fh = (ah - 40) / DPI      # reserve 40px for toolbar

        fig = Figure(facecolor=PANEL)
        fig.set_size_inches(fw, fh)
        fig.subplots_adjust(hspace=0.52, wspace=0.38,
                            left=0.10, right=0.97,
                            top=0.92, bottom=0.14)
        self.current_fig = fig

        try:
            fn(self.df[ds_key], fig)
        except Exception as e:
            messagebox.showerror("Plot Error", str(e))
            self._show_placeholder()
            return

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        tb_frame = tk.Frame(self.plot_frame, bg=PANEL, height=36)
        tb_frame.pack(fill='x', side='bottom')
        tb_frame.pack_propagate(False)
        tb = NavigationToolbar2Tk(canvas, tb_frame)
        tb.config(bg=PANEL)
        for child in tb.winfo_children():
            try: child.config(bg=PANEL, fg=TEXT, highlightbackground=PANEL)
            except: pass
        tb.update()

        self.status_var.set(f"Showing: {name}  |  {len(self.df[ds_key]):,} rows")

    def _save(self):
        if not self.current_fig:
            messagebox.showinfo("Nothing to save", "Draw a plot first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("PDF","*.pdf"),("SVG","*.svg")])
        if path:
            self.current_fig.savefig(path, dpi=180,
                                     bbox_inches='tight', facecolor=PANEL)
            messagebox.showinfo("Saved", f"Saved to:\n{path}")


if __name__ == "__main__":
    app = Dashboard()
    app.mainloop()