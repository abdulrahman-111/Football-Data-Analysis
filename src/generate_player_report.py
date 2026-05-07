"""
generate_player_report.py
─────────────────────────
Loads pre-trained models, runs inference from a JSON input file,
and produces a professional Word (.docx) scouting report.

Usage:
    python generate_player_report.py --input player_input.json \
                                     --output player_report.docx \
                                     --models_dir models/
"""

import argparse
import json
import os
import sys
import numpy as np
import joblib
from datetime import datetime
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
import pandas as pd
from docx.oxml import OxmlElement


# ─────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────
DARK_BLUE  = RGBColor(0x0D, 0x2B, 0x4E)   # header background
ACCENT     = RGBColor(0x1A, 0x78, 0xC2)   # section titles
GREEN      = RGBColor(0x27, 0xAE, 0x60)
ORANGE     = RGBColor(0xE6, 0x7E, 0x22)
RED_CLR    = RGBColor(0xC0, 0x39, 0x2B)
LIGHT_GREY = "F2F4F7"
WHITE_HEX  = "FFFFFF"
DARK_HEX   = "0D2B4E"
ACCENT_HEX = "1A78C2"


# ─────────────────────────────────────────────────────────────────
# XML HELPERS
# ─────────────────────────────────────────────────────────────────
def set_cell_bg(cell, hex_color):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement('w:shd')
    shd.set(qn('w:val'),   'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'),  hex_color)
    tcPr.append(shd)

def set_cell_border(cell, **kwargs):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    for edge in ('top','left','bottom','right','insideH','insideV'):
        if edge in kwargs:
            tag = OxmlElement(f'w:{edge}')
            for k,v in kwargs[edge].items():
                tag.set(qn(f'w:{k}'), str(v))
            tcBorders.append(tag)
    tcPr.append(tcBorders)

def set_para_border_bottom(para, color="1A78C2", size=12):
    pPr  = para._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bot  = OxmlElement('w:bottom')
    bot.set(qn('w:val'),   'single')
    bot.set(qn('w:sz'),    str(size))
    bot.set(qn('w:space'), '1')
    bot.set(qn('w:color'), color)
    pBdr.append(bot)
    pPr.append(pBdr)


# ─────────────────────────────────────────────────────────────────
# SCORING LOGIC  (heuristic thresholds – replace with your own)
# ─────────────────────────────────────────────────────────────────
def score_goals(predicted_goals):
    """Return (score 0-10, label, colour_hex)"""
    if predicted_goals >= 18:   return 9.5, "Elite",       "1A9641"
    if predicted_goals >= 14:   return 8.0, "Excellent",   "27AE60"
    if predicted_goals >= 10:   return 6.5, "Good",        "F0A500"
    if predicted_goals >= 6:    return 5.0, "Average",     "E67E22"
    return                             3.0, "Below Avg",   "C0392B"

def score_assists(predicted_assists):
    if predicted_assists >= 10: return 9.5, "Elite",       "1A9641"
    if predicted_assists >= 7:  return 8.0, "Excellent",   "27AE60"
    if predicted_assists >= 4:  return 6.5, "Good",        "F0A500"
    if predicted_assists >= 2:  return 5.0, "Average",     "E67E22"
    return                             3.0, "Below Avg",   "C0392B"

def score_injury(injury_prob):
    """injury_prob = probability of injury (0-1)"""
    risk_pct = injury_prob * 100
    if risk_pct < 20:  return 9.0, "Low Risk",      "1A9641"
    if risk_pct < 40:  return 7.0, "Moderate Risk", "F0A500"
    if risk_pct < 60:  return 5.0, "High Risk",     "E67E22"
    return                    2.5, "Very High Risk", "C0392B"

def score_market_value(predicted_value):
    if predicted_value >= 60e6:  return 9.5, "World Class",  "1A9641"
    if predicted_value >= 30e6:  return 8.0, "High Value",   "27AE60"
    if predicted_value >= 15e6:  return 6.5, "Good Value",   "F0A500"
    if predicted_value >= 5e6:   return 5.0, "Moderate",     "E67E22"
    return                             3.0, "Low Value",     "C0392B"

def overall_recommendation(scores):
    avg = np.mean(scores)
    if avg >= 8.0: return "STRONGLY RECOMMENDED", "1A9641"
    if avg >= 6.5: return "RECOMMENDED",          "27AE60"
    if avg >= 5.0: return "CONDITIONAL",          "F0A500"
    return               "NOT RECOMMENDED",       "C0392B"


# ─────────────────────────────────────────────────────────────────
# MODEL INFERENCE
# ─────────────────────────────────────────────────────────────────
GOALS_FEATURES = [
    'Expected_xG','Expected_npxG','Standard_Gls','Performance_Gls',
    'Standard_SoT','Expected_npxG+xAG','Touches_Att Pen','Performance_G-PK',
    'Performance_G+A','Standard_Sh','Per 90 Minutes_xG',
    'Per 90 Minutes_xG+xAG','Per 90 Minutes_npxG+xAG','Per 90 Minutes_npxG',
    'assists_next','Per 90 Minutes_G+A','Per 90 Minutes_Gls',
    'Standard_SoT/90','Performance_Off','Standard_Sh/90','SCA Types_Sh',
    'Per 90 Minutes_G+A-PK','Carries_CPA','Progression_PrgR'
]

ASSISTS_FEATURES = [
    'Expected_xA','xAG_','Expected_xAG','Touches_Att 3rd','SCA_SCA',
    'goals_next','KP_','GCA_GCA','PPA_','SCA Types_PassLive',
    'Expected_npxG+xAG','Progression_PrgR','Receiving_PrgR',
    'GCA Types_PassLive','Performance_Ast','Ast_','Performance_G+A',
    'Progression_PrgC','Carries_PrgC','SCA_SCA90','Carries_1/3',
    'Carries_CPA','Touches_Att Pen','Standard_Sh'
]

INJURY_FEATURES = [
    'Age','Height_cm','Weight_kg','Training_Hours_Per_Week',
    'Matches_Played_Past_Season','Previous_Injury_Count','Knee_Strength_Score',
    'Hamstring_Flexibility','Reaction_Time_ms','Balance_Test_Score',
    'Sprint_Speed_10m_s','Agility_Score','Sleep_Hours_Per_Night',
    'Stress_Level_Score','Nutrition_Quality_Score','Warmup_Routine_Adherence','BMI'
]

TRANSFER_FEATURES = [
    'age','appearance','Goals_per90_min','Assists_per90_min',
    'Goal Conceded_per90_min','minutes played','days_injured','games_injured',
    'award','highest_value','position_encoded','team_encoded'
]


def build_input_array(feature_names, data_dict):
    row = []
    for f in feature_names:
        val = data_dict.get(f)
        if val is None:
            raise KeyError(f"Missing feature '{f}' in JSON input.")
        row.append(float(val))
    return pd.DataFrame([row], columns=feature_names)


def run_inference(data, models_dir):
    """
    Returns dict with keys:
        goals_pred, assists_pred, injury_prob, market_value
    Falls back to heuristic estimates when model files are absent
    (so the report generator works without trained .pkl files).
    """
    results = {}

    def load(fname):
        p = os.path.join(models_dir, fname)
        if os.path.exists(p):
            return joblib.load(p)
        return None

    # ── Goals ──────────────────────────────────────────────
    model_g = load("performance_prediction_goals_next_model.pkl")
    if model_g:
        X = build_input_array(GOALS_FEATURES, data["goals_model_features"])
        results["goals_pred"] = int(model_g.predict(X)[0])
    else:
        # heuristic fallback
        results["goals_pred"] = data["goals_model_features"].get("Performance_Gls", 10) * 1.05
        results["goals_source"] = "heuristic"

    # ── Assists ─────────────────────────────────────────────
    model_a = load("performance_prediction_assists_next_model.pkl")
    if model_a:
        X = build_input_array(ASSISTS_FEATURES, data["assists_model_features"])
        results["assists_pred"] = int(model_a.predict(X)[0])
    else:
        results["assists_pred"] = data["assists_model_features"].get("Performance_Ast", 5) * 1.05
        results["assists_source"] = "heuristic"

    # ── Injury ──────────────────────────────────────────────
    model_i = load("Injury_classifier_model.pkl")
    if model_i:
        X = build_input_array(INJURY_FEATURES, data["injury_model_features"])
        if hasattr(model_i, "predict_proba"):
            results["injury_prob"] = float(model_i.predict_proba(X)[0][1])
        else:
            results["injury_prob"] = float(model_i.predict(X)[0])
    else:
        prev = data["injury_model_features"].get("Previous_Injury_Count", 1)
        results["injury_prob"] = min(0.15 + prev * 0.08, 0.9)
        results["injury_source"] = "heuristic"

    # ── Transfer Value ──────────────────────────────────────
    model_t = load("transfer_value_prediction_model.pkl")
    if model_t:
        X = build_input_array(TRANSFER_FEATURES, data["transfer_model_features"])
        results["market_value"] = float(model_t.predict(X)[0])
    else:
        results["market_value"] = data["transfer_model_features"].get("highest_value", 20e6) * 0.9
        results["market_value_source"] = "heuristic"

    return results


# ─────────────────────────────────────────────────────────────────
# DOCUMENT BUILDER
# ─────────────────────────────────────────────────────────────────
def add_run(para, text, bold=False, italic=False, size=None, color=None, font=None):
    run = para.add_run(text)
    run.bold   = bold
    run.italic = italic
    if size:  run.font.size = Pt(size)
    if color: run.font.color.rgb = color
    if font:  run.font.name = font
    return run

def section_heading(doc, title):
    para = doc.add_paragraph()
    para.paragraph_format.space_before = Pt(14)
    para.paragraph_format.space_after  = Pt(4)
    set_para_border_bottom(para, color=ACCENT_HEX, size=12)
    add_run(para, title.upper(), bold=True, size=13, color=ACCENT, font="Arial")
    return para

def score_bar_table(doc, label, score, max_score=10, color_hex="1A78C2"):
    """Renders a label + filled bar + numeric score."""
    tbl = doc.add_table(rows=1, cols=3)
    tbl.alignment = WD_TABLE_ALIGNMENT.LEFT

    # col widths: label | bar | value
    widths = [Cm(4.5), Cm(9.0), Cm(2.0)]
    for i, cell in enumerate(tbl.rows[0].cells):
        cell.width = widths[i]

    # Label
    c0 = tbl.rows[0].cells[0]
    p0 = c0.paragraphs[0]
    p0.alignment = WD_ALIGN_PARAGRAPH.LEFT
    add_run(p0, label, bold=True, size=10, font="Arial")

    # Bar
    c1 = tbl.rows[0].cells[1]
    bar_tbl = c1.add_table(rows=1, cols=2)
    filled = int(round(score / max_score * 20))   # 20 segments
    empty  = 20 - filled
    bc_fill = bar_tbl.rows[0].cells[0]
    bc_empty= bar_tbl.rows[0].cells[1]
    bc_fill.width  = Cm(filled * 0.44)
    bc_empty.width = Cm(empty  * 0.44)
    set_cell_bg(bc_fill,  color_hex)
    set_cell_bg(bc_empty, "E0E0E0")
    bc_fill.paragraphs[0].add_run(" ")
    bc_empty.paragraphs[0].add_run(" ")

    # Numeric score
    c2 = tbl.rows[0].cells[2]
    p2 = c2.paragraphs[0]
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_run(p2, f"{score:.1f}/10", bold=True, size=10, font="Arial")

    doc.add_paragraph()   # spacer


def kv_table(doc, rows_data, col_widths=(Cm(6), Cm(9.5))):
    """Two-column key-value table."""
    border = {'val':'single','sz':'4','space':'0','color':'D0D0D0'}
    tbl = doc.add_table(rows=len(rows_data), cols=2)
    for i, (key, val) in enumerate(rows_data):
        row = tbl.rows[i]
        row.cells[0].width = col_widths[0]
        row.cells[1].width = col_widths[1]

        bg = LIGHT_GREY if i % 2 == 0 else WHITE_HEX
        for ci, cell in enumerate(row.cells):
            set_cell_bg(cell, bg)
            set_cell_border(cell,
                top={'val':'single','sz':'4','space':'0','color':'D5D5D5'},
                bottom={'val':'single','sz':'4','space':'0','color':'D5D5D5'},
                left={'val':'single','sz':'4','space':'0','color':'D5D5D5'},
                right={'val':'single','sz':'4','space':'0','color':'D5D5D5'})
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after  = Pt(3)

        p0 = tbl.rows[i].cells[0].paragraphs[0]
        p1 = tbl.rows[i].cells[1].paragraphs[0]
        add_run(p0, str(key), bold=True,  size=9.5, font="Arial")
        add_run(p1, str(val), bold=False, size=9.5, font="Arial")

    doc.add_paragraph()


def build_report(data, predictions, output_path):
    doc = Document()

    # ── Page setup ─────────────────────────────────────────
    section = doc.sections[0]
    section.page_width  = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = section.bottom_margin = Inches(0.8)
    section.left_margin = section.right_margin = Inches(0.9)

    info = data["player_info"]
    preds = predictions

    # ─────────────────────────────────────────────────────
    # HEADER BANNER
    # ─────────────────────────────────────────────────────
    banner = doc.add_table(rows=2, cols=1)
    banner.alignment = WD_TABLE_ALIGNMENT.CENTER

    r0 = banner.rows[0].cells[0]
    set_cell_bg(r0, DARK_HEX)
    p = r0.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after  = Pt(2)
    add_run(p, "⚽  PLAYER SCOUTING REPORT", bold=True, size=22,
            color=RGBColor(0xFF,0xFF,0xFF), font="Arial")

    r1 = banner.rows[1].cells[0]
    set_cell_bg(r1, ACCENT_HEX)
    p2 = r1.paragraphs[0]
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.paragraph_format.space_before = Pt(4)
    p2.paragraph_format.space_after  = Pt(6)
    add_run(p2,
            f"{info['name']}  |  {info['position']}  |  {info['current_team']}  |  "
            f"Generated: {info.get('report_date', datetime.today().strftime('%Y-%m-%d'))}",
            bold=False, size=11, color=RGBColor(0xFF,0xFF,0xFF), font="Arial")

    doc.add_paragraph()

    # ─────────────────────────────────────────────────────
    # 1. PLAYER OVERVIEW
    # ─────────────────────────────────────────────────────
    section_heading(doc, "1. Player Overview")
    kv_table(doc, [
        ("Full Name",        info["name"]),
        ("Position",         info["position"]),
        ("Age",              f"{info['age']} years"),
        ("Nationality",      info.get("nationality", "N/A")),
        ("Current Club",     info["current_team"]),
        ("Report Date",      info.get("report_date", datetime.today().strftime("%Y-%m-%d"))),
    ])

    # ─────────────────────────────────────────────────────
    # 2. PREDICTED PERFORMANCE
    # ─────────────────────────────────────────────────────
    section_heading(doc, "2. Next-Season Performance Predictions")

    goals_val   = preds["goals_pred"]
    assists_val  = preds["assists_pred"]

    kv_table(doc, [
        ("Predicted Goals (next season)",   f"{goals_val:.1f}"),
        ("Predicted Assists (next season)",  f"{assists_val:.1f}"),
        ("Predicted G+A",                    f"{goals_val + assists_val:.1f}"),
        ("Prediction Method",                "ML Model (Random Forest / LightGBM)"),
    ])

    # Goals score
    g_score, g_label, g_color = score_goals(goals_val)
    a_score, a_label, a_color = score_assists(assists_val)

    score_bar_table(doc, f"Goals Score  ({g_label})",   g_score, color_hex=g_color)
    score_bar_table(doc, f"Assists Score  ({a_label})", a_score, color_hex=a_color)

    # ─────────────────────────────────────────────────────
    # 3. INJURY RISK
    # ─────────────────────────────────────────────────────
    section_heading(doc, "3. Injury Risk Assessment")

    injury_prob = preds["injury_prob"]
    i_score, i_label, i_color = score_injury(injury_prob)
    inj = data["injury_model_features"]

    kv_table(doc, [
        ("Injury Probability",        f"{injury_prob*100:.1f}%"),
        ("Risk Category",             i_label),
        ("Previous Injuries",         str(int(inj.get("Previous_Injury_Count", 0)))),
        ("Matches Played Last Season",str(int(inj.get("Matches_Played_Past_Season", 0)))),
        ("Knee Strength Score",       f"{inj.get('Knee_Strength_Score', 'N/A')}/100"),
        ("Sleep (hrs/night)",         str(inj.get("Sleep_Hours_Per_Night", "N/A"))),
        ("Nutrition Quality",         f"{inj.get('Nutrition_Quality_Score', 'N/A')}/100"),
    ])

    score_bar_table(doc, f"Injury Safety Score  ({i_label})", i_score, color_hex=i_color)

    # ─────────────────────────────────────────────────────
    # 4. MARKET VALUE
    # ─────────────────────────────────────────────────────
    section_heading(doc, "4. Transfer Market Value")

    mval = preds["market_value"]
    mv_score, mv_label, mv_color = score_market_value(mval)
    tf = data["transfer_model_features"]

    kv_table(doc, [
        ("Predicted Market Value",  f"€{mval:,.0f}"),
        ("Value Tier",              mv_label),
        ("Appearances Last Season", str(int(tf.get("appearance", 0)))),
        ("Minutes Played",          f"{int(tf.get('minutes played', 0)):,}"),
        ("Days Injured",            str(int(tf.get("days_injured", 0)))),
        ("Awards Won",              str(int(tf.get("award", 0)))),
        ("Historical Peak Value",   f"€{tf.get('highest_value', 0):,.0f}"),
    ])

    score_bar_table(doc, f"Market Value Score  ({mv_label})", mv_score, color_hex=mv_color)

    # ─────────────────────────────────────────────────────
    # 5. SCORE SUMMARY TABLE
    # ─────────────────────────────────────────────────────
    section_heading(doc, "5. Score Summary")

    categories = [
        ("Goals Output",    g_score,  g_label,  g_color),
        ("Assists Output",  a_score,  a_label,  a_color),
        ("Injury Safety",  i_score,  i_label,  i_color),
        ("Market Value",   mv_score, mv_label, mv_color),
    ]
    all_scores = [c[1] for c in categories]
    overall_score = np.mean(all_scores)
    rec_label, rec_color = overall_recommendation(all_scores)

    # header row
    hdr_tbl = doc.add_table(rows=1+len(categories)+1, cols=3)
    hdr_tbl.alignment = WD_TABLE_ALIGNMENT.LEFT
    col_w = [Cm(6), Cm(5.5), Cm(4)]

    def fill_cell(cell, text, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT,
                  bg=None, fcolor=RGBColor(0,0,0), size=10):
        if bg: set_cell_bg(cell, bg)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = align
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after  = Pt(4)
        add_run(p, text, bold=bold, size=size, color=fcolor, font="Arial")

    # header
    for ci, (txt, w) in enumerate(zip(["Category","Score / 10","Verdict"], col_w)):
        hdr_tbl.rows[0].cells[ci].width = w
        fill_cell(hdr_tbl.rows[0].cells[ci], txt, bold=True,
                  bg=DARK_HEX, fcolor=RGBColor(255,255,255),
                  align=WD_ALIGN_PARAGRAPH.CENTER)

    for ri, (cat, sc, lbl, col) in enumerate(categories, start=1):
        row = hdr_tbl.rows[ri]
        row.cells[0].width = col_w[0]
        row.cells[1].width = col_w[1]
        row.cells[2].width = col_w[2]
        bg = LIGHT_GREY if ri % 2 == 0 else WHITE_HEX
        fill_cell(row.cells[0], cat,           bg=bg)
        fill_cell(row.cells[1], f"{sc:.1f}",   bg=bg, align=WD_ALIGN_PARAGRAPH.CENTER)
        fill_cell(row.cells[2], lbl,            bg=col,
                  fcolor=RGBColor(255,255,255), align=WD_ALIGN_PARAGRAPH.CENTER)

    # overall row
    ov_row = hdr_tbl.rows[-1]
    for ci, (txt, w) in enumerate(zip(
            [f"OVERALL SCORE",f"{overall_score:.2f} / 10", rec_label], col_w)):
        ov_row.cells[ci].width = w
        fill_cell(ov_row.cells[ci], txt, bold=True, bg=rec_color,
                  fcolor=RGBColor(255,255,255), align=WD_ALIGN_PARAGRAPH.CENTER, size=11)

    doc.add_paragraph()

    # ─────────────────────────────────────────────────────
    # 6. RECOMMENDATION
    # ─────────────────────────────────────────────────────
    section_heading(doc, "6. Final Recommendation")

    rec_para = doc.add_paragraph()
    rec_para.paragraph_format.space_before = Pt(6)
    rec_para.paragraph_format.space_after  = Pt(6)

    def verdict_text(score):
        if score >= 8.0:
            return (f"{info['name']} is an exceptional talent who fits the profile of a top signing. "
                    f"With a predicted {goals_val:.0f} goals and {assists_val:.0f} assists next season, "
                    f"low injury risk ({injury_prob*100:.0f}%), and a market value of €{mval:,.0f}, "
                    f"this player represents outstanding value. We STRONGLY RECOMMEND pursuing this transfer.")
        if score >= 6.5:
            return (f"{info['name']} is a solid signing candidate. Predicted output of "
                    f"{goals_val:.0f} goals and {assists_val:.0f} assists next season, with manageable "
                    f"injury risk ({injury_prob*100:.0f}%). Market value estimated at €{mval:,.0f}. "
                    f"We RECOMMEND pursuing negotiations.")
        if score >= 5.0:
            return (f"{info['name']} has potential but comes with some concerns. "
                    f"Conditional recommendation — address injury history and verify fit "
                    f"with tactical requirements before proceeding.")
        return (f"{info['name']} does not meet the required thresholds across the assessed "
                f"categories. Transfer is NOT RECOMMENDED at this stage.")

    add_run(rec_para, verdict_text(overall_score), size=10.5, font="Arial")

    # ─────────────────────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────────────────────
    footer_para = doc.add_paragraph()
    footer_para.paragraph_format.space_before = Pt(20)
    set_para_border_bottom(footer_para, color="BBBBBB", size=6)

    disc = doc.add_paragraph()
    disc.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_run(disc,
            "CONFIDENTIAL — For internal scouting use only.  "
            "Predictions generated by ML models trained on historical data.  "
            "Results should be combined with human assessment.",
            italic=True, size=8, color=RGBColor(0x80,0x80,0x80), font="Arial")

    doc.save(output_path)
    print(f"\n✅  Report saved → {output_path}\n")


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────
def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(BASE_DIR, "data", "processed","player_input.json")
    parser = argparse.ArgumentParser(description="Generate player scouting report")
    parser.add_argument("--input",      default=input_path,
                        help="Path to JSON input file")
    output_path = os.path.join(BASE_DIR, "outputs","player_report.docx")
    parser.add_argument("--output",     default=output_path,
                        help="Output .docx file path")
    model_path = os.path.join(BASE_DIR, "models")
    parser.add_argument("--models_dir", default=model_path,
                        help="Directory containing .pkl model files")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"❌  Input file not found: {args.input}")

    with open(args.input, "r") as f:
        data = json.load(f)

    print(f"📄  Loaded input: {args.input}")
    print(f"🔍  Running model inference ...")

    predictions = run_inference(data, args.models_dir)

    print(f"   Goals predicted  : {predictions['goals_pred']}")
    print(f"   Assists predicted: {predictions['assists_pred']}")
    print(f"   Injury prob      : {predictions['injury_prob']*100:.1f}%")
    print(f"   Market value     : €{predictions['market_value']:,.0f}")

    print(f"📝  Building report ...")
    build_report(data, predictions, args.output)


if __name__ == "__main__":
    main()
