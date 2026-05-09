"""
report_generator.py — Generate styled PDF scouting reports using ReportLab.
"""

import os
from datetime import datetime
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate
from reportlab.lib.pagesizes import letter

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False


# ── Colours ──────────────────────────────────────────────────────
C_DARK   = colors.HexColor("#0A0E1A")
C_PANEL  = colors.HexColor("#141C30")
C_CYAN   = colors.HexColor("#00D4FF")
C_GREEN  = colors.HexColor("#00FF88")
C_GOLD   = colors.HexColor("#FFD700")
C_RED    = colors.HexColor("#FF3B5C")
C_ORANGE = colors.HexColor("#FF8C00")
C_WHITE  = colors.white
C_GREY   = colors.HexColor("#8892A4")
C_BORDER = colors.HexColor("#1E2A45")


def _get_styles():
    base = getSampleStyleSheet()
    styles = {}
    styles["title"] = ParagraphStyle(
        "title", fontSize=22, fontName="Helvetica-Bold",
        textColor=C_WHITE, alignment=TA_CENTER, spaceAfter=4)
    styles["subtitle"] = ParagraphStyle(
        "subtitle", fontSize=11, fontName="Helvetica",
        textColor=C_CYAN, alignment=TA_CENTER, spaceAfter=18)
    styles["section"] = ParagraphStyle(
        "section", fontSize=13, fontName="Helvetica-Bold",
        textColor=C_CYAN, spaceBefore=14, spaceAfter=6)
    styles["body"] = ParagraphStyle(
        "body", fontSize=10, fontName="Helvetica",
        textColor=C_WHITE, spaceAfter=4, leading=16)
    styles["small"] = ParagraphStyle(
        "small", fontSize=8.5, fontName="Helvetica",
        textColor=C_GREY, alignment=TA_CENTER, spaceBefore=20)
    return styles


def _table_style_kv():
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_PANEL),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [C_PANEL, colors.HexColor("#0F1628")]),
        ("TEXTCOLOR", (0, 0), (0, -1), C_GREY),
        ("TEXTCOLOR", (1, 0), (1, -1), C_WHITE),
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",  (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",  (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("LINEBELOW",  (0, 0), (-1, -1), 0.5, C_BORDER),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
    ])


def _score_color(score: float) -> colors.HexColor:
    if score >= 8:
        return C_GREEN
    elif score >= 6.5:
        return colors.HexColor("#4FC3F7")
    elif score >= 5:
        return C_ORANGE
    return C_RED


def generate_pdf_report(player_info: dict, predictions: dict, output_path: str) -> bool:
    """
    Generate a FIFA-themed PDF scouting report.

    Args:
        player_info: dict with name, age, position, current_team, nationality
        predictions: dict with goals_pred, assists_pred, injury_prob, market_value
        output_path: where to save the .pdf

    Returns:
        True on success, False on failure.
    """
    if not REPORTLAB_OK:
        return False

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    styles = _get_styles()
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    story = []

    # ── Header banner ────────────────────────────────────────────
    header_data = [
        [Paragraph("⚽  PLAYER SCOUTING REPORT", styles["title"])],
        [Paragraph(
            f"{player_info.get('name','Unknown')}  •  "
            f"{player_info.get('position','—')}  •  "
            f"{player_info.get('current_team','—')}  •  "
            f"Generated: {datetime.today().strftime('%Y-%m-%d')}",
            styles["subtitle"]
        )],
    ]
    header_tbl = Table(header_data, colWidths=[17*cm])
    header_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_DARK),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#1A78C2")),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 0.4*cm))

    # ── 1. Player Overview ───────────────────────────────────────
    story.append(Paragraph("1. PLAYER OVERVIEW", styles["section"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_CYAN))
    story.append(Spacer(1, 0.2*cm))
    kv = [
        ["Full Name",    player_info.get("name", "—")],
        ["Position",     player_info.get("position", "—")],
        ["Age",          f"{player_info.get('age', '—')} years"],
        ["Nationality",  player_info.get("nationality", "—")],
        ["Current Club", player_info.get("current_team", "—")],
    ]
    t = Table([[Paragraph(k, styles["body"]), Paragraph(v, styles["body"])]
               for k, v in kv], colWidths=[6*cm, 11*cm])
    t.setStyle(_table_style_kv())
    story.append(t)
    story.append(Spacer(1, 0.4*cm))

    # ── 2. Performance Predictions ───────────────────────────────
    goals   = predictions.get("goals_pred", 0)
    assists = predictions.get("assists_pred", 0)
    inj     = predictions.get("injury_prob", 0.3)
    mval    = predictions.get("market_value", 0)

    story.append(Paragraph("2. NEXT-SEASON PERFORMANCE PREDICTIONS", styles["section"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_CYAN))
    story.append(Spacer(1, 0.2*cm))
    kv2 = [
        ["Predicted Goals",    f"{goals:.1f}"],
        ["Predicted Assists",  f"{assists:.1f}"],
        ["Predicted G + A",    f"{goals + assists:.1f}"],
    ]
    t2 = Table([[Paragraph(k, styles["body"]), Paragraph(v, styles["body"])]
                for k, v in kv2], colWidths=[6*cm, 11*cm])
    t2.setStyle(_table_style_kv())
    story.append(t2)
    story.append(Spacer(1, 0.4*cm))

    # ── 3. Injury Risk ───────────────────────────────────────────
    story.append(Paragraph("3. INJURY RISK ASSESSMENT", styles["section"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_CYAN))
    story.append(Spacer(1, 0.2*cm))
    risk_label = ("Low Risk" if inj < 0.25 else
                  "Moderate Risk" if inj < 0.5 else
                  "High Risk" if inj < 0.75 else "Very High Risk")
    kv3 = [
        ["Injury Probability", f"{inj*100:.1f}%"],
        ["Risk Category",      risk_label],
    ]
    t3 = Table([[Paragraph(k, styles["body"]), Paragraph(v, styles["body"])]
                for k, v in kv3], colWidths=[6*cm, 11*cm])
    t3.setStyle(_table_style_kv())
    story.append(t3)
    story.append(Spacer(1, 0.4*cm))

    # ── 4. Market Value ──────────────────────────────────────────
    story.append(Paragraph("4. TRANSFER MARKET VALUE", styles["section"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_CYAN))
    story.append(Spacer(1, 0.2*cm))
    kv4 = [["Predicted Market Value", f"€{mval:,.0f}"]]
    t4 = Table([[Paragraph(k, styles["body"]), Paragraph(v, styles["body"])]
                for k, v in kv4], colWidths=[6*cm, 11*cm])
    t4.setStyle(_table_style_kv())
    story.append(t4)
    story.append(Spacer(1, 0.4*cm))

    # ── 5. Score Summary ─────────────────────────────────────────
    story.append(Paragraph("5. SCORE SUMMARY", styles["section"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_CYAN))
    story.append(Spacer(1, 0.2*cm))

    def goal_score(g):
        if g >= 18: return 9.5, "Elite"
        if g >= 14: return 8.0, "Excellent"
        if g >= 10: return 6.5, "Good"
        if g >= 6:  return 5.0, "Average"
        return 3.0, "Below Avg"

    def assist_score(a):
        if a >= 10: return 9.5, "Elite"
        if a >= 7:  return 8.0, "Excellent"
        if a >= 4:  return 6.5, "Good"
        if a >= 2:  return 5.0, "Average"
        return 3.0, "Below Avg"

    def inj_score(p):
        if p < 0.2: return 9.0, "Low Risk"
        if p < 0.4: return 7.0, "Moderate"
        if p < 0.6: return 5.0, "High Risk"
        return 2.5, "Very High"

    def mval_score(v):
        if v >= 60e6: return 9.5, "World Class"
        if v >= 30e6: return 8.0, "High Value"
        if v >= 15e6: return 6.5, "Good Value"
        if v >= 5e6:  return 5.0, "Moderate"
        return 3.0, "Low Value"

    gs, gl = goal_score(goals)
    as_, al = assist_score(assists)
    is_, il = inj_score(inj)
    ms, ml = mval_score(mval)
    overall = (gs + as_ + is_ + ms) / 4

    categories = [
        ("Goals Output", gs, gl),
        ("Assists Output", as_, al),
        ("Injury Safety", is_, il),
        ("Market Value", ms, ml),
        ("OVERALL", overall, "RECOMMENDED" if overall >= 6.5 else "NOT RECOMMENDED"),
    ]

    tbl_data = [["Category", "Score / 10", "Verdict"]]
    for cat, sc, lbl in categories:
        tbl_data.append([cat, f"{sc:.1f}", lbl])

    ts = Table(tbl_data, colWidths=[7*cm, 5*cm, 5*cm])
    clr_map = []
    for ri, (cat, sc, _) in enumerate(categories, start=1):
        c = _score_color(sc)
        clr_map.append(("BACKGROUND", (2, ri), (2, ri), c))
        clr_map.append(("TEXTCOLOR", (2, ri), (2, ri), C_WHITE))
    ts.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), C_DARK),
        ("TEXTCOLOR",   (0, 0), (-1, 0), C_CYAN),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2),
            [colors.HexColor("#0F1628"), C_PANEL]),
        ("BACKGROUND",  (0, -1), (-1, -1), C_DARK),
        ("TEXTCOLOR",   (0, -1), (-1, -1), C_WHITE),
        ("FONTNAME",    (0, -1), (-1, -1), "Helvetica-Bold"),
        ("TOPPADDING",    (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("LINEBELOW",   (0, 0), (-1, -1), 0.5, C_BORDER),
        *clr_map,
    ]))
    story.append(ts)
    story.append(Spacer(1, 0.5*cm))

    # ── Footer ───────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER))
    story.append(Paragraph(
        "CONFIDENTIAL — For internal scouting use only. "
        "Predictions generated by ML models. "
        "Results should be combined with human assessment.",
        styles["small"]))

    doc.build(story)
    return True
