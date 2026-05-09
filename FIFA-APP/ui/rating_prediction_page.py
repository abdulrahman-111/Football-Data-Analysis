"""
rating_prediction_page.py — Player Rating Prediction & Recommendation Page.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QScrollArea, QGridLayout, QGroupBox,
    QDoubleSpinBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QSlider
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QColor

from models.rating_predictor import RatingPredictor, RATING_FEATURES
from utils import charts
from utils.theme import (
    ACCENT_CYAN, ACCENT_GREEN, ACCENT_GOLD, ACCENT_RED,
    ACCENT_ORANGE, TEXT_PRIMARY, TEXT_SECONDARY, BG_CARD,
    BG_PANEL, BORDER_COLOR, get_rating_color
)
from ui.widgets.widgets import (
    StatCard, SectionHeader, AttributeBar, PlayerCard
)

ATTR_GROUPS = {
    "⚡ Pace":        ["acceleration", "sprint_speed"],
    "🎯 Shooting":    ["finishing", "shot_power", "long_shots", "volleys", "positioning"],
    "🎲 Passing":     ["short_passing", "long_passing", "vision", "crossing", "curve", "fk_accuracy"],
    "🏃 Dribbling":   ["dribbling", "ball_control", "agility", "balance", "reactions"],
    "🛡️ Defending":   ["defensive_awareness", "standing_tackle", "sliding_tackle", "interceptions", "aggression"],
    "💪 Physical":    ["stamina", "strength", "jumping", "heading_accuracy"],
    "🧤 GK":          ["gk_diving", "gk_handling", "gk_kicking", "gk_positioning", "gk_reflexes"],
    "📋 General":     ["height_cm", "weight_kg", "weak_foot", "skill_moves", "preferred_foot_encoded", "penalties", "composure"],
}

DEFAULTS = {
    "height_cm": 180, "weight_kg": 75, "weak_foot": 3, "skill_moves": 3,
    "preferred_foot_encoded": 1, "penalties": 70, "composure": 72,
    "acceleration": 72, "sprint_speed": 72,
    "finishing": 70, "shot_power": 72, "long_shots": 65, "volleys": 65, "positioning": 70,
    "short_passing": 74, "long_passing": 68, "vision": 70, "crossing": 66,
    "curve": 65, "fk_accuracy": 60,
    "dribbling": 72, "ball_control": 74, "agility": 70, "balance": 68, "reactions": 72,
    "defensive_awareness": 58, "standing_tackle": 60, "sliding_tackle": 58,
    "interceptions": 55, "aggression": 62,
    "stamina": 72, "strength": 70, "jumping": 68, "heading_accuracy": 65,
    "gk_diving": 10, "gk_handling": 10, "gk_kicking": 10,
    "gk_positioning": 10, "gk_reflexes": 10,
}


class RatingWorker(QThread):
    done = Signal(dict)

    def __init__(self, predictor, attrs):
        super().__init__()
        self._p = predictor
        self._a = attrs

    def run(self):
        self.done.emit(self._p.predict_rating(self._a))


class RatingPredictionPage(QWidget):
    notify = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._predictor = RatingPredictor()
        self._spinners: dict[str, QDoubleSpinBox] = {}
        self._result: dict | None = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        self._main_lay = QVBoxLayout(container)
        self._main_lay.setContentsMargins(28, 24, 28, 24)
        self._main_lay.setSpacing(20)

        # ── Freeze repaints while building all children ────────────
        container.setUpdatesEnabled(False)
        try:
            self._main_lay.addWidget(SectionHeader("⭐  Player Rating Prediction & Recommendations"))

            body = QHBoxLayout()
            body.setSpacing(20)
            body.addWidget(self._input_panel(), 1)
            self._result_panel = self._placeholder_panel()
            body.addWidget(self._result_panel, 2)
            self._body = body
            self._main_lay.addLayout(body)
            self._main_lay.addStretch()
        finally:
            container.setUpdatesEnabled(True)

        scroll.setWidget(container)
        root.addWidget(scroll)

    def _input_panel(self) -> QFrame:
        outer = QFrame()
        outer.setObjectName("card")
        outer_lay = QVBoxLayout(outer)
        outer_lay.setContentsMargins(0, 0, 0, 14)
        outer_lay.setSpacing(0)

        hdr = QLabel("  🎮  Player Attributes")
        hdr.setFont(QFont("Segoe UI", 13, QFont.Bold))
        hdr.setFixedHeight(44)
        hdr.setStyleSheet(f"color: {ACCENT_CYAN}; background: {BG_CARD}; padding-left: 16px;")
        outer_lay.addWidget(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumHeight(480)

        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(10)

        for group_name, attrs in ATTR_GROUPS.items():
            gb = QGroupBox(group_name)
            gb.setFont(QFont("Segoe UI", 9, QFont.Bold))
            gl = QGridLayout(gb)
            gl.setSpacing(6)
            for i, attr in enumerate(attrs):
                lbl = QLabel(attr.replace("_", " ").title())
                lbl.setFont(QFont("Segoe UI", 9))
                lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent;")
                spin = QDoubleSpinBox()
                spin.setRange(1, 200)
                spin.setValue(DEFAULTS.get(attr, 65))
                spin.setSingleStep(1)
                spin.setDecimals(0)
                spin.setFixedWidth(80)
                spin.setFont(QFont("Segoe UI", 9))
                self._spinners[attr] = spin
                row = i // 2
                col = (i % 2) * 2
                gl.addWidget(lbl,  row, col)
                gl.addWidget(spin, row, col + 1)
            lay.addWidget(gb)

        lay.addStretch()
        scroll.setWidget(inner)
        outer_lay.addWidget(scroll)

        btn = QPushButton("⭐  Predict Rating")
        btn.setFixedHeight(44)
        btn.setFont(QFont("Segoe UI", 12, QFont.Bold))
        btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {ACCENT_GOLD}, stop:1 #B8960A);
                color: #0A0E1A; border: none; border-radius: 6px;
                font-weight: 700; margin: 0 14px;
            }}
            QPushButton:hover {{ background: {ACCENT_GOLD}; }}
        """)
        btn.clicked.connect(self._run_prediction)
        outer_lay.addWidget(btn)
        return outer

    def _placeholder_panel(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        card.setMinimumHeight(300)
        lay = QVBoxLayout(card)
        lay.setAlignment(Qt.AlignCenter)
        lbl = QLabel("⭐  Set player attributes and click\n'Predict Rating' to see results")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFont(QFont("Segoe UI", 12))
        lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent;")
        lay.addWidget(lbl)
        return card

    def _run_prediction(self):
        attrs = {k: float(v.value()) for k, v in self._spinners.items()}
        self._worker = RatingWorker(self._predictor, attrs)
        self._worker.done.connect(self._show_result)
        self._worker.start()
        self.notify.emit("Running prediction…", "info")

    def _show_result(self, result: dict):
        self._result = result
        old = self._result_panel
        self._result_panel = self._build_result_panel(result)
        for i in range(self._body.count()):
            item = self._body.itemAt(i)
            if item and item.widget() is old:
                self._body.removeWidget(old)
                old.deleteLater()
                self._body.addWidget(self._result_panel, 2)
                return

    def _build_result_panel(self, r: dict) -> QFrame:
        rating   = r["rating"]
        position = r["position"]
        similar  = r["similar_players"]
        tips     = r["improvements"]
        color    = get_rating_color(rating)

        card = QFrame()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(16)

        # ── Freeze repaints while building result panel ────────────
        card.setUpdatesEnabled(False)
        try:
            top = QHBoxLayout()
            top.setSpacing(20)
            top.addWidget(PlayerCard("Your Player", position, rating))

            summary = QVBoxLayout()
            summary.setSpacing(10)
            for title, val, clr in [
                ("Overall Rating", str(rating), color),
                ("Best Position",  position,    ACCENT_CYAN),
                ("Tier",           self._tier(rating), color),
            ]:
                sc = StatCard(title, val, color=clr)
                sc.setFixedHeight(76)
                summary.addWidget(sc)
            top.addLayout(summary, 1)
            lay.addLayout(top)

            tabs = QTabWidget()
            tabs.addTab(self._similar_tab(similar), "🔍 Similar Players")
            tabs.addTab(self._improvements_tab(tips), "📈 Improvements")
            tabs.addTab(self._radar_tab(r), "📊 Radar Chart")
            lay.addWidget(tabs)
        finally:
            card.setUpdatesEnabled(True)

        return card

    def _tier(self, rating: int) -> str:
        if rating >= 88: return "⭐ Icon"
        if rating >= 83: return "🥇 Elite"
        if rating >= 76: return "🥈 Gold"
        if rating >= 65: return "🥉 Silver"
        return "🔘 Bronze"

    def _similar_tab(self, similar: list) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)

        table = QTableWidget(len(similar), 4)
        table.setHorizontalHeaderLabels(["Player", "Rating", "Position", "Similarity"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in (1, 2, 3):
            table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        table.setShowGrid(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(False)

        for row, p in enumerate(similar):
            rating = p["rating"]
            color  = get_rating_color(rating)

            name_item = QTableWidgetItem(p["name"])
            name_item.setForeground(QColor("white"))
            name_item.setFont(QFont("Segoe UI", 10, QFont.Bold))

            rating_item = QTableWidgetItem(str(rating))
            rating_item.setTextAlignment(Qt.AlignCenter)
            rating_item.setForeground(QColor(color))
            rating_item.setFont(QFont("Segoe UI", 10, QFont.Bold))

            pos_item = QTableWidgetItem(p["position"])
            pos_item.setTextAlignment(Qt.AlignCenter)
            pos_item.setForeground(QColor(ACCENT_CYAN))

            dist = p.get("distance", 0)
            sim_pct = max(0, int(100 - dist))
            sim_item = QTableWidgetItem(f"{sim_pct}%")
            sim_item.setTextAlignment(Qt.AlignCenter)
            clr = ACCENT_GREEN if sim_pct > 75 else (ACCENT_ORANGE if sim_pct > 50 else TEXT_SECONDARY)
            sim_item.setForeground(QColor(clr))

            table.setItem(row, 0, name_item)
            table.setItem(row, 1, rating_item)
            table.setItem(row, 2, pos_item)
            table.setItem(row, 3, sim_item)
            table.setRowHeight(row, 38)

        lay.addWidget(table)
        return w

    def _improvements_tab(self, tips: list) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        hdr = QLabel("💡  Suggested Improvements to Boost Rating")
        hdr.setFont(QFont("Segoe UI", 12, QFont.Bold))
        hdr.setStyleSheet(f"color: {ACCENT_GOLD}; background: transparent;")
        lay.addWidget(hdr)

        for i, tip in enumerate(tips, 1):
            row = QFrame()
            row.setStyleSheet(f"""
                QFrame {{
                    background: {ACCENT_GOLD}12;
                    border-left: 3px solid {ACCENT_GOLD};
                    border-radius: 4px;
                }}
            """)
            rl = QHBoxLayout(row)
            rl.setContentsMargins(12, 8, 12, 8)
            num = QLabel(f"{i}")
            num.setFont(QFont("Segoe UI", 11, QFont.Bold))
            num.setFixedWidth(22)
            num.setStyleSheet(f"color: {ACCENT_GOLD}; background: transparent;")
            lbl = QLabel(tip)
            lbl.setFont(QFont("Segoe UI", 10))
            lbl.setStyleSheet("color: white; background: transparent;")
            lbl.setWordWrap(True)
            rl.addWidget(num)
            rl.addWidget(lbl, 1)
            lay.addWidget(row)

        lay.addStretch()
        return w

    def _radar_tab(self, r: dict) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)

        attrs = self._spinners
        labels = ["Pace", "Shooting", "Passing", "Dribbling", "Defending", "Physical"]
        group_attrs = {
            "Pace":      ["acceleration", "sprint_speed"],
            "Shooting":  ["finishing", "shot_power", "long_shots"],
            "Passing":   ["short_passing", "long_passing", "vision"],
            "Dribbling": ["dribbling", "ball_control", "agility"],
            "Defending": ["defensive_awareness", "standing_tackle", "sliding_tackle"],
            "Physical":  ["stamina", "strength", "jumping"],
        }
        values = []
        for group in labels:
            keys = group_attrs.get(group, [])
            vals = [float(attrs[k].value()) for k in keys if k in attrs]
            values.append(int(sum(vals) / len(vals)) if vals else 65)

        try:
            color = get_rating_color(r["rating"])
            pix = charts.radar_chart(labels, values, f"Player Profile (OVR {r['rating']})", color)
            lay.addWidget(charts.chart_label(pix, 420, 380), alignment=Qt.AlignCenter)
        except Exception:
            pass

        return w