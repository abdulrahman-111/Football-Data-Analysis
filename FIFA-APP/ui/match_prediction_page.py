"""
match_prediction_page.py — Match Prediction UI page.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QLineEdit, QComboBox, QDoubleSpinBox, QPushButton,
    QScrollArea, QGridLayout, QGroupBox, QSizePolicy,
    QSpacerItem, QTextEdit
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

from models.match_predictor import MatchPredictor
from utils import charts
from utils.theme import (
    ACCENT_CYAN, ACCENT_GREEN, ACCENT_GOLD, ACCENT_RED,
    ACCENT_ORANGE, TEXT_PRIMARY, TEXT_SECONDARY, BG_CARD,
    BG_PANEL, BG_INPUT, BORDER_COLOR
)
from ui.widgets.widgets import (
    StatCard, SectionHeader, ProbabilityGauge, LoadingOverlay, NotificationPopup
)

TEAMS = [
    "Real Madrid", "Manchester City", "Bayern Munich", "Barcelona",
    "Arsenal", "Liverpool", "Chelsea", "PSG", "Atletico Madrid",
    "Juventus", "Inter Milan", "AC Milan", "Borussia Dortmund",
    "Napoli", "Tottenham", "Custom Team",
]


class PredictWorker(QThread):
    done = Signal(dict)

    def __init__(self, predictor, kwargs):
        super().__init__()
        self._p = predictor
        self._kw = kwargs

    def run(self):
        self.done.emit(self._p.predict(**self._kw))


class MatchPredictionPage(QWidget):
    notify = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._predictor = MatchPredictor()
        self._result: dict | None = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(28, 24, 28, 24)
        self._layout.setSpacing(20)

        # ── Freeze repaints while building all children ────────────
        container.setUpdatesEnabled(False)
        try:
            self._layout.addWidget(SectionHeader("⚽  Match Prediction"))

            body = QHBoxLayout()
            body.setSpacing(20)
            body.addWidget(self._input_panel(), 1)
            self._result_panel = self._empty_result_panel()
            body.addWidget(self._result_panel, 2)
            self._layout.addLayout(body)
            self._layout.addStretch()
        finally:
            container.setUpdatesEnabled(True)

        scroll.setWidget(container)
        root.addWidget(scroll)

    def _input_panel(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        card.setMinimumWidth(300)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(14)

        hdr = QLabel("🏟️  Match Setup")
        hdr.setFont(QFont("Segoe UI", 13, QFont.Bold))
        hdr.setStyleSheet(f"color: {ACCENT_CYAN}; background: transparent;")
        lay.addWidget(hdr)

        lay.addWidget(self._field_label("Home Team"))
        self._team1_combo = QComboBox()
        self._team1_combo.addItems(TEAMS)
        lay.addWidget(self._team1_combo)

        self._goals1, self._shots1, self._poss1 = self._stat_row("Avg Goals/Match", "Shots/Match", "Possession %")
        for w in self._stat_widgets[-3:]:
            lay.addWidget(w)

        lay.addWidget(self._divider())

        lay.addWidget(self._field_label("Away Team"))
        self._team2_combo = QComboBox()
        self._team2_combo.addItems(TEAMS)
        self._team2_combo.setCurrentIndex(1)
        lay.addWidget(self._team2_combo)

        g2, s2, p2 = self._stat_row("Avg Goals/Match", "Shots/Match", "Possession %")
        self._goals2, self._shots2, self._poss2 = g2, s2, p2
        for w in self._stat_widgets[-3:]:
            lay.addWidget(w)

        lay.addWidget(self._divider())

        lay.addWidget(self._field_label("Player to Score (optional)"))
        self._player_input = QLineEdit()
        self._player_input.setPlaceholderText("e.g. Erling Haaland")
        lay.addWidget(self._player_input)

        predict_btn = QPushButton("⚡  Predict Match")
        predict_btn.setObjectName("accentBtn")
        predict_btn.setFixedHeight(44)
        predict_btn.setFont(QFont("Segoe UI", 12, QFont.Bold))
        predict_btn.setCursor(Qt.PointingHandCursor)
        predict_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {ACCENT_CYAN}, stop:1 #0090B8);
                color: #0A0E1A; border: none; border-radius: 8px;
                font-weight: 700;
            }}
            QPushButton:hover {{ background: {ACCENT_CYAN}; }}
        """)
        predict_btn.clicked.connect(self._run_prediction)
        lay.addWidget(predict_btn)

        return card

    def _empty_result_panel(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setAlignment(Qt.AlignCenter)
        placeholder = QLabel("Fill in match details and click\n⚡ Predict Match")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setFont(QFont("Segoe UI", 13))
        placeholder.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent;")
        lay.addWidget(placeholder)
        return card

    def _stat_row(self, *labels):
        spins = []
        for lbl in labels:
            w = QWidget(self)
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            l = QLabel(lbl, w)
            l.setFont(QFont("Segoe UI", 9))
            l.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent;")
            l.setFixedWidth(130)
            spin = QDoubleSpinBox(w)
            spin.setRange(0, 200)
            spin.setValue(1.5 if "Goal" in lbl else (12 if "Shot" in lbl else 50))
            spin.setSingleStep(0.5)
            h.addWidget(l)
            h.addWidget(spin, 1)
            if not hasattr(self, "_stat_widgets"):
                self._stat_widgets = []
            self._stat_widgets.append(w)
            spins.append((spin, w))
        return tuple(s for s, _ in spins)

    def _field_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent;")
        return lbl

    def _divider(self) -> QFrame:
        d = QFrame()
        d.setFixedHeight(1)
        d.setStyleSheet(f"background: {BORDER_COLOR};")
        return d

    def _run_prediction(self):
        t1 = self._team1_combo.currentText()
        t2 = self._team2_combo.currentText()
        if t1 == t2:
            self.notify.emit("Please choose two different teams.", "warning")
            return
        kwargs = dict(
            team1=t1, team2=t2,
            goals_t1=self._goals1.value(), shots_t1=int(self._shots1.value()),
            poss_t1=self._poss1.value() / 100,
            goals_t2=self._goals2.value(), shots_t2=int(self._shots2.value()),
            poss_t2=self._poss2.value() / 100,
            player_name=self._player_input.text().strip(),
        )
        self._worker = PredictWorker(self._predictor, kwargs)
        self._worker.done.connect(self._show_result)
        self._worker.start()

    def _show_result(self, res: dict):
        self._result = res
        old = self._result_panel
        self._result_panel = self._build_result_panel(res)
        for i in range(self._layout.count()):
            item = self._layout.itemAt(i)
            if item and item.layout():
                hbox = item.layout()
                for j in range(hbox.count()):
                    w = hbox.itemAt(j)
                    if w and w.widget() is old:
                        hbox.removeWidget(old)
                        old.deleteLater()
                        hbox.addWidget(self._result_panel, 2)
                        return

    def _build_result_panel(self, res: dict) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        outer = QVBoxLayout(card)
        outer.setContentsMargins(20, 18, 20, 18)
        outer.setSpacing(16)

        # ── Freeze result panel repaints while building ────────────
        card.setUpdatesEnabled(False)
        try:
            score_frame = QFrame()
            score_frame.setStyleSheet(f"""
                QFrame {{
                    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                        stop:0 {BG_PANEL}, stop:0.5 {ACCENT_CYAN}18, stop:1 {BG_PANEL});
                    border-radius: 10px;
                }}
            """)
            sf_lay = QHBoxLayout(score_frame)
            sf_lay.setContentsMargins(20, 14, 20, 14)

            t1_lbl = QLabel(res["team1"])
            t1_lbl.setAlignment(Qt.AlignCenter)
            t1_lbl.setFont(QFont("Segoe UI", 14, QFont.Bold))
            t1_lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; background: transparent;")

            score_lbl = QLabel(f"{res['predicted_home']}  –  {res['predicted_away']}")
            score_lbl.setAlignment(Qt.AlignCenter)
            score_lbl.setFont(QFont("Segoe UI", 28, QFont.Bold))
            score_lbl.setStyleSheet(f"color: {ACCENT_CYAN}; background: transparent;")

            t2_lbl = QLabel(res["team2"])
            t2_lbl.setAlignment(Qt.AlignCenter)
            t2_lbl.setFont(QFont("Segoe UI", 14, QFont.Bold))
            t2_lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; background: transparent;")

            sf_lay.addWidget(t1_lbl, 1)
            sf_lay.addWidget(score_lbl)
            sf_lay.addWidget(t2_lbl, 1)
            outer.addWidget(score_frame)

            gauges_row = QHBoxLayout()
            gauges_row.setSpacing(8)
            for label, val in [
                (f"{res['team1']} Win", res["win1"]),
                ("Draw",               res["draw"]),
                (f"{res['team2']} Win", res["win2"]),
            ]:
                g = ProbabilityGauge(label, val)
                g.setMinimumHeight(120)
                gauges_row.addWidget(g)
            outer.addLayout(gauges_row)

            stats_row = QHBoxLayout()
            stats_row.setSpacing(10)
            conf_color = ACCENT_GREEN if res["confidence"] > 0.7 else ACCENT_ORANGE
            for title, val, color in [
                ("xG Home",    f"{res['xg1']:.2f}",            ACCENT_CYAN),
                ("xG Away",    f"{res['xg2']:.2f}",            ACCENT_CYAN),
                ("Confidence", f"{res['confidence']*100:.0f}%", conf_color),
            ]:
                stats_row.addWidget(StatCard(title, val, color=color))
            outer.addLayout(stats_row)

            if res.get("scorer_prob", 0) > 0 and self._player_input.text().strip():
                player = self._player_input.text().strip()
                sp = res["scorer_prob"]
                sp_color = ACCENT_GREEN if sp > 0.5 else ACCENT_ORANGE
                scorer_frame = QFrame()
                scorer_frame.setStyleSheet(f"""
                    QFrame {{
                        background: {sp_color}18;
                        border: 1px solid {sp_color}50;
                        border-radius: 8px;
                    }}
                """)
                sl = QHBoxLayout(scorer_frame)
                sl.setContentsMargins(14, 10, 14, 10)
                icon_lbl = QLabel("🎯")
                icon_lbl.setFont(QFont("Segoe UI", 16))
                icon_lbl.setStyleSheet("background: transparent;")
                text_lbl = QLabel(f"<b>{player}</b> scoring probability: <b style='color:{sp_color}'>{sp*100:.0f}%</b>")
                text_lbl.setFont(QFont("Segoe UI", 11))
                text_lbl.setStyleSheet("background: transparent; color: white;")
                sl.addWidget(icon_lbl)
                sl.addWidget(text_lbl, 1)
                outer.addWidget(scorer_frame)

            try:
                pix = charts.bar_chart(
                    labels=[res["team1"], "Draw", res["team2"]],
                    values=[res["win1"]*100, res["draw"]*100, res["win2"]*100],
                    title="Win Probabilities (%)",
                    colors=["#00D4FF", "#FFD700", "#FF3B5C"],
                    ylabel="%",
                )
                outer.addWidget(charts.chart_label(pix, 560, 250))
            except Exception:
                pass

            analysis_lbl = QLabel(res.get("analysis", ""))
            analysis_lbl.setWordWrap(True)
            analysis_lbl.setFont(QFont("Segoe UI", 10))
            analysis_lbl.setStyleSheet(f"""
                color: {TEXT_SECONDARY};
                background: {BG_PANEL};
                border-radius: 6px;
                padding: 10px;
            """)
            outer.addWidget(analysis_lbl)
        finally:
            card.setUpdatesEnabled(True)

        return card