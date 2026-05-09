"""
player_evaluation_page.py — Player Evaluation with JSON upload and PDF report.
"""
import os, json
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QScrollArea, QTabWidget, QFileDialog,
    QTextEdit, QProgressBar, QSizePolicy, QGridLayout,
    QGroupBox
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QDragEnterEvent, QDropEvent

from models.player_evaluator import PlayerEvaluator
from utils import charts
from utils.report_generator import generate_pdf_report
from utils.theme import (
    ACCENT_CYAN, ACCENT_GREEN, ACCENT_GOLD, ACCENT_RED,
    ACCENT_ORANGE, TEXT_PRIMARY, TEXT_SECONDARY,
    BG_CARD, BG_PANEL, BORDER_COLOR, get_risk_color
)
from ui.widgets.widgets import (
    StatCard, SectionHeader, AttributeBar, PlayerCard,
    LoadingOverlay, NotificationPopup
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
SAMPLE_JSON = os.path.join(DATA_DIR, "player_input.json")


class EvalWorker(QThread):
    done = Signal(dict)
    error = Signal(str)

    def __init__(self, evaluator, data):
        super().__init__()
        self._ev = evaluator
        self._data = data

    def run(self):
        try:
            self.done.emit(self._ev.evaluate(self._data))
        except Exception as e:
            self.error.emit(str(e))


class DropZone(QFrame):
    file_dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFixedHeight(100)
        self.setStyleSheet(f"""
            QFrame {{
                background: {BG_PANEL};
                border: 2px dashed {BORDER_COLOR};
                border-radius: 10px;
            }}
            QFrame:hover {{ border-color: {ACCENT_CYAN}; }}
        """)
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        self._lbl = QLabel("📂  Drop player JSON here  or  click Browse")
        self._lbl.setAlignment(Qt.AlignCenter)
        self._lbl.setFont(QFont("Segoe UI", 11))
        self._lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent;")
        lay.addWidget(self._lbl)

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self.setStyleSheet(self.styleSheet().replace(BORDER_COLOR, ACCENT_CYAN))

    def dragLeaveEvent(self, e):
        self.setStyleSheet(self.styleSheet().replace(ACCENT_CYAN, BORDER_COLOR))

    def dropEvent(self, e: QDropEvent):
        urls = e.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.endswith(".json"):
                self._lbl.setText(f"✅  {os.path.basename(path)}")
                self.file_dropped.emit(path)
            else:
                self._lbl.setText("⚠️  Please drop a .json file")

    def set_loaded(self, name: str):
        self._lbl.setText(f"✅  {name}")


class PlayerEvaluationPage(QWidget):
    notify = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._evaluator = PlayerEvaluator()
        self._data: dict | None = None
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
        self._main_lay.setSpacing(18)

        # ── Freeze repaints while building all children ────────────
        container.setUpdatesEnabled(False)
        try:
            self._main_lay.addWidget(SectionHeader("👤  Player Evaluation"))
            self._main_lay.addWidget(self._upload_bar())
            self._results_area = self._placeholder_panel()
            self._main_lay.addWidget(self._results_area)
            self._main_lay.addStretch()
        finally:
            container.setUpdatesEnabled(True)

        scroll.setWidget(container)
        root.addWidget(scroll)

    def _upload_bar(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        lay = QHBoxLayout(card)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(12)

        self._drop_zone = DropZone()
        self._drop_zone.file_dropped.connect(self._load_json_file)

        browse_btn = QPushButton("📁  Browse")
        browse_btn.setFixedSize(120, 44)
        browse_btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        browse_btn.setStyleSheet(f"""
            QPushButton {{
                background: {BG_CARD}; color: {ACCENT_CYAN};
                border: 1px solid {ACCENT_CYAN}; border-radius: 6px;
            }}
            QPushButton:hover {{ background: {ACCENT_CYAN}20; }}
        """)
        browse_btn.clicked.connect(self._browse_file)

        sample_btn = QPushButton("📄  Load Sample")
        sample_btn.setFixedSize(130, 44)
        sample_btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        sample_btn.setStyleSheet(f"""
            QPushButton {{
                background: {BG_CARD}; color: {ACCENT_GREEN};
                border: 1px solid {ACCENT_GREEN}; border-radius: 6px;
            }}
            QPushButton:hover {{ background: {ACCENT_GREEN}20; }}
        """)
        sample_btn.clicked.connect(self._load_sample)

        self._analyze_btn = QPushButton("⚡  Analyze Player")
        self._analyze_btn.setFixedSize(150, 44)
        self._analyze_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self._analyze_btn.setEnabled(False)
        self._analyze_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {ACCENT_CYAN}, stop:1 #0090B8);
                color: #0A0E1A; border: none; border-radius: 6px; font-weight: 700;
            }}
            QPushButton:disabled {{
                background: #1A2238; color: #4A5568; border: none;
            }}
            QPushButton:hover:!disabled {{ background: {ACCENT_CYAN}; }}
        """)
        self._analyze_btn.clicked.connect(self._run_analysis)

        lay.addWidget(self._drop_zone, 1)
        lay.addWidget(browse_btn)
        lay.addWidget(sample_btn)
        lay.addWidget(self._analyze_btn)
        return card

    def _placeholder_panel(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        card.setMinimumHeight(300)
        lay = QVBoxLayout(card)
        lay.setAlignment(Qt.AlignCenter)
        lbl = QLabel("📋  Load a player JSON file and click Analyze Player\n\nUse 'Load Sample' to try with the built-in demo player")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFont(QFont("Segoe UI", 12))
        lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent;")
        lay.addWidget(lbl)
        return card

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Player JSON", DATA_DIR, "JSON Files (*.json)"
        )
        if path:
            self._load_json_file(path)

    def _load_sample(self):
        if os.path.exists(SAMPLE_JSON):
            self._load_json_file(SAMPLE_JSON)
        else:
            self.notify.emit("Sample file not found.", "error")

    def _load_json_file(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
            self._drop_zone.set_loaded(os.path.basename(path))
            self._analyze_btn.setEnabled(True)
            name = self._data.get("player_info", {}).get("name", "Player")
            self.notify.emit(f"Loaded: {name}", "success")
        except Exception as e:
            self.notify.emit(f"Failed to load JSON: {e}", "error")
            self._data = None
            self._analyze_btn.setEnabled(False)

    def _run_analysis(self):
        if not self._data:
            return
        self._analyze_btn.setEnabled(False)
        self._analyze_btn.setText("⏳ Analyzing...")
        self._worker = EvalWorker(self._evaluator, self._data)
        self._worker.done.connect(self._show_results)
        self._worker.error.connect(lambda e: self.notify.emit(e, "error"))
        self._worker.start()

    def _show_results(self, result: dict):
        self._result = result
        self._analyze_btn.setEnabled(True)
        self._analyze_btn.setText("⚡  Analyze Player")
        self.notify.emit("Analysis complete!", "success")

        old = self._results_area
        self._results_area = self._build_results_panel(result)
        idx = self._main_lay.indexOf(old)
        self._main_lay.removeWidget(old)
        old.deleteLater()
        self._main_lay.insertWidget(idx, self._results_area)

    def _build_results_panel(self, r: dict) -> QFrame:
        info = self._data.get("player_info", {})
        name = info.get("name", "Unknown")
        pos  = info.get("position", "—")
        age  = info.get("age", "—")
        team = info.get("current_team", "—")
        nat  = info.get("nationality", "—")

        g = r["goals_pred"]; a = r["assists_pred"]
        inj = r["injury_prob"]; mv = r["market_value"]
        overall = min(99, max(55, int(55 + g * 1.2 + a * 0.8 - inj * 20)))

        card = QFrame()
        card.setObjectName("card")
        outer = QVBoxLayout(card)
        outer.setContentsMargins(20, 18, 20, 18)
        outer.setSpacing(16)

        # ── Freeze repaints while building result panel ────────────
        card.setUpdatesEnabled(False)
        try:
            top = QHBoxLayout()
            top.setSpacing(20)
            top.addWidget(PlayerCard(name, pos, overall, nat, team))

            metrics_col = QVBoxLayout()
            metrics_col.setSpacing(10)
            risk_color = get_risk_color(inj)
            for title, val, color in [
                ("Predicted Goals",   f"{g:.1f}",        ACCENT_GREEN),
                ("Predicted Assists", f"{a:.1f}",        ACCENT_CYAN),
                ("Injury Risk",       f"{inj*100:.1f}%", risk_color),
                ("Market Value",      f"€{mv:,.0f}",     ACCENT_GOLD),
            ]:
                sc = StatCard(title, val, color=color)
                sc.setFixedHeight(80)
                metrics_col.addWidget(sc)
            top.addLayout(metrics_col, 1)
            outer.addLayout(top)

            tabs = QTabWidget()
            tabs.addTab(self._analytics_tab(r, overall), "📊 Analytics")
            tabs.addTab(self._strengths_tab(r), "💪 Strengths & Weaknesses")
            tabs.addTab(self._growth_tab(r), "📈 Career Growth")
            outer.addWidget(tabs)

            pdf_btn = QPushButton("📥  Download PDF Report")
            pdf_btn.setFixedHeight(40)
            pdf_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
            pdf_btn.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                        stop:0 {ACCENT_GOLD}, stop:1 #B8960A);
                    color: #0A0E1A; border: none; border-radius: 6px; font-weight: 700;
                }}
                QPushButton:hover {{ background: {ACCENT_GOLD}; }}
            """)
            pdf_btn.clicked.connect(lambda: self._save_pdf(r, info))
            outer.addWidget(pdf_btn)
        finally:
            card.setUpdatesEnabled(True)

        return card

    def _analytics_tab(self, r: dict, overall: int) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(16)

        try:
            t = r["trends"]
            pix = charts.radar_chart(t["labels"], t["values"], "Performance Radar")
            lay.addWidget(charts.chart_label(pix, 300, 300))
        except Exception:
            pass

        bars_col = QVBoxLayout()
        bars_col.setSpacing(6)
        bars_col.addWidget(QLabel("Key Attributes"))
        for lbl, val, color in [
            ("Goals (next)",   min(99, int(r["goals_pred"] * 4)),    ACCENT_GREEN),
            ("Assists (next)", min(99, int(r["assists_pred"] * 6)),   ACCENT_CYAN),
            ("Injury Safety",  int((1 - r["injury_prob"]) * 100),    get_risk_color(r["injury_prob"])),
            ("Market Value",   min(99, int(r["market_value"] / 1e6)), ACCENT_GOLD),
            ("Overall Est.",   overall,                               ACCENT_CYAN),
        ]:
            bars_col.addWidget(AttributeBar(lbl, val, 100, color))
        bars_col.addStretch()
        lay.addLayout(bars_col, 1)
        return w

    def _strengths_tab(self, r: dict) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)

        s_hdr = QLabel("✅  Strengths")
        s_hdr.setFont(QFont("Segoe UI", 12, QFont.Bold))
        s_hdr.setStyleSheet(f"color: {ACCENT_GREEN}; background: transparent;")
        lay.addWidget(s_hdr)

        for s in r.get("strengths", []):
            lbl = QLabel(f"  •  {s}")
            lbl.setFont(QFont("Segoe UI", 10))
            lbl.setStyleSheet(f"""
                color: white; background: {ACCENT_GREEN}15;
                border-left: 3px solid {ACCENT_GREEN};
                border-radius: 4px; padding: 6px 10px;
            """)
            lay.addWidget(lbl)

        lay.addSpacing(8)
        w_hdr = QLabel("⚠️  Weaknesses")
        w_hdr.setFont(QFont("Segoe UI", 12, QFont.Bold))
        w_hdr.setStyleSheet(f"color: {ACCENT_ORANGE}; background: transparent;")
        lay.addWidget(w_hdr)

        for wk in r.get("weaknesses", []):
            lbl = QLabel(f"  •  {wk}")
            lbl.setFont(QFont("Segoe UI", 10))
            lbl.setStyleSheet(f"""
                color: white; background: {ACCENT_ORANGE}15;
                border-left: 3px solid {ACCENT_ORANGE};
                border-radius: 4px; padding: 6px 10px;
            """)
            lay.addWidget(lbl)

        lay.addStretch()
        return w

    def _growth_tab(self, r: dict) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)

        cg = r.get("career_growth", {})
        if cg:
            try:
                pix = charts.line_chart(
                    x_vals=cg["seasons"],
                    y_series={"Goals": cg["goals"], "Assists": cg["assists"]},
                    title="5-Season Career Growth Forecast",
                    xlabel="Season", ylabel="Count"
                )
                lay.addWidget(charts.chart_label(pix, 700, 320))
            except Exception:
                pass

        note = QLabel("📌 Projections based on current trajectory and ML model outputs.")
        note.setFont(QFont("Segoe UI", 9))
        note.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent;")
        lay.addWidget(note)
        lay.addStretch()
        return w

    def _save_pdf(self, r: dict, info: dict):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF Report", os.path.join(BASE_DIR, "outputs", "player_report.pdf"),
            "PDF Files (*.pdf)"
        )
        if path:
            ok = generate_pdf_report(info, r, path)
            if ok:
                self.notify.emit(f"PDF saved: {os.path.basename(path)}", "success")
            else:
                self.notify.emit("PDF generation failed (install reportlab).", "error")