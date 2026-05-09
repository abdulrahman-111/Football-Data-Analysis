"""
dashboard.py — Main window: topbar + sidebar + stacked content pages.
"""
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QStackedWidget, QLineEdit,
    QPushButton, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QGraphicsOpacityEffect

from utils.theme import (
    BG_DARK, BG_PANEL, BG_CARD, BG_SIDEBAR,
    ACCENT_CYAN, ACCENT_GREEN, ACCENT_GOLD,
    TEXT_PRIMARY, TEXT_SECONDARY, BORDER_COLOR
)
from ui.sidebar import Sidebar
from ui.home_page import HomePage
from ui.match_prediction_page import MatchPredictionPage
from ui.player_evaluation_page import PlayerEvaluationPage
from ui.rating_prediction_page import RatingPredictionPage
from ui.settings_page import SettingsPage
from ui.widgets.widgets import NotificationPopup


class TopBar(QFrame):
    """Top application bar with title, search, and status indicator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(58)
        self.setStyleSheet(f"""
            QFrame {{
                background: {BG_PANEL};
                border-bottom: 1px solid {BORDER_COLOR};
            }}
        """)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 0, 20, 0)
        lay.setSpacing(16)

        self._title = QLabel("Home")
        self._title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self._title.setStyleSheet(f"color: {TEXT_PRIMARY}; background: transparent;")

        self._search = QLineEdit()
        self._search.setPlaceholderText("🔍  Search players, teams...")
        self._search.setFixedWidth(280)
        self._search.setFixedHeight(34)

        self._status = QLabel("● Live")
        self._status.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self._status.setStyleSheet(f"color: {ACCENT_GREEN}; background: transparent;")

        self._bell = QPushButton("🔔")
        self._bell.setFixedSize(34, 34)
        self._bell.setStyleSheet(f"""
            QPushButton {{
                background: transparent; border: 1px solid {BORDER_COLOR};
                border-radius: 6px; font-size: 14px;
            }}
            QPushButton:hover {{ border-color: {ACCENT_CYAN}; }}
        """)

        self._profile = QLabel("👤  Scout")
        self._profile.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self._profile.setStyleSheet(f"""
            color: {ACCENT_CYAN};
            background: {ACCENT_CYAN}15;
            border: 1px solid {ACCENT_CYAN}40;
            border-radius: 6px;
            padding: 4px 12px;
        """)

        lay.addWidget(self._title)
        lay.addStretch()
        lay.addWidget(self._search)
        lay.addWidget(self._status)
        lay.addWidget(self._bell)
        lay.addWidget(self._profile)

        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._pulse_status)
        self._pulse_timer.start(1800)
        self._pulse_state = True

    def set_title(self, title: str):
        self._title.setText(title)

    def _pulse_status(self):
        self._pulse_state = not self._pulse_state
        color = ACCENT_GREEN if self._pulse_state else "#005530"
        self._status.setStyleSheet(f"color: {color}; background: transparent;")


PAGE_TITLES = {
    0: "🏠  Home",
    1: "⚽  Match Prediction",
    2: "👤  Player Evaluation",
    3: "⭐  Rating & Recommendations",
    4: "⚙️  Settings",
}


class Dashboard(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FIFA Analytics Dashboard")
        self.setMinimumSize(1180, 720)
        self.resize(1380, 820)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._topbar = TopBar()
        root.addWidget(self._topbar)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self._sidebar = Sidebar()
        self._sidebar.page_changed.connect(self._switch_page)
        body.addWidget(self._sidebar)

        self._stack = QStackedWidget()

        # ── Freeze repaints while constructing all pages ────────────
        # Without this each page __init__ triggers incremental repaints
        # causing the visible sequential "pop-in" on startup.
        central.setUpdatesEnabled(False)
        try:
            self._home_page   = HomePage(switch_page_cb=self._switch_page)
            self._match_page  = MatchPredictionPage()
            self._eval_page   = PlayerEvaluationPage()
            self._rating_page = RatingPredictionPage()
            self._settings    = SettingsPage()

            for page in [self._home_page, self._match_page,
                         self._eval_page, self._rating_page, self._settings]:
                self._stack.addWidget(page)
        finally:
            central.setUpdatesEnabled(True)

        # Wire notification signals
        for page in [self._match_page, self._eval_page,
                     self._rating_page, self._settings]:
            if hasattr(page, "notify"):
                page.notify.connect(self._show_notification)

        body.addWidget(self._stack, 1)
        root.addLayout(body, 1)

        # ── Page-switch fade overlay ────────────────────────────────
        # We animate a thin semi-transparent overlay instead of applying
        # QGraphicsOpacityEffect directly to pages. This avoids the nested-
        # painter crash that occurs when an effect is on a widget that
        # contains custom paintEvent children (e.g. ProbabilityGauge).
        self._fade_overlay = QFrame(central)
        self._fade_overlay.setStyleSheet(f"background: {BG_DARK};")
        self._fade_overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._fade_overlay.hide()

        self._fade_effect = QGraphicsOpacityEffect(self._fade_overlay)
        self._fade_overlay.setGraphicsEffect(self._fade_effect)

        self._fade_anim = QPropertyAnimation(self._fade_effect, b"opacity")
        self._fade_anim.setDuration(180)
        self._fade_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._fade_anim.finished.connect(self._fade_overlay.hide)

        # Notification area (floating bottom-right)
        self._notif_container = QWidget(central)
        self._notif_container.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        notif_lay = QVBoxLayout(self._notif_container)
        notif_lay.setAlignment(Qt.AlignBottom | Qt.AlignRight)
        notif_lay.setContentsMargins(0, 0, 16, 16)
        notif_lay.setSpacing(6)
        self._notif_container.setLayout(notif_lay)
        self._notif_container.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cw = self.centralWidget()
        if cw and hasattr(self, "_notif_container"):
            self._notif_container.setGeometry(cw.rect())
        if cw and hasattr(self, "_fade_overlay"):
            self._fade_overlay.setGeometry(cw.rect())

    # ── Page switching ─────────────────────────────────────────────
    def _switch_page(self, idx: int):
        self._stack.setCurrentIndex(idx)
        self._topbar.set_title(PAGE_TITLES.get(idx, ""))
        self._sidebar.navigate_to(idx)

        # Fade the overlay from semi-opaque to transparent over the new page.
        # The overlay sits on top; no effect is applied to the page itself.
        cw = self.centralWidget()
        if cw:
            self._fade_overlay.setGeometry(cw.rect())
        self._fade_overlay.show()
        self._fade_overlay.raise_()
        self._fade_anim.stop()
        self._fade_anim.setStartValue(0.55)
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.start()

    # ── Notifications ──────────────────────────────────────────────
    def _show_notification(self, message: str, kind: str = "info"):
        popup = NotificationPopup(message, kind, self._notif_container)
        self._notif_container.layout().addWidget(popup)
        popup.show()