"""
widgets.py — Reusable custom PySide6 widgets for the FIFA dashboard.
"""
from PySide6.QtWidgets import (
    QFrame, QLabel, QVBoxLayout, QHBoxLayout, QProgressBar,
    QWidget, QGraphicsOpacityEffect, QSizePolicy
)
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QSize, QTimer
from PySide6.QtGui import QFont, QPainter, QColor, QPen, QBrush, QPainterPath

from utils.theme import (
    BG_CARD, BG_PANEL, ACCENT_CYAN, ACCENT_GREEN, ACCENT_GOLD,
    ACCENT_RED, ACCENT_ORANGE, TEXT_PRIMARY, TEXT_SECONDARY,
    BORDER_COLOR, get_rating_color, get_risk_color
)


# ─────────────────────────────────────────────────────────────────
def batch_add(layout, widgets: list):
    """
    Add multiple widgets to a layout in one shot without triggering
    a repaint after each insertion. Eliminates the sequential pop-in
    effect when building panels with many child widgets.

    Usage:
        batch_add(self.main_layout, [card1, card2, card3, gauge])
    """
    parent_widget = layout.parentWidget()
    if parent_widget:
        parent_widget.setUpdatesEnabled(False)
    try:
        for w in widgets:
            layout.addWidget(w)
    finally:
        if parent_widget:
            parent_widget.setUpdatesEnabled(True)
            parent_widget.update()


# ─────────────────────────────────────────────────────────────────
class StatCard(QFrame):
    """Small KPI card with a title, value, and optional subtitle."""

    def __init__(self, title: str, value: str, subtitle: str = "",
                 color: str = ACCENT_CYAN, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setMinimumWidth(140)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(4)

        title_lbl = QLabel(title)
        title_lbl.setObjectName("subtitleLabel")
        title_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))

        value_lbl = QLabel(value)
        value_lbl.setFont(QFont("Segoe UI", 22, QFont.Bold))
        value_lbl.setStyleSheet(f"color: {color}; background: transparent; border: none;")

        layout.addWidget(title_lbl)
        layout.addWidget(value_lbl)

        if subtitle:
            sub = QLabel(subtitle)
            sub.setObjectName("subtitleLabel")
            sub.setFont(QFont("Segoe UI", 9))
            layout.addWidget(sub)

        # NOTE: QGraphicsOpacityEffect intentionally removed.
        # Attaching an effect — even at opacity 1.0 — forces Qt to re-render
        # the entire widget subtree into an offscreen pixmap on every repaint,
        # causing lag and nested-painter conflicts on any custom-painted children
        # (e.g. ProbabilityGauge). Add an effect only when you actually animate it.


# ─────────────────────────────────────────────────────────────────
class AttributeBar(QWidget):
    """FIFA-style labeled attribute progress bar."""

    def __init__(self, label: str, value: int, max_val: int = 100,
                 color: str = ACCENT_CYAN, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(10)

        lbl = QLabel(label)
        lbl.setFixedWidth(160)
        lbl.setFont(QFont("Segoe UI", 10))
        lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent;")

        bar = QProgressBar()
        bar.setRange(0, max_val)
        bar.setValue(value)
        bar.setFixedHeight(8)
        bar.setStyleSheet(f"""
            QProgressBar {{
                background: #1A2238; border: none; border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {color}, stop:1 {ACCENT_GREEN});
                border-radius: 4px;
            }}
        """)

        val_lbl = QLabel(str(value))
        val_lbl.setFixedWidth(35)
        val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        val_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        val_lbl.setStyleSheet(f"color: {color}; background: transparent;")

        layout.addWidget(lbl)
        layout.addWidget(bar, 1)
        layout.addWidget(val_lbl)

    def set_value(self, value: int):
        self.findChild(QProgressBar).setValue(value)


# ─────────────────────────────────────────────────────────────────
class PlayerCard(QFrame):
    """FIFA Ultimate Team style player card widget."""

    def __init__(self, name: str, position: str, rating: int,
                 nationality: str = "—", team: str = "—", parent=None):
        super().__init__(parent)
        self.setObjectName("glowCard")
        self.setFixedSize(200, 280)

        color = get_rating_color(rating)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Top banner
        banner = QFrame()
        banner.setFixedHeight(90)
        banner.setStyleSheet(f"""
            background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                stop:0 {color}30, stop:1 {BG_CARD});
            border-top-left-radius: 10px; border-top-right-radius: 10px;
        """)
        banner_layout = QVBoxLayout(banner)
        banner_layout.setAlignment(Qt.AlignCenter)

        rating_lbl = QLabel(str(rating))
        rating_lbl.setAlignment(Qt.AlignCenter)
        rating_lbl.setFont(QFont("Segoe UI", 36, QFont.Bold))
        rating_lbl.setStyleSheet(f"color: {color}; background: transparent;")

        pos_lbl = QLabel(position)
        pos_lbl.setAlignment(Qt.AlignCenter)
        pos_lbl.setFont(QFont("Segoe UI", 12, QFont.Bold))
        pos_lbl.setStyleSheet(f"color: {color}80; background: transparent;")

        banner_layout.addWidget(rating_lbl)
        banner_layout.addWidget(pos_lbl)
        layout.addWidget(banner)

        # Player info
        info = QFrame()
        info_layout = QVBoxLayout(info)
        info_layout.setAlignment(Qt.AlignCenter)
        info_layout.setSpacing(4)

        name_lbl = QLabel(name)
        name_lbl.setAlignment(Qt.AlignCenter)
        name_lbl.setWordWrap(True)
        name_lbl.setFont(QFont("Segoe UI", 11, QFont.Bold))
        name_lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; background: transparent;")

        team_lbl = QLabel(team)
        team_lbl.setAlignment(Qt.AlignCenter)
        team_lbl.setFont(QFont("Segoe UI", 9))
        team_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent;")

        nat_lbl = QLabel(f"🌐 {nationality}")
        nat_lbl.setAlignment(Qt.AlignCenter)
        nat_lbl.setFont(QFont("Segoe UI", 9))
        nat_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent;")

        info_layout.addSpacing(12)
        info_layout.addWidget(name_lbl)
        info_layout.addWidget(team_lbl)
        info_layout.addWidget(nat_lbl)
        layout.addWidget(info, 1)

        # Bottom bar
        bottom = QFrame()
        bottom.setFixedHeight(36)
        bottom.setStyleSheet(f"""
            background: {color}20;
            border-bottom-left-radius: 10px; border-bottom-right-radius: 10px;
        """)
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setAlignment(Qt.AlignCenter)
        badge = QLabel("⚽ FIFA Analytics")
        badge.setFont(QFont("Segoe UI", 8))
        badge.setStyleSheet(f"color: {color}; background: transparent;")
        bottom_layout.addWidget(badge)
        layout.addWidget(bottom)


# ─────────────────────────────────────────────────────────────────
class ProbabilityGauge(QWidget):
    """Animated semi-circle probability gauge."""

    def __init__(self, label: str, value: float = 0.5, parent=None):
        super().__init__(parent)
        self.label = label
        self._value = value
        self.setMinimumSize(180, 120)

    def setValue(self, v: float):
        self._value = max(0.0, min(1.0, v))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w // 2, h - 20
        r = min(w, h * 2) // 2 - 10

        # Background arc
        bg_pen = QPen(QColor(BORDER_COLOR), 10, Qt.SolidLine, Qt.RoundCap)
        p.setPen(bg_pen)
        p.drawArc(int(cx - r), int(cy - r), int(r * 2), int(r * 2), 0 * 16, 180 * 16)

        # Value arc
        if self._value > 0.01:
            clr = (ACCENT_GREEN if self._value >= 0.6
                   else ACCENT_ORANGE if self._value >= 0.4
                   else ACCENT_RED)
            val_pen = QPen(QColor(clr), 10, Qt.SolidLine, Qt.RoundCap)
            p.setPen(val_pen)
            span = int(self._value * 180)
            p.drawArc(int(cx - r), int(cy - r), int(r * 2), int(r * 2), 0 * 16, span * 16)
            clr_val = clr
        else:
            clr_val = ACCENT_RED

        # Text
        p.setPen(QColor(clr_val))
        p.setFont(QFont("Segoe UI", 18, QFont.Bold))
        p.drawText(0, cy - 20, w, 30, Qt.AlignCenter, f"{self._value * 100:.0f}%")
        p.setPen(QColor(TEXT_SECONDARY))
        p.setFont(QFont("Segoe UI", 9))
        p.drawText(0, cy + 5, w, 20, Qt.AlignCenter, self.label)
        # NOTE: Never call p.end() inside paintEvent — Qt owns the painter lifecycle here


# ─────────────────────────────────────────────────────────────────
class SectionHeader(QLabel):
    """Styled section header with cyan underline accent."""

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setFont(QFont("Segoe UI", 15, QFont.Bold))
        self.setStyleSheet(f"""
            color: {TEXT_PRIMARY};
            border-bottom: 2px solid {ACCENT_CYAN};
            padding-bottom: 6px;
            background: transparent;
        """)


# ─────────────────────────────────────────────────────────────────
class NotificationPopup(QFrame):
    """Toast notification that auto-hides after 3s."""

    def __init__(self, message: str, kind: str = "info", parent=None):
        super().__init__(parent)
        colors_map = {"info": ACCENT_CYAN, "success": ACCENT_GREEN,
                      "warning": ACCENT_ORANGE, "error": ACCENT_RED}
        color = colors_map.get(kind, ACCENT_CYAN)
        self.setStyleSheet(f"""
            QFrame {{
                background: {BG_CARD};
                border-left: 4px solid {color};
                border-radius: 8px;
                padding: 4px;
            }}
        """)
        self.setFixedWidth(320)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lbl = QLabel(message)
        lbl.setWordWrap(True)
        lbl.setFont(QFont("Segoe UI", 10))
        lbl.setStyleSheet("background: transparent; color: white;")
        lay.addWidget(lbl)

        # Effect is kept here because it IS actively animated by _fade_out
        self._effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._effect)
        self._effect.setOpacity(1.0)

        QTimer.singleShot(3000, self._fade_out)

    def _fade_out(self):
        self._anim = QPropertyAnimation(self._effect, b"opacity")
        self._anim.setDuration(600)
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.0)
        self._anim.finished.connect(self.deleteLater)
        self._anim.start()


# ─────────────────────────────────────────────────────────────────
class LoadingOverlay(QWidget):
    """Full-screen dark overlay with spinning indicator text."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setStyleSheet(f"background: {BG_CARD}CC;")
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        self._lbl = QLabel("⏳ Analyzing...")
        self._lbl.setAlignment(Qt.AlignCenter)
        self._lbl.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self._lbl.setStyleSheet(f"color: {ACCENT_CYAN}; background: transparent;")
        lay.addWidget(self._lbl)

        self._dots = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(400)

    def _tick(self):
        self._dots = (self._dots + 1) % 4
        self._lbl.setText("⏳ Analyzing" + "." * self._dots)

    def showEvent(self, e):
        super().showEvent(e)
        if self.parent():
            self.setGeometry(self.parent().rect())

    def stop(self):
        self._timer.stop()
        self.hide()