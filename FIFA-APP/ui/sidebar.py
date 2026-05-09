"""
sidebar.py — Animated collapsible sidebar navigation.
"""
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QWidget, QSpacerItem, QSizePolicy
)
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, Signal, QSize
from PySide6.QtGui import QFont, QIcon

from utils.theme import (
    BG_SIDEBAR, ACCENT_CYAN, TEXT_PRIMARY, TEXT_SECONDARY,
    BORDER_COLOR, HOVER_COLOR, SELECTED_COLOR
)

NAV_ITEMS = [
    ("🏠", "Home",               0),
    ("⚽", "Match Prediction",    1),
    ("👤", "Player Evaluation",   2),
    ("⭐", "Rating & Recommend",  3),
    ("⚙️", "Settings",            4),
]


class SidebarButton(QPushButton):
    """Single sidebar navigation button."""

    def __init__(self, icon: str, label: str, page_idx: int, parent=None):
        super().__init__(parent)
        self.page_idx = page_idx
        self._icon_str = icon
        self._label = label
        self._active = False
        self.setFixedHeight(48)
        self.setCursor(Qt.PointingHandCursor)
        self._update_style()

    def set_active(self, active: bool):
        self._active = active
        self._update_style()

    def _update_style(self):
        if self._active:
            self.setObjectName("sidebarBtnActive")
            self.setText(f"  {self._icon_str}   {self._label}")
        else:
            self.setObjectName("sidebarBtn")
            self.setText(f"  {self._icon_str}   {self._label}")
        self.style().unpolish(self)
        self.style().polish(self)


class Sidebar(QFrame):
    """Collapsible left sidebar with navigation buttons."""

    page_changed = Signal(int)

    EXPANDED_W  = 220
    COLLAPSED_W = 56

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(self.EXPANDED_W)
        self._expanded = True
        self._buttons: list[SidebarButton] = []
        self._build_ui()

    # ── Build ──────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Logo bar ───────────────────────────────────────────
        logo_bar = QFrame()
        logo_bar.setFixedHeight(64)
        logo_bar.setStyleSheet(
            f"background: {BG_SIDEBAR}; border-bottom: 1px solid {BORDER_COLOR};"
        )
        logo_lay = QHBoxLayout(logo_bar)
        logo_lay.setContentsMargins(12, 0, 8, 0)

        self._logo_lbl = QLabel("⚽ FIFA Analytics")
        self._logo_lbl.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self._logo_lbl.setStyleSheet(f"color: {ACCENT_CYAN}; background: transparent;")

        self._toggle_btn = QPushButton("◀")
        self._toggle_btn.setFixedSize(28, 28)
        self._toggle_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {TEXT_SECONDARY};
                border: 1px solid {BORDER_COLOR}; border-radius: 4px; font-size: 10px;
            }}
            QPushButton:hover {{ color: {ACCENT_CYAN}; border-color: {ACCENT_CYAN}; }}
        """)
        self._toggle_btn.clicked.connect(self.toggle)

        logo_lay.addWidget(self._logo_lbl, 1)
        logo_lay.addWidget(self._toggle_btn)
        root.addWidget(logo_bar)

        # ── Nav buttons ────────────────────────────────────────
        self._nav_area = QWidget()
        nav_lay = QVBoxLayout(self._nav_area)
        nav_lay.setContentsMargins(8, 12, 8, 0)
        nav_lay.setSpacing(4)

        for icon, label, idx in NAV_ITEMS:
            btn = SidebarButton(icon, label, idx)
            btn.clicked.connect(lambda checked=False, i=idx: self._on_nav(i))
            self._buttons.append(btn)
            nav_lay.addWidget(btn)

        nav_lay.addStretch()
        root.addWidget(self._nav_area, 1)

        # ── Version footer ─────────────────────────────────────
        footer = QLabel("v1.0.0")
        footer.setAlignment(Qt.AlignCenter)
        footer.setFixedHeight(28)
        footer.setFont(QFont("Segoe UI", 8))
        footer.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent;")
        root.addWidget(footer)

        # Start with first button active
        self._set_active(0)

    # ── Slots ──────────────────────────────────────────────────────
    def _on_nav(self, page_idx: int):
        self._set_active(page_idx)
        self.page_changed.emit(page_idx)

    def _set_active(self, page_idx: int):
        for btn in self._buttons:
            btn.set_active(btn.page_idx == page_idx)

    def navigate_to(self, page_idx: int):
        """Called externally to sync active state."""
        self._set_active(page_idx)

    # ── Toggle collapse ────────────────────────────────────────────
    def toggle(self):
        target = self.COLLAPSED_W if self._expanded else self.EXPANDED_W
        self._anim = QPropertyAnimation(self, b"minimumWidth")
        self._anim.setDuration(250)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.setStartValue(self.width())
        self._anim.setEndValue(target)
        self._anim2 = QPropertyAnimation(self, b"maximumWidth")
        self._anim2.setDuration(250)
        self._anim2.setEasingCurve(QEasingCurve.OutCubic)
        self._anim2.setStartValue(self.width())
        self._anim2.setEndValue(target)
        self._anim.start()
        self._anim2.start()
        self._expanded = not self._expanded
        self._toggle_btn.setText("▶" if not self._expanded else "◀")
        self._logo_lbl.setVisible(self._expanded)

        # Show/hide labels in buttons
        for btn in self._buttons:
            if self._expanded:
                btn.setText(f"  {btn._icon_str}   {btn._label}")
            else:
                btn.setText(f"  {btn._icon_str}")
