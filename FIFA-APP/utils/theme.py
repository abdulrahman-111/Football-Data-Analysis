"""
FIFA Career Mode inspired theme — colors, QSS stylesheet, and helpers.
"""

# ── Palette ──────────────────────────────────────────────────────
BG_DARK       = "#0A0E1A"       # main window background
BG_PANEL      = "#0F1628"       # panel / card background
BG_SIDEBAR    = "#080C18"       # sidebar background
BG_CARD       = "#141C30"       # elevated card
BG_INPUT      = "#1A2238"       # input field background
ACCENT_CYAN   = "#00D4FF"       # primary neon cyan
ACCENT_GREEN  = "#00FF88"       # success / positive
ACCENT_GOLD   = "#FFD700"       # gold / highlight
ACCENT_RED    = "#FF3B5C"       # danger / negative
ACCENT_ORANGE = "#FF8C00"       # warning
TEXT_PRIMARY  = "#FFFFFF"
TEXT_SECONDARY= "#8892A4"
TEXT_MUTED    = "#4A5568"
BORDER_COLOR  = "#1E2A45"
HOVER_COLOR   = "#1A2540"
SELECTED_COLOR= "#0D1F3C"

# ── QSS Master Stylesheet ─────────────────────────────────────────
QSS = f"""
/* ─── Global ─────────────────────────────────────────────────── */
QWidget {{
    background-color: {BG_DARK};
    color: {TEXT_PRIMARY};
    font-family: "Segoe UI", "Arial", sans-serif;
    font-size: 13px;
}}

QMainWindow {{
    background-color: {BG_DARK};
}}

/* ─── Scroll bars ─────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {BG_PANEL};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {ACCENT_CYAN};
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {BG_PANEL};
    height: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background: {ACCENT_CYAN};
    border-radius: 4px;
    min-width: 20px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ─── Buttons ─────────────────────────────────────────────────── */
QPushButton {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: 600;
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: {HOVER_COLOR};
    border: 1px solid {ACCENT_CYAN};
    color: {ACCENT_CYAN};
}}
QPushButton:pressed {{
    background-color: {SELECTED_COLOR};
}}

QPushButton#accentBtn {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {ACCENT_CYAN}, stop:1 #0090B8);
    color: {BG_DARK};
    border: none;
    font-weight: 700;
}}
QPushButton#accentBtn:hover {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #33DEFF, stop:1 {ACCENT_CYAN});
    color: {BG_DARK};
    border: none;
}}

QPushButton#greenBtn {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {ACCENT_GREEN}, stop:1 #00B860);
    color: {BG_DARK};
    border: none;
    font-weight: 700;
}}
QPushButton#greenBtn:hover {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #33FFaa, stop:1 {ACCENT_GREEN});
    border: none;
}}

QPushButton#dangerBtn {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {ACCENT_RED}, stop:1 #CC1F40);
    color: white;
    border: none;
    font-weight: 700;
}}

/* ─── Line Edits ──────────────────────────────────────────────── */
QLineEdit {{
    background-color: {BG_INPUT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 13px;
    selection-background-color: {ACCENT_CYAN};
}}
QLineEdit:focus {{
    border: 1px solid {ACCENT_CYAN};
}}
QLineEdit::placeholder {{
    color: {TEXT_MUTED};
}}

/* ─── ComboBox ────────────────────────────────────────────────── */
QComboBox {{
    background-color: {BG_INPUT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    padding: 7px 12px;
    font-size: 13px;
    min-width: 140px;
}}
QComboBox:focus {{ border: 1px solid {ACCENT_CYAN}; }}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid {TEXT_SECONDARY};
    width: 0; height: 0;
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_COLOR};
    selection-background-color: {HOVER_COLOR};
    outline: none;
}}

/* ─── Labels ──────────────────────────────────────────────────── */
QLabel#titleLabel {{
    font-size: 22px;
    font-weight: 700;
    color: {TEXT_PRIMARY};
}}
QLabel#subtitleLabel {{
    font-size: 14px;
    color: {TEXT_SECONDARY};
}}
QLabel#accentLabel {{
    font-size: 28px;
    font-weight: 800;
    color: {ACCENT_CYAN};
}}
QLabel#goldLabel {{
    font-size: 28px;
    font-weight: 800;
    color: {ACCENT_GOLD};
}}
QLabel#greenLabel {{
    color: {ACCENT_GREEN};
    font-weight: 700;
}}
QLabel#redLabel {{
    color: {ACCENT_RED};
    font-weight: 700;
}}

/* ─── Cards / Panels ─────────────────────────────────────────── */
QFrame#card {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_COLOR};
    border-radius: 10px;
}}
QFrame#glowCard {{
    background-color: {BG_CARD};
    border: 1px solid {ACCENT_CYAN};
    border-radius: 10px;
}}

/* ─── TabWidget ───────────────────────────────────────────────── */
QTabWidget::pane {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER_COLOR};
    border-radius: 8px;
}}
QTabBar::tab {{
    background-color: {BG_CARD};
    color: {TEXT_SECONDARY};
    padding: 9px 20px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: 600;
}}
QTabBar::tab:selected {{
    background-color: {BG_PANEL};
    color: {ACCENT_CYAN};
    border-bottom: 2px solid {ACCENT_CYAN};
}}
QTabBar::tab:hover:!selected {{
    background-color: {HOVER_COLOR};
    color: {TEXT_PRIMARY};
}}

/* ─── ProgressBar ─────────────────────────────────────────────── */
QProgressBar {{
    background-color: {BG_INPUT};
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {ACCENT_CYAN}, stop:1 {ACCENT_GREEN});
    border-radius: 4px;
}}

/* ─── Slider ──────────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    background: {BG_INPUT};
    height: 6px;
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT_CYAN};
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {ACCENT_CYAN}, stop:1 {ACCENT_GREEN});
    border-radius: 3px;
}}

/* ─── Table ───────────────────────────────────────────────────── */
QTableWidget {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER_COLOR};
    border-radius: 8px;
    gridline-color: {BORDER_COLOR};
    selection-background-color: {HOVER_COLOR};
    selection-color: {TEXT_PRIMARY};
    outline: none;
}}
QTableWidget::item {{
    padding: 8px 12px;
    border-bottom: 1px solid {BORDER_COLOR};
}}
QTableWidget::item:selected {{
    background-color: {HOVER_COLOR};
    color: {ACCENT_CYAN};
}}
QHeaderView::section {{
    background-color: {BG_CARD};
    color: {TEXT_SECONDARY};
    padding: 10px 12px;
    border: none;
    border-bottom: 2px solid {ACCENT_CYAN};
    font-weight: 700;
    font-size: 11px;
    text-transform: uppercase;
}}

/* ─── Sidebar ─────────────────────────────────────────────────── */
QFrame#sidebar {{
    background-color: {BG_SIDEBAR};
    border-right: 1px solid {BORDER_COLOR};
}}

QPushButton#sidebarBtn {{
    background-color: transparent;
    color: {TEXT_SECONDARY};
    border: none;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: left;
    font-size: 13px;
    font-weight: 600;
}}
QPushButton#sidebarBtn:hover {{
    background-color: {HOVER_COLOR};
    color: {TEXT_PRIMARY};
}}
QPushButton#sidebarBtnActive {{
    background-color: {SELECTED_COLOR};
    color: {ACCENT_CYAN};
    border: none;
    border-left: 3px solid {ACCENT_CYAN};
    border-radius: 0px;
    padding: 12px 16px;
    text-align: left;
    font-size: 13px;
    font-weight: 700;
}}

/* ─── Tooltip ─────────────────────────────────────────────────── */
QToolTip {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {ACCENT_CYAN};
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 12px;
}}

/* ─── Spinbox ─────────────────────────────────────────────────── */
QDoubleSpinBox, QSpinBox {{
    background-color: {BG_INPUT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 13px;
}}
QDoubleSpinBox:focus, QSpinBox:focus {{
    border: 1px solid {ACCENT_CYAN};
}}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QSpinBox::up-button, QSpinBox::down-button {{
    background-color: {BG_CARD};
    border: none;
    width: 18px;
}}

/* ─── GroupBox ────────────────────────────────────────────────── */
QGroupBox {{
    border: 1px solid {BORDER_COLOR};
    border-radius: 8px;
    margin-top: 14px;
    padding: 10px;
    font-weight: 700;
    color: {TEXT_SECONDARY};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: {ACCENT_CYAN};
    font-size: 12px;
}}

/* ─── TextEdit ────────────────────────────────────────────────── */
QTextEdit {{
    background-color: {BG_INPUT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    padding: 8px;
    font-size: 13px;
    line-height: 1.5;
}}
QTextEdit:focus {{
    border: 1px solid {ACCENT_CYAN};
}}

/* ─── Splitter ────────────────────────────────────────────────── */
QSplitter::handle {{
    background-color: {BORDER_COLOR};
    width: 1px;
}}
"""


def get_rating_color(rating: int) -> str:
    """Return hex color string based on FIFA-style rating."""
    if rating >= 85:
        return ACCENT_GOLD
    elif rating >= 75:
        return ACCENT_GREEN
    elif rating >= 65:
        return "#4FC3F7"
    else:
        return ACCENT_ORANGE


def get_risk_color(prob: float) -> str:
    """Return color for injury risk probability 0-1."""
    if prob < 0.25:
        return ACCENT_GREEN
    elif prob < 0.5:
        return ACCENT_ORANGE
    elif prob < 0.75:
        return "#FF6600"
    else:
        return ACCENT_RED
