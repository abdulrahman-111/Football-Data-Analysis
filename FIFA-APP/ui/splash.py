"""
splash.py — FIFA-themed animated splash screen.
"""
from PySide6.QtWidgets import QSplashScreen, QWidget, QVBoxLayout, QLabel, QProgressBar
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QColor, QPainter, QLinearGradient, QPixmap

from utils.theme import BG_DARK, BG_PANEL, ACCENT_CYAN, ACCENT_GREEN, TEXT_PRIMARY


class SplashScreen(QSplashScreen):
    """Animated FIFA-style splash screen with loading bar."""

    def __init__(self):
        # Create a background pixmap
        pm = QPixmap(680, 380)
        pm.fill(QColor(BG_DARK))
        super().__init__(pm, Qt.WindowStaysOnTopHint)
        self.setWindowFlag(Qt.FramelessWindowHint)
        self._progress = 0
        self._build()

    def _build(self):
        self._container = QWidget(self)
        self._container.setGeometry(0, 0, 680, 380)
        self._container.setStyleSheet(f"""
            background: transparent;
        """)
        lay = QVBoxLayout(self._container)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(14)

        # Ball icon
        ball = QLabel("⚽")
        ball.setAlignment(Qt.AlignCenter)
        ball.setFont(QFont("Segoe UI", 56))
        ball.setStyleSheet("background: transparent; color: white;")
        lay.addWidget(ball)

        # Title
        title = QLabel("FIFA Analytics Dashboard")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 26, QFont.Bold))
        title.setStyleSheet(f"color: {ACCENT_CYAN}; background: transparent;")
        lay.addWidget(title)

        # Subtitle
        sub = QLabel("AI-Powered Football Intelligence Platform")
        sub.setAlignment(Qt.AlignCenter)
        sub.setFont(QFont("Segoe UI", 12))
        sub.setStyleSheet(f"color: #8892A4; background: transparent;")
        lay.addWidget(sub)

        lay.addSpacing(12)

        # Progress bar
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setFixedSize(440, 8)
        self._bar.setTextVisible(False)
        self._bar.setStyleSheet(f"""
            QProgressBar {{
                background: #1A2238; border: none; border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {ACCENT_CYAN}, stop:1 {ACCENT_GREEN});
                border-radius: 4px;
            }}
        """)
        lay.addWidget(self._bar, alignment=Qt.AlignCenter)

        # Status
        self._status_lbl = QLabel("Loading models…")
        self._status_lbl.setAlignment(Qt.AlignCenter)
        self._status_lbl.setFont(QFont("Segoe UI", 9))
        self._status_lbl.setStyleSheet(f"color: #4A5568; background: transparent;")
        lay.addWidget(self._status_lbl)

        lay.addSpacing(8)
        version = QLabel("v1.0.0")
        version.setAlignment(Qt.AlignCenter)
        version.setFont(QFont("Segoe UI", 8))
        version.setStyleSheet("color: #2A3548; background: transparent;")
        lay.addWidget(version)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # Draw gradient background
        grad = QLinearGradient(0, 0, 680, 380)
        grad.setColorAt(0.0, QColor("#070C18"))
        grad.setColorAt(0.5, QColor("#0A1022"))
        grad.setColorAt(1.0, QColor("#060A14"))
        painter.fillRect(self.rect(), grad)
        # Cyan top border
        painter.setPen(QColor(ACCENT_CYAN))
        painter.drawLine(0, 0, 680, 0)
        painter.end()

    def advance(self, value: int, status: str = ""):
        self._progress = value
        self._bar.setValue(value)
        if status:
            self._status_lbl.setText(status)
        self.repaint()
