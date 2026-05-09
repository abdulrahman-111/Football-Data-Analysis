"""
main.py — Entry point for the FIFA Analytics Dashboard.

Run with:
    python main.py
"""
import sys
import os

# ── Make imports work from any cwd ────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont

from utils.theme import QSS
from ui.splash import SplashScreen
from ui.dashboard import Dashboard
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate
from reportlab.lib.pagesizes import letter


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("FIFA Analytics Dashboard")
    app.setOrganizationName("FIFA Analytics")
    app.setStyle("Fusion")

    # Global stylesheet
    #app.setStyleSheet(QSS)

    # Default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # ── Splash screen ─────────────────────────────────────────
    splash = SplashScreen()
    splash.show()
    app.processEvents()

    # Simulate loading stages
    stages = [
        (15,  "Initializing UI framework…"),
        (30,  "Loading ML models…"),
        (55,  "Preparing player database…"),
        (75,  "Building analytics engine…"),
        (90,  "Applying theme…"),
        (100, "Ready!"),
    ]

    window = None

    def load_stage(idx=0):
        nonlocal window
        if idx < len(stages):
            val, msg = stages[idx]
            splash.advance(val, msg)
            QTimer.singleShot(220, lambda: load_stage(idx + 1))
        else:
            window = Dashboard()
            window.show()
            splash.finish(window)

    QTimer.singleShot(100, lambda: load_stage(0))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
