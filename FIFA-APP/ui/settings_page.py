"""
settings_page.py — Application settings page.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QComboBox, QScrollArea, QGroupBox, QCheckBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from utils.theme import (
    ACCENT_CYAN, ACCENT_GREEN, ACCENT_GOLD, ACCENT_RED,
    TEXT_PRIMARY, TEXT_SECONDARY, BG_CARD, BORDER_COLOR
)
from ui.widgets.widgets import SectionHeader


class SettingsPage(QWidget):
    notify = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(28, 24, 28, 24)
        lay.setSpacing(20)

        lay.addWidget(SectionHeader("⚙️  Settings"))

        # ── Appearance ─────────────────────────────────────────
        app_group = QGroupBox("🎨  Appearance")
        app_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        ag = QVBoxLayout(app_group)
        ag.setSpacing(12)

        theme_row = QHBoxLayout()
        theme_lbl = QLabel("Theme Accent")
        theme_lbl.setFont(QFont("Segoe UI", 10))
        theme_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent;")
        theme_combo = QComboBox()
        theme_combo.addItems(["Cyan (Default)", "Green", "Gold", "Red"])
        theme_row.addWidget(theme_lbl)
        theme_row.addWidget(theme_combo, 1)
        ag.addLayout(theme_row)

        anim_check = QCheckBox("Enable animations")
        anim_check.setChecked(True)
        anim_check.setFont(QFont("Segoe UI", 10))
        anim_check.setStyleSheet("color: white;")
        ag.addWidget(anim_check)

        lay.addWidget(app_group)

        # ── Models ─────────────────────────────────────────────
        model_group = QGroupBox("🤖  AI Models")
        model_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        mg = QVBoxLayout(model_group)
        mg.setSpacing(12)

        import os
        BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODELS = os.path.join(BASE, "models")

        model_files = [
            ("Goals Model",      "performance_prediction_goals_next_model.pkl"),
            ("Assists Model",    "performance_prediction_assists_next_model.pkl"),
            ("Injury Model",     "Injury_classifier_model.pkl"),
            ("Transfer Model",   "transfer_value_prediction_model.pkl"),
            ("Rating Model",     "overall_rating_prediction.pkl"),
            ("Position Model",   "position_classification.pkl"),
            ("Recommender",      "player_recommender.pkl"),
        ]

        for name, fname in model_files:
            row = QFrame()
            row.setStyleSheet(f"""
                QFrame {{
                    background: {BG_CARD};
                    border-radius: 6px;
                    border: 1px solid {BORDER_COLOR};
                }}
            """)
            rl = QHBoxLayout(row)
            rl.setContentsMargins(12, 8, 12, 8)

            n_lbl = QLabel(name)
            n_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
            n_lbl.setStyleSheet("background: transparent; color: white;")

            path = os.path.join(MODELS, fname)
            exists = os.path.exists(path)
            status_lbl = QLabel("✅ Loaded" if exists else "❌ Not Found")
            status_lbl.setFont(QFont("Segoe UI", 9))
            status_lbl.setStyleSheet(
                f"background: transparent; color: {'#00FF88' if exists else '#FF3B5C'};"
            )

            rl.addWidget(n_lbl, 1)
            rl.addWidget(status_lbl)
            mg.addWidget(row)

        lay.addWidget(model_group)

        # ── About ──────────────────────────────────────────────
        about_group = QGroupBox("ℹ️  About")
        about_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        abg = QVBoxLayout(about_group)

        about_lbl = QLabel(
            "FIFA Analytics Dashboard  v1.0.0\n\n"
            "A PySide6 desktop application powered by ML models for\n"
            "football analytics — match prediction, player evaluation,\n"
            "rating prediction and player recommendations.\n\n"
            "Models: LightGBM · Random Forest · KNN · Logistic Regression\n"
            "GUI: PySide6 · Matplotlib · ReportLab"
        )
        about_lbl.setFont(QFont("Segoe UI", 10))
        about_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent;")
        abg.addWidget(about_lbl)
        lay.addWidget(about_group)

        lay.addStretch()
        scroll.setWidget(container)
        root.addWidget(scroll)
