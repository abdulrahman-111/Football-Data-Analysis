"""
home_page.py — Main dashboard home page, FIFA Career Mode style.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea, QGridLayout, QPushButton, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

from utils.theme import (
    BG_DARK, BG_CARD, BG_PANEL, ACCENT_CYAN, ACCENT_GREEN,
    ACCENT_GOLD, ACCENT_RED, ACCENT_ORANGE, TEXT_PRIMARY,
    TEXT_SECONDARY, BORDER_COLOR
)
from ui.widgets.widgets import StatCard, SectionHeader


QUICK_STATS = [
    ("Total Players",   "18,944",  "+1,204 this month",  ACCENT_CYAN),
    ("Avg Rating",      "72.4",    "League average",      ACCENT_GREEN),
    ("Market Value",    "€2.1B",   "Total squad value",   ACCENT_GOLD),
    ("Injury Risk",     "23%",     "League average risk", ACCENT_ORANGE),
]

RECENT_MATCHES = [
    ("Real Madrid",      "3 – 1",  "Barcelona",       ACCENT_GREEN),
    ("Man City",         "2 – 2",  "Arsenal",         ACCENT_CYAN),
    ("Bayern Munich",    "4 – 0",  "Wolfsburg",       ACCENT_GREEN),
    ("PSG",              "1 – 2",  "Lyon",            ACCENT_RED),
    ("Liverpool",        "3 – 0",  "Everton",         ACCENT_GREEN),
]

TOP_SCORERS = [
    ("Lionel Messi",   "RW", "Miami",       28, ACCENT_GOLD),
    ("Kylian Mbappé",    "FW", "Real Madrid",    25, ACCENT_CYAN),
    ("Mohamed Salah",    "RW", "Liverpool",      22, ACCENT_GREEN),
    ("Harry Kane",       "ST", "Bayern Munich",  21, ACCENT_CYAN),
    ("Lamine Yamal", "RW", "Barcelona",    19, ACCENT_ORANGE),
]


class HomePage(QWidget):
    """FIFA Career Mode style home / overview dashboard."""

    def __init__(self, switch_page_cb, parent=None):
        super().__init__(parent)
        self._switch = switch_page_cb
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(24)

        # ── Freeze container repaints while adding all children ────
        container.setUpdatesEnabled(False)
        try:
            layout.addWidget(self._hero_banner())
            layout.addWidget(SectionHeader("📊  Overview"))

            stats_row = QHBoxLayout()
            stats_row.setSpacing(12)
            for title, val, sub, color in QUICK_STATS:
                stats_row.addWidget(StatCard(title, val, sub, color))
            layout.addLayout(stats_row)

            mid = QHBoxLayout()
            mid.setSpacing(16)
            mid.addWidget(self._recent_matches_panel(), 1)
            mid.addWidget(self._top_scorers_panel(), 1)
            layout.addLayout(mid)

            layout.addWidget(SectionHeader("🚀  Quick Actions"))
            layout.addWidget(self._quick_actions())
            layout.addStretch()
        finally:
            container.setUpdatesEnabled(True)

        scroll.setWidget(container)
        root.addWidget(scroll)

    def _hero_banner(self) -> QFrame:
        banner = QFrame()
        banner.setFixedHeight(130)
        banner.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0A1628, stop:0.5 #0D2040, stop:1 #071020
                );
                border: 1px solid {ACCENT_CYAN}40;
                border-radius: 12px;
            }}
        """)
        lay = QHBoxLayout(banner)
        lay.setContentsMargins(28, 0, 28, 0)

        left = QVBoxLayout()
        title = QLabel("⚽  FIFA Analytics Dashboard")
        title.setFont(QFont("Segoe UI", 22, QFont.Bold))
        title.setStyleSheet(f"color: {ACCENT_CYAN}; background: transparent;")
        sub = QLabel("AI-powered football intelligence — Career Mode Edition")
        sub.setFont(QFont("Segoe UI", 11))
        sub.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent;")
        left.addWidget(title)
        left.addWidget(sub)

        right = QVBoxLayout()
        right.setAlignment(Qt.AlignCenter)
        badge = QLabel("SEASON\n2024/25")
        badge.setAlignment(Qt.AlignCenter)
        badge.setFont(QFont("Segoe UI", 14, QFont.Bold))
        badge.setStyleSheet(f"""
            color: {ACCENT_GOLD};
            background: {ACCENT_GOLD}18;
            border: 1px solid {ACCENT_GOLD}50;
            border-radius: 8px; padding: 10px 18px;
        """)
        right.addWidget(badge)

        lay.addLayout(left, 1)
        lay.addLayout(right)
        return banner

    def _recent_matches_panel(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        hdr = QLabel("🏟️  Recent Matches")
        hdr.setFont(QFont("Segoe UI", 13, QFont.Bold))
        hdr.setStyleSheet(f"color: {TEXT_PRIMARY}; background: transparent;")
        layout.addWidget(hdr)

        for home, score, away, color in RECENT_MATCHES:
            row = QFrame()
            row.setStyleSheet(f"""
                QFrame {{ background: {BG_PANEL}; border-radius: 6px;
                          border-left: 3px solid {color}; }}
            """)
            row_lay = QHBoxLayout(row)
            row_lay.setContentsMargins(10, 6, 10, 6)

            home_lbl = QLabel(home)
            home_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
            home_lbl.setStyleSheet("background: transparent; color: white;")

            score_lbl = QLabel(score)
            score_lbl.setAlignment(Qt.AlignCenter)
            score_lbl.setFont(QFont("Segoe UI", 11, QFont.Bold))
            score_lbl.setFixedWidth(60)
            score_lbl.setStyleSheet(f"color: {color}; background: transparent;")

            away_lbl = QLabel(away)
            away_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            away_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
            away_lbl.setStyleSheet("background: transparent; color: white;")

            row_lay.addWidget(home_lbl, 1)
            row_lay.addWidget(score_lbl)
            row_lay.addWidget(away_lbl, 1)
            layout.addWidget(row)

        return card

    def _top_scorers_panel(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        hdr = QLabel("🥇  Top Scorers")
        hdr.setFont(QFont("Segoe UI", 13, QFont.Bold))
        hdr.setStyleSheet(f"color: {TEXT_PRIMARY}; background: transparent;")
        layout.addWidget(hdr)

        for rank, (name, pos, team, goals, color) in enumerate(TOP_SCORERS, 1):
            row = QFrame()
            row.setStyleSheet(f"""
                QFrame {{ background: {BG_PANEL}; border-radius: 6px; }}
            """)
            row_lay = QHBoxLayout(row)
            row_lay.setContentsMargins(10, 6, 10, 6)
            row_lay.setSpacing(10)

            rank_lbl = QLabel(f"#{rank}")
            rank_lbl.setFixedWidth(28)
            rank_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
            rank_lbl.setStyleSheet(f"color: {color}; background: transparent;")

            pos_badge = QLabel(pos)
            pos_badge.setFixedWidth(38)
            pos_badge.setAlignment(Qt.AlignCenter)
            pos_badge.setFont(QFont("Segoe UI", 8, QFont.Bold))
            pos_badge.setStyleSheet(f"""
                color: {color}; background: {color}20;
                border: 1px solid {color}50; border-radius: 4px;
                padding: 2px 4px;
            """)

            name_lbl = QLabel(name)
            name_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
            name_lbl.setStyleSheet("background: transparent; color: white;")

            team_lbl = QLabel(team)
            team_lbl.setFont(QFont("Segoe UI", 9))
            team_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent;")

            goals_lbl = QLabel(f"⚽ {goals}")
            goals_lbl.setFont(QFont("Segoe UI", 11, QFont.Bold))
            goals_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            goals_lbl.setStyleSheet(f"color: {color}; background: transparent;")

            info = QVBoxLayout()
            info.setSpacing(0)
            info.addWidget(name_lbl)
            info.addWidget(team_lbl)

            row_lay.addWidget(rank_lbl)
            row_lay.addWidget(pos_badge)
            row_lay.addLayout(info, 1)
            row_lay.addWidget(goals_lbl)
            layout.addWidget(row)

        return card

    def _quick_actions(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setSpacing(14)
        lay.setContentsMargins(0, 0, 0, 0)

        actions = [
            ("⚽  Predict Match",   1, ACCENT_CYAN),
            ("👤  Evaluate Player", 2, ACCENT_GREEN),
            ("⭐  Rate & Recommend",3, ACCENT_GOLD),
        ]
        for label, page, color in actions:
            btn = QPushButton(label)
            btn.setFixedHeight(48)
            btn.setFont(QFont("Segoe UI", 12, QFont.Bold))
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                        stop:0 {color}, stop:1 {color}99);
                    color: #0A0E1A; border: none; border-radius: 8px;
                    padding: 0 24px; font-weight: 700;
                }}
                QPushButton:hover {{ background: {color}; color: #0A0E1A; }}
            """)
            btn.clicked.connect(lambda checked=False, p=page: self._switch(p))
            lay.addWidget(btn)
        return w