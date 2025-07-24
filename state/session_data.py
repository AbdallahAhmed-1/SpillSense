# state/session_data.py

import pandas as pd
from typing import Optional, Any


class SessionData:
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all session data to initial state."""
        self.data: Optional[pd.DataFrame] = None
        self.cleaned_data: Optional[pd.DataFrame] = None
        self.model: Any = None
        self.state: dict = {}
        self.latest_predictions: Optional[pd.DataFrame] = None
        self.generated_texts: list[str] = []
        self.generated_plots: list[str] = []
        self.last_report_path: Optional[str] = None
        self.chat_history: list[tuple[str, str]] = []

        # ✅ Add this line
        self.latest_csv_path: Optional[str] = None

    def has_data(self) -> bool:
        return self.data is not None


# ✅ Global session instance to be used across the app
session = SessionData()
