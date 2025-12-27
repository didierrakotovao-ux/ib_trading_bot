import pandas as pd
class Scoring:
    name: str
    df: pd.DataFrame

    def score(self) -> int:
        """Score optionnel (0–100)"""
        return 0