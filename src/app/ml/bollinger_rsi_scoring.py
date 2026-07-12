import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.ml.scoring import Scoring
import pandas_ta as ta  # type: ignore


class BollingerRsiScoring(Scoring):
    """
    Score de retour à la moyenne (mean reversion) pour marché de côté.
    Achat près de la bande basse de Bollinger avec RSI survendu en reprise.
    df est le dataframe de chaque symbole à scorer.
    """
    df: pd.DataFrame
    name = "BollingerRsiScoring"

    def __init__(self, bb_length=20, bb_std=2.0, rsi_length=14):
        self.bb_length = bb_length
        self.bb_std = bb_std
        self.rsi_length = rsi_length
        self.df = None

    def score(self, df: pd.DataFrame) -> int:
        """
        Retourne un score entier (0-100) basé sur :
          - proximité de la bande basse de Bollinger (jusqu'à 50 pts)
          - niveau de survente du RSI (jusqu'à 35 pts)
          - reprise du RSI (creux en formation, +15 pts)
        """
        try:
            sma = df["close"].rolling(self.bb_length).mean()
            std = df["close"].rolling(self.bb_length).std()
            df["bb_mid"] = sma
            df["bb_low"] = sma - self.bb_std * std
            df["bb_high"] = sma + self.bb_std * std
            df["RSI"] = ta.rsi(df["close"], length=self.rsi_length)

            self.df = df

            close = df["close"].iloc[-1]
            bb_low = df["bb_low"].iloc[-1]
            bb_high = df["bb_high"].iloc[-1]
            rsi = df["RSI"].iloc[-1]

            if pd.isna(bb_low) or pd.isna(bb_high) or pd.isna(rsi) or bb_high <= bb_low:
                return 0

            score = 0
            band_width = bb_high - bb_low
            distance_pct = (close - bb_low) / band_width  # 0 = sur la bande basse, 1 = sur la bande haute

            if distance_pct <= 0.05:
                score += 50
            elif distance_pct <= 0.15:
                score += 30
            elif distance_pct <= 0.25:
                score += 15

            if rsi < 25:
                score += 35
            elif rsi < 30:
                score += 25
            elif rsi < 35:
                score += 15
            elif rsi < 40:
                score += 5

            if len(df) > 1:
                prev_rsi = df["RSI"].iloc[-2]
                if not pd.isna(prev_rsi) and rsi > prev_rsi:
                    score += 15

            return min(score, 100)

        except Exception as e:
            print(f"[ERREUR] Erreur lors du calcul BollingerRsiScoring: {e}")
            return 0
