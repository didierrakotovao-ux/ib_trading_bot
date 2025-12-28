
import pandas as pd
import numpy as np
from ml.scoring import Scoring
import yfinance as yf # type: ignore
from scipy.signal import argrelextrema # type: ignore
import pandas_ta as ta

class AdDivergenceScoring(Scoring):
    """Implémente un score basé sur la détection de divergences haussières entre le prix et l'indicateur A/D. df est le dataframe de chaque symbole à scorer."""
    df: pd.DataFrame
    name = "AdDivergenceScoring"

    def __init__(self):
        self.df = None
        
    def score(self, df: pd.DataFrame):
        """
        Implemente un score basé sur la détection de divergences haussières entre le prix et l'indicateur A/D.
        Utilise pandas_ta pour les indicateurs techniques (EMA, RSI, AD).
        Retourne un score entier (0–100).
        """
        try:
            df["EMA50"] = ta.ema(df["close"], length=50)
            df["EMA200"] = ta.ema(df["close"], length=200)
            df["RSI"] = ta.rsi(df["close"], length=14)
            df["AD"] = ta.ad(df["high"], df["low"], df["close"], df["volume"])

            self.df = df
            score = 0
            if self.detect_ad_bullish_divergence():
                score += 40
            if self.df["close"].iloc[-1] > self.df["EMA50"].iloc[-1] > self.df["EMA200"].iloc[-1]:
                score += 30
            if self.df["EMA50"].iloc[-1] > self.df["EMA200"].iloc[-1]:
                score += 10
            if 70 > self.df["RSI"].iloc[-1] > 30:
                score += 20

            return score

        except Exception as e:
            print(f"❌ Erreur lors du calcul des indicateurs avec pandas_ta: {e}")
            return 0

    
    def detect_ad_bullish_divergence(self, order=5, max_dist=6):
        try:
            if self.df is None or len(self.df) < 50:
                return False

            # S'assurer que l'index est un DateTimeIndex
            if not isinstance(self.df.index, pd.DatetimeIndex):
                self.df.index = pd.to_datetime(self.df.index)

            self.df["price_low"] = np.nan
            self.df["ad_low"] = np.nan

            price_idx = argrelextrema(self.df["low"].values, np.less, order=order)[0]
            ad_idx = argrelextrema(self.df["AD"].values, np.less, order=order)[0]

            self.df.loc[self.df.index[price_idx], "price_low"] = self.df["low"].iloc[price_idx]
            self.df.loc[self.df.index[ad_idx], "ad_low"] = self.df["AD"].iloc[ad_idx]

            price_lows = self.df.dropna(subset=["price_low"])
            ad_lows = self.df.dropna(subset=["ad_low"])

            if len(price_lows) < 2 or len(ad_lows) < 2:
                return False

            p1, p2 = price_lows.iloc[-2], price_lows.iloc[-1]

            # Lower Low prix
            if p2.price_low >= p1.price_low:
                return False

            # Pivot A/D proche
            delta_days = abs((ad_lows.index - p2.name) / np.timedelta64(1, 'D'))
            ad_candidates = ad_lows.loc[delta_days <= max_dist]

            if len(ad_candidates) < 2:
                return False

            ad1, ad2 = ad_candidates.iloc[-2], ad_candidates.iloc[-1]

            # Higher Low A/D
            if ad2.ad_low <= ad1.ad_low:
                return False

            idx = self.df.index.get_loc(p2.name)

            # Confirmation RSI
            if not (self.df["RSI"].iloc[idx-1] < 40 and self.df["RSI"].iloc[idx] > 40):
                return False

            return True

        except Exception as e:
            print(f"❌ Erreur lors de la détection de divergence: {e}")
            return False

