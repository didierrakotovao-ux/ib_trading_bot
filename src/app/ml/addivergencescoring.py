import pandas as pd
import numpy as np
from ml.scoring import Scoring
import yfinance as yf # type: ignore
from scipy.signal import argrelextrema # type: ignore

class AdDivergenceScoring(Scoring):
    """Implémente un score basé sur la détection de divergences haussières entre le prix et l'indicateur A/D. df est le dataframe de chaque symbole à scorer."""
    df: pd.DataFrame
    name = "AdDivergenceScoring"

    def __init__(self):
        self.df = None
        
    def score(self, df: pd.DataFrame):
        """
        Implemente un score basé sur la détection de divergences haussières entre le prix et l'indicateur A/D.
        prend en compte aussi la confirmation par le RSI.
        et les ema50 et ema200 pour la tendance générale.
        Retourne un score entier (0–100).
        """
        # Placeholder for actual scoring logic
        try:
            df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()
            df["EMA200"] = df["close"].ewm(span=200, adjust=False).mean()
            delta = df["close"].diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            avg_gain = up.rolling(window=14).mean()
            avg_loss = down.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df["RSI"] = 100 - (100 / (1 + rs))
            mfv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"])
            mfv = mfv.fillna(0)
            mfv_volume = mfv * df["volume"]
            df["AD"] = mfv_volume.cumsum()

        except Exception as e:
            print(f"❌ Erreur lors du calcul des indicateurs: {e}")
            return 0
        
        self.df = df
        score = 0
        if(self.detect_ad_bullish_divergence()):
            score += 40
        if self.df["close"].iloc[-1] > self.df["EMA50"].iloc[-1] > self.df["EMA200"].iloc[-1]:
            score += 30
        if self.df["EMA50"].iloc[-1] > self.df["EMA200"].iloc[-1]:
            score += 10
        if 70 >self.df["RSI"].iloc[-1] > 30:
            score += 20

        return score
    
    def detect_ad_bullish_divergence(self, order=5, max_dist=6):
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
        ad_candidates = ad_lows.loc[
            abs((ad_lows.index - p2.name).days) <= max_dist
        ]

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
