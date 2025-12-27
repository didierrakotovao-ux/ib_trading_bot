import pandas as pd
import numpy as np
from app.ml.scoring import Scoring
import yfinance as yf # type: ignore
from scipy.signal import argrelextrema # type: ignore

class AdDivergenceScoring(Scoring):
    """Implémente un score basé sur la détection de divergences haussières entre le prix et l'indicateur A/D. df est le dataframe de chaque symbole à scorer."""
    df: pd.DataFrame
    name = "AdDivergenceScoring"

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        
    def score(self):
        """
        Implemente un score basé sur la détection de divergences haussières entre le prix et l'indicateur A/D.
        prend en compte aussi la confirmation par le RSI.
        et les ema50 et ema200 pour la tendance générale.
        Retourne un score entier (0–100).
        """
        # Placeholder for actual scoring logic
        score = 0
        if(self.detect_ad_bullish_divergence()):
            score += 40
        if self.df["Close"].iloc[-1] > self.df["EMA50"].iloc[-1] > self.df["EMA200"].iloc[-1]:
            score += 30
        if self.df["EMA50"].iloc[-1] > self.df["EMA200"].iloc[-1]:
            score += 10
        if 70 >self.df["RSI"].iloc[-1] > 30:
            score += 20

        return score
    
    def detect_ad_bullish_divergence(self, order=5, max_dist=6):
        self.df["price_low"] = np.nan
        self.df["ad_low"] = np.nan

        price_idx = argrelextrema(self.df["Low"].values, np.less, order=order)[0]
        ad_idx = argrelextrema(self.df["AD"].values, np.less, order=order)[0]

        self.df.loc[self.df.index[price_idx], "price_low"] = self.df["Low"].iloc[price_idx]
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
