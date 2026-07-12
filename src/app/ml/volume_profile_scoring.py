import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.ml.scoring import Scoring
import pandas_ta as ta  # type: ignore


class VolumeProfileScoring(Scoring):
    """
    Score basé sur le Volume Profile (Volume at Price) combiné à un filtre de régime ADX.
    Achat à l'approche d'un nœud de support à fort volume (HVN) quand le marché
    est confirmé en range (ADX faible). df est le dataframe de chaque symbole à scorer.

    Approximation du profil de volume à partir de barres OHLCV journalières :
    le volume de chaque jour est réparti proportionnellement sur les buckets de prix
    chevauchant l'intervalle [low, high] de ce jour (pas de données intrajournalières).
    """
    df: pd.DataFrame
    name = "VolumeProfileScoring"

    def __init__(self, vp_window=60, n_bins=24, adx_threshold=20.0, hvn_percentile=70):
        self.vp_window = vp_window
        self.n_bins = n_bins
        self.adx_threshold = adx_threshold
        self.hvn_percentile = hvn_percentile
        self.df = None

    def _volume_profile(self, window: pd.DataFrame):
        """Calcule (bin_centers, bin_volume) sur la fenêtre donnée, vectorisé numpy."""
        low = window["low"].values
        high = window["high"].values
        vol = window["volume"].values

        price_min = low.min()
        price_max = high.max()
        if price_max <= price_min:
            return None, None

        bin_edges = np.linspace(price_min, price_max, self.n_bins + 1)
        bin_volume = np.zeros(self.n_bins)
        rng = high - low
        rng_safe = np.where(rng <= 0, 1e-9, rng)

        for i in range(self.n_bins):
            b_lo, b_hi = bin_edges[i], bin_edges[i + 1]
            overlap = np.clip(np.minimum(high, b_hi) - np.maximum(low, b_lo), 0, None)
            bin_volume[i] = np.sum(vol * overlap / rng_safe)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, bin_volume

    def score(self, df: pd.DataFrame) -> int:
        """
        Retourne un score entier (0-100) basé sur :
          - confirmation de régime range via ADX (jusqu'à 35 pts)
          - proximité d'un nœud de support à fort volume (HVN) (jusqu'à 40 pts)
          - RSI neutre/bas, pas de surachat (jusqu'à 15 pts)
          - pas de cassure du bas de la fenêtre (10 pts)
        """
        try:
            adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
            df["ADX"] = adx_df["ADX_14"] if adx_df is not None else None
            df["RSI"] = ta.rsi(df["close"], length=14)

            self.df = df

            if len(df) < self.vp_window:
                return 0

            window = df.iloc[-self.vp_window:]
            close = df["close"].iloc[-1]
            adx = df["ADX"].iloc[-1]
            rsi = df["RSI"].iloc[-1]

            if pd.isna(adx) or pd.isna(rsi):
                return 0

            bin_centers, bin_volume = self._volume_profile(window)
            if bin_centers is None:
                return 0

            nonzero = bin_volume[bin_volume > 0]
            if len(nonzero) == 0:
                return 0
            hvn_threshold = np.percentile(nonzero, self.hvn_percentile)
            hvn_mask = bin_volume >= hvn_threshold
            hvn_prices = bin_centers[hvn_mask]

            supports = hvn_prices[hvn_prices <= close]
            nearest_support = supports.max() if len(supports) else None

            score = 0
            if adx < self.adx_threshold:
                score += 35

            if nearest_support is not None and close > 0:
                dist_pct = (close - nearest_support) / close
                if dist_pct <= 0.015:
                    score += 40
                elif dist_pct <= 0.03:
                    score += 25
                elif dist_pct <= 0.05:
                    score += 10

            if 30 < rsi < 55:
                score += 15

            if close >= window["low"].min() * 1.01:
                score += 10

            return min(score, 100)

        except Exception as e:
            print(f"[ERREUR] Erreur lors du calcul VolumeProfileScoring: {e}")
            return 0
