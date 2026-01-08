    
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.ml.scoring import Scoring
import yfinance as yf # type: ignore
from scipy.signal import argrelextrema # type: ignore
import pandas_ta as ta

class MomentumScoring(Scoring):
    df: pd.DataFrame
    name = "MomentumScoring"

    def __init__(self):
        self.df = None

    def score(self, df: pd.DataFrame):
        """
        Score momentum amélioré pour détecter les mouvements forts.
        Indicateurs: Volume, ADX, Breakout, RSI, Accélération ROC, MACD
        """
        try:
            # ============================================
            # 1. CALCUL DES INDICATEURS
            # ============================================

            # Rendements
            df["returns_1m"] = df["close"].pct_change(20)
            df["returns_3m"] = df["close"].pct_change(60)
            df["returns_6m"] = df["close"].pct_change(120)

            # MACD
            macd_data = ta.macd(df["close"])
            df["MACD_hist"] = macd_data["MACDh_12_26_9"]

            # ROC et accélération
            df["ROC_20"] = ta.roc(df["close"], length=20)
            df["ROC_5"] = ta.roc(df["close"], length=5)
            df["ROC_acceleration"] = df["ROC_5"].diff()

            # Volume relatif (vs moyenne 20 jours)
            df["volume_ma20"] = df["volume"].rolling(20).mean()
            df["volume_ratio"] = df["volume"] / (df["volume_ma20"] + 1e-6)

            # ADX - Force de la tendance
            adx_data = ta.adx(df["high"], df["low"], df["close"], length=14)
            df["ADX"] = adx_data["ADX_14"]
            df["DI_plus"] = adx_data["DMP_14"]
            df["DI_minus"] = adx_data["DMN_14"]

            # Breakout detection - Position vs plus haut 52 semaines
            df["high_52w"] = df["high"].rolling(252, min_periods=50).max()
            df["breakout_strength"] = df["close"] / (df["high_52w"] + 1e-6)

            # RSI
            df["RSI"] = ta.rsi(df["close"], length=14)

            # Volatilité pour Sharpe
            df["volatility_60d"] = df["close"].pct_change().rolling(60).std()
            df["sharpe_momentum"] = df["returns_3m"] / (df["volatility_60d"] * np.sqrt(60) + 1e-6)

            self.df = df

            # ============================================
            # 2. SCORING (100 points max)
            # ============================================
            score = 0

            # --- ADX: Force de la tendance (25 points) ---
            adx_val = self.df["ADX"].iloc[-1]
            di_plus = self.df["DI_plus"].iloc[-1]
            di_minus = self.df["DI_minus"].iloc[-1]

            if adx_val > 40 and di_plus > di_minus:
                score += 25  # Tendance très forte haussière
            elif adx_val > 25 and di_plus > di_minus:
                score += 18  # Tendance forte haussière
            elif adx_val > 20 and di_plus > di_minus:
                score += 10  # Tendance modérée haussière

            # --- Breakout: Proximité du plus haut (20 points) ---
            breakout = self.df["breakout_strength"].iloc[-1]
            if breakout >= 1.0:
                score += 20  # Nouveau plus haut !
            elif breakout >= 0.98:
                score += 15  # Très proche du plus haut
            elif breakout >= 0.95:
                score += 10  # Proche du plus haut
            elif breakout >= 0.90:
                score += 5

            # --- Volume: Confirmation du mouvement (15 points) ---
            vol_ratio = self.df["volume_ratio"].iloc[-1]
            vol_ratio_avg_5d = self.df["volume_ratio"].tail(5).mean()

            if vol_ratio > 2.0 and vol_ratio_avg_5d > 1.5:
                score += 15  # Volume très élevé et soutenu
            elif vol_ratio > 1.5 or vol_ratio_avg_5d > 1.3:
                score += 10  # Volume élevé
            elif vol_ratio > 1.2:
                score += 5   # Volume légèrement élevé

            # --- Accélération du momentum (15 points) ---
            roc_accel = self.df["ROC_acceleration"].iloc[-1]
            roc_5 = self.df["ROC_5"].iloc[-1]

            if roc_5 > 0 and roc_accel > 0.5:
                score += 15  # Momentum positif et accélère fortement
            elif roc_5 > 0 and roc_accel > 0:
                score += 10  # Momentum positif et accélère
            elif roc_5 > 2:
                score += 5   # Momentum positif fort

            # --- RSI: Zone de momentum sain (10 points) ---
            rsi_val = self.df["RSI"].iloc[-1]
            if 55 <= rsi_val <= 70:
                score += 10  # Zone idéale de momentum haussier
            elif 50 <= rsi_val < 55:
                score += 7   # Momentum naissant
            elif 70 < rsi_val <= 80:
                score += 5   # Fort mais attention surachat

            # --- MACD: Confirmation tendance (10 points) ---
            if len(self.df) > 1:
                macd_hist = self.df["MACD_hist"].iloc[-1]
                macd_hist_prev = self.df["MACD_hist"].iloc[-2]

                if macd_hist > 0 and macd_hist > macd_hist_prev:
                    score += 10  # MACD positif et croissant
                elif macd_hist > 0:
                    score += 5   # MACD positif

            # --- Rendements (5 points bonus) ---
            ret_3m = self.df["returns_3m"].iloc[-1]
            if ret_3m > 0.15:
                score += 5
            elif ret_3m > 0.10:
                score += 3

            return max(0, min(score, 100))

        except Exception as e:
            print(f"[ERREUR MomentumScoring] {e}")
            return 0