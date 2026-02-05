"""
ML Earnings Scoring - Scoring avec le modèle earnings momentum (SUE/CAR3).
"""
import pandas as pd
import numpy as np
import sqlite3
import joblib
from pathlib import Path
from typing import Optional, List

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.ml.scoring import Scoring
from src.app.ml.earnings_features import EarningsFeatures


class EarningsMLScoring(Scoring):
    """
    Scoring utilisant le modèle earnings momentum avec features
    SUE, CAR3, smoothness, saisonnalité et secteur.
    """
    name = "EarningsMLScoring"
    df: pd.DataFrame

    def __init__(self, model_path: str = "models/earnings_momentum_model.pkl",
                 db_path: str = "trading_data.db"):
        self.df = None
        self.model = None
        self.scaler = None
        self.model_loaded = False
        # Résoudre les chemins par rapport à la racine du projet
        project_root = Path(__file__).parent.parent.parent.parent
        self.model_path = project_root / model_path
        self.db_path = str(project_root / db_path)

        # Colonnes de features
        self.feature_columns: List[str] = []
        self.base_feature_columns: List[str] = []
        self.smooth_feature_columns: List[str] = []
        self.earnings_feature_columns: List[str] = []
        self.sector_columns: List[str] = []

        # Cache des métadonnées
        self._sector_cache: dict = {}
        self._earnings_cache: dict = {}

        # Earnings features helper
        self.earnings_features = EarningsFeatures(self.db_path)

        self._load_model()
        self._load_sector_cache()

    def _load_model(self):
        """Charge le modèle ML earnings momentum."""
        try:
            if self.model_path.exists():
                data = joblib.load(self.model_path)
                self.model = data['model']
                self.scaler = data['scaler']
                self.feature_columns = data['feature_columns']
                self.base_feature_columns = data.get('base_feature_columns', [])
                self.smooth_feature_columns = data.get('smooth_feature_columns', [])
                self.earnings_feature_columns = data.get('earnings_feature_columns', [])
                self.sector_columns = data.get('sector_columns', [])
                self.model_loaded = True
                print(f"[OK] Modèle earnings ML chargé: {self.model_path}")
                print(f"     {len(self.feature_columns)} features "
                      f"(dont {len(self.earnings_feature_columns)} earnings, "
                      f"{len(self.sector_columns)} secteurs)")
            else:
                print(f"[WARN] Modèle earnings ML non trouvé: {self.model_path}")
        except Exception as e:
            print(f"[ERROR] Erreur chargement modèle earnings: {e}")

    def _load_sector_cache(self):
        """Charge le cache des secteurs depuis la BD."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT symbol, sector FROM symbol_metadata")
            self._sector_cache = {row[0]: row[1] for row in cursor.fetchall()}
            conn.close()
            print(f"[OK] Cache secteurs chargé: {len(self._sector_cache)} symboles")
        except Exception as e:
            print(f"[WARN] Impossible de charger les secteurs: {e}")
            self._sector_cache = {}

    def _rolling_r2(self, prices: pd.Series, window: int) -> pd.Series:
        """Calcule le R² sur une fenêtre glissante."""
        x = np.arange(window, dtype=np.float64)
        sum_x = x.sum()
        sum_x2 = (x ** 2).sum()
        n = window
        denom_x = n * sum_x2 - sum_x ** 2

        def calc_r2(y):
            if len(y) < window or np.isnan(y).any():
                return np.nan
            sum_y = y.sum()
            sum_y2 = (y ** 2).sum()
            sum_xy = (x * y).sum()
            denom_y = n * sum_y2 - sum_y ** 2
            if denom_y <= 0 or denom_x <= 0:
                return np.nan
            r = (n * sum_xy - sum_x * sum_y) / (np.sqrt(denom_x * denom_y) + 1e-10)
            return r ** 2

        return prices.rolling(window=window).apply(calc_r2, raw=True)

    def _get_earnings_features(self, symbol: str) -> dict:
        """Récupère les features earnings pour un symbole."""
        if symbol not in self._earnings_cache:
            try:
                self._earnings_cache[symbol] = self.earnings_features.get_latest_earnings_features(symbol)
            except:
                self._earnings_cache[symbol] = {
                    'sue': 0, 'sue_avg4': 0, 'car3': 0,
                    'car3_avg4': 0, 'earnings_momentum': 0
                }
        return self._earnings_cache[symbol]

    def _create_features(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Crée toutes les features nécessaires pour le modèle."""
        df = df.copy()
        df.columns = df.columns.str.lower()

        # === Features de base ===
        if 'sma20_volume' not in df.columns:
            df['sma20_volume'] = df['volume'].rolling(window=20).mean()

        if 'hl_sma20vol' not in df.columns:
            df['hl_sma20vol'] = (df['high'] - df['low']) / (df['sma20_volume'] + 1e-6)

        if 'oc_sma20vol' not in df.columns:
            df['oc_sma20vol'] = (df['open'] - df['close']) / (df['sma20_volume'] + 1e-6)

        # MACD
        if 'macd' not in df.columns:
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()

        # RSI
        if 'rsi' not in df.columns:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-6)
            df['rsi'] = 100 - (100 / (1 + rs))

        # ADX simplifié
        if 'adx' not in df.columns:
            df['adx'] = 25

        # Bollinger Bands
        if 'bb_high' not in df.columns:
            sma20 = df['close'].rolling(20).mean()
            std20 = df['close'].rolling(20).std()
            df['bb_high'] = sma20 + 2 * std20
            df['bb_low'] = sma20 - 2 * std20

        if 'pct_close' not in df.columns:
            df['pct_close'] = df['close'].pct_change()

        # Features calculées
        df['macd_hist'] = df['macd'] - df['macd_signal']

        bb_range = df['bb_high'] - df['bb_low']
        df['bb_position'] = np.where(bb_range > 0,
            (df['close'] - df['bb_low']) / bb_range, 0.5)

        df['rsi_momentum'] = df['rsi'] - 50
        df['volume_ratio'] = df['volume'] / (df['sma20_volume'] + 1e-6)
        df['trend_strength'] = df['adx'] * np.sign(df['macd'])

        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        df['return_20d'] = df['close'].pct_change(20)

        df['volatility_10d'] = df['close'].pct_change().rolling(10).std()

        high_52w = df['high'].rolling(252, min_periods=50).max()
        df['high_52w_pct'] = df['close'] / (high_52w + 1e-6)

        # === Features Smoothness ===
        df['smoothness_20d'] = self._rolling_r2(df['close'], 20)
        df['smoothness_50d'] = self._rolling_r2(df['close'], 50)

        # === Features Saisonnalité ===
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            month = df['date'].dt.month
        else:
            month = pd.Series([pd.Timestamp.now().month] * len(df))

        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)

        # === Features Earnings ===
        earnings = self._get_earnings_features(symbol) if symbol else {}
        for col in self.earnings_feature_columns:
            df[col] = earnings.get(col, 0) if earnings.get(col) is not None else 0

        # === Features Secteur ===
        sector = self._sector_cache.get(symbol, 'Unknown') if symbol else 'Unknown'
        for col in self.sector_columns:
            expected_sector = col.replace('sector_', '')
            df[col] = 1.0 if sector == expected_sector else 0.0

        return df

    def score(self, df: pd.DataFrame, symbol: str = None) -> int:
        """
        Score un DataFrame avec le modèle ML earnings.

        Args:
            df: DataFrame avec données OHLCV (minimum 60 jours)
            symbol: Symbole pour récupérer secteur et earnings

        Returns:
            Score de 0 à 100
        """
        if not self.model_loaded:
            print("[WARN] Modèle earnings ML non chargé, retourne 0")
            return 0

        try:
            df = self._create_features(df, symbol)
            self.df = df

            if len(df) < 60:
                return 0

            # Vérifier les features disponibles
            missing = [c for c in self.feature_columns if c not in df.columns]
            if missing:
                print(f"[WARN] Features manquantes: {missing[:5]}...")
                return 0

            df_clean = df.dropna(subset=self.feature_columns)
            if df_clean.empty:
                return 0

            X = df_clean[self.feature_columns].iloc[-1:].values
            X_scaled = self.scaler.transform(X)
            probability = self.model.predict_proba(X_scaled)[0, 1]

            score = int(probability * 100)
            return max(0, min(score, 100))

        except Exception as e:
            print(f"[ERREUR EarningsMLScoring] {e}")
            return 0
