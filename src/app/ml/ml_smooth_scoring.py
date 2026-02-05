"""
ML Smooth Scoring - Scoring avec le modèle smooth momentum (smoothness + saisonnalité + secteur).
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


class SmoothMLScoring(Scoring):
    """
    Scoring utilisant le modèle smooth momentum avec features de
    smoothness, saisonnalité et secteur.
    """
    name = "SmoothMLScoring"
    df: pd.DataFrame

    def __init__(self, model_path: str = "models/smooth_momentum_model.pkl",
                 db_path: str = "trading_data.db"):
        self.df = None
        self.model = None
        self.scaler = None
        self.model_loaded = False
        # Résoudre le chemin par rapport à la racine du projet
        project_root = Path(__file__).parent.parent.parent.parent
        self.model_path = project_root / model_path
        self.db_path = str(project_root / db_path)

        # Colonnes chargées depuis le modèle
        self.feature_columns: List[str] = []
        self.sector_columns: List[str] = []

        # Cache des métadonnées de secteur
        self._sector_cache: dict = {}

        self._load_model()
        self._load_sector_cache()

    def _load_model(self):
        """Charge le modèle ML smooth momentum."""
        try:
            if self.model_path.exists():
                data = joblib.load(self.model_path)
                self.model = data['model']
                self.scaler = data['scaler']
                self.feature_columns = data['feature_columns']
                self.sector_columns = data.get('sector_columns', [])
                self.model_loaded = True
                print(f"[OK] Modele smooth ML charge: {self.model_path}")
                print(f"     {len(self.feature_columns)} features "
                      f"(dont {len(self.sector_columns)} secteurs)")
            else:
                print(f"[WARN] Modele smooth ML non trouve: {self.model_path}")
                print("       Entrainez-le avec: python src/app/ml/ml_smooth_momentum_predictor.py")
        except Exception as e:
            print(f"[ERROR] Erreur chargement modele smooth: {e}")

    def _load_sector_cache(self):
        """Charge le cache des secteurs depuis la BD."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT symbol, sector FROM symbol_metadata")
            self._sector_cache = {row[0]: row[1] for row in cursor.fetchall()}
            conn.close()
            print(f"[OK] Cache secteurs charge: {len(self._sector_cache)} symboles")
        except Exception as e:
            print(f"[WARN] Impossible de charger les secteurs: {e}")
            self._sector_cache = {}

    def _rolling_r2(self, prices: pd.Series, window: int) -> pd.Series:
        """Calcule le R² sur une fenêtre glissante (formule analytique)."""
        x = np.arange(window, dtype=np.float64)
        sum_x = x.sum()
        sum_x2 = (x ** 2).sum()
        n = window

        def r2_calc(y):
            if len(y) < window:
                return np.nan
            sum_y = y.sum()
            sum_y2 = (y ** 2).sum()
            sum_xy = (x * y).sum()

            numerator = (n * sum_xy - sum_x * sum_y) ** 2
            denom_x = n * sum_x2 - sum_x ** 2
            denom_y = n * sum_y2 - sum_y ** 2

            if denom_x == 0 or denom_y == 0:
                return 0.0

            return numerator / (denom_x * denom_y)

        return prices.rolling(window, min_periods=window).apply(r2_calc, raw=True)

    def _create_features(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Crée les features pour le scoring (base + smoothness + saisonnalité + secteur).
        """
        df = df.copy()
        df.columns = df.columns.str.lower()

        # === Features de base (identiques à MLScoring) ===

        if 'sma20_volume' not in df.columns:
            df['sma20_volume'] = df['volume'].rolling(window=20).mean()

        if 'hl_sma20vol' not in df.columns:
            df['hl_sma20vol'] = (df['high'] - df['low']) / (df['sma20_volume'] + 1e-6)

        if 'oc_sma20vol' not in df.columns:
            df['oc_sma20vol'] = (df['open'] - df['close']) / (df['sma20_volume'] + 1e-6)

        if 'macd' not in df.columns:
            try:
                import pandas_ta as ta
                macd_df = ta.macd(df['close'], fast=12, slow=26)
                df['macd'] = macd_df['MACD_12_26_9']
                df['macd_signal'] = macd_df['MACDs_12_26_9']
            except:
                ema12 = df['close'].ewm(span=12).mean()
                ema26 = df['close'].ewm(span=26).mean()
                df['macd'] = ema12 - ema26
                df['macd_signal'] = df['macd'].ewm(span=9).mean()

        if 'rsi' not in df.columns:
            try:
                import pandas_ta as ta
                df['rsi'] = ta.rsi(df['close'], length=14)
            except:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / (loss + 1e-6)
                df['rsi'] = 100 - (100 / (1 + rs))

        if 'adx' not in df.columns:
            try:
                import pandas_ta as ta
                adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
                df['adx'] = adx_df['ADX_14']
            except:
                df['adx'] = 25

        if 'bb_high' not in df.columns:
            sma20 = df['close'].rolling(20).mean()
            std20 = df['close'].rolling(20).std()
            df['bb_high'] = sma20 + 2 * std20
            df['bb_low'] = sma20 - 2 * std20

        if 'pct_close' not in df.columns:
            df['pct_close'] = df['close'].pct_change()

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

        # === Smoothness (R²) ===
        df['smoothness_20d'] = self._rolling_r2(df['close'], 20)
        df['smoothness_50d'] = self._rolling_r2(df['close'], 50)

        # === Saisonnalité ===
        if 'date' in df.columns:
            month = pd.to_datetime(df['date']).dt.month
        else:
            # Fallback: utiliser l'index si c'est un DatetimeIndex
            month = pd.Series(df.index.month if hasattr(df.index, 'month')
                              else [1] * len(df))

        df['month_sin'] = np.sin(2 * np.pi * month.values / 12)
        df['month_cos'] = np.cos(2 * np.pi * month.values / 12)

        # === Secteur (one-hot) ===
        sector = self._sector_cache.get(symbol, 'Unknown') if symbol else 'Unknown'
        for col in self.sector_columns:
            df[col] = 0
        sector_col = f'sector_{sector}'
        if sector_col in self.sector_columns:
            df[sector_col] = 1

        return df

    def score(self, df: pd.DataFrame, symbol: str = None) -> int:
        """
        Score un DataFrame avec le modèle smooth momentum.

        Args:
            df: DataFrame avec donnees OHLCV (minimum 60 jours)
            symbol: Symbole pour le lookup du secteur

        Returns:
            Score de 0 a 100 (probabilite de hausse >5%)
        """
        if not self.model_loaded:
            print("[WARN] Modele smooth ML non charge, retourne 0")
            return 0

        try:
            df = self._create_features(df, symbol=symbol)
            self.df = df

            if len(df) < 60:
                print(f"[WARN] Pas assez de donnees ({len(df)} < 60)")
                return 0

            df_clean = df.dropna(subset=self.feature_columns)

            if df_clean.empty:
                print("[WARN] Pas de donnees valides apres nettoyage")
                return 0

            X = df_clean[self.feature_columns].iloc[-1:].values
            X_scaled = self.scaler.transform(X)
            probability = self.model.predict_proba(X_scaled)[0, 1]

            score = int(probability * 100)
            return max(0, min(score, 100))

        except Exception as e:
            print(f"[ERREUR SmoothMLScoring] {e}")
            import traceback
            traceback.print_exc()
            return 0

    def get_prediction_details(self, df: pd.DataFrame, symbol: str = None) -> dict:
        """Retourne les details de la prediction."""
        score = self.score(df, symbol=symbol)
        probability = score / 100.0

        if probability >= 0.6:
            signal = "BUY"
        elif probability >= 0.4:
            signal = "HOLD"
        else:
            signal = "AVOID"

        details = {
            'score': score,
            'probability': probability,
            'signal': signal,
        }

        if self.df is not None and not self.df.empty:
            last = self.df.iloc[-1]
            details['features'] = {
                'smoothness_20d': float(last.get('smoothness_20d', 0)),
                'smoothness_50d': float(last.get('smoothness_50d', 0)),
                'month_sin': float(last.get('month_sin', 0)),
                'month_cos': float(last.get('month_cos', 0)),
                'rsi': float(last.get('rsi', 0)),
                'adx': float(last.get('adx', 0)),
                'macd_hist': float(last.get('macd_hist', 0)),
            }
            # Ajouter le secteur
            if symbol and symbol in self._sector_cache:
                details['sector'] = self._sector_cache[symbol]

        return details


# Test standalone
if __name__ == "__main__":
    import sqlite3

    db_path = "trading_data.db"
    conn = sqlite3.connect(db_path)

    symbols_to_test = ['AAPL', 'NVDA', 'GOOGL', 'QCOM', 'TXN', 'INTU',
                       'TFC', 'USB', 'PNC', 'ABBV', 'MRK', 'PFE',
                       'ORLY', 'AZO', 'YUM', 'RTX', 'GE', 'LMT']

    scoring = SmoothMLScoring(
        model_path="models/smooth_momentum_model.pkl",
        db_path=db_path
    )

    print("\n" + "=" * 60)
    print("TEST SmoothMLScoring")
    print("=" * 60)

    for symbol in symbols_to_test:
        query = f"""
            SELECT date, open, high, low, close, volume
            FROM historical_data
            WHERE symbol = '{symbol}'
            ORDER BY date
        """
        df = pd.read_sql_query(query, conn)

        if not df.empty:
            details = scoring.get_prediction_details(df, symbol=symbol)
            sector = details.get('sector', 'N/A')
            smooth = details.get('features', {}).get('smoothness_20d', 0)
            print(f"{symbol:6} [{sector:20}] Score={details['score']:3} "
                  f"-> {details['signal']:5} (smooth20={smooth:.3f})")
        else:
            print(f"{symbol}: Pas de donnees")

    conn.close()
