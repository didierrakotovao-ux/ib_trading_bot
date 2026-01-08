"""
ML Scoring - Scoring basé sur le modèle ML de prédiction momentum.
Prédit la probabilité qu'un stock augmente de >5% dans les 20 prochains jours.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Optional
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.ml.scoring import Scoring


class MLScoring(Scoring):
    """
    Scoring utilisant le modèle ML entraîné pour prédire les hausses >5%.
    """
    name = "MLScoring"
    df: pd.DataFrame

    # Features requises par le modèle
    FEATURE_COLUMNS = [
        'hl_sma20vol', 'oc_sma20vol', 'macd', 'macd_signal', 'rsi', 'adx',
        'bb_high', 'bb_low', 'pct_close', 'macd_hist', 'bb_position',
        'rsi_momentum', 'volume_ratio', 'trend_strength', 'return_5d',
        'return_10d', 'return_20d', 'volatility_10d', 'high_52w_pct',
    ]

    def __init__(self, model_path: str = "models/momentum_model.pkl"):
        """
        Initialise le scoring ML.

        Args:
            model_path: Chemin vers le modèle sauvegardé
        """
        self.df = None
        self.model = None
        self.scaler = None
        self.model_loaded = False
        self.model_path = Path(model_path)

        # Essayer de charger le modèle
        self._load_model()

    def _load_model(self):
        """Charge le modèle ML."""
        try:
            if self.model_path.exists():
                data = joblib.load(self.model_path)
                self.model = data['model']
                self.scaler = data['scaler']
                self.model_loaded = True
                print(f"[OK] Modèle ML chargé: {self.model_path}")
            else:
                print(f"[WARN] Modèle ML non trouvé: {self.model_path}")
                print("       Entraînez-le avec: python src/app/ml/ml_momentum_predictor.py")
        except Exception as e:
            print(f"[ERROR] Erreur chargement modèle: {e}")

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée les features nécessaires pour le modèle.

        Args:
            df: DataFrame avec données OHLCV

        Returns:
            DataFrame enrichi avec les features
        """
        df = df.copy()

        # Normaliser les noms de colonnes
        df.columns = df.columns.str.lower()

        # --- Features de base (si pas déjà présentes) ---

        # SMA20 du volume
        if 'sma20_volume' not in df.columns:
            df['sma20_volume'] = df['volume'].rolling(window=20).mean()

        # hl_sma20vol et oc_sma20vol
        if 'hl_sma20vol' not in df.columns:
            df['hl_sma20vol'] = (df['high'] - df['low']) / (df['sma20_volume'] + 1e-6)

        if 'oc_sma20vol' not in df.columns:
            df['oc_sma20vol'] = (df['open'] - df['close']) / (df['sma20_volume'] + 1e-6)

        # MACD
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

        # RSI
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

        # ADX
        if 'adx' not in df.columns:
            try:
                import pandas_ta as ta
                adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
                df['adx'] = adx_df['ADX_14']
            except:
                df['adx'] = 25  # Valeur par défaut

        # Bollinger Bands
        if 'bb_high' not in df.columns:
            sma20 = df['close'].rolling(20).mean()
            std20 = df['close'].rolling(20).std()
            df['bb_high'] = sma20 + 2 * std20
            df['bb_low'] = sma20 - 2 * std20

        # pct_close
        if 'pct_close' not in df.columns:
            df['pct_close'] = df['close'].pct_change()

        # --- Features calculées ---

        # MACD Histogram
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Position dans les Bandes de Bollinger
        bb_range = df['bb_high'] - df['bb_low']
        df['bb_position'] = np.where(
            bb_range > 0,
            (df['close'] - df['bb_low']) / bb_range,
            0.5
        )

        # RSI centré
        df['rsi_momentum'] = df['rsi'] - 50

        # Volume ratio
        df['volume_ratio'] = df['volume'] / (df['sma20_volume'] + 1e-6)

        # Trend strength
        df['trend_strength'] = df['adx'] * np.sign(df['macd'])

        # Rendements passés
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        df['return_20d'] = df['close'].pct_change(20)

        # Volatilité
        df['volatility_10d'] = df['close'].pct_change().rolling(10).std()

        # Position vs plus haut 52 semaines
        high_52w = df['high'].rolling(252, min_periods=50).max()
        df['high_52w_pct'] = df['close'] / (high_52w + 1e-6)

        return df

    def score(self, df: pd.DataFrame) -> int:
        """
        Score un DataFrame avec le modèle ML.

        Args:
            df: DataFrame avec données OHLCV (minimum 60 jours)

        Returns:
            Score de 0 à 100 (probabilité de hausse >5%)
        """
        if not self.model_loaded:
            print("[WARN] Modèle ML non chargé, retourne 0")
            return 0

        try:
            # Créer les features
            df = self._create_features(df)
            self.df = df

            # Vérifier qu'on a assez de données
            if len(df) < 60:
                print(f"[WARN] Pas assez de données ({len(df)} < 60)")
                return 0

            # Prendre la dernière ligne avec toutes les features
            df_clean = df.dropna(subset=self.FEATURE_COLUMNS)

            if df_clean.empty:
                print("[WARN] Pas de données valides après nettoyage")
                return 0

            # Extraire les features de la dernière ligne
            X = df_clean[self.FEATURE_COLUMNS].iloc[-1:].values

            # Normaliser
            X_scaled = self.scaler.transform(X)

            # Prédire la probabilité
            probability = self.model.predict_proba(X_scaled)[0, 1]

            # Convertir en score 0-100
            score = int(probability * 100)

            return max(0, min(score, 100))

        except Exception as e:
            print(f"[ERREUR MLScoring] {e}")
            return 0

    def get_prediction_details(self, df: pd.DataFrame) -> dict:
        """
        Retourne les détails de la prédiction.

        Args:
            df: DataFrame avec données OHLCV

        Returns:
            Dictionnaire avec probabilité, signal, et features clés
        """
        score = self.score(df)
        probability = score / 100.0

        # Signal basé sur la probabilité
        if probability >= 0.6:
            signal = "BUY"
        elif probability >= 0.4:
            signal = "HOLD"
        else:
            signal = "AVOID"

        # Features clés
        details = {
            'score': score,
            'probability': probability,
            'signal': signal,
        }

        if self.df is not None and not self.df.empty:
            last = self.df.iloc[-1]
            details['features'] = {
                'volatility_10d': float(last.get('volatility_10d', 0)),
                'high_52w_pct': float(last.get('high_52w_pct', 0)),
                'rsi': float(last.get('rsi', 0)),
                'adx': float(last.get('adx', 0)),
                'macd_hist': float(last.get('macd_hist', 0)),
            }

        return details


# Test standalone
if __name__ == "__main__":
    import sqlite3

    # Charger des données de test
    db_path = "trading_data.db"
    conn = sqlite3.connect(db_path)

    symbols_to_test = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'AMD']

    scoring = MLScoring(model_path="models/momentum_model.pkl")

    print("\n" + "="*60)
    print("TEST MLScoring")
    print("="*60)

    for symbol in symbols_to_test:
        query = f"""
            SELECT date, open, high, low, close, volume,
                   sma20_volume, hl_sma20vol, oc_sma20vol,
                   macd, macd_signal, rsi, adx, bb_high, bb_low, pct_close
            FROM historical_data
            WHERE symbol = '{symbol}'
            ORDER BY date
        """
        df = pd.read_sql_query(query, conn)

        if not df.empty:
            details = scoring.get_prediction_details(df)
            print(f"{symbol}: Score={details['score']} -> {details['signal']}")
            if 'features' in details:
                print(f"        RSI={details['features']['rsi']:.1f}, "
                      f"ADX={details['features']['adx']:.1f}, "
                      f"Vol10d={details['features']['volatility_10d']:.4f}")
        else:
            print(f"{symbol}: Pas de données")

    conn.close()
