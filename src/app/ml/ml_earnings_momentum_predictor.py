"""
ML Earnings Momentum Predictor - Extension du modèle smooth momentum avec features earnings.

Features additionnelles (Novy-Marx):
- SUE: Standardized Unexpected Earnings (surprise %)
- SUE_avg4: Moyenne des 4 derniers quarters
- CAR3: Cumulative Abnormal Return 3 jours autour des earnings
- Earnings Momentum: Tendance des surprises

Usage: python ml_earnings_momentum_predictor.py
"""
import pandas as pd
import numpy as np
import sqlite3
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.ml.earnings_features import EarningsFeatures


class MLEarningsMomentumPredictor:
    """
    Modèle ML étendu avec smoothness, saisonnalité, secteur ET earnings features.
    Prédit si un stock va augmenter de >5% dans les N prochains jours.
    """

    # Features de base
    BASE_FEATURE_COLUMNS = [
        'hl_sma20vol', 'oc_sma20vol', 'macd', 'macd_signal', 'rsi', 'adx',
        'bb_high', 'bb_low', 'pct_close', 'macd_hist', 'bb_position',
        'rsi_momentum', 'volume_ratio', 'trend_strength', 'return_5d',
        'return_10d', 'return_20d', 'volatility_10d', 'high_52w_pct',
    ]

    # Features smooth momentum
    SMOOTH_FEATURE_COLUMNS = [
        'smoothness_20d',
        'smoothness_50d',
        'month_sin',
        'month_cos',
    ]

    # Features earnings (Novy-Marx)
    EARNINGS_FEATURE_COLUMNS = [
        'sue',              # Dernière surprise earnings (%)
        'sue_avg4',         # Moyenne 4 derniers quarters
        'car3',             # CAR3 dernière annonce
        'car3_avg4',        # Moyenne CAR3 4 quarters
        'earnings_momentum', # Tendance des surprises
    ]

    def __init__(self, db_path: str = "trading_data.db",
                 model_path: str = "models/earnings_momentum_model.pkl"):
        self.db_path = db_path
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance: Optional[pd.DataFrame] = None

        # Paramètres de prédiction
        self.prediction_horizon = 20
        self.target_return = 0.05

        # Colonnes de secteur
        self.sector_columns: List[str] = []

        # Liste complète des features
        self.FEATURE_COLUMNS: List[str] = []

        # Cache des features earnings
        self._earnings_cache: Dict[str, Dict] = {}
        self.earnings_features = EarningsFeatures(db_path)

    def load_data(self, min_date: Optional[str] = None) -> pd.DataFrame:
        """Charge les données depuis la DB avec les métadonnées de secteur."""
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT h.symbol, h.date, h.open, h.high, h.low, h.close, h.volume,
                   h.sma20_volume, h.hl_sma20vol, h.oc_sma20vol,
                   h.macd, h.macd_signal, h.rsi, h.adx, h.bb_high, h.bb_low, h.pct_close,
                   COALESCE(m.sector, 'Unknown') as sector
            FROM historical_data h
            LEFT JOIN symbol_metadata m ON h.symbol = m.symbol
            WHERE h.close > 0 AND h.volume > 0
        """

        if min_date:
            query += f" AND h.date >= '{min_date}'"

        query += " ORDER BY h.symbol, h.date"

        df = pd.read_sql_query(query, conn)
        conn.close()

        df['date'] = pd.to_datetime(df['date'])
        print(f"[OK] {len(df):,} lignes chargées depuis la DB")

        # Stats des secteurs
        sector_counts = df.groupby('sector')['symbol'].nunique()
        print(f"[INFO] Secteurs trouvés: {len(sector_counts)}")

        return df

    def _rolling_r2(self, prices: pd.Series, window: int) -> pd.Series:
        """Calcule le R² sur une fenêtre glissante."""
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

    def fetch_earnings_features(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Récupère les features earnings pour une liste de symboles.
        Utilise un cache pour éviter les appels API répétés.
        """
        print(f"[EARNINGS] Récupération des features earnings pour {len(symbols)} symboles...")

        # Filtrer les symboles déjà en cache
        symbols_to_fetch = [s for s in symbols if s not in self._earnings_cache]

        if symbols_to_fetch:
            for i, symbol in enumerate(symbols_to_fetch):
                try:
                    features = self.earnings_features.get_latest_earnings_features(symbol)
                    self._earnings_cache[symbol] = features

                    if (i + 1) % 50 == 0:
                        print(f"  [{i+1}/{len(symbols_to_fetch)}] symboles traités...")

                except Exception as e:
                    self._earnings_cache[symbol] = {
                        'sue': None, 'sue_avg4': None,
                        'car3': None, 'car3_avg4': None,
                        'earnings_momentum': None
                    }

        print(f"[EARNINGS] {len(self._earnings_cache)} symboles en cache")

        # Stats sur les données disponibles
        valid_sue = sum(1 for v in self._earnings_cache.values() if v.get('sue') is not None)
        valid_car3 = sum(1 for v in self._earnings_cache.values() if v.get('car3') is not None)
        print(f"[EARNINGS] SUE disponible: {valid_sue} | CAR3 disponible: {valid_car3}")

        return self._earnings_cache

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée toutes les features: base + smooth + secteur + earnings."""
        df = df.copy()
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

        # === Features de base ===
        df['macd_hist'] = df['macd'] - df['macd_signal']

        bb_range = df['bb_high'] - df['bb_low']
        df['bb_position'] = np.where(bb_range > 0,
            (df['close'] - df['bb_low']) / bb_range, 0.5)

        df['rsi_momentum'] = df['rsi'] - 50

        df['volume_ratio'] = np.where(df['sma20_volume'] > 0,
            df['volume'] / df['sma20_volume'], 1.0)

        df['trend_strength'] = df['adx'] * np.sign(df['macd'])

        df['return_5d'] = df.groupby('symbol')['close'].pct_change(5)
        df['return_10d'] = df.groupby('symbol')['close'].pct_change(10)
        df['return_20d'] = df.groupby('symbol')['close'].pct_change(20)

        df['volatility_10d'] = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(10).std())

        df['high_52w'] = df.groupby('symbol')['high'].transform(
            lambda x: x.rolling(252, min_periods=50).max())
        df['high_52w_pct'] = df['close'] / df['high_52w']
        df = df.drop(columns=['high_52w'])

        # === Features Smoothness (R²) ===
        print("[FEATURES] Calcul du smoothness (R²)...")
        df['smoothness_20d'] = df.groupby('symbol')['close'].transform(
            lambda x: self._rolling_r2(x, 20))
        df['smoothness_50d'] = df.groupby('symbol')['close'].transform(
            lambda x: self._rolling_r2(x, 50))

        # === Features Saisonnalité ===
        month = df['date'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)

        # === Features Secteur ===
        if 'sector' not in df.columns:
            df['sector'] = 'Unknown'

        sector_dummies = pd.get_dummies(df['sector'], prefix='sector')
        self.sector_columns = sorted(sector_dummies.columns.tolist())
        df = pd.concat([df, sector_dummies], axis=1)

        # === Features Earnings (Novy-Marx) ===
        print("[FEATURES] Ajout des features earnings...")
        symbols = df['symbol'].unique().tolist()
        earnings_cache = self.fetch_earnings_features(symbols)

        # Mapper les features earnings par symbole
        for col in self.EARNINGS_FEATURE_COLUMNS:
            df[col] = df['symbol'].map(lambda s: earnings_cache.get(s, {}).get(col))

        # Remplir les valeurs manquantes avec 0 (neutre)
        for col in self.EARNINGS_FEATURE_COLUMNS:
            df[col] = df[col].fillna(0)

        # Construire la liste complète des features
        self.FEATURE_COLUMNS = (
            self.BASE_FEATURE_COLUMNS +
            self.SMOOTH_FEATURE_COLUMNS +
            self.EARNINGS_FEATURE_COLUMNS +
            self.sector_columns
        )

        print(f"[FEATURES] {len(self.FEATURE_COLUMNS)} features au total:")
        print(f"  - Base: {len(self.BASE_FEATURE_COLUMNS)}")
        print(f"  - Smooth: {len(self.SMOOTH_FEATURE_COLUMNS)}")
        print(f"  - Earnings: {len(self.EARNINGS_FEATURE_COLUMNS)}")
        print(f"  - Secteurs: {len(self.sector_columns)}")

        return df

    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée les labels: le stock a-t-il atteint +5% dans les N prochains jours ?"""
        df = df.copy()
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

        def calculate_future_return(group):
            future_max = group['high'].shift(-1).rolling(
                window=self.prediction_horizon,
                min_periods=1
            ).max().shift(-self.prediction_horizon + 1)
            max_return = (future_max - group['close']) / group['close']
            return (max_return >= self.target_return).astype(int)

        df['target'] = df.groupby('symbol', group_keys=False).apply(
            calculate_future_return).values

        return df

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prépare les données pour l'entraînement."""
        df_clean = df.dropna(subset=self.FEATURE_COLUMNS + ['target'])

        cutoff_date = df_clean['date'].max() - timedelta(days=self.prediction_horizon + 5)
        df_clean = df_clean[df_clean['date'] <= cutoff_date]

        X = df_clean[self.FEATURE_COLUMNS].values
        y = df_clean['target'].values

        print(f"[OK] Données préparées: {len(X):,} échantillons")
        print(f"     Distribution: {y.mean()*100:.1f}% positifs ({y.sum():,})")

        return X, y, self.FEATURE_COLUMNS

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict:
        """Entraîne le modèle XGBoost."""
        # Split temporel 80/20
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        print(f"[INFO] Train: {len(X_train):,} | Test: {len(X_test):,}")

        # Normalisation
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        scale_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)

        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_weight,
            random_state=42,
            eval_metric='auc',
            early_stopping_rounds=20
        )

        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=True
        )

        # Evaluation
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        roc_auc = roc_auc_score(y_test, y_proba)

        print("\n" + "=" * 50)
        print("RESULTATS - EARNINGS MOMENTUM MODEL")
        print("=" * 50)
        print(f"\nROC-AUC Score: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['< 5%', '>= 5%']))

        cm = confusion_matrix(y_test, y_pred)
        print(f"Matrice de confusion:")
        print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
        print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")

        # Feature importance
        importance = self.model.feature_importances_
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        print("\nTop 15 Features les plus importantes:")
        print(self.feature_importance.head(15).to_string(index=False))

        # Importance des features earnings spécifiquement
        earnings_imp = self.feature_importance[
            self.feature_importance['feature'].isin(self.EARNINGS_FEATURE_COLUMNS)
        ]
        print("\nImportance des features EARNINGS:")
        print(earnings_imp.to_string(index=False))

        return {
            'roc_auc': roc_auc,
            'accuracy': (y_pred == y_test).mean(),
            'precision': cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0,
            'recall': cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0,
        }

    def save_model(self):
        """Sauvegarde le modèle avec les métadonnées."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.FEATURE_COLUMNS,
            'base_feature_columns': self.BASE_FEATURE_COLUMNS,
            'smooth_feature_columns': self.SMOOTH_FEATURE_COLUMNS,
            'earnings_feature_columns': self.EARNINGS_FEATURE_COLUMNS,
            'sector_columns': self.sector_columns,
            'feature_importance': self.feature_importance,
            'prediction_horizon': self.prediction_horizon,
            'target_return': self.target_return,
        }, self.model_path)

        print(f"\n[OK] Modèle sauvegardé: {self.model_path}")

    def run_full_training(self, min_date: str = "2018-01-01"):
        """Execute l'entraînement complet."""
        print("\n" + "=" * 60)
        print("ENTRAINEMENT - EARNINGS MOMENTUM MODEL")
        print("=" * 60)
        print(f"Horizon: {self.prediction_horizon} jours | Target: >={self.target_return*100:.0f}%")
        print(f"Features: base + smooth + secteur + EARNINGS (SUE/CAR3)")
        print(f"Date min: {min_date}")
        print("=" * 60 + "\n")

        print("[1/4] Chargement des données...")
        df = self.load_data(min_date)

        print("[2/4] Création des features...")
        df = self.create_features(df)

        print("[3/4] Création des labels...")
        df = self.create_labels(df)

        print("[4/4] Entraînement du modèle...")
        X, y, feature_names = self.prepare_training_data(df)
        metrics = self.train(X, y, feature_names)

        self.save_model()

        return metrics


if __name__ == "__main__":
    predictor = MLEarningsMomentumPredictor(
        db_path="trading_data.db",
        model_path="models/earnings_momentum_model.pkl"
    )

    metrics = predictor.run_full_training(min_date="2018-01-01")
