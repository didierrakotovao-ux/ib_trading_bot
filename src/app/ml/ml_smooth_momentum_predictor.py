"""
ML Smooth Momentum Predictor - Extension du modèle momentum avec features de smoothness,
saisonnalité et secteur.

Features additionnelles:
- Smoothness: R² de régression linéaire sur fenêtres glissantes (régularité du prix)
- Saisonnalité: Encodage cyclique du mois (sin/cos)
- Secteur: One-hot encoding du secteur GICS depuis symbol_metadata

Usage: python ml_smooth_momentum_predictor.py
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


class MLSmoothMomentumPredictor:
    """
    Modèle ML étendu avec smoothness, saisonnalité et secteur.
    Prédit si un stock va augmenter de >5% dans les N prochains jours.
    """

    # Features de base (identiques au modèle original)
    BASE_FEATURE_COLUMNS = [
        'hl_sma20vol', 'oc_sma20vol', 'macd', 'macd_signal', 'rsi', 'adx',
        'bb_high', 'bb_low', 'pct_close', 'macd_hist', 'bb_position',
        'rsi_momentum', 'volume_ratio', 'trend_strength', 'return_5d',
        'return_10d', 'return_20d', 'volatility_10d', 'high_52w_pct',
    ]

    # Nouvelles features (hors secteur, qui est dynamique)
    NEW_FEATURE_COLUMNS = [
        'smoothness_20d',   # R² régression linéaire 20 jours
        'smoothness_50d',   # R² régression linéaire 50 jours
        'month_sin',        # sin(2π * mois / 12)
        'month_cos',        # cos(2π * mois / 12)
    ]

    def __init__(self, db_path: str = "trading_data.db",
                 model_path: str = "models/smooth_momentum_model.pkl"):
        self.db_path = db_path
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance: Optional[pd.DataFrame] = None

        # Paramètres de prédiction (identiques au modèle original)
        self.prediction_horizon = 20
        self.target_return = 0.05

        # Colonnes de secteur (déterminées à l'entraînement)
        self.sector_columns: List[str] = []

        # Liste complète des features (construite dynamiquement)
        self.FEATURE_COLUMNS: List[str] = []

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
        print(f"[OK] {len(df):,} lignes chargees depuis la DB")

        # Stats des secteurs
        sector_counts = df.groupby('sector')['symbol'].nunique()
        print(f"[INFO] Secteurs trouves: {len(sector_counts)}")
        for sector, count in sector_counts.sort_values(ascending=False).items():
            print(f"       {sector}: {count} symboles")

        return df

    def _rolling_r2(self, prices: pd.Series, window: int) -> pd.Series:
        """
        Calcule le R² d'une régression linéaire sur une fenêtre glissante.
        Utilise la formule analytique pour la performance (pas de sklearn).

        R² = [n*Σ(xy) - Σx*Σy]² / ([n*Σ(x²) - (Σx)²] * [n*Σ(y²) - (Σy)²])
        où x = indices (0, 1, ..., n-1) et y = prix
        """
        # x est constant pour une fenêtre donnée: [0, 1, ..., window-1]
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

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée toutes les features: base + smoothness + saisonnalité + secteur.

        Toutes les features utilisent uniquement des données historiques
        (pas de lookahead bias).
        """
        df = df.copy()
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

        # === Features de base (identiques au modèle original) ===

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

        # === Nouvelles features: Smoothness (R²) ===
        print("[FEATURES] Calcul du smoothness (R²)...")
        df['smoothness_20d'] = df.groupby('symbol')['close'].transform(
            lambda x: self._rolling_r2(x, 20))
        df['smoothness_50d'] = df.groupby('symbol')['close'].transform(
            lambda x: self._rolling_r2(x, 50))

        # === Nouvelles features: Saisonnalité ===
        month = df['date'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)

        # === Nouvelles features: Secteur (one-hot encoding) ===
        if 'sector' not in df.columns:
            df['sector'] = 'Unknown'

        sector_dummies = pd.get_dummies(df['sector'], prefix='sector')
        # Stocker les colonnes de secteur pour le scoring
        self.sector_columns = sorted(sector_dummies.columns.tolist())
        df = pd.concat([df, sector_dummies], axis=1)

        # Construire la liste complète des features
        self.FEATURE_COLUMNS = (
            self.BASE_FEATURE_COLUMNS +
            self.NEW_FEATURE_COLUMNS +
            self.sector_columns
        )

        print(f"[FEATURES] {len(self.FEATURE_COLUMNS)} features au total "
              f"({len(self.BASE_FEATURE_COLUMNS)} base + "
              f"{len(self.NEW_FEATURE_COLUMNS)} nouvelles + "
              f"{len(self.sector_columns)} secteurs)")

        return df

    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée les labels: le stock a-t-il atteint +5% dans les N prochains jours ?
        Identique au modèle original.
        """
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

        print(f"[OK] Donnees preparees: {len(X):,} echantillons")
        print(f"     Distribution: {y.mean()*100:.1f}% positifs ({y.sum():,})")

        return X, y, self.FEATURE_COLUMNS

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict:
        """Entraîne le modèle XGBoost."""
        # Split temporel 80/20
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        print(f"[INFO] Train: {len(X_train):,} | Test: {len(X_test):,}")
        print(f"[INFO] Train - Classe 0: {sum(y_train==0):,}, Classe 1: {sum(y_train==1):,}")
        print(f"[INFO] Test  - Classe 0: {sum(y_test==0):,}, Classe 1: {sum(y_test==1):,}")

        # Normalisation
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        scale_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
        print(f"[INFO] Scale pos weight: {scale_weight:.2f}")

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
        print("RESULTATS - SMOOTH MOMENTUM MODEL")
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

        # Afficher l'importance des nouvelles features
        new_features = self.NEW_FEATURE_COLUMNS + self.sector_columns
        new_imp = self.feature_importance[
            self.feature_importance['feature'].isin(new_features)
        ]
        print("\nImportance des NOUVELLES features:")
        print(new_imp.to_string(index=False))

        return {
            'roc_auc': roc_auc,
            'accuracy': (y_pred == y_test).mean(),
            'precision': cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0,
            'recall': cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0,
        }

    def save_model(self):
        """Sauvegarde le modèle avec les métadonnées de secteur."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.FEATURE_COLUMNS,
            'base_feature_columns': self.BASE_FEATURE_COLUMNS,
            'new_feature_columns': self.NEW_FEATURE_COLUMNS,
            'sector_columns': self.sector_columns,
            'feature_importance': self.feature_importance,
            'prediction_horizon': self.prediction_horizon,
            'target_return': self.target_return,
        }, self.model_path)

        print(f"\n[OK] Modele sauvegarde: {self.model_path}")

    def run_full_training(self, min_date: str = "2018-01-01"):
        """Execute l'entraînement complet."""
        print("\n" + "=" * 60)
        print("ENTRAINEMENT - SMOOTH MOMENTUM MODEL")
        print("=" * 60)
        print(f"Horizon: {self.prediction_horizon} jours | Target: >={self.target_return*100:.0f}%")
        print(f"Features: base + smoothness + saisonnalite + secteur")
        print(f"Date min: {min_date}")
        print("=" * 60 + "\n")

        print("[1/4] Chargement des donnees...")
        df = self.load_data(min_date)

        print("[2/4] Creation des features...")
        df = self.create_features(df)

        print("[3/4] Creation des labels...")
        df = self.create_labels(df)

        print("[4/4] Entrainement du modele...")
        X, y, feature_names = self.prepare_training_data(df)
        metrics = self.train(X, y, feature_names)

        self.save_model()

        return metrics


if __name__ == "__main__":
    predictor = MLSmoothMomentumPredictor(
        db_path="trading_data.db",
        model_path="models/smooth_momentum_model.pkl"
    )

    metrics = predictor.run_full_training(min_date="2018-01-01")
