"""
ML Momentum Predictor - Prédit la probabilité qu'un stock augmente de >5%
Utilise les features de la DB + indicateurs techniques pour entraîner un modèle XGBoost.
"""
import pandas as pd
import numpy as np
import sqlite3
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import RandomForestClassifier

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


class MLMomentumPredictor:
    """
    Modèle ML pour prédire si un stock va augmenter de >5% dans les N prochains jours.
    """

    # Features utilisées pour l'entraînement
    FEATURE_COLUMNS = [
        # Features de la DB
        'hl_sma20vol',      # Range normalisé par volume
        'oc_sma20vol',      # Body normalisé par volume
        'macd',
        'macd_signal',
        'rsi',
        'adx',
        'bb_high',
        'bb_low',
        'pct_close',

        # Features calculées
        'macd_hist',        # MACD - Signal
        'bb_position',      # Position dans les bandes de Bollinger
        'rsi_momentum',     # RSI - 50 (centré)
        'volume_ratio',     # Volume / SMA20 volume
        'trend_strength',   # ADX > 25 + direction

        # Features de momentum
        'return_5d',        # Rendement 5 jours passés
        'return_10d',       # Rendement 10 jours passés
        'return_20d',       # Rendement 20 jours passés
        'volatility_10d',   # Volatilité 10 jours
        'high_52w_pct',     # % vs plus haut 52 semaines
    ]

    def __init__(self, db_path: str = "trading_data.db", model_path: str = "models/momentum_model.pkl"):
        self.db_path = db_path
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance: Optional[pd.DataFrame] = None

        # Paramètres de prédiction
        self.prediction_horizon = 20  # Jours pour atteindre +5%
        self.target_return = 0.05     # 5% de hausse

    def load_data(self, min_date: Optional[str] = None) -> pd.DataFrame:
        """
        Charge les données depuis la DB.

        Args:
            min_date: Date minimum (format YYYY-MM-DD)

        Returns:
            DataFrame avec toutes les données
        """
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT symbol, date, open, high, low, close, volume,
                   sma20_volume, hl_sma20vol, oc_sma20vol,
                   macd, macd_signal, rsi, adx, bb_high, bb_low, pct_close
            FROM historical_data
            WHERE close > 0 AND volume > 0
        """

        if min_date:
            query += f" AND date >= '{min_date}'"

        query += " ORDER BY symbol, date"

        df = pd.read_sql_query(query, conn)
        conn.close()

        df['date'] = pd.to_datetime(df['date'])
        print(f"[OK] {len(df):,} lignes chargées depuis la DB")

        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée les features supplémentaires pour le ML.

        Args:
            df: DataFrame brut

        Returns:
            DataFrame enrichi avec les features
        """
        df = df.copy()

        # Grouper par symbole pour les calculs
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df.loc[mask].copy()

            # --- Features calculées depuis les données existantes ---

            # MACD Histogram
            df.loc[mask, 'macd_hist'] = symbol_data['macd'] - symbol_data['macd_signal']

            # Position dans les Bandes de Bollinger (0 = bas, 1 = haut)
            bb_range = symbol_data['bb_high'] - symbol_data['bb_low']
            df.loc[mask, 'bb_position'] = np.where(
                bb_range > 0,
                (symbol_data['close'] - symbol_data['bb_low']) / bb_range,
                0.5
            )

            # RSI centré (momentum)
            df.loc[mask, 'rsi_momentum'] = symbol_data['rsi'] - 50

            # Volume ratio
            df.loc[mask, 'volume_ratio'] = np.where(
                symbol_data['sma20_volume'] > 0,
                symbol_data['volume'] / symbol_data['sma20_volume'],
                1.0
            )

            # Trend strength (ADX avec direction basée sur MACD)
            df.loc[mask, 'trend_strength'] = symbol_data['adx'] * np.sign(symbol_data['macd'])

            # --- Features de momentum historique ---

            # Rendements passés
            df.loc[mask, 'return_5d'] = symbol_data['close'].pct_change(5)
            df.loc[mask, 'return_10d'] = symbol_data['close'].pct_change(10)
            df.loc[mask, 'return_20d'] = symbol_data['close'].pct_change(20)

            # Volatilité
            df.loc[mask, 'volatility_10d'] = symbol_data['close'].pct_change().rolling(10).std()

            # Position vs plus haut 52 semaines
            high_52w = symbol_data['high'].rolling(252, min_periods=50).max()
            df.loc[mask, 'high_52w_pct'] = symbol_data['close'] / high_52w

        return df

    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée les labels (target) : le stock a-t-il atteint +5% dans les N prochains jours ?

        Args:
            df: DataFrame avec features

        Returns:
            DataFrame avec colonne 'target'
        """
        df = df.copy()
        df['target'] = 0

        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df.loc[mask].copy()

            # Pour chaque jour, regarder le max des N prochains jours
            future_max = symbol_data['high'].shift(-1).rolling(
                window=self.prediction_horizon,
                min_periods=1
            ).max().shift(-self.prediction_horizon + 1)

            # Calculer le rendement max potentiel
            max_return = (future_max - symbol_data['close']) / symbol_data['close']

            # Label = 1 si le rendement max >= target
            df.loc[mask, 'target'] = (max_return >= self.target_return).astype(int)

        return df

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prépare les données pour l'entraînement.

        Args:
            df: DataFrame complet

        Returns:
            X, y, feature_names
        """
        # Supprimer les lignes avec NaN dans les features
        df_clean = df.dropna(subset=self.FEATURE_COLUMNS + ['target'])

        # Exclure les derniers jours (pas de label fiable)
        cutoff_date = df_clean['date'].max() - timedelta(days=self.prediction_horizon + 5)
        df_clean = df_clean[df_clean['date'] <= cutoff_date]

        X = df_clean[self.FEATURE_COLUMNS].values
        y = df_clean['target'].values

        print(f"[OK] Données préparées: {len(X):,} échantillons")
        print(f"     Distribution: {y.mean()*100:.1f}% positifs ({y.sum():,})")

        return X, y, self.FEATURE_COLUMNS

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict:
        """
        Entraîne le modèle.

        Args:
            X: Features
            y: Labels
            feature_names: Noms des features

        Returns:
            Dictionnaire avec les métriques
        """
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Normalisation
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Créer le modèle
        if HAS_XGBOOST:
            print("[INFO] Utilisation de XGBoost")
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),  # Balance classes
                random_state=42,
                eval_metric='auc',
                early_stopping_rounds=20
            )
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                verbose=False
            )
        else:
            print("[INFO] Utilisation de RandomForest (XGBoost non installé)")
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train_scaled, y_train)

        # Évaluation
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        # Métriques
        roc_auc = roc_auc_score(y_test, y_proba)

        print("\n" + "="*50)
        print("RÉSULTATS D'ENTRAÎNEMENT")
        print("="*50)
        print(f"\nROC-AUC Score: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['< 5%', '>= 5%']))

        print("\nMatrice de confusion:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
        print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")

        # Feature importance
        if HAS_XGBOOST:
            importance = self.model.feature_importances_
        else:
            importance = self.model.feature_importances_

        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Features les plus importantes:")
        print(self.feature_importance.head(10).to_string(index=False))

        return {
            'roc_auc': roc_auc,
            'accuracy': (y_pred == y_test).mean(),
            'precision': cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0,
            'recall': cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0,
        }

    def save_model(self):
        """Sauvegarde le modèle et le scaler."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.FEATURE_COLUMNS,
            'feature_importance': self.feature_importance,
            'prediction_horizon': self.prediction_horizon,
            'target_return': self.target_return,
        }, self.model_path)

        print(f"\n[OK] Modèle sauvegardé: {self.model_path}")

    def load_model(self):
        """Charge un modèle existant."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {self.model_path}")

        data = joblib.load(self.model_path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.FEATURE_COLUMNS = data['feature_columns']
        self.feature_importance = data['feature_importance']
        self.prediction_horizon = data['prediction_horizon']
        self.target_return = data['target_return']

        print(f"[OK] Modèle chargé: {self.model_path}")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prédit la probabilité de hausse >5% pour chaque ligne.

        Args:
            df: DataFrame avec les features (doit avoir les colonnes requises)

        Returns:
            DataFrame avec colonnes 'probability' et 'prediction'
        """
        if self.model is None:
            raise ValueError("Modèle non chargé. Utilisez load_model() ou train() d'abord.")

        # Créer les features si nécessaire
        if 'macd_hist' not in df.columns:
            df = self.create_features(df)

        # Préparer les données
        df_pred = df.dropna(subset=self.FEATURE_COLUMNS)
        X = df_pred[self.FEATURE_COLUMNS].values
        X_scaled = self.scaler.transform(X)

        # Prédictions
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        # Ajouter au DataFrame
        df_pred = df_pred.copy()
        df_pred['probability'] = probabilities
        df_pred['prediction'] = predictions

        return df_pred[['symbol', 'date', 'close', 'probability', 'prediction']]

    def score_symbol(self, symbol: str, latest_only: bool = True) -> Dict:
        """
        Score un symbole spécifique.

        Args:
            symbol: Symbole à scorer
            latest_only: Si True, ne retourne que le dernier score

        Returns:
            Dictionnaire avec le score
        """
        # Charger les données du symbole
        conn = sqlite3.connect(self.db_path)
        query = f"""
            SELECT symbol, date, open, high, low, close, volume,
                   sma20_volume, hl_sma20vol, oc_sma20vol,
                   macd, macd_signal, rsi, adx, bb_high, bb_low, pct_close
            FROM historical_data
            WHERE symbol = '{symbol}'
            ORDER BY date
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return {'symbol': symbol, 'error': 'No data found'}

        df['date'] = pd.to_datetime(df['date'])

        # Prédire
        predictions = self.predict(df)

        if latest_only:
            latest = predictions.iloc[-1]
            return {
                'symbol': symbol,
                'date': latest['date'].strftime('%Y-%m-%d'),
                'close': latest['close'],
                'probability': float(latest['probability']),
                'prediction': int(latest['prediction']),
                'signal': 'BUY' if latest['probability'] >= 0.6 else 'HOLD' if latest['probability'] >= 0.4 else 'AVOID'
            }

        return predictions.to_dict('records')

    def run_full_training(self, min_date: str = "2020-01-01"):
        """
        Exécute l'entraînement complet du modèle.

        Args:
            min_date: Date minimum pour l'entraînement
        """
        print("\n" + "="*60)
        print("ENTRAÎNEMENT DU MODÈLE ML MOMENTUM")
        print("="*60)
        print(f"Horizon de prédiction: {self.prediction_horizon} jours")
        print(f"Target: hausse >= {self.target_return*100:.0f}%")
        print(f"Date min: {min_date}")
        print("="*60 + "\n")

        # 1. Charger les données
        print("[1/4] Chargement des données...")
        df = self.load_data(min_date)

        # 2. Créer les features
        print("[2/4] Création des features...")
        df = self.create_features(df)

        # 3. Créer les labels
        print("[3/4] Création des labels...")
        df = self.create_labels(df)

        # 4. Préparer et entraîner
        print("[4/4] Entraînement du modèle...")
        X, y, feature_names = self.prepare_training_data(df)
        metrics = self.train(X, y, feature_names)

        # 5. Sauvegarder
        self.save_model()

        return metrics


class MLMomentumScoring:
    """
    Classe de scoring utilisant le modèle ML entraîné.
    Compatible avec l'interface Scoring existante.
    """
    name = "MLMomentumScoring"

    def __init__(self, model_path: str = "models/momentum_model.pkl"):
        self.predictor = MLMomentumPredictor(model_path=model_path)
        try:
            self.predictor.load_model()
            self.model_loaded = True
        except FileNotFoundError:
            print("[WARN] Modèle ML non trouvé. Entraînez-le d'abord avec MLMomentumPredictor.run_full_training()")
            self.model_loaded = False

    def score(self, df: pd.DataFrame) -> int:
        """
        Score un DataFrame (compatible avec l'interface Scoring).

        Args:
            df: DataFrame avec les données OHLCV

        Returns:
            Score de 0 à 100
        """
        if not self.model_loaded:
            return 0

        try:
            # S'assurer que les colonnes sont en minuscules
            df.columns = df.columns.str.lower()

            # Ajouter un symbole fictif si absent
            if 'symbol' not in df.columns:
                df['symbol'] = 'UNKNOWN'

            # Prédire
            predictions = self.predictor.predict(df)

            if predictions.empty:
                return 0

            # Prendre la dernière probabilité et la convertir en score 0-100
            probability = predictions['probability'].iloc[-1]
            score = int(probability * 100)

            return max(0, min(score, 100))

        except Exception as e:
            print(f"[ERREUR MLMomentumScoring] {e}")
            return 0


# Script principal pour l'entraînement
if __name__ == "__main__":
    predictor = MLMomentumPredictor(
        db_path="trading_data.db",
        model_path="models/momentum_model.pkl"
    )

    # Entraîner le modèle
    metrics = predictor.run_full_training(min_date="2018-01-01")

    print("\n" + "="*60)
    print("TEST SUR QUELQUES SYMBOLES")
    print("="*60)

    # Tester sur quelques symboles
    for symbol in ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'AMD']:
        result = predictor.score_symbol(symbol)
        print(f"{symbol}: {result['probability']*100:.1f}% -> {result['signal']}")
