"""
ML Smooth Momentum Predictor - Extension du modèle momentum avec features de smoothness,
saisonnalité et secteur.

Features additionnelles:
- Smoothness: R² de régression linéaire sur fenêtres glissantes (régularité du prix)
- Saisonnalité: Encodage cyclique du mois (sin/cos)
- Secteur: One-hot encoding du secteur GICS depuis symbol_metadata

Usage: python ml_smooth_momentum_predictor.py
"""
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.database.pg_connection import get_conn, read_sql
from src.app.ml.triple_barrier import triple_barrier_labels
from src.app.ml.features import compute_base_features_multi, BASE_FEATURE_COLUMNS as SHARED_BASE_COLUMNS
from src.app.ml.market_context import merge_market_features, MARKET_FEATURE_COLUMNS

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
    Labels triple-barrière : prédit si la barrière de profit (+8%) est touchée
    avant le trailing stop (-5%) et avant la barrière de temps (20 jours),
    c'est-à-dire si le trade réel aurait été gagnant.
    """

    # Features de base — liste partagée avec le scoring (src/app/ml/features.py)
    BASE_FEATURE_COLUMNS = SHARED_BASE_COLUMNS

    # Nouvelles features (hors secteur, qui est dynamique)
    NEW_FEATURE_COLUMNS = [
        'smoothness_20d',   # R² régression linéaire 20 jours
        'smoothness_50d',   # R² régression linéaire 50 jours
        'month_sin',        # sin(2π * mois / 12)
        'month_cos',        # cos(2π * mois / 12)
    ]

    def __init__(self, _db_path=None,
                 model_path: str = "models/smooth_momentum_model.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance: Optional[pd.DataFrame] = None

        # Paramètres de labeling triple-barrière (López de Prado, AFML ch. 3)
        # Chargés depuis stop_config.json : le label DOIT décrire les mêmes
        # barrières que celles que le stop_manager applique en réel
        self.prediction_horizon = 20   # barrière temporelle (jours de bourse)
        self.profit_barrier = 0.08     # barrière haute : % depuis le close d'entrée
        self.stop_barrier = 0.08       # barrière basse : trailing % depuis le HWM
        self._load_barriers_from_stop_config()

        # Univers tradable — mêmes filtres que le screener IB de la stratégie.
        # Appliqué point-in-time sur les échantillons d'entraînement (un titre
        # entre/sort de l'univers selon son prix/volume du moment).
        self.min_price = 5.0
        self.max_price = 1000.0
        self.min_avg_volume = 500_000  # sur sma20_volume

    def _load_barriers_from_stop_config(self):
        """
        Synchronise les barrières du label avec stop_config.json (source de
        vérité du stop_manager). Fallback silencieux sur les valeurs par
        défaut si le fichier est absent ou incomplet.
        """
        cfg_path = Path(__file__).resolve().parents[3] / "stop_config.json"
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.stop_barrier = float(cfg["protection"]["pct"]) / 100
            self.profit_barrier = float(cfg["profit"]["fixed_pct"]) / 100
            self.prediction_horizon = int(
                cfg.get("time", {}).get("max_holding_days", self.prediction_horizon)
            ) or self.prediction_horizon
            print(f"[CONFIG] Barrieres chargees depuis stop_config.json: "
                  f"profit +{self.profit_barrier*100:.0f}% | "
                  f"stop trailing -{self.stop_barrier*100:.0f}% | "
                  f"temps {self.prediction_horizon} jours")
        except Exception as e:
            print(f"[CONFIG][WARN] stop_config.json non lu ({e}) — barrieres par defaut")

        # Colonnes de secteur (déterminées à l'entraînement)
        self.sector_columns: List[str] = []

        # Liste complète des features (construite dynamiquement)
        self.FEATURE_COLUMNS: List[str] = []

    def load_data(self, min_date: Optional[str] = None) -> pd.DataFrame:
        """Charge les données depuis la DB avec les métadonnées de secteur."""
        # OHLCV brut uniquement — les colonnes d'indicateurs de la DB sont
        # ignorées (à zéro pour la quasi-totalité des symboles, bug
        # compute_features). Tout est recalculé en code (src/app/ml/features.py).
        query = """
            SELECT h.symbol, h.date, h.open, h.high, h.low, h.close, h.volume,
                   COALESCE(m.sector, 'Unknown') as sector
            FROM historical_data h
            LEFT JOIN symbol_metadata m ON h.symbol = m.symbol
            WHERE h.close > 0 AND h.volume > 0
        """

        if min_date:
            query += f" AND h.date >= '{min_date}'"

        query += " ORDER BY h.symbol, h.date"

        df = read_sql(query)

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
        # === Features de base — calculées en code depuis l'OHLCV ===
        # Module partagé avec le scoring live (mêmes formules, pas de skew)
        print("[FEATURES] Calcul des features de base (OHLCV -> indicateurs)...")
        df = compute_base_features_multi(df)

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

        # === Contexte de marché (régime) ===
        # Le banc d'essai a montré qu'un gate binaire dégrade l'EV : le
        # contexte est fourni au modèle, qui apprend l'interaction
        print("[FEATURES] Ajout du contexte de marché (SPY/QQQ)...")
        df = merge_market_features(df)

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
            MARKET_FEATURE_COLUMNS +
            self.sector_columns
        )

        print(f"[FEATURES] {len(self.FEATURE_COLUMNS)} features au total "
              f"({len(self.BASE_FEATURE_COLUMNS)} base + "
              f"{len(self.NEW_FEATURE_COLUMNS)} nouvelles + "
              f"{len(MARKET_FEATURE_COLUMNS)} marché + "
              f"{len(self.sector_columns)} secteurs)")

        return df

    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Labels triple-barrière (AFML ch. 3) — voir src/app/ml/triple_barrier.py
        pour la sémantique complète (module partagé avec le pipeline wyckoff).
        """
        return triple_barrier_labels(
            df,
            profit_barrier=self.profit_barrier,
            stop_barrier=self.stop_barrier,
            horizon=self.prediction_horizon,
        )

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prépare les données pour l'entraînement.
        Trie par date (indispensable pour un split temporel) et retire les
        échantillons dont la fenêtre de label (prediction_horizon jours de
        bourse) déborde de l'historique disponible.
        """
        df_clean = df.dropna(subset=self.FEATURE_COLUMNS + ['target'])

        # Univers tradable uniquement — appliqué APRÈS le calcul des
        # features/labels (séries continues) mais AVANT l'échantillonnage :
        # le modèle n'apprend et n'est évalué que sur ce que la stratégie
        # peut réellement trader.
        n_before = len(df_clean)
        tradable = (
            df_clean['close'].between(self.min_price, self.max_price)
            & (df_clean['sma20_volume'] >= self.min_avg_volume)
        )
        df_clean = df_clean[tradable]
        print(f"[UNIVERS] {n_before:,} -> {len(df_clean):,} echantillons tradables "
              f"(prix {self.min_price:.0f}-{self.max_price:.0f}$, "
              f"volume moyen >= {self.min_avg_volume:,})")

        # Calendrier global des jours de bourse : permet de raisonner en jours
        # de trading plutôt qu'en jours calendaires pour la fenêtre de label
        unique_dates = np.sort(df_clean['date'].unique())
        day_idx = np.searchsorted(unique_dates, df_clean['date'].values)

        # Retirer les échantillons dont le label est partiellement observé
        max_valid_idx = len(unique_dates) - 1 - self.prediction_horizon
        df_clean = df_clean[day_idx <= max_valid_idx]

        # Tri temporel strict (avant tri par symbole) — le split 80/20 doit
        # couper dans le temps, pas dans l'alphabet des symboles
        df_clean = df_clean.sort_values(['date', 'symbol']).reset_index(drop=True)

        X = df_clean[self.FEATURE_COLUMNS].values
        y = df_clean['target'].values.astype(int)
        dates = df_clean['date'].values
        returns = df_clean['trade_return'].values

        print(f"[OK] Donnees preparees: {len(X):,} echantillons")
        print(f"     Periode: {df_clean['date'].min().date()} -> {df_clean['date'].max().date()}")
        print(f"     Distribution: {y.mean()*100:.1f}% positifs ({y.sum():,})")
        print(f"     Rendement moyen par trade (toutes entrees): {np.nanmean(returns)*100:+.2f}%")

        return X, y, dates, returns, self.FEATURE_COLUMNS

    def train(self, X: np.ndarray, y: np.ndarray, dates: np.ndarray,
              returns: np.ndarray, feature_names: List[str]) -> Dict:
        """
        Entraîne le modèle XGBoost avec un split temporel strict et purge
        des labels chevauchants (López de Prado, AFML ch. 7).

        Découpage : train | purge | validation | purge | test
        - Les labels regardent prediction_horizon jours de bourse dans le
          futur : tout échantillon dont la fenêtre de label déborde dans le
          bloc suivant est purgé (sinon fuite d'information).
        - L'early stopping utilise le set de validation, jamais le test.
        """
        # Indices en jours de bourse sur le calendrier global
        unique_dates = np.sort(np.unique(dates))
        day_idx = np.searchsorted(unique_dates, dates)
        horizon = self.prediction_horizon

        # Bornes temporelles (X est trié par date) : 65% train / 15% val / 20% test
        val_start_i = day_idx[int(len(X) * 0.65)]
        test_start_i = day_idx[int(len(X) * 0.80)]

        # Purge : la fenêtre de label [t+1, t+horizon] ne doit pas déborder
        # dans le bloc suivant
        train_mask = day_idx + horizon < val_start_i
        val_mask = (day_idx >= val_start_i) & (day_idx + horizon < test_start_i)
        test_mask = day_idx >= test_start_i

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        returns_test = returns[test_mask]

        def _period(mask):
            d = dates[mask]
            return f"{np.datetime_as_string(d.min(), unit='D')} -> {np.datetime_as_string(d.max(), unit='D')}"

        purged = len(X) - len(X_train) - len(X_val) - len(X_test)
        print(f"[INFO] Train: {len(X_train):,} ({_period(train_mask)})")
        print(f"[INFO] Val  : {len(X_val):,} ({_period(val_mask)})")
        print(f"[INFO] Test : {len(X_test):,} ({_period(test_mask)})")
        print(f"[INFO] Purge: {purged:,} echantillons retires aux frontieres ({horizon} jours de bourse)")
        print(f"[INFO] Train - Classe 0: {sum(y_train==0):,}, Classe 1: {sum(y_train==1):,}")
        print(f"[INFO] Test  - Classe 0: {sum(y_test==0):,}, Classe 1: {sum(y_test==1):,}")

        # Normalisation (fit sur train uniquement)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
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
            eval_set=[(X_val_scaled, y_val)],
            verbose=50
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
        print(classification_report(y_test, y_pred, target_names=['stop/temps', 'profit +8%']))

        cm = confusion_matrix(y_test, y_pred)
        print(f"Matrice de confusion:")
        print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
        print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")

        # --- Balayage de seuils : du score ML à la décision de trading ---
        # EV/trade = rendement moyen réalisé (barrières) des entrées dont le
        # score dépasse le seuil. C'est ce chiffre qui fixe le seuil d'entrée
        # en production, pas l'AUC.
        n_test_days = max(len(np.unique(dates[test_mask])), 1)
        print(f"\nBalayage de seuils sur le test set "
              f"({n_test_days} jours de bourse, EV brut hors couts):")
        print(f"{'seuil':>6} | {'entrees':>10} | {'/jour':>7} | {'precision':>9} | {'EV/trade':>9}")
        base_ev = np.nanmean(returns_test)
        print(f"{'tous':>6} | {len(y_test):>10,} | {len(y_test)/n_test_days:>7,.0f} | "
              f"{y_test.mean()*100:>8.1f}% | {base_ev*100:>+8.2f}%")
        for th in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            sel = y_proba >= th
            n_sel = int(sel.sum())
            if n_sel == 0:
                print(f"{int(th*100):>6} | {0:>10,} | {'-':>7} | {'-':>9} | {'-':>9}")
                continue
            print(f"{int(th*100):>6} | {n_sel:>10,} | {n_sel/n_test_days:>7,.1f} | "
                  f"{y_test[sel].mean()*100:>8.1f}% | {np.nanmean(returns_test[sel])*100:>+8.2f}%")

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
            'market_feature_columns': MARKET_FEATURE_COLUMNS,
            'feature_importance': self.feature_importance,
            'prediction_horizon': self.prediction_horizon,
            'profit_barrier': self.profit_barrier,
            'stop_barrier': self.stop_barrier,
            'labeling': 'triple_barrier',
        }, self.model_path)

        print(f"\n[OK] Modele sauvegarde: {self.model_path}")

    def run_full_training(self, min_date: str = "2018-01-01"):
        """Execute l'entraînement complet."""
        print("\n" + "=" * 60)
        print("ENTRAINEMENT - SMOOTH MOMENTUM MODEL")
        print("=" * 60)
        print(f"Labels triple-barriere: profit +{self.profit_barrier*100:.0f}% | "
              f"stop trailing -{self.stop_barrier*100:.0f}% | temps {self.prediction_horizon} jours")
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
        X, y, dates, returns, feature_names = self.prepare_training_data(df)
        metrics = self.train(X, y, dates, returns, feature_names)

        self.save_model()

        return metrics


if __name__ == "__main__":
    predictor = MLSmoothMomentumPredictor(
        model_path="models/smooth_momentum_model.pkl"
    )

    metrics = predictor.run_full_training(min_date="2018-01-01")
