"""
ML Wyckoff Predictor — meta-labeling sur événements Volume Price Analysis.

Architecture AFML (López de Prado) :
  - Signal primaire à règles : détecteurs Wyckoff (spring, test, SOS)
    de src/app/strategies/wyckoff_vpa.py — volontairement permissifs.
  - Échantillonnage ÉVÉNEMENTIEL (ch. 2) : seuls les jours d'événement
    deviennent des échantillons → labels beaucoup moins chevauchants que
    l'échantillonnage quotidien du pipeline momentum.
  - Meta-labeling (ch. 3) : le modèle prédit si le trade déclenché par
    l'événement aurait été gagnant (label triple-barrière partagé,
    barrières lues depuis stop_config.json).
  - Évaluation : split temporel purgé + balayage de seuils avec espérance
    de rendement par trade. Critère de succès :
    EV(événements filtrés ML) > EV(événements bruts) > 0, out-of-time.

Usage: python src/app/ml/ml_wyckoff_predictor.py
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.database.pg_connection import read_sql
from src.app.ml.triple_barrier import triple_barrier_labels, load_barriers_from_stop_config
from src.app.ml.market_context import merge_market_features, MARKET_FEATURE_COLUMNS
from src.app.strategies.wyckoff_vpa import (
    compute_vpa_features, detect_events, ML_FEATURE_COLUMNS)

import xgboost as xgb


class MLWyckoffPredictor:
    """Meta-labeling des événements Wyckoff/VPA."""

    # Features VPA + contexte de marché (le modèle apprend l'interaction
    # setup × régime — même approche que le pipeline momentum)
    FEATURE_COLUMNS = ML_FEATURE_COLUMNS + MARKET_FEATURE_COLUMNS

    def __init__(self, model_path: str = "models/wyckoff_model.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None

        # Barrières partagées avec le stop_manager (source de vérité)
        b = load_barriers_from_stop_config()
        self.profit_barrier = b['profit_barrier']
        self.stop_barrier = b['stop_barrier']
        self.prediction_horizon = b['horizon']
        print(f"[CONFIG] Barrieres: profit +{self.profit_barrier*100:.0f}% | "
              f"stop trailing -{self.stop_barrier*100:.0f}% | "
              f"temps {self.prediction_horizon} jours")

        # Univers tradable (mêmes filtres que le screener IB)
        self.min_price = 5.0
        self.max_price = 500.0
        self.min_avg_volume = 500_000

    def load_data(self, min_date: str = "2018-01-01") -> pd.DataFrame:
        """OHLCV brut uniquement — toutes les features sont calculées en code
        (le sma20_volume du filtre d'univers vient de compute_vpa_features)."""
        df = read_sql("""
            SELECT symbol, date, open, high, low, close, volume
            FROM historical_data
            WHERE close > 0 AND volume > 0 AND date >= %s
            ORDER BY symbol, date
        """, (min_date,))
        df['date'] = pd.to_datetime(df['date'])
        print(f"[OK] {len(df):,} lignes chargees depuis la DB")
        return df

    def build_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features VPA -> événements -> labels -> échantillons tradables."""
        print("[1/3] Features VPA...")
        df = compute_vpa_features(df)

        print("[2/3] Detection des evenements Wyckoff...")
        df = detect_events(df)
        n_events = int(df['event_any'].sum())
        counts = df.loc[df['event_any'], 'event_type'].value_counts()
        print(f"[EVENTS] {n_events:,} evenements sur {len(df):,} barres "
              f"({n_events/max(len(df),1)*100:.2f}%)")
        for etype, cnt in counts.items():
            print(f"         {etype}: {cnt:,}")

        print("[2b/3] Contexte de marché (SPY/QQQ)...")
        df = merge_market_features(df)

        print("[3/3] Labels triple-barriere...")
        df = triple_barrier_labels(
            df, self.profit_barrier, self.stop_barrier, self.prediction_horizon)

        # Échantillons = événements seulement, univers tradable, labels valides
        samples = df[df['event_any']].dropna(
            subset=self.FEATURE_COLUMNS + ['target'])
        n_before = len(samples)
        samples = samples[
            samples['close'].between(self.min_price, self.max_price)
            & (samples['sma20_volume'] >= self.min_avg_volume)
        ]
        print(f"[UNIVERS] {n_before:,} -> {len(samples):,} evenements tradables")
        samples = samples.sort_values(['date', 'symbol']).reset_index(drop=True)

        print(f"[OK] {len(samples):,} echantillons | "
              f"{samples['target'].mean()*100:.1f}% positifs | "
              f"EV brut {samples['trade_return'].mean()*100:+.2f}%/trade")
        return samples

    def train(self, samples: pd.DataFrame) -> Dict:
        """Split temporel purgé + XGBoost + balayage de seuils EV."""
        X = samples[self.FEATURE_COLUMNS].astype(float).values
        y = samples['target'].values.astype(int)
        dates = samples['date'].values
        returns = samples['trade_return'].values

        unique_dates = np.sort(np.unique(dates))
        day_idx = np.searchsorted(unique_dates, dates)
        horizon = self.prediction_horizon

        val_start_i = day_idx[int(len(X) * 0.65)]
        test_start_i = day_idx[int(len(X) * 0.80)]
        train_mask = day_idx + horizon < val_start_i
        val_mask = (day_idx >= val_start_i) & (day_idx + horizon < test_start_i)
        test_mask = day_idx >= test_start_i

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        returns_test = returns[test_mask]

        def _period(mask):
            d = dates[mask]
            return (f"{np.datetime_as_string(d.min(), unit='D')} -> "
                    f"{np.datetime_as_string(d.max(), unit='D')}")

        print(f"[INFO] Train: {len(X_train):,} ({_period(train_mask)})")
        print(f"[INFO] Val  : {len(X_val):,} ({_period(val_mask)})")
        print(f"[INFO] Test : {len(X_test):,} ({_period(test_mask)})")

        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)
        X_test_s = self.scaler.transform(X_test)

        scale_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,             # échantillons rares -> arbres peu profonds
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_weight,
            random_state=42,
            eval_metric='auc',
            early_stopping_rounds=30,
        )
        self.model.fit(X_train_s, y_train,
                       eval_set=[(X_val_s, y_val)], verbose=50)

        y_pred = self.model.predict(X_test_s)
        y_proba = self.model.predict_proba(X_test_s)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else float('nan')

        print("\n" + "=" * 50)
        print("RESULTATS - WYCKOFF VPA MODEL")
        print("=" * 50)
        print(f"\nROC-AUC Score: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=['stop/temps', 'profit']))
        cm = confusion_matrix(y_test, y_pred)
        print(f"Matrice de confusion:")
        print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
        print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")

        # --- Balayage de seuils : le critère de décision est l'EV/trade ---
        n_test_days = max(len(np.unique(dates[test_mask])), 1)
        print(f"\nBalayage de seuils sur le test set "
              f"({n_test_days} jours de bourse, EV brut hors couts):")
        print(f"{'seuil':>6} | {'entrees':>9} | {'/jour':>6} | {'precision':>9} | {'EV/trade':>9}")
        print(f"{'tous':>6} | {len(y_test):>9,} | {len(y_test)/n_test_days:>6,.1f} | "
              f"{y_test.mean()*100:>8.1f}% | {np.nanmean(returns_test)*100:>+8.2f}%")
        for th in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            sel = y_proba >= th
            n_sel = int(sel.sum())
            if n_sel == 0:
                print(f"{int(th*100):>6} | {0:>9,} | {'-':>6} | {'-':>9} | {'-':>9}")
                continue
            print(f"{int(th*100):>6} | {n_sel:>9,} | {n_sel/n_test_days:>6,.1f} | "
                  f"{y_test[sel].mean()*100:>8.1f}% | "
                  f"{np.nanmean(returns_test[sel])*100:>+8.2f}%")

        self.feature_importance = pd.DataFrame({
            'feature': self.FEATURE_COLUMNS,
            'importance': self.model.feature_importances_,
        }).sort_values('importance', ascending=False)
        print("\nTop 15 Features:")
        print(self.feature_importance.head(15).to_string(index=False))

        return {'roc_auc': roc_auc}

    def save_model(self):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.FEATURE_COLUMNS,
            'feature_importance': self.feature_importance,
            'prediction_horizon': self.prediction_horizon,
            'profit_barrier': self.profit_barrier,
            'stop_barrier': self.stop_barrier,
            'labeling': 'triple_barrier',
            'sampling': 'wyckoff_events',
        }, self.model_path)
        print(f"\n[OK] Modele sauvegarde: {self.model_path}")

    def run_full_training(self, min_date: str = "2018-01-01"):
        print("\n" + "=" * 60)
        print("ENTRAINEMENT - WYCKOFF VPA MODEL (meta-labeling)")
        print("=" * 60 + "\n")
        df = self.load_data(min_date)
        samples = self.build_samples(df)
        if len(samples) < 10_000:
            print(f"[WARN] Seulement {len(samples):,} echantillons — "
                  "verifier les seuils des detecteurs avant d'entrainer")
        metrics = self.train(samples)
        self.save_model()
        return metrics


if __name__ == "__main__":
    predictor = MLWyckoffPredictor()
    predictor.run_full_training(min_date="2018-01-01")
