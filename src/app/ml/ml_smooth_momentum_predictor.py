"""
ML Smooth Momentum Predictor - Extension du modèle momentum avec features de smoothness,
saisonnalité et secteur.

Features additionnelles:
- Smoothness: R² de régression linéaire sur fenêtres glissantes (régularité du prix)
- Saisonnalité: Encodage cyclique du mois (sin/cos)
- Secteur: One-hot encoding du secteur GICS depuis symbol_metadata

Usage: python ml_smooth_momentum_predictor.py
"""
import argparse
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
from src.app.database.pg_connection import get_conn, read_sql, read_sql_ca
from src.app.ml.triple_barrier import triple_barrier_labels
from src.app.ml.features import compute_base_features_multi, BASE_FEATURE_COLUMNS as SHARED_BASE_COLUMNS
from src.app.ml.market_context import merge_market_features, MARKET_FEATURE_COLUMNS
from src.app.strategies.momentum_filters import MomentumFilters

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

    def __init__(self, db_path=None,
                 model_path: str = "models/smooth_momentum_model.pkl",
                 market: str = "us"):
        """
        Args:
            market: 'us' (base stockus) ou 'ca' (base stockca). Le contexte
                marché reste SPY/QQQ dans les deux cas (le beta cross-border
                des titres TSX est réel) ; le filtre earnings ne couvre que
                les États-Unis (EDGAR) — pass-through côté CA.
        """
        # db_path accepté pour compatibilité (train_smooth_ml_ca.py) mais
        # ignoré — la connexion passe par pg_config.py
        _ = db_path
        if market not in ("us", "ca"):
            raise ValueError(f"market={market!r} — attendu 'us' ou 'ca'")
        self.market = market
        self.model_path = Path(model_path)
        self.model = None
        self.calibrator = None
        self.scaler = StandardScaler()
        self.feature_importance: Optional[pd.DataFrame] = None
        self.training_date_min: Optional[str] = None
        self.training_date_max: Optional[str] = None

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
            # label_horizon PRIME sur max_holding_days : la barrière
            # d'EXÉCUTION (40j) peut être plus longue que l'horizon des
            # LABELS (20j) sans faire dériver l'entraînement
            time_cfg = cfg.get("time", {})
            self.prediction_horizon = int(
                time_cfg.get("label_horizon")
                or time_cfg.get("max_holding_days")
                or self.prediction_horizon)
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

    def load_data(self, min_date: Optional[str] = None,
                  max_date: Optional[str] = None) -> pd.DataFrame:
        """
        Charge les données depuis la DB avec les métadonnées de secteur.
        market='us' -> base stockus (read_sql) ; market='ca' -> base stockca
        (read_sql_ca), avec repli sans jointure secteur si symbol_metadata
        n'existe pas côté CA.
        """
        # OHLCV brut uniquement — les colonnes d'indicateurs de la DB sont
        # ignorées (à zéro pour la quasi-totalité des symboles, bug
        # compute_features). Tout est recalculé en code (src/app/ml/features.py).
        reader = read_sql if self.market == "us" else read_sql_ca
        print(f"[DATA] Marché: {self.market.upper()}")

        # La base CA n'a pas forcément symbol_metadata : détection propre
        # plutôt qu'un échec/rollback de la jointure
        has_metadata = True
        probe = reader("""
            SELECT COUNT(*) AS n FROM information_schema.tables
            WHERE table_name = 'symbol_metadata'
        """)
        has_metadata = int(probe['n'].iloc[0]) > 0
        if not has_metadata:
            print("[DATA] Pas de table symbol_metadata — secteurs = 'Unknown'")

        if has_metadata:
            query = """
                SELECT h.symbol, h.date, h.open, h.high, h.low, h.close, h.volume,
                       COALESCE(m.sector, 'Unknown') as sector
                FROM historical_data h
                LEFT JOIN symbol_metadata m ON h.symbol = m.symbol
                WHERE h.close > 0 AND h.volume > 0
            """
        else:
            query = """
                SELECT h.symbol, h.date, h.open, h.high, h.low, h.close, h.volume,
                       'Unknown' as sector
                FROM historical_data h
                WHERE h.close > 0 AND h.volume > 0
            """

        if min_date:
            query += f" AND h.date >= '{min_date}'"
        if max_date:
            query += f" AND h.date <= '{max_date}'"

        query += " ORDER BY h.symbol, h.date"

        df = reader(query)

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

    def _passes_momentum_filters(self, df: pd.DataFrame) -> bool:
        """Applique le filtre momentum 12-1 + FIP avant le scoring."""
        if len(df) < 252:
            return False

        data = df.sort_values('date') if 'date' in df.columns else df
        momentum_12_1 = MomentumFilters.calc_momentum_12_1(data)
        if np.isnan(momentum_12_1) or momentum_12_1 <= 0:
            return False

        fip = MomentumFilters.calc_fip(data)
        if np.isnan(fip) or fip >= 0:
            return False

        return True

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

        # Nombre d'arbres FIXE, sans early stopping : la courbe d'AUC de
        # validation est plate à ±0.005 entre 1 et 60 arbres (runs des
        # 2026-07-11/13) — l'early stopping y choisissait au hasard entre un
        # modèle quasi vide (best_iter=0, AUC test 0.611) et un bon modèle
        # (best_iter~40-60, AUC test 0.644) selon le bruit du jour. 100
        # arbres = zone du bon run ; la validation sert à la calibration.
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_weight,
            random_state=42,
            eval_metric='auc',
        )

        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=50
        )

        # --- Calibration isotonique (ajustée sur la VALIDATION) ---
        # scale_pos_weight compresse les probabilités brutes vers le milieu :
        # le balayage saturait à 70-75. La calibration est monotone (les
        # classements sont préservés) mais redonne aux scores le sens de
        # vraie probabilité de gain et étale la queue haute -> sélection fine.
        from sklearn.isotonic import IsotonicRegression
        proba_val_raw = self.model.predict_proba(X_val_scaled)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(proba_val_raw, y_val)

        # Evaluation sur probabilités CALIBRÉES.
        # NB: pas de rapport de classification à cutoff 0.5 — le plafond des
        # probabilités calibrées dépend du régime de l'année de validation
        # (0.38 pour le modèle 2010-2017, 0.80 pour 2018-2026) : à 0.5 la
        # matrice peut être structurellement vide (TP=FP=0) sans que le
        # modèle soit en cause. L'évaluation se fait au POINT DE
        # FONCTIONNEMENT réel, déterminé par le balayage ci-dessous.
        y_proba_raw = self.model.predict_proba(X_test_scaled)[:, 1]
        y_proba = self.calibrator.predict(y_proba_raw)

        roc_auc = roc_auc_score(y_test, y_proba_raw)
        print(f"\n[CALIBRATION] proba brute max={y_proba_raw.max():.2f} -> "
              f"calibrée max={y_proba.max():.2f} | "
              f"moyenne {y_proba_raw.mean():.2f} -> {y_proba.mean():.2f}")

        print("\n" + "=" * 50)
        print("RESULTATS - SMOOTH MOMENTUM MODEL")
        print("=" * 50)
        print(f"\nROC-AUC Score: {roc_auc:.4f}")

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
        threshold_rows = []
        # Grille étendue vers le bas : le plafond des probabilités calibrées
        # dépend du régime de l'année de validation (ex. modèle 2010-2017 :
        # max 0.38 car sa validation 2015-16 couvrait deux krachs) — une
        # grille qui démarre à 50 peut ne rien voir du tout.
        for th in [0.25, 0.30, 0.35, 0.40, 0.45,
                   0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            sel = y_proba >= th
            n_sel = int(sel.sum())
            if n_sel == 0:
                print(f"{int(th*100):>6} | {0:>10,} | {'-':>7} | {'-':>9} | {'-':>9}")
                continue
            precision_sel = float(y_test[sel].mean())
            ev_sel = float(np.nanmean(returns_test[sel]))
            threshold_rows.append({
                'threshold': th,
                'n_entries': n_sel,
                'entries_per_day': n_sel / n_test_days,
                'precision': precision_sel,
                'ev_per_trade': ev_sel,
            })
            print(f"{int(th*100):>6} | {n_sel:>10,} | {n_sel/n_test_days:>7,.1f} | "
                  f"{precision_sel*100:>8.1f}% | {ev_sel*100:>+8.2f}%")

        recommendations = {}
        for target_precision in (0.50, 0.66):
            eligible = [r for r in threshold_rows if r['precision'] >= target_precision]
            if not eligible:
                recommendations[f'precision_{int(target_precision*100)}'] = None
                continue
            best = max(eligible, key=lambda r: r['ev_per_trade'])
            recommendations[f'precision_{int(target_precision*100)}'] = best

        print("\nRecommandations de seuil (sur test out-of-time):")
        for label in ("precision_50", "precision_66"):
            rec = recommendations[label]
            if rec is None:
                print(f"  - {label}: aucun seuil du balayage n'atteint cette precision")
            else:
                print(
                    f"  - {label}: seuil {int(rec['threshold']*100)} "
                    f"| precision {rec['precision']*100:.1f}% "
                    f"| EV/trade {rec['ev_per_trade']*100:+.2f}% "
                    f"| entrees/jour {rec['entries_per_day']:.1f}"
                )

        # --- POINT DE FONCTIONNEMENT : la matrice qui a du sens ---
        # Seuil retenu = meilleure EV parmi les lignes du balayage avec un
        # minimum d'échantillons (n >= 30). Remplace l'ancien rapport à
        # cutoff 0.5, structurellement vide quand le plafond calibré < 0.5.
        op = None
        eligible_rows = [r for r in threshold_rows if r['n_entries'] >= 30]
        if eligible_rows:
            op = max(eligible_rows, key=lambda r: r['ev_per_trade'])
            sel_op = y_proba >= op['threshold']
            tp = int((y_test[sel_op] == 1).sum())
            fp = int((y_test[sel_op] == 0).sum())
            print(f"\nPoint de fonctionnement (EV max, n>=30) : "
                  f"seuil {int(op['threshold']*100)}")
            print(f"  {op['n_entries']:,} signaux ({op['entries_per_day']:.1f}/j) | "
                  f"TP={tp:,} FP={fp:,} | précision {op['precision']*100:.1f}% | "
                  f"EV/trade {op['ev_per_trade']*100:+.2f}%")
            print("  (le recall n'est pas une métrique pertinente : capacité "
                  "5 slots, on ne cherche pas à capturer tous les gagnants)")

        # --- Précision/EV du TOP-N PAR JOUR : la métrique de production ---
        # La stratégie ne prend pas "tous les scores >= seuil" mais les N
        # meilleurs du jour. La qualité du top-N quotidien est donc la vraie
        # précision opérationnelle — mécaniquement supérieure à celle des
        # buckets de seuil.
        dtest = pd.DataFrame({'date': dates[test_mask], 'proba': y_proba,
                              'y': y_test, 'ret': returns_test})
        print("\nTop-N par jour (probabilités calibrées, test out-of-time):")
        top_n_stats = {}
        for n in (1, 3, 5):
            topn = dtest.sort_values(['date', 'proba'], ascending=[True, False]) \
                        .groupby('date').head(n)
            stats = {
                'n_entries': int(len(topn)),
                'precision': float(topn['y'].mean()),
                'ev_per_trade': float(np.nanmean(topn['ret'])),
                'score_plancher_moyen': float(
                    topn.groupby('date')['proba'].min().mean() * 100),
            }
            top_n_stats[f'top_{n}'] = stats
            print(f"  top-{n}/jour : precision {stats['precision']*100:.1f}% | "
                  f"EV/trade {stats['ev_per_trade']*100:+.2f}% | "
                  f"score plancher moyen {stats['score_plancher_moyen']:.0f}")

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

        # accuracy/precision/recall évalués AU POINT DE FONCTIONNEMENT
        # (plus au cutoff 0.5, vide de sens sur probabilités calibrées)
        if op is not None:
            sel_op = y_proba >= op['threshold']
            precision_op = float(op['precision'])
            recall_op = float((y_test[sel_op] == 1).sum() / max((y_test == 1).sum(), 1))
            accuracy_op = float(((y_proba >= op['threshold']).astype(int) == y_test).mean())
        else:
            precision_op = recall_op = accuracy_op = 0.0

        return {
            'roc_auc': roc_auc,
            'operating_threshold': op['threshold'] if op else None,
            'accuracy': accuracy_op,
            'precision': precision_op,
            'recall': recall_op,
            'threshold_scan': threshold_rows,
            'threshold_recommendations': recommendations,
            'top_n_stats': top_n_stats,
        }

    def save_model(self):
        """Sauvegarde le modèle avec les métadonnées de secteur."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        # Filet de sécurité : conserver la version précédente en .prev
        # (le 2026-07-18, une expérience 2010-2017 a écrasé le modèle de
        # production sans sauvegarde)
        if self.model_path.exists():
            import shutil
            prev = self.model_path.with_suffix('.pkl.prev')
            shutil.copy2(self.model_path, prev)
            print(f"[SAVE] Version précédente conservée: {prev.name}")

        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            # Calibration isotonique (val) : les scorers l'appliquent après
            # predict_proba pour retrouver de vraies probabilités
            'calibrator': getattr(self, 'calibrator', None),
            'feature_columns': self.FEATURE_COLUMNS,
            'base_feature_columns': self.BASE_FEATURE_COLUMNS,
            'new_feature_columns': self.NEW_FEATURE_COLUMNS,
            'sector_columns': self.sector_columns,
            'market_feature_columns': MARKET_FEATURE_COLUMNS,
            'feature_importance': self.feature_importance,
            'prediction_horizon': self.prediction_horizon,
            'profit_barrier': self.profit_barrier,
            'stop_barrier': self.stop_barrier,
            'training_date_min': self.training_date_min,
            'training_date_max': self.training_date_max,
            'threshold_scan': getattr(self, 'threshold_scan', None),
            'threshold_recommendations': getattr(self, 'threshold_recommendations', None),
            'labeling': 'triple_barrier',
        }, self.model_path)

        print(f"\n[OK] Modele sauvegarde: {self.model_path}")

    def run_full_training(self, date_min: str = "2018-01-01",
                          date_max: Optional[str] = None,
                          min_date: Optional[str] = None):
        """Execute l'entraînement complet."""
        if min_date is not None:
            date_min = min_date

        # Garde-fou : une expérience datée (--date-max) qui écrirait sur le
        # chemin de PRODUCTION est redirigée vers un fichier d'expérience.
        # (La production a été écrasée deux fois le 2026-07-18 par des tests
        # d'ère lancés sans --model-path.) Pour écraser volontairement la
        # production avec une fenêtre datée, passer --model-path explicite.
        if date_max is not None and self.model_path.name == "smooth_momentum_model.pkl":
            exp_name = (f"exp_smooth_{date_min.replace('-', '')}"
                        f"_{date_max.replace('-', '')}.pkl")
            self.model_path = self.model_path.parent / exp_name
            print(f"[SAVE] Expérience datée détectée -> modèle redirigé vers "
                  f"{exp_name} (la production n'est pas touchée)")

        print("\n" + "=" * 60)
        print("ENTRAINEMENT - SMOOTH MOMENTUM MODEL")
        print("=" * 60)
        print(f"Labels triple-barriere: profit +{self.profit_barrier*100:.0f}% | "
              f"stop trailing -{self.stop_barrier*100:.0f}% | temps {self.prediction_horizon} jours")
        print(f"Features: base + smoothness + saisonnalite + secteur")
        print(f"Date range: {date_min} -> {date_max or 'fin'}")
        print("=" * 60 + "\n")

        self.training_date_min = date_min
        self.training_date_max = date_max

        print("[1/4] Chargement des donnees...")
        df = self.load_data(date_min, date_max)

        print("[2/4] Creation des features...")
        df = self.create_features(df)

        print("[3/4] Creation des labels...")
        df = self.create_labels(df)

        print("[4/4] Entrainement du modele...")
        X, y, dates, returns, feature_names = self.prepare_training_data(df)
        metrics = self.train(X, y, dates, returns, feature_names)
        self.threshold_scan = metrics.get('threshold_scan')
        self.threshold_recommendations = metrics.get('threshold_recommendations')

        self.save_model()

        return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement du modèle smooth momentum")
    parser.add_argument("--date-min", default="2018-01-01",
                        help="Date de début de l'entraînement (YYYY-MM-DD)")
    parser.add_argument("--min-date", default=None,
                        help="Alias rétrocompatible de --date-min")
    parser.add_argument("--date-max", default=None,
                        help="Date de fin de l'entraînement (YYYY-MM-DD)")
    parser.add_argument("--model-path", default="models/smooth_momentum_model.pkl",
                        help="Chemin du fichier modèle à sauvegarder")
    args = parser.parse_args()

    predictor = MLSmoothMomentumPredictor(
        model_path=args.model_path
    )

    metrics = predictor.run_full_training(
        date_min=args.date_min,
        date_max=args.date_max,
        min_date=args.min_date,
    )
