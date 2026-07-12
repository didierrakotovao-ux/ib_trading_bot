import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import List

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.ml.scoring import Scoring
from src.app.strategies.wyckoff_vpa import compute_vpa_features, detect_events
from src.app.ml.market_context import (
    load_market_features, merge_market_features, MARKET_FEATURE_COLUMNS)

class WyckoffMLScoring(Scoring):
    """
    Scoring utilisant le modèle Wyckoff/VPA (meta-labeling d'événements).

    Discipline meta-labeling : le modèle n'est entraîné QUE sur les jours
    d'événement (spring/test/SOS). Par défaut (require_event=True), le score
    est 0 quand la dernière barre n'est pas un événement — sinon le modèle
    extrapolerait hors de sa distribution d'entraînement.
    """
    name = "WyckoffMLScoring"
    df: pd.DataFrame

    def __init__(self, model_path: str = "models/wyckoff_model.pkl",
                 db_path=None,  # db_path ignoré — connexion via pg_config.py
                 require_event: bool = True):
        self.df = None
        self.model = None
        self.scaler = None
        self.model_loaded = False
        self.require_event = require_event
        project_root = Path(__file__).parent.parent.parent.parent
        self.model_path = project_root / model_path

        # Colonnes chargées depuis le modèle
        self.feature_columns: List[str] = []
        # Cache du contexte marché (SPY/QQQ), partagé pour toute la session
        self._market_cache = None

        self._load_model()

    def _load_model(self):
        """Charge le modèle ML Wyckoff."""
        try:
            if self.model_path.exists():
                data = joblib.load(self.model_path)
                self.model = data['model']
                self.scaler = data['scaler']
                self.feature_columns = data['feature_columns']
                self.model_loaded = True
                print(f"[OK] Modele Wyckoff ML charge: {self.model_path}")
                print(f"     {len(self.feature_columns)} features")
            else:
                print(f"[WARN] Modele Wyckoff ML non trouve: {self.model_path}")
                print("       Entrainez-le avec: python src/app/ml/ml_wyckoff_predictor.py")
        except Exception as e:
            print(f"[ERREUR] Impossible de charger le modele Wyckoff ML: {e}")
            self.model_loaded = False

    def _create_features(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Recalcule les features Wyckoff/VPA de la même manière que le pipeline
        d'entraînement pour éviter tout train/serve skew.
        """
        _ = symbol  # API uniforme avec les autres scoring, non utilisé ici.

        local_df = df.copy()
        local_df.columns = local_df.columns.str.lower()

        # Le calcul Wyckoff/VPA requiert une colonne symbol pour groupby.
        if 'symbol' not in local_df.columns:
            local_df['symbol'] = symbol if symbol else 'UNKNOWN'

        if 'date' not in local_df.columns:
            if hasattr(local_df.index, 'dtype'):
                local_df['date'] = pd.to_datetime(local_df.index)
            else:
                local_df['date'] = pd.date_range(end=pd.Timestamp.today(), periods=len(local_df), freq='B')
        else:
            local_df['date'] = pd.to_datetime(local_df['date'])

        local_df = compute_vpa_features(local_df)
        local_df = detect_events(local_df)

        # Contexte de marché — seulement si le modèle chargé a été entraîné
        # avec (rétro-compatible avec les anciens pkl)
        if any(c in self.feature_columns for c in MARKET_FEATURE_COLUMNS):
            if self._market_cache is None:
                self._market_cache = load_market_features()
            local_df = merge_market_features(local_df, market=self._market_cache)

        return local_df

    def score(self, df: pd.DataFrame, symbol: str):
        if not self.model_loaded:
            print("[WARN] Modele Wyckoff ML non charge, retourne 0")
            return 0

        try:
            df = self._create_features(df, symbol=symbol)
            self.df = df

            if len(df) < 60:
                print(f"[WARN] Pas assez de donnees ({len(df)} < 60)")
                return 0

            # Discipline meta-labeling : ne scorer que les jours d'événement
            # (le modèle n'a jamais vu de jour ordinaire à l'entraînement)
            if self.require_event and not bool(df['event_any'].iloc[-1]):
                return 0

            # Scorer la DERNIÈRE barre ou rien — pas une barre plus ancienne
            # qui aurait survécu au dropna
            last = df[self.feature_columns].iloc[-1:]
            if last.isna().any(axis=None):
                return 0

            X = last.values
            X_scaled = self.scaler.transform(X)
            probability = self.model.predict_proba(X_scaled)[0, 1]

            score = int(probability * 100)
            return max(0, min(score, 100))

        except Exception as e:
            print(f"[ERREUR WyckoffMLScoring] {e}")
            import traceback
            traceback.print_exc()
            return 0


