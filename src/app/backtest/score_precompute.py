"""
Pré-calcul vectorisé des scores ML pour les backtests.

Problème : la boucle de backtest appelle scoring.score(df_400j) pour chaque
symbole à chaque jour simulé — ~120 000 recalculs Python de features
identiques (heures). Ici, tout est calculé UNE FOIS sur la fenêtre complète
(groupby vectorisé, exactement comme l'entraînement), puis scoré en un seul
predict_proba batch. La simulation ne fait plus que des lookups O(1).

Garanties de parité :
  - les features sont construites via les MÊMES objets de scoring que le
    live (SmoothMLScoring / WyckoffMLScoring : mêmes formules, mêmes
    colonnes du pkl, même cache secteurs) ;
  - seule différence connue : les indicateurs EWM (MACD) sont calculés sur
    l'historique complet de la fenêtre au lieu d'une tranche de 400 jours —
    plus proche de l'entraînement, écart de score marginal.

Utilisé uniquement en backtest : le chemin live (scoring.score) est intact.
"""
import time
import numpy as np
import pandas as pd

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.ml.features import compute_base_features_multi
from src.app.ml.market_context import merge_market_features, MARKET_FEATURE_COLUMNS


class PrecomputedScores:
    """Lookup (symbol, date) -> score / momentum_12_1 / fip."""

    def __init__(self, scores: dict, mom121: dict = None, fip: dict = None):
        self._scores = scores
        self._mom = mom121 or {}
        self._fip = fip or {}

    @staticmethod
    def _key(symbol, d):
        return (symbol, pd.Timestamp(d).date())

    def score(self, symbol, d) -> int:
        return self._scores.get(self._key(symbol, d), 0)

    def momentum_12_1(self, symbol, d) -> float:
        return self._mom.get(self._key(symbol, d), np.nan)

    def fip(self, symbol, d) -> float:
        return self._fip.get(self._key(symbol, d), np.nan)


def _concat_dataframes(dataframes: dict) -> pd.DataFrame:
    """dict {symbol: df indexé par date} -> frame long trié (symbol, date)."""
    parts = []
    for symbol, df in dataframes.items():
        g = df.reset_index()
        g.columns = [c.lower() for c in g.columns]
        if 'date' not in g.columns and 'index' in g.columns:
            g = g.rename(columns={'index': 'date'})
        g['symbol'] = symbol
        parts.append(g[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']])
    out = pd.concat(parts, ignore_index=True)
    out['date'] = pd.to_datetime(out['date'])
    return out.sort_values(['symbol', 'date']).reset_index(drop=True)


def _batch_scores(df: pd.DataFrame, scoring) -> dict:
    """predict_proba batch sur les lignes complètes -> dict clé -> score."""
    feats = scoring.feature_columns
    valid = df[feats].notna().all(axis=1)
    sub = df.loc[valid]
    if sub.empty:
        return {}
    X = scoring.scaler.transform(sub[feats].astype(float).values)
    proba = scoring.model.predict_proba(X)[:, 1]
    scores = np.clip((proba * 100).astype(int), 0, 100)
    keys = zip(sub['symbol'], sub['date'].dt.date)
    return {k: int(s) for k, s in zip(keys, scores)}


def precompute_momentum(dataframes: dict, model_path: str = None,
                        verbose: bool = True) -> PrecomputedScores:
    """Scores smooth-momentum + pré-filtres Gray & Vogel (mom 12-1, FIP)."""
    from src.app.ml.ml_smooth_scoring import SmoothMLScoring
    t0 = time.time()
    scoring = SmoothMLScoring(
        model_path=model_path or "models/smooth_momentum_model.pkl")

    df = _concat_dataframes(dataframes)
    if verbose:
        print(f"[PRECOMPUTE] Momentum: {len(df):,} lignes, "
              f"{df['symbol'].nunique():,} symboles")

    # === Features — mêmes sources que SmoothMLScoring._create_features ===
    df = compute_base_features_multi(df)
    g = df.groupby('symbol')['close']
    df['smoothness_20d'] = g.transform(lambda x: scoring._rolling_r2(x, 20))
    df['smoothness_50d'] = g.transform(lambda x: scoring._rolling_r2(x, 50))
    month = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    if any(c in scoring.feature_columns for c in MARKET_FEATURE_COLUMNS):
        df = merge_market_features(df)
    # Secteurs : one-hot selon les colonnes du modèle (comme le scorer live)
    sector = df['symbol'].map(
        lambda s: scoring._sector_cache.get(s, 'Unknown'))
    sector_col = 'sector_' + sector
    for col in scoring.sector_columns:
        df[col] = (sector_col == col).astype(int)

    scores = _batch_scores(df, scoring)

    # === Pré-filtres Gray & Vogel, vectorisés (formules de MomentumFilters) ===
    c = df.groupby('symbol')['close']
    # calc_momentum_12_1 : (close[-21] - close[-252]) / close[-252]
    p12 = c.shift(251)
    df['_mom121'] = (c.shift(20) - p12) / (p12 + 1e-10)
    # calc_fip : 251 rendements des 252 derniers closes
    ret = c.pct_change()
    pos = (ret > 0.005).groupby(df['symbol']).transform(
        lambda x: x.rolling(251, min_periods=251).mean())
    neg = (ret < -0.005).groupby(df['symbol']).transform(
        lambda x: x.rolling(251, min_periods=251).mean())
    df['_fip'] = (1 - pos - neg) * (neg - pos)

    keys = list(zip(df['symbol'], df['date'].dt.date))
    mom121 = {k: v for k, v in zip(keys, df['_mom121']) if not np.isnan(v)}
    fip = {k: v for k, v in zip(keys, df['_fip']) if not np.isnan(v)}

    if verbose:
        print(f"[PRECOMPUTE] Momentum prêt en {time.time()-t0:.1f}s "
              f"({len(scores):,} scores, {len(mom121):,} momentum, {len(fip):,} FIP)")
    return PrecomputedScores(scores, mom121, fip)


def precompute_wyckoff(dataframes: dict, model_path: str = None,
                       verbose: bool = True) -> PrecomputedScores:
    """Scores Wyckoff — uniquement sur les jours d'événement (spring/test/SOS)."""
    from src.app.ml.ml_wyckoff_scoring import WyckoffMLScoring
    from src.app.strategies.wyckoff_vpa import compute_vpa_features, detect_events
    t0 = time.time()
    scoring = WyckoffMLScoring(
        model_path=model_path or "models/wyckoff_model.pkl")

    df = _concat_dataframes(dataframes)
    if verbose:
        print(f"[PRECOMPUTE] Wyckoff: {len(df):,} lignes, "
              f"{df['symbol'].nunique():,} symboles")

    df = compute_vpa_features(df)
    df = detect_events(df)
    if any(c in scoring.feature_columns for c in MARKET_FEATURE_COLUMNS):
        df = merge_market_features(df)

    # Discipline meta-labeling : ne scorer que les jours d'événement
    events = df[df['event_any']].copy()
    scores = _batch_scores(events, scoring)

    if verbose:
        print(f"[PRECOMPUTE] Wyckoff prêt en {time.time()-t0:.1f}s "
              f"({int(df['event_any'].sum()):,} événements, {len(scores):,} scores)")
    return PrecomputedScores(scores)


def precompute_for_strategy(scoring_type: str, dataframes: dict,
                            model_path: str = None,
                            verbose: bool = True) -> PrecomputedScores:
    """Choisit le pré-calcul selon le type de stratégie du wrapper."""
    if scoring_type == "wyckoff_ml":
        return precompute_wyckoff(dataframes, model_path, verbose)
    return precompute_momentum(dataframes, model_path, verbose)
