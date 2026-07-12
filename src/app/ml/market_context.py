"""
Contexte de marché — features de régime injectées dans le modèle ML.

Conclusion du banc d'essai régime (2026-07-11, scratchpad eval_regime_*):
les gates binaires (SPY>MM200 etc.) sélectionnent systématiquement les
mauvaises entrées dans les années difficiles (réouvertures = bear market
rallies, fermetures = V-bottoms). À la place, le contexte de marché est
fourni AU MODÈLE comme features : le meta-modèle apprend l'interaction
entre le setup du titre et l'état du marché (AFML ch. 3).

Source : séries SPY/QQQ de historical_data (mises à jour quotidiennement).
"""
import numpy as np
import pandas as pd

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.database.pg_connection import read_sql

MARKET_FEATURE_COLUMNS = [
    'spy_mom_20d',   # momentum 20 jours du SPY (aligné sur l'horizon de trade)
    'spy_vol_20d',   # volatilité réalisée 20j annualisée du SPY
    'spy_dd_60d',    # drawdown du SPY vs son max 60 jours
    'qqq_mom_20d',   # momentum 20 jours du QQQ (univers NASDAQ)
]


def load_market_features(min_date: str = "2016-01-01") -> pd.DataFrame:
    """
    Charge SPY/QQQ depuis la DB et calcule les features de contexte marché.

    Returns:
        DataFrame avec colonnes: date (datetime64) + MARKET_FEATURE_COLUMNS.
        Aucun lookahead : chaque ligne n'utilise que les données <= date.
    """
    idx = read_sql("""
        SELECT DISTINCT ON (symbol, date) symbol, date, close
        FROM historical_data
        WHERE symbol IN ('SPY', 'QQQ') AND date >= %s
        ORDER BY symbol, date, source
    """, (min_date,))
    idx['date'] = pd.to_datetime(idx['date'])

    out = None
    for sym, prefix in [('SPY', 'spy'), ('QQQ', 'qqq')]:
        s = idx[idx['symbol'] == sym].sort_values('date').set_index('date')['close']
        f = pd.DataFrame(index=s.index)
        f[f'{prefix}_mom_20d'] = s / s.shift(20) - 1
        if prefix == 'spy':
            f['spy_vol_20d'] = s.pct_change().rolling(20).std() * np.sqrt(252)
            f['spy_dd_60d'] = s / s.rolling(60).max() - 1
        out = f if out is None else out.join(f, how='outer')

    out = out[MARKET_FEATURE_COLUMNS].reset_index().rename(columns={'index': 'date'})
    return out


def merge_market_features(df: pd.DataFrame,
                          market: pd.DataFrame = None) -> pd.DataFrame:
    """
    Ajoute les features de contexte marché à un DataFrame par jointure sur
    la date. Accepte une colonne 'date' ou un DatetimeIndex. À défaut,
    utilise la dernière valeur de marché disponible (mode live sans dates)
    avec un avertissement — à éviter en backtest (lookahead).
    """
    if market is None:
        market = load_market_features()

    if 'date' in df.columns:
        dates = pd.to_datetime(df['date']).dt.normalize()
    elif isinstance(df.index, pd.DatetimeIndex):
        dates = pd.Series(df.index.normalize(), index=df.index)
    else:
        print("[MARKET][WARN] Pas de dates dans le DataFrame — utilisation du "
              "dernier contexte marché connu (ne PAS utiliser en backtest)")
        last = market.sort_values('date').iloc[-1]
        for col in MARKET_FEATURE_COLUMNS:
            df[col] = last[col]
        return df

    m = market.copy()
    m['date'] = pd.to_datetime(m['date']).dt.normalize()
    m = m.set_index('date')[MARKET_FEATURE_COLUMNS]
    joined = m.reindex(dates.values)
    for col in MARKET_FEATURE_COLUMNS:
        df[col] = joined[col].values
    return df
