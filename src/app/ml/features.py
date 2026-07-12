"""
Features de base — SOURCE UNIQUE, toujours calculées depuis l'OHLCV brut.

Règle : les colonnes d'indicateurs de la DB (sma20_volume, rsi, macd, …)
ne sont JAMAIS utilisées pour le scoring ou l'entraînement. Elles se sont
révélées à zéro pour la quasi-totalité des symboles (et pour tous depuis
mi-janvier 2026, bug compute_features), et le pattern « recalculer si la
colonne est absente » laissait passer ces zéros silencieusement.

Ce module est partagé par le scoring live (ml_smooth_scoring) et
l'entraînement (ml_smooth_momentum_predictor) : mêmes formules des deux
côtés, plus de train/serve skew. Formules déterministes uniquement
(pas de dépendance pandas_ta).
"""
import numpy as np
import pandas as pd

# Colonnes produites par compute_base_features (l'ordre est celui attendu
# par les modèles momentum)
BASE_FEATURE_COLUMNS = [
    'hl_sma20vol', 'oc_sma20vol', 'macd', 'macd_signal', 'rsi', 'adx',
    'bb_high', 'bb_low', 'pct_close', 'macd_hist', 'bb_position',
    'rsi_momentum', 'volume_ratio', 'trend_strength', 'return_5d',
    'return_10d', 'return_20d', 'volatility_10d', 'high_52w_pct',
]

# Colonnes potentiellement présentes depuis la DB, à écraser systématiquement
DB_STALE_COLUMNS = BASE_FEATURE_COLUMNS + ['sma20_volume']


def _adx(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 14) -> pd.Series:
    """ADX simplifié (même formule que l'ancien compute_features.py)."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.rolling(window=period).mean()


def compute_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule toutes les features de base pour UN symbole (df trié par date,
    colonnes open/high/low/close/volume). Toute colonne homonyme existante
    (venant de la DB) est écrasée.
    """
    df = df.copy()
    # Écraser les colonnes DB potentiellement à zéro/périmées
    df = df.drop(columns=[c for c in DB_STALE_COLUMNS if c in df.columns])

    o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df['volume']

    # Volume
    df['sma20_volume'] = v.rolling(window=20).mean()
    df['hl_sma20vol'] = (h - l) / (df['sma20_volume'] + 1e-6)
    df['oc_sma20vol'] = (o - c) / (df['sma20_volume'] + 1e-6)
    df['volume_ratio'] = v / (df['sma20_volume'] + 1e-6)

    # MACD
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # RSI 14
    delta = c.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['rsi'] = 100 - 100 / (1 + gain / (loss + 1e-10))
    df['rsi_momentum'] = df['rsi'] - 50

    # ADX + force de tendance
    df['adx'] = _adx(h, l, c)
    df['trend_strength'] = df['adx'] * np.sign(df['macd'])

    # Bandes de Bollinger
    sma20 = c.rolling(window=20).mean()
    std20 = c.rolling(window=20).std()
    df['bb_high'] = sma20 + 2 * std20
    df['bb_low'] = sma20 - 2 * std20
    bb_range = df['bb_high'] - df['bb_low']
    df['bb_position'] = np.where(
        bb_range > 0, (c - df['bb_low']) / bb_range, 0.5)

    # Rendements et volatilité
    df['pct_close'] = c.pct_change()
    df['return_5d'] = c.pct_change(5)
    df['return_10d'] = c.pct_change(10)
    df['return_20d'] = c.pct_change(20)
    df['volatility_10d'] = c.pct_change().rolling(10).std()

    # Position vs plus haut 52 semaines
    high_52w = h.rolling(252, min_periods=50).max()
    df['high_52w_pct'] = c / (high_52w + 1e-6)

    return df


def compute_base_features_multi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Version multi-symboles : applique compute_base_features par symbole
    (aucune fenêtre glissante ne traverse une frontière de symbole).
    """
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    symbols = df['symbol']
    out = df.groupby('symbol', group_keys=False).apply(compute_base_features)
    out = out.sort_index()
    # pandas récents excluent la colonne de groupby du frame passé à apply :
    # la restaurer (alignement par index d'origine, préservé par group_keys=False)
    if 'symbol' not in out.columns:
        out['symbol'] = symbols
    return out
