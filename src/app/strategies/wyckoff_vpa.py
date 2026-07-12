"""
Wyckoff / Volume Price Analysis — features et détecteurs d'événements.

Traduction en règles calculables des concepts de Wyckoff tels que détaillés
par Anna Coulling (A Complete Guide to Volume Price Analysis) :
  - anomalies effort/résultat : gros volume pour petit déplacement = absorption
  - no demand / no supply : test de l'absence d'intérêt sur volume faible
  - stopping volume / selling climax : fin de markdown, entrée des pros
  - trading range (accumulation), spring (shakeout), test, sign of strength

Principes d'implémentation :
  - Tout est calculé au temps t sans regarder le futur (pas de lookahead).
  - Toutes les features sont NORMALISÉES (volume relatif au symbole, spread
    en multiples d'ATR) — sinon le ML réapprend simplement la volatilité.
  - Les détecteurs sont volontairement permissifs (rappel élevé) : c'est le
    meta-labeling ML qui filtre la précision (AFML ch. 3). Les seuils sont
    des constantes de module, ajustables.

Les fonctions opèrent sur un DataFrame multi-symboles trié (symbol, date)
avec colonnes : symbol, date, open, high, low, close, volume.
"""
import numpy as np
import pandas as pd

# --- Seuils des détecteurs (constantes v1, à calibrer) ---
NARROW_SPREAD = 0.7      # spread <= 0.7 × ATR20 = barre étroite
WIDE_SPREAD = 1.5        # spread >= 1.5 × ATR20 = barre large
HIGH_VOLUME = 2.0        # volume >= 2 × SMA20 = effort marqué
CLIMAX_VOLUME = 3.0      # volume >= 3 × SMA20 = climax
LOW_VOLUME = 0.7         # volume <= 0.7 × SMA20 = pas d'intérêt
RANGE_WINDOW = 15        # fenêtre de détection du trading range
RANGE_MAX_WIDTH_ATR = 5.0  # largeur max du range en multiples d'ATR20
RANGE_MIN_DAYS = 10      # jours consécutifs en range avant spring/SOS
TEST_LOOKBACK = 10       # fenêtre de recherche d'un spring avant un test


def _g(df: pd.DataFrame):
    """Raccourci groupby symbol (les séries doivent rester par symbole)."""
    return df.groupby('symbol', group_keys=False)


def compute_vpa_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les primitives VPA et les features de contexte.
    Toutes les colonnes ajoutées sont normalisées et sans lookahead.
    """
    df = df.copy()
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    close = df['close']
    prev_close = _g(df)['close'].shift(1)

    # --- Primitives ---
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df['atr20'] = tr.groupby(df['symbol']).transform(
        lambda x: x.rolling(20, min_periods=10).mean())

    spread = df['high'] - df['low']
    df['rel_spread'] = spread / (df['atr20'] + 1e-9)

    vol_sma20 = _g(df)['volume'].transform(
        lambda x: x.rolling(20, min_periods=10).mean())
    vol_std20 = _g(df)['volume'].transform(
        lambda x: x.rolling(20, min_periods=10).std())
    df['rel_volume'] = df['volume'] / (vol_sma20 + 1e-9)
    df['volume_z'] = (df['volume'] - vol_sma20) / (vol_std20 + 1e-9)
    # Exposé pour le filtre d'univers tradable (calculé ici, jamais lu en DB)
    df['sma20_volume'] = vol_sma20

    # Close Location Value : position du close dans le range du jour
    df['clv'] = np.where(spread > 0, (close - df['low']) / spread, 0.5)
    df['body_pct'] = np.where(spread > 0,
                              (close - df['open']).abs() / spread, 0.0)

    up_bar = close > prev_close
    down_bar = close < prev_close
    df['up_bar'] = up_bar

    vol_lt_prev2 = (df['volume'] < _g(df)['volume'].shift(1)) & \
                   (df['volume'] < _g(df)['volume'].shift(2))

    # --- Anomalies VPA (barre par barre) ---
    # Effort sans résultat : gros volume, petit déplacement = absorption
    df['effort_no_result_up'] = up_bar & (df['rel_volume'] >= HIGH_VOLUME) \
        & (df['rel_spread'] <= NARROW_SPREAD)
    df['effort_no_result_down'] = down_bar & (df['rel_volume'] >= HIGH_VOLUME) \
        & (df['rel_spread'] <= NARROW_SPREAD)
    # No demand / no supply (Coulling) : barre étroite, volume sous les 2 précédentes
    df['no_demand'] = up_bar & (df['rel_spread'] <= NARROW_SPREAD) & vol_lt_prev2
    df['no_supply'] = down_bar & (df['rel_spread'] <= NARROW_SPREAD) & vol_lt_prev2
    # Stopping volume : les pros absorbent la vente, close dans le haut du range
    df['stopping_volume'] = down_bar & (df['rel_volume'] >= HIGH_VOLUME) \
        & (df['clv'] >= 0.5)
    # Climax
    df['selling_climax'] = down_bar & (df['rel_volume'] >= CLIMAX_VOLUME) \
        & (df['rel_spread'] >= WIDE_SPREAD) & (df['clv'] >= 0.35)
    df['buying_climax'] = up_bar & (df['rel_volume'] >= CLIMAX_VOLUME) \
        & (df['rel_spread'] >= WIDE_SPREAD) & (df['clv'] <= 0.65)

    # --- Agrégats courts (comptes d'anomalies récentes, décalés d'un jour non:
    #     la barre du jour est connue au close, on trade au close/lendemain) ---
    for col, win in [('no_supply', 5), ('no_demand', 5),
                     ('stopping_volume', 10), ('selling_climax', 20),
                     ('effort_no_result_down', 10)]:
        df[f'{col}_{win}d'] = df.groupby('symbol')[col].transform(
            lambda x: x.rolling(win, min_periods=1).sum())

    # Assèchement du volume (accumulation : le flottant se raréfie)
    df['vol_dryup'] = _g(df)['rel_volume'].transform(
        lambda x: x.rolling(5, min_periods=3).mean()
        / (x.rolling(20, min_periods=10).mean() + 1e-9))

    # --- Contexte tendance ---
    df['trend_40d'] = _g(df)['close'].pct_change(40)
    high_52w = _g(df)['high'].transform(
        lambda x: x.rolling(252, min_periods=50).max())
    low_52w = _g(df)['low'].transform(
        lambda x: x.rolling(252, min_periods=50).min())
    df['dist_52w_high'] = close / (high_52w + 1e-9)
    df['dist_52w_low'] = close / (low_52w + 1e-9)

    # --- Trading range (support/résistance des RANGE_WINDOW derniers jours,
    #     hors jour courant) ---
    df['range_support'] = _g(df)['low'].transform(
        lambda x: x.rolling(RANGE_WINDOW, min_periods=RANGE_WINDOW).min().shift(1))
    df['range_resistance'] = _g(df)['high'].transform(
        lambda x: x.rolling(RANGE_WINDOW, min_periods=RANGE_WINDOW).max().shift(1))
    range_width = df['range_resistance'] - df['range_support']
    df['range_width_atr'] = range_width / (df['atr20'] + 1e-9)
    df['pos_in_range'] = np.where(
        range_width > 0, (close - df['range_support']) / range_width, 0.5)

    in_range = (df['range_width_atr'] <= RANGE_MAX_WIDTH_ATR) \
        & df['range_width_atr'].notna()
    # Jours consécutifs en range (compteur remis à zéro à chaque sortie
    # de range ou changement de symbole)
    new_symbol = df['symbol'] != df['symbol'].shift(1)
    block = (~in_range | new_symbol).cumsum()
    df['days_in_range'] = in_range.groupby(block).cumsum()

    return df


def detect_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Détecte les événements d'entrée Wyckoff. Requiert compute_vpa_features.

    - event_spring : en range établi, le low casse le support puis le close
      revient au-dessus le jour même (shakeout). Le volume du spring est
      laissé en feature (fort = shakeout violent, faible = no supply).
    - event_test  : dans les TEST_LOOKBACK jours après un spring, retour vers
      le support sur volume faible, close tenu — confirmation no supply.
    - event_sos   : sign of strength — barre large qui casse la résistance
      du range sur volume fort, close dans le haut de la barre.

    event_any = union ; event_type = 'spring' | 'test' | 'sos' (priorité au
    plus fort signal si plusieurs le même jour : sos > spring > test).
    """
    df = df.copy()

    established_range = (df['days_in_range'].shift(1).fillna(0) >= RANGE_MIN_DAYS) \
        & (df['symbol'] == df['symbol'].shift(1))

    support = df['range_support']
    resistance = df['range_resistance']

    # --- Spring ---
    df['event_spring'] = established_range \
        & (df['low'] < support) \
        & (df['close'] > support)
    # Profondeur du shakeout (feature de contexte)
    df['spring_depth_atr'] = np.where(
        df['event_spring'],
        (support - df['low']) / (df['atr20'] + 1e-9), 0.0)

    # --- Test (après spring) ---
    spring_recent = df.groupby('symbol')['event_spring'].transform(
        lambda x: x.shift(1).rolling(TEST_LOOKBACK, min_periods=1).max()
    ).fillna(0).astype(bool)
    df['event_test'] = spring_recent \
        & (df['low'] <= support + 0.5 * df['atr20']) \
        & (df['close'] > support) \
        & (df['rel_volume'] <= LOW_VOLUME) \
        & (df['clv'] >= 0.5)

    # --- Sign of Strength ---
    df['event_sos'] = established_range \
        & (df['close'] > resistance) \
        & (df['rel_spread'] >= 1.2) \
        & (df['rel_volume'] >= 1.5) \
        & (df['clv'] >= 0.7)

    df['event_any'] = df['event_spring'] | df['event_test'] | df['event_sos']
    df['event_type'] = np.select(
        [df['event_sos'], df['event_spring'], df['event_test']],
        ['sos', 'spring', 'test'], default='')

    return df


# Features destinées au modèle ML (toutes normalisées, sans lookahead)
ML_FEATURE_COLUMNS = [
    # Barre courante
    'rel_spread', 'rel_volume', 'volume_z', 'clv', 'body_pct',
    # Anomalies récentes
    'no_supply_5d', 'no_demand_5d', 'stopping_volume_10d',
    'selling_climax_20d', 'effort_no_result_down_10d',
    'vol_dryup',
    # Contexte
    'trend_40d', 'dist_52w_high', 'dist_52w_low',
    'range_width_atr', 'pos_in_range', 'days_in_range',
    # Type d'événement déclencheur
    'event_spring', 'event_test', 'event_sos', 'spring_depth_atr',
]
