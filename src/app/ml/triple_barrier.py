"""
Triple-barrier labeling (López de Prado, AFML ch. 3) — module partagé.

Utilisé par les pipelines momentum (ml_smooth_momentum_predictor) et
wyckoff (ml_wyckoff_predictor). Les barrières doivent décrire les mêmes
règles de sortie que celles que le stop_manager applique en réel
(stop_config.json est la source de vérité).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Racine du projet (src/app/ml/ -> 3 niveaux au-dessus)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_barriers_from_stop_config(config_path: Path = None) -> dict:
    """
    Lit les barrières depuis stop_config.json.

    Returns:
        dict avec profit_barrier, stop_barrier (fractions, ex: 0.08)
        et horizon (jours de bourse). Valeurs par défaut si le fichier
        est absent ou incomplet.
    """
    defaults = {"profit_barrier": 0.08, "stop_barrier": 0.08, "horizon": 20}
    path = config_path or PROJECT_ROOT / "stop_config.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        time_cfg = cfg.get("time", {})
        # label_horizon prime : la barrière d'EXÉCUTION (max_holding_days)
        # peut être plus longue sans faire dériver les labels des modèles
        horizon = int(time_cfg.get("label_horizon")
                      or time_cfg.get("max_holding_days")
                      or defaults["horizon"])
        return {
            "profit_barrier": float(cfg["profit"]["fixed_pct"]) / 100,
            "stop_barrier": float(cfg["protection"]["pct"]) / 100,
            "horizon": horizon,
        }
    except Exception as e:
        print(f"[CONFIG][WARN] stop_config.json non lu ({e}) — barrieres par defaut")
        return defaults


def triple_barrier_labels(df: pd.DataFrame, profit_barrier: float,
                          stop_barrier: float, horizon: int) -> pd.DataFrame:
    """
    Labels triple-barrière alignés sur les sorties réelles du stop_manager :
      - barrière profit : close d'entrée × (1 + profit_barrier)
      - barrière stop   : trailing — HWM × (1 - stop_barrier), HWM initialisé
        au close d'entrée puis remonté avec les highs
      - barrière temps  : horizon jours de bourse, sortie au close

    target = 1 si la barrière profit est touchée en premier, 0 si le stop
    ou la barrière temps arrive avant. Si stop et profit sont touchés le
    même jour (indécidable en daily), le stop est prioritaire (pessimiste).
    Le HWM intègre le high du jour courant avant le test du stop (le
    stop_manager suit le prix en intraday) — également pessimiste.
    target = NaN si l'historique du symbole s'arrête avant qu'une barrière
    soit touchée (fenêtre incomplète, échantillon inutilisable).

    trade_return = rendement réalisé à la sortie : +profit_barrier (profit),
    HWM×(1-stop_barrier)/entrée - 1 (stop, peut être positif), ou
    close(t+horizon)/entrée - 1 (temps).

    Args:
        df: DataFrame avec colonnes symbol, date, high, low, close

    Returns:
        df trié par (symbol, date) avec colonnes 'target' et 'trade_return'
    """
    df = df.copy()
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    n = len(df)
    codes = pd.factorize(df['symbol'])[0]
    high = df['high'].to_numpy(dtype=np.float64)
    low = df['low'].to_numpy(dtype=np.float64)
    entry = df['close'].to_numpy(dtype=np.float64)

    profit_level = entry * (1 + profit_barrier)
    hwm = entry.copy()
    label = np.zeros(n)
    trade_return = np.full(n, np.nan)  # rendement réalisé à la sortie
    decided = np.zeros(n, dtype=bool)
    observed_days = np.zeros(n, dtype=np.int32)

    # Une passe vectorisée par jour d'offset : état trailing maintenu
    # pour toutes les entrées simultanément
    for k in range(1, horizon + 1):
        h_k = np.full(n, np.nan)
        l_k = np.full(n, np.nan)
        h_k[:-k] = high[k:]
        l_k[:-k] = low[k:]
        # Jour futur valide seulement s'il appartient au même symbole
        valid = np.zeros(n, dtype=bool)
        valid[:-k] = codes[k:] == codes[:-k]
        observed_days += valid

        hwm = np.where(valid, np.maximum(hwm, h_k), hwm)
        stop_hit = valid & (l_k <= hwm * (1 - stop_barrier))
        profit_hit = valid & (h_k >= profit_level)

        newly_profit = ~decided & profit_hit & ~stop_hit
        newly_stop = ~decided & stop_hit
        label[newly_profit] = 1
        # Sortie au niveau de la barrière touchée
        trade_return[newly_profit] = profit_barrier
        trade_return[newly_stop] = (
            hwm[newly_stop] * (1 - stop_barrier) / entry[newly_stop] - 1
        )
        decided |= stop_hit | profit_hit

    # Barrière de temps : sortie au close du jour t+horizon
    complete = observed_days >= horizon
    time_exit = ~decided & complete
    close_h = np.full(n, np.nan)
    close_h[:-horizon] = entry[horizon:]
    trade_return[time_exit] = close_h[time_exit] / entry[time_exit] - 1

    # Fenêtre incomplète (fin d'historique du symbole) sans barrière touchée
    label[~decided & ~complete] = np.nan

    df['target'] = label
    df['trade_return'] = trade_return
    return df
