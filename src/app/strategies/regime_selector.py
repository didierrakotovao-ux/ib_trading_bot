"""
Sélecteur de régime VIX — port live du regime_switch validé en backtest.

Machine à états avec hystérésis :
    VIX > risk_off  -> régime 'wyckoff'  (marché stressé : ranges/springs)
    VIX < risk_on   -> régime 'momentum' (marché calme : tendances)
    entre les deux  -> conserve le régime précédent

Validation (compare-all 2022-01 -> 2026-06, trailing alignés 8%) :
    momentum seul +96.9% / DD 27.4% | wyckoff seul +48.0% / DD 21.7%
    regime_switch +139.6% / DD 17.4% — out-of-time +103.7k$ vs +65.1k$.
Seuils 18/27 = zone stable de la grille de sensibilité du 2026-07-12.

Design SANS ÉTAT PERSISTÉ : l'état d'hystérésis est obtenu en rejouant la
machine sur l'historique VIX complet (déterministe, survit aux redémarrages,
identique à la logique du wrapper de backtest). Le VIX (^VIX) est en base et
maintenu par l'update quotidien.
"""
from typing import Optional, Tuple

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.database.pg_connection import read_sql

MOMENTUM = "momentum"
WYCKOFF = "wyckoff"


class RegimeSelector:
    """Détermine le régime de marché courant depuis la série VIX en base."""

    def __init__(self, risk_on: float = 18.0, risk_off: float = 27.0,
                 vix_symbol: str = "^VIX"):
        if risk_on >= risk_off:
            raise ValueError(
                f"risk_on ({risk_on}) doit être < risk_off ({risk_off}) — "
                "l'hystérésis exige une bande morte entre les deux seuils.")
        self.risk_on = risk_on
        self.risk_off = risk_off
        self.vix_symbol = vix_symbol

    def current_regime(self, as_of=None) -> Tuple[str, Optional[float], Optional[str]]:
        """
        Rejoue l'hystérésis sur l'historique VIX jusqu'à as_of (défaut: tout).

        Returns:
            (regime, dernier_vix, date_du_vix) — regime = 'momentum' par
            défaut prudent si le VIX est indisponible (comportement
            historique du bot, signalé par un warning).
        """
        params = [self.vix_symbol]
        date_clause = ""
        if as_of is not None:
            date_clause = " AND date <= %s"
            params.append(str(as_of)[:10])
        df = read_sql(f"""
            SELECT DISTINCT ON (date) date, close
            FROM historical_data
            WHERE symbol = %s{date_clause}
            ORDER BY date, source
        """, tuple(params))

        if df.empty:
            print(f"[REGIME][WARN] Aucune donnée {self.vix_symbol} en base — "
                  f"régime par défaut: {MOMENTUM}")
            return MOMENTUM, None, None

        state = MOMENTUM
        for v in df['close'].values:
            if v > self.risk_off:
                state = WYCKOFF
            elif v < self.risk_on:
                state = MOMENTUM

        last_vix = float(df['close'].iloc[-1])
        last_date = str(df['date'].iloc[-1])[:10]
        print(f"[REGIME] {self.vix_symbol}={last_vix:.1f} ({last_date}) | "
              f"hystérésis {self.risk_on}/{self.risk_off} -> régime {state.upper()}")
        return state, last_vix, last_date


def current_strategy_name(as_of=None, risk_on: float = 18.0,
                          risk_off: float = 27.0) -> str:
    """
    Nom de stratégie du régime courant pour la journalisation
    ('Momentum' | 'Wyckoff'). Utilisé par stop_manager et live_journal
    quand ils créent des entrées de journal a posteriori — passer as_of =
    date d'entrée de la position pour attribuer au régime de l'époque.
    Fallback silencieux sur 'Momentum' (comportement historique) en cas
    d'indisponibilité du VIX.
    """
    try:
        regime, _, _ = RegimeSelector(risk_on, risk_off).current_regime(as_of=as_of)
        return "Wyckoff" if regime == WYCKOFF else "Momentum"
    except Exception as e:
        print(f"[REGIME][WARN] Régime indéterminable ({e}) — attribution 'Momentum'")
        return "Momentum"
