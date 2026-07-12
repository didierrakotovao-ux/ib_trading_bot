"""
Filtre earnings — blackout des entrées autour des annonces de résultats.

Motivation (backtest TrailingOnly_Smooth, fenêtre 2026-01-28 → 2026-03-30) :
les 3 pires trades (TTMI −18%, PRIM −17%, CIEN −13%, tous en 1 jour) sont
des gaps post-earnings AU TRAVERS du trailing stop — 64% de la perte totale.
Un stop ne protège pas d'un gap overnight ; la seule protection est de ne
pas être exposé pendant l'annonce.

Source : table earnings_dates (annonces EDGAR 8-K).
- Backtest : les annonces réelles postérieures à la date simulée sont en
  base → blackout exact.
- Live : si aucune annonce future connue, estimation = dernière annonce
  + multiples de ~91 jours (cadence trimestrielle).
- Symbole absent de la table → pas de blocage (on ne pénalise pas
  l'absence de données), mais comptabilisé pour le diagnostic.
"""
import bisect
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.database.pg_connection import get_conn

QUARTER_DAYS = 91  # cadence trimestrielle approximative


def _as_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d)[:10], "%Y-%m-%d").date()


class EarningsFilter:
    """Blackout d'entrée si une annonce de résultats est proche."""

    def __init__(self, blackout_days: int = 14):
        """
        Args:
            blackout_days: fenêtre calendaire avant l'annonce (14 j
                calendaires ≈ 10 jours de bourse) pendant laquelle on
                n'ouvre pas de position.
        """
        self.blackout_days = blackout_days
        self._dates: Dict[str, List[date]] = {}
        self._missing: set = set()

    def preload(self, symbols: List[str], as_of) -> None:
        """
        Charge en une requête les annonces utiles pour un lot de symboles :
        la dernière année (pour l'estimation live) + toutes les futures
        (pour le blackout exact en backtest).
        """
        todo = [s for s in symbols
                if s not in self._dates and s not in self._missing]
        if not todo:
            return
        min_date = _as_date(as_of) - timedelta(days=400)
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT symbol, announcement_date FROM earnings_dates
            WHERE symbol = ANY(%s) AND announcement_date >= %s
            ORDER BY symbol, announcement_date
        """, (todo, min_date))
        for sym, ann in cur.fetchall():
            self._dates.setdefault(sym, []).append(_as_date(ann))
        cur.close()
        conn.close()
        for s in todo:
            if s not in self._dates:
                self._missing.add(s)

    def next_announcement(self, symbol: str, as_of) -> Optional[date]:
        """
        Prochaine annonce connue (>= as_of, JOUR MÊME INCLUS) ou estimée
        (dernière + ~91j). None si aucune donnée pour ce symbole.

        Le jour même compte : une annonce après clôture le jour de l'entrée
        provoque le gap le lendemain (cas TTMI/PRIM du backtest, −17/−18%).
        Avec des données daily on ne peut pas distinguer avant/après clôture
        — on bloque les deux (conservateur).
        """
        dates = self._dates.get(symbol)
        if not dates:
            return None
        d = _as_date(as_of)
        i = bisect.bisect_left(dates, d)
        if i < len(dates):
            return dates[i]          # annonce réelle connue (backtest)
        # Estimation live : dernière annonce + multiples de ~91 jours
        nxt = dates[-1]
        while nxt <= d:
            nxt += timedelta(days=QUARTER_DAYS)
        return nxt

    def is_in_blackout(self, symbol: str, as_of) -> bool:
        """True si une annonce (réelle ou estimée) tombe dans la fenêtre."""
        nxt = self.next_announcement(symbol, as_of)
        if nxt is None:
            return False
        return (nxt - _as_date(as_of)).days <= self.blackout_days
