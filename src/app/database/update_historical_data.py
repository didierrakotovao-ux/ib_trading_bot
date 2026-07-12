"""
Script de mise à jour incrémentale des données historiques.
Récupère uniquement les nouvelles barres depuis la dernière date en base.

Améliorations vs version précédente :
  - Téléchargement par lots (100 symboles / appel yfinance) → ~100× moins d'appels HTTP
  - Toutes les dernières dates chargées en 1 seule requête SQL
  - Connexion PostgreSQL réutilisée sur toute la durée
  - Reprise automatique : les symboles déjà à jour sont ignorés
  - Progression détaillée avec ETA

Usage :
    python update_historical_data.py              # Met à jour tous les symboles
    python update_historical_data.py AAPL MSFT    # Met à jour des symboles spécifiques
    python update_historical_data.py --batch 50   # Taille de lot (défaut: 100)
    python update_historical_data.py --compute    # Met à jour et recalcule les features
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.database.pg_connection import get_conn


class HistoricalDataUpdater:
    """Met à jour les données historiques de façon incrémentale."""

    ALWAYS_UPDATE_SYMBOLS = ['^VIX']

    def __init__(self, db_path=None):  # db_path ignoré — connexion via pg_config.py
        pass

    # ------------------------------------------------------------------
    # Méthodes unitaires (conservées pour compatibilité)
    # ------------------------------------------------------------------

    def get_symbols_to_update(self) -> list:
        """Récupère tous les symboles de la base de données + symboles forcés."""
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT symbol FROM historical_data ORDER BY symbol")
        symbols = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()

        # Garantit l'inclusion d'indices macro requis pour le backtest/risk overlay.
        for forced_symbol in self.ALWAYS_UPDATE_SYMBOLS:
            if forced_symbol not in symbols:
                symbols.append(forced_symbol)

        symbols.sort()
        return symbols

    def get_last_date(self, symbol: str) -> str:
        """Récupère la dernière date en base pour un symbole."""
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT MAX(date) FROM historical_data WHERE symbol = %s", (symbol,))
        result = cur.fetchone()
        cur.close()
        conn.close()
        return str(result[0]) if result and result[0] else None

    def update_symbol(self, symbol: str) -> int:
        """Met à jour les données pour un symbole (mode unitaire)."""
        last_date = self.get_last_date(symbol)
        if last_date:
            start_date = (datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

        end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        if start_date >= end_date:
            return 0

        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if df.empty:
                return 0
            return self._insert_df(df, symbol)
        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")
            return 0

    # ------------------------------------------------------------------
    # Méthode principale : mise à jour par lots
    # ------------------------------------------------------------------

    def update_all(self, symbols: list = None, verbose: bool = True,
                   batch_size: int = 100):
        """
        Met à jour tous les symboles via téléchargement par lots.

        Args:
            symbols   : liste de symboles (None = tous depuis la base)
            verbose   : afficher les détails par symbole
            batch_size: nombre de symboles par appel yfinance (défaut: 100)
        """
        if symbols is None:
            symbols = self.get_symbols_to_update()

        today     = datetime.now().strftime('%Y-%m-%d')
        end_date  = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

        print(f"[UPDATE] {len(symbols):,} symboles à vérifier...")

        # --- Charger TOUTES les dernières dates en 1 requête ---
        conn = get_conn()
        cur  = conn.cursor()
        cur.execute("SELECT symbol, MAX(date) FROM historical_data GROUP BY symbol")
        last_dates = {row[0]: str(row[1]) for row in cur.fetchall()}
        cur.close()
        conn.close()

        # --- Filtrer les symboles qui ont besoin d'une mise à jour ---
        to_update = {}   # symbol → start_date
        for sym in symbols:
            ld = last_dates.get(sym)
            if ld and ld >= today:
                continue   # déjà à jour aujourd'hui
            if ld:
                start = (datetime.strptime(ld, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                start = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
            if start < end_date:
                to_update[sym] = start

        already_up_to_date = len(symbols) - len(to_update)
        print(f"[UPDATE] {already_up_to_date:,} déjà à jour | "
              f"{len(to_update):,} à télécharger | lots de {batch_size}")

        if not to_update:
            print("[UPDATE] Tout est à jour.")
            return 0

        # --- Grouper par date de départ pour optimiser les lots ---
        # Les symboles avec la même start_date peuvent être téléchargés ensemble
        from collections import defaultdict
        by_start: dict = defaultdict(list)
        for sym, start in to_update.items():
            by_start[start].append(sym)

        total_new      = 0
        updated_count  = 0
        error_count    = 0
        processed      = 0
        t_start        = time.time()

        for start_date, sym_group in sorted(by_start.items()):
            # Découper le groupe en lots de batch_size
            for batch_idx in range(0, len(sym_group), batch_size):
                batch = sym_group[batch_idx: batch_idx + batch_size]

                new_bars, errors = self._download_and_insert_batch(
                    batch, start_date, end_date, verbose
                )

                total_new     += new_bars
                updated_count += sum(1 for s in batch if s in to_update)
                error_count   += errors
                processed     += len(batch)

                # Progression
                elapsed   = time.time() - t_start
                rate      = processed / elapsed if elapsed > 0 else 1
                remaining = (len(to_update) - processed) / rate if rate > 0 else 0
                print(f"[UPDATE] {processed:5}/{len(to_update):5} symboles | "
                      f"+{total_new:,} barres | "
                      f"{error_count} erreurs | "
                      f"~{remaining/60:.0f} min restantes")

                # Pause anti rate-limit (Yahoo Finance : ~2 req/s max)
                time.sleep(2)

        elapsed = time.time() - t_start
        print(f"\n[UPDATE] Terminé en {elapsed/60:.1f} min : "
              f"{updated_count} symboles mis à jour, "
              f"{total_new:,} nouvelles barres, "
              f"{error_count} erreurs")
        return total_new

    # ------------------------------------------------------------------
    # Téléchargement et insertion d'un lot
    # ------------------------------------------------------------------

    def _download_and_insert_batch(self, symbols: list, start_date: str,
                                   end_date: str, verbose: bool,
                                   max_retries: int = 4) -> tuple:
        """
        Télécharge et insère un lot de symboles en une fois.
        Retry automatique avec backoff exponentiel si rate-limited.

        Returns:
            (total_barres_insérées, nb_erreurs)
        """
        total_inserted = 0
        errors         = 0
        wait           = 30   # secondes initiales avant retry

        for attempt in range(1, max_retries + 1):
            try:
                # Un seul appel HTTP pour tout le lot
                df_raw = yf.download(
                    symbols,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    group_by='ticker',
                    threads=False   # évite les race conditions sur rate-limit
                )

                if df_raw.empty:
                    return 0, 0

                # Ouvrir une connexion unique pour tout le lot
                conn   = get_conn()
                cursor = conn.cursor()

                for symbol in symbols:
                    try:
                        df_sym = self._extract_symbol(df_raw, symbol, len(symbols))
                        if df_sym is None or df_sym.empty:
                            continue

                        n = self._insert_rows(cursor, df_sym, symbol)
                        total_inserted += n
                        if verbose and n > 0:
                            print(f"  {symbol}: +{n} barres")

                    except Exception as e:
                        errors += 1
                        print(f"[ERROR] {symbol}: {e}")

                conn.commit()
                cursor.close()
                conn.close()
                return total_inserted, errors   # succès

            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = any(kw in err_str for kw in
                                    ['too many requests', 'rate limit', '429', 'try after'])
                if is_rate_limit and attempt < max_retries:
                    print(f"  [RATE-LIMIT] Attente {wait}s avant retry "
                          f"(tentative {attempt}/{max_retries})...")
                    time.sleep(wait)
                    wait *= 2   # 30s → 60s → 120s → 240s
                else:
                    if not is_rate_limit:
                        print(f"[ERROR] lot {symbols[0]}…{symbols[-1]}: {e}")
                    else:
                        print(f"  [RATE-LIMIT] Lot ignoré après {max_retries} tentatives "
                              f"({symbols[0]}…{symbols[-1]})")
                    errors += len(symbols)
                    return total_inserted, errors

        return total_inserted, errors

    # ------------------------------------------------------------------
    # Extraction d'un symbole depuis un DataFrame multi-symboles
    # ------------------------------------------------------------------

    def _extract_symbol(self, df_raw: pd.DataFrame, symbol: str,
                        nb_symbols: int) -> pd.DataFrame:
        """
        Extrait les données d'un symbole depuis un DataFrame yfinance multi-symboles.
        Gère les deux formats : lot unique (MultiIndex) et lot multiple (xs).
        """
        if nb_symbols == 1:
            # Lot d'un seul symbole : MultiIndex (Price, Symbol) ou simple
            df = df_raw.copy()
            if isinstance(df.columns, pd.MultiIndex):
                price_types = {'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'}
                level0 = set(df.columns.get_level_values(0))
                if level0 & price_types:
                    df.columns = df.columns.get_level_values(0)
                else:
                    df.columns = df.columns.get_level_values(1)
        else:
            # Lot multi-symboles : group_by='ticker' → level 0 = ticker
            if isinstance(df_raw.columns, pd.MultiIndex):
                try:
                    df = df_raw[symbol].copy()
                except KeyError:
                    return None
            else:
                return None

        if df.empty:
            return None

        # Normaliser les colonnes
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume', 'Adj Close': 'adjusted_close'
        })

        # Normaliser l'index date
        df.index.name = 'date'
        df = df.reset_index()
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        # Garder seulement les colonnes utiles
        cols_needed = ['date', 'open', 'high', 'low', 'close', 'volume']
        if 'adjusted_close' in df.columns:
            cols_needed.append('adjusted_close')
        df = df[[c for c in cols_needed if c in df.columns]]

        return df.dropna(subset=['open', 'close'])

    # ------------------------------------------------------------------
    # Insertion en base
    # ------------------------------------------------------------------

    def _insert_rows(self, cursor, df: pd.DataFrame, symbol: str) -> int:
        """Insère les lignes d'un DataFrame pour un symbole. Retourne le nb inséré."""
        inserted = 0
        for _, row in df.iterrows():
            try:
                cursor.execute("""
                    INSERT INTO historical_data
                    (symbol, date, open, high, low, close, volume, adjusted_close, source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, date, source) DO NOTHING
                """, (
                    symbol,
                    row['date'],
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    int(row['volume']),
                    float(row.get('adjusted_close', row['close'])),
                    'yfinance'
                ))
                if cursor.rowcount > 0:
                    inserted += 1
            except Exception:
                pass
        return inserted

    def _insert_df(self, df: pd.DataFrame, symbol: str) -> int:
        """Insère un DataFrame (mode unitaire). Retourne le nb de lignes insérées."""
        df = self._extract_symbol(df, symbol, nb_symbols=1)
        if df is None or df.empty:
            return 0
        conn   = get_conn()
        cursor = conn.cursor()
        n = self._insert_rows(cursor, df, symbol)
        conn.commit()
        cursor.close()
        conn.close()
        return n


# ----------------------------------------------------------------------
# Point d'entrée
# ----------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Mise à jour incrémentale des données historiques"
    )
    parser.add_argument('symbols', nargs='*',
                        help='Symboles spécifiques (optionnel)')
    parser.add_argument('--no-features', action='store_true',
                        help='Ne pas mettre à jour les features après l\'import '
                             '(défaut: mise à jour incrémentale automatique de '
                             'historical_data et computed_features)')
    parser.add_argument('--compute', '-c', action='store_true',
                        help='Obsolète (les features sont mises à jour '
                             'automatiquement) — conservé pour compatibilité')
    parser.add_argument('--batch', type=int, default=100,
                        help='Taille de lot yfinance (défaut: 100)')
    parser.add_argument('--db', default='trading_data.db',
                        help='Ignoré (PostgreSQL)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Mode silencieux')

    args = parser.parse_args()

    updater = HistoricalDataUpdater()
    symbols = args.symbols if args.symbols else None

    new_bars = updater.update_all(
        symbols    = symbols,
        verbose    = not args.quiet,
        batch_size = args.batch,
    )

    # Mise à jour automatique des features après import (incrémentale) :
    #   1. colonnes d'indicateurs de historical_data
    #   2. table computed_features (jeu complet, smoothness/momentum_12_1/FIP)
    # Sans elle, les nouvelles lignes gardent des indicateurs NULL/0 pour
    # toujours — c'est ce qui a empoisonné le scoring de janvier à juillet 2026.
    if new_bars > 0 and not args.no_features:
        from compute_features import FeatureComputer
        computer = FeatureComputer()
        print("\n[COMPUTE] Mise à jour des indicateurs de historical_data...")
        computer.update_indicator_columns(symbols=symbols, full=False)
        print("\n[COMPUTE] Mise à jour de computed_features...")
        computer.compute_all(symbols=symbols, incremental=True)

        # Rafraîchissement glissant des dates d'earnings (filtre blackout à
        # l'entrée — earnings_filter.py). EDGAR ne publie les 8-K qu'au jour
        # de l'annonce : en live, le filtre estime la prochaine annonce
        # depuis la dernière connue (+~91j), la table doit donc rester
        # fraîche. refresh_days=7 => seuls les symboles non rafraîchis
        # depuis 7 jours sont re-téléchargés (~1/7 de l'univers par jour).
        try:
            from src.app.value.populate_earnings_dates import populate
            print("\n[EARNINGS] Rafraîchissement des dates d'annonces (EDGAR)...")
            populate(symbols=symbols, max_workers=3, refresh_days=7,
                     force=False, recent_only=True, verbose=not args.quiet)
        except Exception as e:
            print(f"[EARNINGS][WARN] Rafraîchissement earnings_dates impossible: {e}")


if __name__ == "__main__":
    main()
