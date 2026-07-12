"""
Script pour calculer et sauvegarder les features ML dans la base de données.
À exécuter après import_yahoo_bulk.py ou update_historical_data.py

Deux modes :
1. (défaut) Mise à jour des colonnes d'indicateurs de historical_data
   (sma20_volume, rsi, macd, …) via le module partagé src/app/ml/features.py
   — mêmes formules que le scoring/entraînement ML.
2. (--legacy-table) Ancien comportement : insertion dans computed_features.

Usage:
    python compute_features.py                  # incrémental, tous les symboles
    python compute_features.py --full           # recalcule tout l'historique
    python compute_features.py AAPL MSFT        # symboles spécifiques
    python compute_features.py --legacy-table   # ancienne table computed_features
"""
import io
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.database.db_manager import DatabaseManager
from src.app.database.pg_connection import get_engine, get_conn
from src.app.ml.features import compute_base_features


class FeatureComputer:
    """Calcule les features ML et les sauvegarde en base de données."""

    def __init__(self, db_path=None):  # db_path ignoré — connexion via pg_config.py
        self.db = DatabaseManager()

    def _calc_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcule le RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _calc_macd(self, prices: pd.Series) -> tuple:
        """Calcule MACD, Signal et Histogramme."""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        hist = macd - signal
        return macd, signal, hist

    def _calc_adx(self, high: pd.Series, low: pd.Series, close: pd.Series,
                  period: int = 14) -> pd.Series:
        """Calcule l'ADX simplifié."""
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()

        # Direction
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
        minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

        plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=period).mean()

        return adx

    def _calc_bollinger(self, prices: pd.Series, period: int = 20) -> tuple:
        """Calcule les bandes de Bollinger."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        return upper, lower

    def _calc_smoothness(self, prices: pd.Series, window: int) -> pd.Series:
        """Calcule le R² sur une fenêtre glissante (smoothness)."""
        x = np.arange(window, dtype=np.float64)
        sum_x = x.sum()
        sum_x2 = (x ** 2).sum()
        n = window
        denom_x = n * sum_x2 - sum_x ** 2

        def calc_r2(y):
            if len(y) < window or np.isnan(y).any():
                return np.nan
            sum_y = y.sum()
            sum_y2 = (y ** 2).sum()
            sum_xy = (x * y).sum()
            denom_y = n * sum_y2 - sum_y ** 2
            if denom_y <= 0 or denom_x <= 0:
                return np.nan
            r = (n * sum_xy - sum_x * sum_y) / (np.sqrt(denom_x * denom_y) + 1e-10)
            return r ** 2

        return prices.rolling(window=window).apply(calc_r2, raw=True)

    def _calc_momentum_12_1(self, close: pd.Series) -> pd.Series:
        """Calcule le momentum 12-1 mois (Gray & Vogel)."""
        # Rendement entre M-12 et M-1 (~252 - 21 jours)
        price_12m = close.shift(252)
        price_1m = close.shift(21)
        return (price_1m - price_12m) / (price_12m + 1e-10)

    def _calc_fip(self, close: pd.Series, flat_threshold: float = 0.005) -> pd.Series:
        """Calcule le Frog-in-the-Pan (Gray & Vogel)."""
        def calc_fip_window(window_data):
            if len(window_data) < 252:
                return np.nan
            daily_returns = np.diff(window_data) / window_data[:-1]
            n_days = len(daily_returns)
            pos_days = np.sum(daily_returns > flat_threshold)
            neg_days = np.sum(daily_returns < -flat_threshold)
            flat_days = n_days - pos_days - neg_days
            pct_pos = pos_days / n_days
            pct_neg = neg_days / n_days
            pct_flat = flat_days / n_days
            return pct_flat * (pct_neg - pct_pos)

        return close.rolling(window=252).apply(calc_fip_window, raw=True)

    def compute_features_for_symbol(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule toutes les features pour un symbole.

        Args:
            symbol: Symbole de l'action
            df: DataFrame avec OHLCV

        Returns:
            DataFrame avec les features calculées
        """
        if len(df) < 260:  # Besoin de ~252 jours pour les features
            return pd.DataFrame()

        df = df.copy()
        df.columns = df.columns.str.lower()

        # S'assurer que les colonnes existent
        required = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(c in df.columns for c in required):
            print(f"[WARN] {symbol}: Colonnes manquantes")
            return pd.DataFrame()

        # --- Calcul des features ---

        # SMA20 Volume
        df['sma20_volume'] = df['volume'].rolling(20).mean()

        # HL et OC normalisés par volume
        df['hl_sma20vol'] = (df['high'] - df['low']) / (df['sma20_volume'] + 1e-10)
        df['oc_sma20vol'] = (df['open'] - df['close']) / (df['sma20_volume'] + 1e-10)

        # RSI
        df['rsi'] = self._calc_rsi(df['close'])

        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calc_macd(df['close'])

        # ADX
        df['adx'] = self._calc_adx(df['high'], df['low'], df['close'])

        # Bollinger Bands
        df['bb_high'], df['bb_low'] = self._calc_bollinger(df['close'])
        bb_range = df['bb_high'] - df['bb_low']
        df['bb_position'] = np.where(
            bb_range > 0,
            (df['close'] - df['bb_low']) / bb_range,
            0.5
        )

        # RSI momentum
        df['rsi_momentum'] = df['rsi'] - 50

        # Volume ratio
        df['volume_ratio'] = df['volume'] / (df['sma20_volume'] + 1e-10)

        # Trend strength
        df['trend_strength'] = df['adx'] * np.sign(df['macd'])

        # Returns
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        df['return_20d'] = df['close'].pct_change(20)
        df['pct_close'] = df['close'].pct_change()

        # Volatility
        df['volatility_10d'] = df['close'].pct_change().rolling(10).std()

        # Position vs 52-week high
        high_52w = df['high'].rolling(252, min_periods=50).max()
        df['high_52w_pct'] = df['close'] / (high_52w + 1e-10)

        # Smoothness (R²)
        df['smoothness_20d'] = self._calc_smoothness(df['close'], 20)
        df['smoothness_50d'] = self._calc_smoothness(df['close'], 50)

        # Seasonality (month encoding)
        df['date'] = pd.to_datetime(df['date'])
        month = df['date'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)

        # Gray & Vogel filters
        df['momentum_12_1'] = self._calc_momentum_12_1(df['close'])
        df['fip'] = self._calc_fip(df['close'])

        # Ajouter le symbole
        df['symbol'] = symbol

        # Sélectionner les colonnes pertinentes
        feature_columns = [
            'symbol', 'date', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'adx',
            'bb_high', 'bb_low', 'bb_position', 'rsi_momentum',
            'volume_ratio', 'trend_strength', 'return_5d', 'return_10d',
            'return_20d', 'volatility_10d', 'high_52w_pct', 'hl_sma20vol',
            'oc_sma20vol', 'pct_close', 'smoothness_20d', 'smoothness_50d',
            'month_sin', 'month_cos', 'momentum_12_1', 'fip'
        ]

        result = df[feature_columns].copy()
        result['date'] = result['date'].dt.strftime('%Y-%m-%d')

        # Supprimer les lignes avec trop de NaN
        result = result.dropna(subset=['rsi', 'macd', 'momentum_12_1'])

        return result

    # ------------------------------------------------------------------
    # Mise à jour des colonnes d'indicateurs de historical_data
    # ------------------------------------------------------------------

    # Colonnes d'indicateurs présentes dans historical_data
    INDICATOR_COLUMNS = [
        'sma20_volume', 'hl_sma20vol', 'oc_sma20vol', 'macd', 'macd_signal',
        'rsi', 'adx', 'bb_high', 'bb_low', 'pct_close',
    ]

    def update_indicator_columns(self, symbols: List[str] = None,
                                 full: bool = False,
                                 lookback_days: int = 300,
                                 batch_size: int = 300):
        """
        Met à jour les colonnes d'indicateurs de historical_data.

        Calcul via src/app/ml/features.py (mêmes formules que le ML), par
        groupe (symbol, source), écriture en masse via COPY + UPDATE.
        Les valeurs de warm-up (fenêtres incomplètes) sont écrites en NULL,
        jamais en 0 — un zéro est indistinguable d'une vraie valeur.

        Args:
            symbols: sous-ensemble de symboles (défaut: tous)
            full: True = recalcule tout l'historique.
                  False = incrémental : recalcule à partir de la dernière
                  date valide (sma20_volume > 0) moins lookback_days pour
                  amorcer les fenêtres (252j pour high_52w + marge).
        """
        if symbols is None:
            symbols = self.db.get_all_symbols()
        print(f"[FEATURES-DB] Mise à jour des indicateurs de historical_data "
              f"({len(symbols)} symboles, mode {'FULL' if full else 'incrémental'})")

        total_updated = 0
        errors = 0
        for start in range(0, len(symbols), batch_size):
            batch = symbols[start:start + batch_size]
            try:
                total_updated += self._update_batch(batch, full, lookback_days)
            except Exception as e:
                errors += 1
                print(f"[FEATURES-DB][ERROR] batch {start//batch_size}: {e}")
            done = min(start + batch_size, len(symbols))
            print(f"[FEATURES-DB] {done}/{len(symbols)} symboles | "
                  f"{total_updated:,} lignes mises à jour")

        print(f"[FEATURES-DB] Terminé: {total_updated:,} lignes, {errors} erreurs batch")
        return total_updated

    def _update_batch(self, batch: List[str], full: bool,
                      lookback_days: int) -> int:
        """Calcule et écrit les indicateurs pour un lot de symboles."""
        conn = get_conn()
        try:
            # 1. Fenêtre de recalcul par symbole (mode incrémental) :
            #    depuis la dernière date valide, moins le lookback
            write_from = {}   # symbol -> première date à écrire
            if not full:
                cur = conn.cursor()
                cur.execute("""
                    SELECT symbol, MAX(date) FILTER (WHERE sma20_volume > 0)
                    FROM historical_data WHERE symbol = ANY(%s)
                    GROUP BY symbol
                """, (batch,))
                for sym, last_valid in cur.fetchall():
                    if last_valid is not None:
                        write_from[sym] = last_valid
                cur.close()

            # 2. Charger l'OHLCV nécessaire (avec lookback pour les fenêtres)
            min_load = {}
            for sym in batch:
                if full or sym not in write_from:
                    min_load[sym] = None  # tout l'historique
                else:
                    min_load[sym] = write_from[sym] - timedelta(days=lookback_days + 130)

            cur = conn.cursor()
            cur.execute("""
                SELECT symbol, date, source, open, high, low, close, volume
                FROM historical_data
                WHERE symbol = ANY(%s)
                ORDER BY symbol, source, date
            """, (batch,))
            rows = cur.fetchall()
            cur.close()
            if not rows:
                return 0
            df = pd.DataFrame(rows, columns=[
                'symbol', 'date', 'source', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['date']).dt.date

            # Filtrer le lookback par symbole (en gardant assez d'historique)
            if not full:
                keep = np.ones(len(df), dtype=bool)
                for sym, min_d in min_load.items():
                    if min_d is not None:
                        keep &= ~((df['symbol'] == sym).values
                                  & (df['date'] < min_d).values)
                df = df[keep]
            if df.empty:
                return 0

            # 3. Calcul par (symbol, source) — jamais de fenêtre inter-séries
            parts = []
            for (sym, src), g in df.groupby(['symbol', 'source'], sort=False):
                g = g.sort_values('date').reset_index(drop=True)
                out = compute_base_features(g)
                out['symbol'], out['source'] = sym, src
                parts.append(out)
            feat = pd.concat(parts, ignore_index=True)

            # 4. Ne réécrire que la fenêtre nécessaire (incrémental)
            if not full:
                keep = np.ones(len(feat), dtype=bool)
                for sym, from_d in write_from.items():
                    keep &= ~((feat['symbol'] == sym).values
                              & (feat['date'] < from_d).values)
                feat = feat[keep]
            if feat.empty:
                return 0

            # 5. Écriture en masse : COPY vers table temporaire puis UPDATE
            cols = self.INDICATOR_COLUMNS
            out_df = feat[['symbol', 'date', 'source'] + cols].copy()
            # NaN -> NULL (champ vide en CSV) — jamais 0
            buf = io.StringIO()
            out_df.to_csv(buf, index=False, header=False, na_rep='')
            buf.seek(0)

            cur = conn.cursor()
            col_defs = ", ".join(f"{c} double precision" for c in cols)
            cur.execute(f"""
                CREATE TEMP TABLE tmp_indicators
                (symbol text, date date, source text, {col_defs})
                ON COMMIT DROP
            """)
            cur.copy_expert(
                "COPY tmp_indicators FROM STDIN WITH (FORMAT csv, NULL '')", buf)
            set_clause = ", ".join(f"{c} = t.{c}" for c in cols)
            cur.execute(f"""
                UPDATE historical_data h SET {set_clause}
                FROM tmp_indicators t
                WHERE h.symbol = t.symbol AND h.date = t.date AND h.source = t.source
            """)
            updated = cur.rowcount
            conn.commit()
            cur.close()
            return updated
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Table computed_features (jeu COMPLET de features, dérivées incluses)
    # ------------------------------------------------------------------

    COMPUTED_TABLE_COLUMNS = [
        'rsi', 'macd', 'macd_signal', 'macd_hist', 'adx',
        'bb_high', 'bb_low', 'bb_position', 'rsi_momentum',
        'volume_ratio', 'trend_strength', 'return_5d', 'return_10d',
        'return_20d', 'volatility_10d', 'high_52w_pct', 'hl_sma20vol',
        'oc_sma20vol', 'pct_close', 'smoothness_20d', 'smoothness_50d',
        'month_sin', 'month_cos', 'momentum_12_1', 'fip',
    ]

    def compute_all(self, symbols: List[str] = None, incremental: bool = False,
                    batch_size: int = 200, lookback_days: int = 600):
        """
        Peuple/actualise la table computed_features : le jeu complet de
        features (base via src/app/ml/features.py + smoothness, saisonnalité,
        momentum_12_1, FIP). Upsert sur (symbol, date).

        Args:
            symbols: sous-ensemble (défaut: tous)
            incremental: True = seulement les dates postérieures à la
                dernière ligne existante (lookback_days de données chargées
                en amont pour amorcer les fenêtres 252j)
        """
        if symbols is None:
            symbols = self.db.get_all_symbols()

        # Index unique requis pour l'upsert
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_computed_features_symbol_date
            ON computed_features(symbol, date)
        """)
        conn.commit()
        cur.close()
        conn.close()

        print(f"[COMPUTE] computed_features: {len(symbols)} symboles "
              f"(mode {'incrémental' if incremental else 'FULL'})")

        total_rows = 0
        errors = 0
        for start in range(0, len(symbols), batch_size):
            batch = symbols[start:start + batch_size]
            try:
                total_rows += self._compute_features_batch(
                    batch, incremental, lookback_days)
            except Exception as e:
                errors += 1
                print(f"[COMPUTE][ERROR] batch {start//batch_size}: {e}")
            done = min(start + batch_size, len(symbols))
            print(f"[COMPUTE] {done}/{len(symbols)} symboles | "
                  f"{total_rows:,} lignes upsertées")

        print(f"[COMPUTE] Terminé: {total_rows:,} lignes, {errors} erreurs batch")
        return total_rows

    def _compute_features_batch(self, batch: List[str], incremental: bool,
                                lookback_days: int) -> int:
        """Calcule et upserte le jeu complet de features pour un lot."""
        conn = get_conn()
        try:
            # Dernière date déjà calculée par symbole (mode incrémental)
            last_date = {}
            if incremental:
                cur = conn.cursor()
                cur.execute("""
                    SELECT symbol, MAX(date) FROM computed_features
                    WHERE symbol = ANY(%s) GROUP BY symbol
                """, (batch,))
                last_date = {s: d for s, d in cur.fetchall() if d is not None}
                cur.close()

            # OHLCV dédupliqué par (symbol, date) — choix déterministe de la
            # source si plusieurs existent
            cur = conn.cursor()
            cur.execute("""
                SELECT DISTINCT ON (symbol, date)
                       symbol, date, open, high, low, close, volume
                FROM historical_data
                WHERE symbol = ANY(%s) AND close > 0
                ORDER BY symbol, date, source
            """, (batch,))
            rows = cur.fetchall()
            cur.close()
            if not rows:
                return 0
            df = pd.DataFrame(rows, columns=[
                'symbol', 'date', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['date'])

            parts = []
            for sym, g in df.groupby('symbol', sort=False):
                g = g.sort_values('date').reset_index(drop=True)
                if incremental and sym in last_date:
                    cutoff = pd.Timestamp(last_date[sym])
                    g = g[g['date'] >= cutoff - timedelta(days=lookback_days)]
                    g = g.reset_index(drop=True)
                if len(g) < 30:
                    continue
                # Features de base — module partagé (mêmes formules que le ML)
                out = compute_base_features(g)
                # Features étendues
                out['smoothness_20d'] = self._calc_smoothness(out['close'], 20)
                out['smoothness_50d'] = self._calc_smoothness(out['close'], 50)
                month = out['date'].dt.month
                out['month_sin'] = np.sin(2 * np.pi * month / 12)
                out['month_cos'] = np.cos(2 * np.pi * month / 12)
                out['momentum_12_1'] = self._calc_momentum_12_1(out['close'])
                out['fip'] = self._calc_fip(out['close'])
                out['symbol'] = sym
                # N'écrire que les nouvelles dates (incrémental)
                if incremental and sym in last_date:
                    out = out[out['date'] > pd.Timestamp(last_date[sym])]
                parts.append(out)
            if not parts:
                return 0
            feat = pd.concat(parts, ignore_index=True)

            cols = self.COMPUTED_TABLE_COLUMNS
            out_df = feat[['symbol', 'date'] + cols].copy()
            out_df['date'] = out_df['date'].dt.strftime('%Y-%m-%d')
            buf = io.StringIO()
            out_df.to_csv(buf, index=False, header=False, na_rep='')
            buf.seek(0)

            cur = conn.cursor()
            col_defs = ", ".join(f"{c} double precision" for c in cols)
            cur.execute(f"""
                CREATE TEMP TABLE tmp_computed
                (symbol text, date date, {col_defs})
                ON COMMIT DROP
            """)
            cur.copy_expert(
                "COPY tmp_computed FROM STDIN WITH (FORMAT csv, NULL '')", buf)
            col_list = ", ".join(cols)
            set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols)
            cur.execute(f"""
                INSERT INTO computed_features (symbol, date, {col_list})
                SELECT symbol, date, {col_list} FROM tmp_computed
                ON CONFLICT (symbol, date) DO UPDATE SET {set_clause}
            """)
            upserted = cur.rowcount
            conn.commit()
            cur.close()
            return upserted
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Calcul des features ML")
    parser.add_argument('symbols', nargs='*', help='Symboles spécifiques (optionnel)')
    parser.add_argument('--full', action='store_true',
                        help='Recalcule tout l\'historique (défaut: incrémental)')
    parser.add_argument('--indicators-only', action='store_true',
                        help='Seulement les colonnes d\'indicateurs de historical_data')
    parser.add_argument('--features-table-only', action='store_true',
                        help='Seulement la table computed_features (jeu complet)')
    parser.add_argument('--db', default='trading_data.db', help='Ignoré (PostgreSQL)')

    args = parser.parse_args()

    computer = FeatureComputer(args.db)
    symbols = args.symbols if args.symbols else None

    if not args.features_table_only:
        computer.update_indicator_columns(symbols=symbols, full=args.full)
    if not args.indicators_only:
        computer.compute_all(symbols=symbols, incremental=not args.full)


if __name__ == "__main__":
    main()
