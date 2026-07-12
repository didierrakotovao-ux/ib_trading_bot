"""
Module de gestion de base de données pour le stockage des résultats de screening,
données historiques et signaux de trading.
"""
import pandas as pd
import pandas_ta as ta
from datetime import datetime
from typing import List, Dict, Optional, Any
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.database.pg_connection import get_conn, get_engine, dict_cursor, read_sql


class DatabaseManager:
    """Gestionnaire de base de données PostgreSQL pour le trading."""

    def __init__(self, db_path: str = None):  # noqa: db_path ignoré, connexion via pg_config.py
        self.conn = None
        self._create_tables()

    def connect(self):
        """Établit la connexion à la base de données."""
        if self.conn is None or self.conn.closed:
            self.conn = get_conn()
            print("[OK] Base de donnees connectee (PostgreSQL)")

    def close(self):
        """Ferme la connexion à la base de données."""
        if self.conn and not self.conn.closed:
            self.conn.close()
            self.conn = None
            print("Base de donnees fermee")

    def _create_tables(self):
        """Crée les tables si elles n'existent pas."""
        self.connect()
        cur = self.conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS scanner_results (
                id        SERIAL PRIMARY KEY,
                symbol    TEXT NOT NULL,
                exchange  TEXT,
                scan_type TEXT NOT NULL,
                rank      INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata  TEXT
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_scanner_symbol
            ON scanner_results(symbol, scan_type, timestamp DESC)
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS historical_data (
                id             SERIAL PRIMARY KEY,
                symbol         TEXT NOT NULL,
                date           DATE NOT NULL,
                open           DOUBLE PRECISION NOT NULL,
                high           DOUBLE PRECISION NOT NULL,
                low            DOUBLE PRECISION NOT NULL,
                close          DOUBLE PRECISION NOT NULL,
                volume         BIGINT NOT NULL,
                adjusted_close DOUBLE PRECISION,
                dividends      DOUBLE PRECISION DEFAULT 0,
                stock_splits   DOUBLE PRECISION DEFAULT 0,
                source         TEXT NOT NULL DEFAULT 'unknown',
                sma20_volume   DOUBLE PRECISION DEFAULT 0,
                hl_sma20vol    DOUBLE PRECISION DEFAULT 0,
                oc_sma20vol    DOUBLE PRECISION DEFAULT 0,
                macd           DOUBLE PRECISION DEFAULT 0,
                macd_signal    DOUBLE PRECISION DEFAULT 0,
                rsi            DOUBLE PRECISION DEFAULT 0,
                adx            DOUBLE PRECISION DEFAULT 0,
                bb_high        DOUBLE PRECISION DEFAULT 0,
                bb_low         DOUBLE PRECISION DEFAULT 0,
                pct_close      DOUBLE PRECISION DEFAULT 0,
                timestamp      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date, source)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_historical_symbol_date
            ON historical_data(symbol, date DESC)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_historical_date
            ON historical_data(date DESC)
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS technical_indicators (
                id             SERIAL PRIMARY KEY,
                symbol         TEXT NOT NULL,
                date           DATE NOT NULL,
                sma_20         DOUBLE PRECISION,
                sma_50         DOUBLE PRECISION,
                sma_200        DOUBLE PRECISION,
                ema_12         DOUBLE PRECISION,
                ema_26         DOUBLE PRECISION,
                rsi_14         DOUBLE PRECISION,
                macd           DOUBLE PRECISION,
                macd_signal    DOUBLE PRECISION,
                macd_hist      DOUBLE PRECISION,
                bb_upper       DOUBLE PRECISION,
                bb_middle      DOUBLE PRECISION,
                bb_lower       DOUBLE PRECISION,
                atr_14         DOUBLE PRECISION,
                volume_sma_20  DOUBLE PRECISION,
                volume_ratio   DOUBLE PRECISION,
                pct_from_high_52w DOUBLE PRECISION,
                timestamp      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_indicators_symbol_date
            ON technical_indicators(symbol, date DESC)
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS trading_signals (
                id           SERIAL PRIMARY KEY,
                symbol       TEXT NOT NULL,
                signal_type  TEXT NOT NULL
                    CHECK(signal_type IN ('BUY','SELL','HOLD','ACCUMULATION','DISTRIBUTION','WATCH')),
                strategy     TEXT NOT NULL,
                price        DOUBLE PRECISION,
                confidence   DOUBLE PRECISION CHECK(confidence >= 0 AND confidence <= 1),
                target_price DOUBLE PRECISION,
                stop_loss    DOUBLE PRECISION,
                metadata     TEXT,
                timestamp    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_symbol
            ON trading_signals(symbol, timestamp DESC)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_type_strategy
            ON trading_signals(signal_type, strategy, timestamp DESC)
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS signal_outcomes (
                id             SERIAL PRIMARY KEY,
                signal_id      INTEGER NOT NULL,
                symbol         TEXT NOT NULL,
                entry_price    DOUBLE PRECISION NOT NULL,
                entry_date     DATE NOT NULL,
                price_5d       DOUBLE PRECISION,
                return_5d      DOUBLE PRECISION,
                price_10d      DOUBLE PRECISION,
                return_10d     DOUBLE PRECISION,
                price_20d      DOUBLE PRECISION,
                return_20d     DOUBLE PRECISION,
                max_gain       DOUBLE PRECISION,
                max_loss       DOUBLE PRECISION,
                max_gain_days  INTEGER,
                max_loss_days  INTEGER,
                profitable     BOOLEAN,
                roi            DOUBLE PRECISION,
                updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (signal_id) REFERENCES trading_signals(id),
                UNIQUE(signal_id)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_outcomes_symbol
            ON signal_outcomes(symbol, entry_date DESC)
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                id         SERIAL PRIMARY KEY,
                symbol     TEXT NOT NULL UNIQUE,
                reason     TEXT,
                added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                active     BOOLEAN DEFAULT TRUE,
                notes      TEXT
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_watchlist_active
            ON watchlist(active, added_date DESC)
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS symbol_metadata (
                symbol       TEXT PRIMARY KEY,
                company_name TEXT,
                sector       TEXT,
                industry     TEXT,
                market_cap   DOUBLE PRECISION,
                exchange     TEXT,
                currency     TEXT DEFAULT 'USD',
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS computed_features (
                id              SERIAL PRIMARY KEY,
                symbol          TEXT NOT NULL,
                date            DATE NOT NULL,
                rsi             DOUBLE PRECISION,
                macd            DOUBLE PRECISION,
                macd_signal     DOUBLE PRECISION,
                macd_hist       DOUBLE PRECISION,
                adx             DOUBLE PRECISION,
                bb_high         DOUBLE PRECISION,
                bb_low          DOUBLE PRECISION,
                bb_position     DOUBLE PRECISION,
                rsi_momentum    DOUBLE PRECISION,
                volume_ratio    DOUBLE PRECISION,
                trend_strength  DOUBLE PRECISION,
                return_5d       DOUBLE PRECISION,
                return_10d      DOUBLE PRECISION,
                return_20d      DOUBLE PRECISION,
                volatility_10d  DOUBLE PRECISION,
                high_52w_pct    DOUBLE PRECISION,
                hl_sma20vol     DOUBLE PRECISION,
                oc_sma20vol     DOUBLE PRECISION,
                pct_close       DOUBLE PRECISION,
                smoothness_20d  DOUBLE PRECISION,
                smoothness_50d  DOUBLE PRECISION,
                month_sin       DOUBLE PRECISION,
                month_cos       DOUBLE PRECISION,
                momentum_12_1   DOUBLE PRECISION,
                fip             DOUBLE PRECISION,
                ml_score        INTEGER,
                timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_features_symbol_date
            ON computed_features(symbol, date DESC)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_features_date
            ON computed_features(date DESC)
        """)

        self.conn.commit()
        cur.close()
        print("[OK] Tables de base de donnees initialisees")

    def save_scanner_results(self, results: List[Dict[str, Any]], scan_type: str):
        """Sauvegarde les résultats d'un scanner."""
        self.connect()
        cur = self.conn.cursor()
        for result in results:
            cur.execute("""
                INSERT INTO scanner_results (symbol, exchange, scan_type, rank, metadata)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                result.get('symbol'),
                result.get('exchange'),
                scan_type,
                result.get('rank'),
                str(result)
            ))
        self.conn.commit()
        cur.close()
        print(f"[OK] {len(results)} resultats de scanner sauvegardes")

    def save_historical_data(self, symbol: str, df: pd.DataFrame, source: str = "unknown") -> bool:
        """Sauvegarde les données historiques dans la DB (cache)."""
        self.connect()

        df_copy = df.copy()

        if isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy['date'] = df_copy.index.strftime('%Y-%m-%d')
            df_copy = df_copy.reset_index(drop=True)

        df_copy['symbol'] = symbol
        df_copy['source'] = source
        df_copy.columns = df_copy.columns.str.lower()

        if 'date' in df_copy.columns:
            df_copy['date'] = pd.to_datetime(df_copy['date']).dt.strftime('%Y-%m-%d')

        base_columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'source']
        optional_columns = ['dividends', 'stock splits']
        columns_to_keep = base_columns.copy()
        for col in optional_columns:
            if col in df_copy.columns:
                columns_to_keep.append(col)

        df_copy = df_copy[[c for c in columns_to_keep if c in df_copy.columns]]

        if 'stock splits' in df_copy.columns:
            df_copy = df_copy.rename(columns={'stock splits': 'stock_splits'})

        df_copy = df_copy.reset_index(drop=True)
        self.enrich_features(df_copy)

        try:
            engine = get_engine()
            df_copy.to_sql(
                'historical_data',
                engine,
                if_exists='append',
                index=False,
                method=_pg_upsert_ignore
            )
            print(f"[OK] Donnees historiques pour {symbol} sauvegardees ({len(df)} barres)")
            return True
        except Exception as e:
            print(f"[ERROR] Erreur sauvegarde {symbol}: {e}")
            return False

    def enrich_features(self, df):
        try:
            close = df['close'].dropna()

            if len(close) >= 26:
                macd_df = ta.macd(df["close"], fast=12, slow=26)
                df['macd'] = macd_df['MACD_12_26_9']
                df['macd_signal'] = macd_df['MACDs_12_26_9']
            else:
                df['macd'] = 0
                df['macd_signal'] = 0

            if len(close) >= 14:
                df['rsi'] = ta.rsi(df["close"], length=14)
                adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
                df['adx'] = adx_df['ADX_14']
            else:
                df['rsi'] = 0
                df['adx'] = 0

            if len(close) >= 20:
                bb_df = ta.bbands(close, length=20, std=2)
                df['bb_high'] = bb_df['BBU_20_2.0_2.0'] if 'BBU_20_2.0_2.0' in bb_df else 0
                df['bb_low'] = bb_df['BBL_20_2.0_2.0'] if 'BBL_20_2.0_2.0' in bb_df else 0
                df['sma20_volume'] = df['volume'].rolling(window=20).mean()
                df['hl_sma20vol'] = (df['high'] - df['low']) / df['sma20_volume']
                df['oc_sma20vol'] = (df['open'] - df['close']) / df['sma20_volume']
            else:
                df['bb_high'] = 0
                df['bb_low'] = 0
                df['sma20_volume'] = 0
                df['hl_sma20vol'] = 0
                df['oc_sma20vol'] = 0

            df['pct_close'] = df['close'].pct_change()
            return df
        except Exception as e:
            print(f"[ERROR] Erreur lors de l'enrichissement des features: {e}")
            return None

    def get_historical_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        source: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Récupère les données historiques depuis la DB."""
        self.connect()

        query = "SELECT * FROM historical_data WHERE symbol = %s"
        params: List[Any] = [symbol]

        if start_date:
            query += " AND date >= %s"
            params.append(start_date.strftime('%Y-%m-%d'))

        if end_date:
            query += " AND date <= %s"
            params.append(end_date.strftime('%Y-%m-%d'))

        if source:
            query += " AND source = %s"
            params.append(source)

        query += " ORDER BY date ASC"

        df = read_sql(query, params)

        if df.empty:
            return None

        df['date'] = pd.to_datetime(df['date'])
        return df

    def save_trading_signal(
        self,
        symbol: str,
        signal_type: str,
        strategy: str,
        price: Optional[float] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Sauvegarde un signal de trading."""
        self.connect()
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO trading_signals
            (symbol, signal_type, strategy, price, confidence, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (symbol, signal_type, strategy, price, confidence, str(metadata or {})))
        self.conn.commit()
        cur.close()
        print(f"[OK] Signal {signal_type} pour {symbol} sauvegarde")

    def get_trading_signals(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """Récupère les signaux de trading."""
        self.connect()

        query = "SELECT * FROM trading_signals WHERE 1=1"
        params: List[Any] = []

        if symbol:
            query += " AND symbol = %s"
            params.append(symbol)

        if strategy:
            query += " AND strategy = %s"
            params.append(strategy)

        query += " ORDER BY timestamp DESC LIMIT %s"
        params.append(limit)

        return read_sql(query, params)

    def add_to_watchlist(self, symbol: str, reason: str = ""):
        """Ajoute un symbole à la watchlist."""
        self.connect()
        cur = self.conn.cursor()
        try:
            cur.execute("""
                INSERT INTO watchlist (symbol, reason)
                VALUES (%s, %s)
                ON CONFLICT (symbol) DO NOTHING
            """, (symbol, reason))
            self.conn.commit()
            print(f"[OK] {symbol} ajoute a la watchlist")
        except Exception as e:
            self.conn.rollback()
            print(f"[INFO] {symbol} deja dans la watchlist: {e}")
        finally:
            cur.close()

    def get_watchlist(self, active_only: bool = True) -> List[str]:
        """Récupère la liste des symboles en watchlist."""
        self.connect()
        query = "SELECT symbol FROM watchlist"
        if active_only:
            query += " WHERE active = TRUE"
        query += " ORDER BY added_date DESC"
        cur = self.conn.cursor()
        cur.execute(query)
        result = [row[0] for row in cur.fetchall()]
        cur.close()
        return result

    def export_to_csv(self, table_name: str, output_path: str):
        """Exporte une table vers un fichier CSV."""
        df = read_sql(f"SELECT * FROM {table_name}")
        df.to_csv(output_path, index=False)
        print(f"[OK] Table {table_name} exportee vers {output_path}")

    def get_all_symbols(self) -> List[str]:
        """Récupère tous les symboles uniques ayant des données historiques."""
        self.connect()
        cur = self.conn.cursor()
        cur.execute("SELECT DISTINCT symbol FROM historical_data ORDER BY symbol")
        result = [row[0] for row in cur.fetchall()]
        cur.close()
        return result

    def get_all_signals(self) -> List[Dict]:
        """Récupère tous les signaux de trading."""
        self.connect()
        cur = dict_cursor(self.conn)
        cur.execute("""
            SELECT id, symbol, signal_type, price AS entry_price, timestamp AS entry_date,
                   metadata AS reason, confidence, metadata, timestamp
            FROM trading_signals
            ORDER BY timestamp DESC
        """)
        result = [dict(row) for row in cur.fetchall()]
        cur.close()
        return result

    def save_computed_features(self, symbol: str, date: str, features: Dict[str, Any]) -> bool:
        """Sauvegarde les features calculées pour un symbole et une date."""
        self.connect()
        cur = self.conn.cursor()

        feature_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_hist', 'adx',
            'bb_high', 'bb_low', 'bb_position', 'rsi_momentum',
            'volume_ratio', 'trend_strength', 'return_5d', 'return_10d',
            'return_20d', 'volatility_10d', 'high_52w_pct', 'hl_sma20vol',
            'oc_sma20vol', 'pct_close', 'smoothness_20d', 'smoothness_50d',
            'month_sin', 'month_cos', 'momentum_12_1', 'fip', 'ml_score'
        ]

        values = [symbol, date] + [features.get(col) for col in feature_columns]
        columns_str = 'symbol, date, ' + ', '.join(feature_columns)
        placeholders = ', '.join(['%s'] * len(values))
        update_set = ', '.join(f"{col}=EXCLUDED.{col}" for col in feature_columns)

        try:
            cur.execute(f"""
                INSERT INTO computed_features ({columns_str})
                VALUES ({placeholders})
                ON CONFLICT (symbol, date) DO UPDATE SET {update_set}
            """, values)
            self.conn.commit()
            cur.close()
            return True
        except Exception as e:
            self.conn.rollback()
            cur.close()
            print(f"[ERROR] save_computed_features: {e}")
            return False

    def save_computed_features_bulk(self, df: pd.DataFrame) -> int:
        """Sauvegarde les features calculées en bulk depuis un DataFrame."""
        self.connect()

        feature_columns = [
            'symbol', 'date', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'adx',
            'bb_high', 'bb_low', 'bb_position', 'rsi_momentum',
            'volume_ratio', 'trend_strength', 'return_5d', 'return_10d',
            'return_20d', 'volatility_10d', 'high_52w_pct', 'hl_sma20vol',
            'oc_sma20vol', 'pct_close', 'smoothness_20d', 'smoothness_50d',
            'month_sin', 'month_cos', 'momentum_12_1', 'fip', 'ml_score'
        ]

        available_columns = [c for c in feature_columns if c in df.columns]
        if 'symbol' not in available_columns or 'date' not in available_columns:
            print("[ERROR] DataFrame doit contenir 'symbol' et 'date'")
            return 0

        df_to_save = df[available_columns].copy()

        try:
            engine = get_engine()
            df_to_save.to_sql('computed_features', engine,
                              if_exists='append', index=False,
                              method=_pg_upsert_ignore)
            return len(df_to_save)
        except Exception as e:
            print(f"[ERROR] save_computed_features_bulk: {e}")
            return 0

    def get_computed_features(self, symbol: str, start_date: str = None,
                              end_date: str = None) -> pd.DataFrame:
        """Récupère les features calculées pour un symbole."""
        self.connect()

        query = "SELECT * FROM computed_features WHERE symbol = %s"
        params = [symbol]

        if start_date:
            query += " AND date >= %s"
            params.append(start_date)
        if end_date:
            query += " AND date <= %s"
            params.append(end_date)

        query += " ORDER BY date"

        return read_sql(query, params)

    def get_latest_features_date(self, symbol: str = None) -> Optional[str]:
        """Récupère la date la plus récente des features calculées."""
        self.connect()
        cur = self.conn.cursor()

        if symbol:
            cur.execute("SELECT MAX(date) FROM computed_features WHERE symbol = %s", (symbol,))
        else:
            cur.execute("SELECT MAX(date) FROM computed_features")

        result = cur.fetchone()
        cur.close()
        return str(result[0]) if result and result[0] else None


def _pg_upsert_ignore(table, conn, keys, data_iter):
    """Méthode pour pandas to_sql : ignore les conflits UNIQUE (ON CONFLICT DO NOTHING)."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    data = [dict(zip(keys, row)) for row in data_iter]
    stmt = pg_insert(table.table).values(data).on_conflict_do_nothing()
    conn.execute(stmt)
