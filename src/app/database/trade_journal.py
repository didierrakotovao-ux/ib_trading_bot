"""
Module de journalisation des trades en base de données.
Supporte trois modes : backtest, paper, live.
"""
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.database.pg_connection import get_conn, dict_cursor


class TradeMode(Enum):
    """Mode de trading."""
    BACKTEST = "backtest"
    PAPER    = "paper"
    LIVE     = "live"


class TradeJournal:
    """
    Gestionnaire de journal de trades en base de données PostgreSQL.
    Catégorise les trades par mode (backtest, paper, live) et par stratégie.
    """

    def __init__(self, db_path: str = None):  # noqa: db_path ignoré, connexion via pg_config.py
        self.conn = None
        self._create_table()

    def connect(self):
        """Établit la connexion à la base de données."""
        if self.conn is None or self.conn.closed:
            self.conn = get_conn()

    def close(self):
        """Ferme la connexion."""
        if self.conn and not self.conn.closed:
            self.conn.close()
            self.conn = None

    def _create_table(self):
        """Crée les tables trades et trade_exits si elles n'existent pas."""
        self.connect()
        cur = self.conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id                  SERIAL PRIMARY KEY,
                trade_mode          TEXT NOT NULL
                    CHECK(trade_mode IN ('backtest','paper','live')),
                strategy_name       TEXT NOT NULL,
                symbol              TEXT NOT NULL,
                date_entree         TIMESTAMP NOT NULL,
                prix_entree         DOUBLE PRECISION NOT NULL,
                quantite            INTEGER NOT NULL,
                quantite_restante   INTEGER NOT NULL DEFAULT 0,
                date_sortie         TIMESTAMP,
                prix_sortie         DOUBLE PRECISION,
                cause_sortie        TEXT,
                pnl_brut            DOUBLE PRECISION,
                commission          DOUBLE PRECISION,
                pnl_net             DOUBLE PRECISION,
                bars_held           INTEGER,
                score_entree        DOUBLE PRECISION,
                exhaustion_signals  TEXT,
                backtest_start_date DATE,
                backtest_end_date   DATE,
                notes               TEXT,
                created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS trade_exits (
                id              SERIAL PRIMARY KEY,
                trade_id        INTEGER NOT NULL,
                symbol          TEXT NOT NULL,
                date_sortie     TIMESTAMP NOT NULL,
                prix_sortie     DOUBLE PRECISION NOT NULL,
                quantite_vendue INTEGER NOT NULL,
                cause_sortie    TEXT,
                pnl_brut        DOUBLE PRECISION,
                commission      DOUBLE PRECISION,
                pnl_net         DOUBLE PRECISION,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (trade_id) REFERENCES trades(id)
            )
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_trade_exits_trade_id
            ON trade_exits(trade_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_trade_exits_symbol_date
            ON trade_exits(symbol, date_sortie DESC)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_mode_strategy
            ON trades(trade_mode, strategy_name)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_symbol
            ON trades(symbol, date_entree DESC)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_date
            ON trades(date_entree DESC)
        """)

        # Migration : ajouter quantite_restante si absente (idempotent)
        cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='trades' AND column_name='quantite_restante'
                ) THEN
                    ALTER TABLE trades ADD COLUMN quantite_restante INTEGER NOT NULL DEFAULT 0;
                    UPDATE trades SET quantite_restante = quantite
                    WHERE quantite_restante = 0 AND date_sortie IS NULL;
                END IF;
            END $$
        """)

        self.conn.commit()
        cur.close()
        print("[OK] Table trades initialisée")

    def clear_backtest_trades(self, strategy_name: str,
                              backtest_start_date=None, backtest_end_date=None):
        """
        Efface les trades de backtest pour une stratégie donnée.

        Si backtest_start_date/backtest_end_date sont fournis, seuls les
        trades de CETTE période sont effacés (re-run de la même fenêtre) —
        les résultats des autres périodes sont conservés, ce qui permet de
        comparer plusieurs fenêtres pour une même stratégie.
        Sans période : ancien comportement (tout effacer pour la stratégie).
        """
        self.connect()
        cur = self.conn.cursor()
        if backtest_start_date is not None and backtest_end_date is not None:
            cur.execute("""
                DELETE FROM trades
                WHERE trade_mode = 'backtest' AND strategy_name = %s
                  AND backtest_start_date = %s AND backtest_end_date = %s
            """, (strategy_name, backtest_start_date, backtest_end_date))
            scope = f" (période {backtest_start_date} -> {backtest_end_date})"
        else:
            cur.execute("""
                DELETE FROM trades
                WHERE trade_mode = 'backtest' AND strategy_name = %s
            """, (strategy_name,))
            scope = " (toutes périodes)"
        deleted = cur.rowcount
        self.conn.commit()
        cur.close()
        if deleted > 0:
            print(f"[CLEAN] {deleted} trades backtest effacés pour {strategy_name}{scope}")

    def log_trade(
        self,
        trade_mode: TradeMode,
        strategy_name: str,
        symbol: str,
        date_entree: datetime,
        prix_entree: float,
        quantite: int,
        date_sortie: Optional[datetime] = None,
        prix_sortie: Optional[float] = None,
        cause_sortie: Optional[str] = None,
        pnl_brut: Optional[float] = None,
        commission: Optional[float] = None,
        pnl_net: Optional[float] = None,
        bars_held: Optional[int] = None,
        score_entree: Optional[float] = None,
        exhaustion_signals: Optional[str] = None,
        backtest_start_date: Optional[datetime] = None,
        backtest_end_date: Optional[datetime] = None,
        notes: Optional[str] = None
    ) -> int:
        """Enregistre un trade dans le journal. Retourne l'ID inséré."""
        self.connect()
        cur = self.conn.cursor()

        quantite_restante_init = 0 if date_sortie else quantite

        cur.execute("""
            INSERT INTO trades (
                trade_mode, strategy_name, symbol,
                date_entree, prix_entree, quantite, quantite_restante,
                date_sortie, prix_sortie, cause_sortie,
                pnl_brut, commission, pnl_net,
                bars_held, score_entree, exhaustion_signals,
                backtest_start_date, backtest_end_date, notes
            ) VALUES (
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
            ) RETURNING id
        """, (
            trade_mode.value, strategy_name, symbol,
            date_entree.strftime('%Y-%m-%d %H:%M:%S') if date_entree else None,
            prix_entree, quantite, quantite_restante_init,
            date_sortie.strftime('%Y-%m-%d %H:%M:%S') if date_sortie else None,
            prix_sortie, cause_sortie,
            pnl_brut, commission, pnl_net,
            bars_held, score_entree, exhaustion_signals,
            backtest_start_date.strftime('%Y-%m-%d') if backtest_start_date else None,
            backtest_end_date.strftime('%Y-%m-%d') if backtest_end_date else None,
            notes
        ))

        trade_id = cur.fetchone()[0]
        self.conn.commit()
        cur.close()
        return trade_id or 0

    def log_partial_exit(
        self,
        trade_id: int,
        symbol: str,
        date_sortie: datetime,
        prix_sortie: float,
        quantite_vendue: int,
        prix_entree: float,
        cause_sortie: str = "MANUAL"
    ) -> int:
        """
        Enregistre une sortie partielle ou totale d'une position.
        Insère dans trade_exits, décrémente quantite_restante, ferme si complet.
        """
        self.connect()

        commission_rate = 0.001
        pnl_brut  = (prix_sortie - prix_entree) * quantite_vendue
        commission = (prix_entree * quantite_vendue + prix_sortie * quantite_vendue) * commission_rate
        pnl_net   = pnl_brut - commission

        cur = self.conn.cursor()

        cur.execute("""
            INSERT INTO trade_exits
                (trade_id, symbol, date_sortie, prix_sortie,
                 quantite_vendue, cause_sortie, pnl_brut, commission, pnl_net)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
        """, (
            trade_id, symbol,
            date_sortie.strftime('%Y-%m-%d %H:%M:%S'),
            prix_sortie, quantite_vendue, cause_sortie,
            round(pnl_brut, 2), round(commission, 2), round(pnl_net, 2)
        ))
        exit_id = cur.fetchone()[0]

        cur.execute("""
            UPDATE trades SET quantite_restante = quantite_restante - %s WHERE id = %s
        """, (quantite_vendue, trade_id))

        cur.execute("SELECT quantite_restante FROM trades WHERE id = %s", (trade_id,))
        row = cur.fetchone()
        if row and row[0] <= 0:
            cur.execute("""
                SELECT SUM(pnl_brut), SUM(commission), SUM(pnl_net), MAX(date_sortie)
                FROM trade_exits WHERE trade_id = %s
            """, (trade_id,))
            totals = cur.fetchone()
            total_pnl_brut   = totals[0] or 0
            total_commission = totals[1] or 0
            total_pnl_net    = totals[2] or 0
            last_exit_date   = totals[3]

            cur.execute("""
                UPDATE trades SET
                    date_sortie = %s, prix_sortie = %s, cause_sortie = %s,
                    pnl_brut = %s, commission = %s, pnl_net = %s, quantite_restante = 0
                WHERE id = %s
            """, (
                last_exit_date, prix_sortie, cause_sortie,
                round(total_pnl_brut, 2), round(total_commission, 2), round(total_pnl_net, 2),
                trade_id
            ))
            print(f"[JOURNAL] Position fermée (trade_id={trade_id}): PnL net total = {total_pnl_net:.2f}$")
        else:
            restant = row[0] if row else '?'
            print(f"[JOURNAL] Sortie partielle enregistrée (trade_id={trade_id}): "
                  f"{quantite_vendue} vendues @ {prix_sortie:.2f}, PnL={pnl_net:.2f}$, restant={restant}")

        self.conn.commit()
        cur.close()
        return exit_id or 0

    def get_trades(
        self,
        trade_mode: Optional[TradeMode] = None,
        strategy_name: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Récupère les trades selon les filtres."""
        self.connect()

        query = "SELECT * FROM trades WHERE 1=1"
        params: List[Any] = []

        if trade_mode:
            query += " AND trade_mode = %s"
            params.append(trade_mode.value)
        if strategy_name:
            query += " AND strategy_name = %s"
            params.append(strategy_name)
        if symbol:
            query += " AND symbol = %s"
            params.append(symbol)
        if start_date:
            query += " AND date_entree >= %s"
            params.append(start_date.strftime('%Y-%m-%d'))
        if end_date:
            query += " AND date_entree <= %s"
            params.append(end_date.strftime('%Y-%m-%d'))

        query += " ORDER BY date_entree DESC LIMIT %s"
        params.append(limit)

        # Utiliser le curseur psycopg2 directement pour éviter le warning
        # pandas ne supporte officiellement que SQLAlchemy ou sqlite3 dans read_sql_query
        cur = self.conn.cursor()
        cur.execute(query, params)
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        cur.close()
        df = pd.DataFrame(rows, columns=cols)

        if not df.empty:
            df['date_entree'] = pd.to_datetime(df['date_entree'])
            df['date_sortie'] = pd.to_datetime(df['date_sortie'])

        return df

    def get_performance_summary(
        self,
        trade_mode: Optional[TradeMode] = None,
        strategy_name: Optional[str] = None,
        backtest_start_date=None,
        backtest_end_date=None
    ) -> Dict[str, Any]:
        """
        Calcule un résumé de performance.
        Si backtest_start_date/backtest_end_date sont fournis, ne considère
        que les trades de cette période (plusieurs fenêtres de backtest
        peuvent coexister en base pour une même stratégie).
        """
        df = self.get_trades(trade_mode=trade_mode, strategy_name=strategy_name)

        if df.empty:
            return {"error": "Aucun trade trouvé"}

        if backtest_start_date is not None and 'backtest_start_date' in df.columns:
            df = df[pd.to_datetime(df['backtest_start_date'])
                    == pd.to_datetime(backtest_start_date).normalize()]
        if backtest_end_date is not None and 'backtest_end_date' in df.columns:
            df = df[pd.to_datetime(df['backtest_end_date'])
                    == pd.to_datetime(backtest_end_date).normalize()]
        if df.empty:
            return {"error": "Aucun trade trouvé pour cette période"}

        closed = df[df['date_sortie'].notna()].copy()
        if closed.empty:
            return {"error": "Aucun trade fermé"}

        total_trades   = len(closed)
        winning_trades = len(closed[closed['pnl_net'] > 0])
        losing_trades  = len(closed[closed['pnl_net'] < 0])

        return {
            "total_trades":    total_trades,
            "winning_trades":  winning_trades,
            "losing_trades":   losing_trades,
            "win_rate":        winning_trades / total_trades * 100 if total_trades > 0 else 0,
            "total_pnl_net":   closed['pnl_net'].sum(),
            "avg_pnl_net":     closed['pnl_net'].mean(),
            "avg_winning_trade": closed[closed['pnl_net'] > 0]['pnl_net'].mean() if winning_trades > 0 else 0,
            "avg_losing_trade":  closed[closed['pnl_net'] < 0]['pnl_net'].mean() if losing_trades > 0 else 0,
            "max_win":          closed['pnl_net'].max(),
            "max_loss":         closed['pnl_net'].min(),
            "avg_bars_held":    closed['bars_held'].mean() if 'bars_held' in closed.columns else 0,
            "total_commission": closed['commission'].sum() if 'commission' in closed.columns else 0
        }

    def export_to_csv(
        self,
        filepath: str,
        trade_mode: Optional[TradeMode] = None,
        strategy_name: Optional[str] = None
    ):
        """Exporte les trades vers un fichier CSV."""
        df = self.get_trades(trade_mode=trade_mode, strategy_name=strategy_name)
        df.to_csv(filepath, index=False)
        print(f"[EXPORT] {len(df)} trades exportés vers {filepath}")


if __name__ == "__main__":
    journal = TradeJournal()
    journal.clear_backtest_trades("TestStrategy")
    trade_id = journal.log_trade(
        trade_mode=TradeMode.BACKTEST,
        strategy_name="TestStrategy",
        symbol="AAPL",
        date_entree=datetime(2025, 1, 10, 10, 30),
        prix_entree=150.0,
        quantite=100,
        date_sortie=datetime(2025, 1, 15, 14, 0),
        prix_sortie=155.0,
        cause_sortie="TRAILING_STOP",
        pnl_brut=500.0,
        commission=2.0,
        pnl_net=498.0,
        bars_held=5
    )
    print(f"Trade ajouté avec ID: {trade_id}")
    trades = journal.get_trades(trade_mode=TradeMode.BACKTEST)
    print(f"Trades backtest: {len(trades)}")
    summary = journal.get_performance_summary(trade_mode=TradeMode.BACKTEST)
    print(f"Résumé: {summary}")
    journal.close()
