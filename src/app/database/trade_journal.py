"""
Module de journalisation des trades en base de données.
Supporte trois modes : backtest, paper, live.
"""
import sqlite3
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class TradeMode(Enum):
    """Mode de trading."""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class TradeJournal:
    """
    Gestionnaire de journal de trades en base de données.

    Catégorise les trades par mode (backtest, paper, live) et par stratégie.
    Pour les backtests, efface automatiquement les trades existants de la stratégie.
    """

    def __init__(self, db_path: str = "trading_data.db"):
        """
        Initialise le journal de trades.

        Args:
            db_path: Chemin vers le fichier SQLite
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._create_table()

    def connect(self):
        """Établit la connexion à la base de données."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row

    def close(self):
        """Ferme la connexion."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def _create_table(self):
        """Crée la table trades si elle n'existe pas."""
        self.connect()
        if self.conn is None:
            raise RuntimeError("Connexion BD échouée")

        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                -- Identification
                trade_mode TEXT NOT NULL CHECK(trade_mode IN ('backtest', 'paper', 'live')),
                strategy_name TEXT NOT NULL,
                symbol TEXT NOT NULL,

                -- Entrée
                date_entree DATETIME NOT NULL,
                prix_entree REAL NOT NULL,
                quantite INTEGER NOT NULL,

                -- Sortie
                date_sortie DATETIME,
                prix_sortie REAL,
                cause_sortie TEXT,

                -- Performance
                pnl_brut REAL,
                commission REAL,
                pnl_net REAL,

                -- Métriques
                bars_held INTEGER,
                score_entree REAL,

                -- Signaux (pour exhaustion stop)
                exhaustion_signals TEXT,

                -- Métadonnées
                backtest_start_date DATE,
                backtest_end_date DATE,
                notes TEXT,

                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Index pour les requêtes fréquentes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_mode_strategy
            ON trades(trade_mode, strategy_name)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_symbol
            ON trades(symbol, date_entree DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_date
            ON trades(date_entree DESC)
        """)

        self.conn.commit()
        print("[OK] Table trades initialisée")

    def clear_backtest_trades(self, strategy_name: str):
        """
        Efface tous les trades de backtest pour une stratégie donnée.

        Args:
            strategy_name: Nom de la stratégie
        """
        self.connect()
        if self.conn is None:
            raise RuntimeError("Connexion BD échouée")

        cursor = self.conn.cursor()
        cursor.execute("""
            DELETE FROM trades
            WHERE trade_mode = 'backtest' AND strategy_name = ?
        """, (strategy_name,))

        deleted = cursor.rowcount
        self.conn.commit()

        if deleted > 0:
            print(f"[CLEAN] {deleted} trades backtest effacés pour {strategy_name}")

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
        """
        Enregistre un trade dans le journal.

        Args:
            trade_mode: Mode de trading (backtest, paper, live)
            strategy_name: Nom de la stratégie
            symbol: Symbole tradé
            date_entree: Date/heure d'entrée
            prix_entree: Prix d'entrée
            quantite: Quantité
            date_sortie: Date/heure de sortie (optionnel)
            prix_sortie: Prix de sortie (optionnel)
            cause_sortie: Cause de la sortie (TRAILING_STOP, EXHAUSTION_STOP, etc.)
            pnl_brut: P&L brut
            commission: Commissions payées
            pnl_net: P&L net
            bars_held: Nombre de bars tenus
            score_entree: Score ML à l'entrée
            exhaustion_signals: Signaux d'essoufflement détectés
            backtest_start_date: Date de début du backtest
            backtest_end_date: Date de fin du backtest
            notes: Notes additionnelles

        Returns:
            ID du trade inséré
        """
        self.connect()
        if self.conn is None:
            raise RuntimeError("Connexion BD échouée")

        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO trades (
                trade_mode, strategy_name, symbol,
                date_entree, prix_entree, quantite,
                date_sortie, prix_sortie, cause_sortie,
                pnl_brut, commission, pnl_net,
                bars_held, score_entree, exhaustion_signals,
                backtest_start_date, backtest_end_date, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_mode.value,
            strategy_name,
            symbol,
            date_entree.strftime('%Y-%m-%d %H:%M:%S') if date_entree else None,
            prix_entree,
            quantite,
            date_sortie.strftime('%Y-%m-%d %H:%M:%S') if date_sortie else None,
            prix_sortie,
            cause_sortie,
            pnl_brut,
            commission,
            pnl_net,
            bars_held,
            score_entree,
            exhaustion_signals,
            backtest_start_date.strftime('%Y-%m-%d') if backtest_start_date else None,
            backtest_end_date.strftime('%Y-%m-%d') if backtest_end_date else None,
            notes
        ))

        self.conn.commit()
        return cursor.lastrowid or 0

    def get_trades(
        self,
        trade_mode: Optional[TradeMode] = None,
        strategy_name: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Récupère les trades selon les filtres.

        Args:
            trade_mode: Filtrer par mode
            strategy_name: Filtrer par stratégie
            symbol: Filtrer par symbole
            start_date: Date de début
            end_date: Date de fin
            limit: Nombre max de résultats

        Returns:
            DataFrame des trades
        """
        self.connect()
        if self.conn is None:
            raise RuntimeError("Connexion BD échouée")

        query = "SELECT * FROM trades WHERE 1=1"
        params: List[Any] = []

        if trade_mode:
            query += " AND trade_mode = ?"
            params.append(trade_mode.value)

        if strategy_name:
            query += " AND strategy_name = ?"
            params.append(strategy_name)

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if start_date:
            query += " AND date_entree >= ?"
            params.append(start_date.strftime('%Y-%m-%d'))

        if end_date:
            query += " AND date_entree <= ?"
            params.append(end_date.strftime('%Y-%m-%d'))

        query += " ORDER BY date_entree DESC LIMIT ?"
        params.append(limit)

        df = pd.read_sql_query(query, self.conn, params=params)

        # Convertir les dates
        if not df.empty:
            df['date_entree'] = pd.to_datetime(df['date_entree'])
            df['date_sortie'] = pd.to_datetime(df['date_sortie'])

        return df

    def get_performance_summary(
        self,
        trade_mode: Optional[TradeMode] = None,
        strategy_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calcule un résumé de performance.

        Args:
            trade_mode: Filtrer par mode
            strategy_name: Filtrer par stratégie

        Returns:
            Dictionnaire avec les métriques de performance
        """
        df = self.get_trades(trade_mode=trade_mode, strategy_name=strategy_name)

        if df.empty:
            return {"error": "Aucun trade trouvé"}

        # Filtrer les trades fermés
        closed = df[df['date_sortie'].notna()].copy()

        if closed.empty:
            return {"error": "Aucun trade fermé"}

        total_trades = len(closed)
        winning_trades = len(closed[closed['pnl_net'] > 0])
        losing_trades = len(closed[closed['pnl_net'] < 0])

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": winning_trades / total_trades * 100 if total_trades > 0 else 0,
            "total_pnl_net": closed['pnl_net'].sum(),
            "avg_pnl_net": closed['pnl_net'].mean(),
            "avg_winning_trade": closed[closed['pnl_net'] > 0]['pnl_net'].mean() if winning_trades > 0 else 0,
            "avg_losing_trade": closed[closed['pnl_net'] < 0]['pnl_net'].mean() if losing_trades > 0 else 0,
            "max_win": closed['pnl_net'].max(),
            "max_loss": closed['pnl_net'].min(),
            "avg_bars_held": closed['bars_held'].mean() if 'bars_held' in closed.columns else 0,
            "total_commission": closed['commission'].sum() if 'commission' in closed.columns else 0
        }

    def export_to_csv(
        self,
        filepath: str,
        trade_mode: Optional[TradeMode] = None,
        strategy_name: Optional[str] = None
    ):
        """
        Exporte les trades vers un fichier CSV.

        Args:
            filepath: Chemin du fichier CSV
            trade_mode: Filtrer par mode
            strategy_name: Filtrer par stratégie
        """
        df = self.get_trades(trade_mode=trade_mode, strategy_name=strategy_name)
        df.to_csv(filepath, index=False)
        print(f"[EXPORT] {len(df)} trades exportés vers {filepath}")


# Test standalone
if __name__ == "__main__":
    journal = TradeJournal()

    # Test: effacer et ajouter des trades de backtest
    journal.clear_backtest_trades("TestStrategy")

    # Ajouter un trade test
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

    # Récupérer les trades
    trades = journal.get_trades(trade_mode=TradeMode.BACKTEST)
    print(f"\nTrades backtest: {len(trades)}")
    print(trades)

    # Résumé de performance
    summary = journal.get_performance_summary(trade_mode=TradeMode.BACKTEST)
    print(f"\nRésumé: {summary}")

    journal.close()
