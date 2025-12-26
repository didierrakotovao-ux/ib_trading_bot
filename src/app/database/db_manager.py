"""
Module de gestion de base de données pour le stockage des résultats de screening,
données historiques et signaux de trading.
"""
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any


class DatabaseManager:
    """Gestionnaire de base de données SQLite pour le trading."""
    
    def __init__(self, db_path: str = "trading_data.db"):
        """
        Initialize le gestionnaire de base de données.
        
        Args:
            db_path: Chemin vers le fichier SQLite
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._create_tables()
    
    def connect(self):
        """Établit la connexion à la base de données."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # Accès par nom de colonne
            print(f"[OK] Base de donnees connectee: {self.db_path}")
    
    def close(self):
        """Ferme la connexion à la base de données."""
        if self.conn:
            self.conn.close()
            self.conn = None
            print("🔌 Base de données fermée")
    
    def _create_tables(self):
        """Crée les tables si elles n'existent pas."""
        self.connect()
        
        if self.conn is None:
            raise RuntimeError("La connexion à la base de données a échoué")
        
        cursor = self.conn.cursor()
        
        # Table pour les résultats de scanner
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scanner_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                exchange TEXT,
                scan_type TEXT NOT NULL,
                rank INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)
        
        # Index pour scanner_results
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_scanner_symbol 
            ON scanner_results(symbol, scan_type, timestamp DESC)
        """)
        
        # Table pour les données historiques (cache)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                adjusted_close REAL,
                dividends REAL DEFAULT 0,
                stock_splits REAL DEFAULT 0,
                source TEXT NOT NULL DEFAULT 'unknown',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date, source)
            )
        """)
        
        # Index pour améliorer les performances
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_historical_symbol_date 
            ON historical_data(symbol, date DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_historical_date 
            ON historical_data(date DESC)
        """)
        
        # Table pour les indicateurs techniques (features)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS technical_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                
                -- Moving Averages
                sma_20 REAL,
                sma_50 REAL,
                sma_200 REAL,
                ema_12 REAL,
                ema_26 REAL,
                
                -- Momentum
                rsi_14 REAL,
                macd REAL,
                macd_signal REAL,
                macd_hist REAL,
                
                -- Volatility
                bb_upper REAL,
                bb_middle REAL,
                bb_lower REAL,
                atr_14 REAL,
                
                -- Volume
                volume_sma_20 REAL,
                volume_ratio REAL,
                
                -- Custom
                pct_from_high_52w REAL,
                
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date)
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_indicators_symbol_date 
            ON technical_indicators(symbol, date DESC)
        """)
        
        # Table pour les signaux de trading
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL CHECK(signal_type IN ('BUY', 'SELL', 'HOLD', 'ACCUMULATION', 'DISTRIBUTION', 'WATCH')),
                strategy TEXT NOT NULL,
                price REAL,
                confidence REAL CHECK(confidence >= 0 AND confidence <= 1),
                target_price REAL,
                stop_loss REAL,
                metadata TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_symbol 
            ON trading_signals(symbol, timestamp DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_type_strategy 
            ON trading_signals(signal_type, strategy, timestamp DESC)
        """)
        
        # Table pour les labels (résultats réels pour ML)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                entry_price REAL NOT NULL,
                entry_date DATE NOT NULL,
                
                -- Résultats à différents horizons
                price_5d REAL,
                return_5d REAL,
                price_10d REAL,
                return_10d REAL,
                price_20d REAL,
                return_20d REAL,
                
                -- Extremes
                max_gain REAL,
                max_loss REAL,
                max_gain_days INTEGER,
                max_loss_days INTEGER,
                
                -- Label pour ML
                profitable BOOLEAN,
                roi REAL,
                
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (signal_id) REFERENCES trading_signals(id),
                UNIQUE(signal_id)
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_outcomes_symbol 
            ON signal_outcomes(symbol, entry_date DESC)
        """)
        
        # Table pour les watchlists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL UNIQUE,
                reason TEXT,
                added_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                active BOOLEAN DEFAULT 1,
                notes TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_watchlist_active 
            ON watchlist(active, added_date DESC)
        """)
        
        # Table pour les métadonnées des symboles
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS symbol_metadata (
                symbol TEXT PRIMARY KEY,
                company_name TEXT,
                sector TEXT,
                industry TEXT,
                market_cap REAL,
                exchange TEXT,
                currency TEXT DEFAULT 'USD',
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.commit()
        print("[OK] Tables de base de donnees initialisees")
    
    def save_scanner_results(self, results: List[Dict[str, Any]], scan_type: str):
        """
        Sauvegarde les résultats d'un scanner.
        
        Args:
            results: Liste de résultats du scanner
            scan_type: Type de scan utilisé
        """
        self.connect()
        if self.conn is None:
            raise RuntimeError("La connexion à la base de données a échoué")
        
        cursor = self.conn.cursor()
        
        for result in results:
            cursor.execute("""
                INSERT INTO scanner_results (symbol, exchange, scan_type, rank, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                result.get('symbol'),
                result.get('exchange'),
                scan_type,
                result.get('rank'),
                str(result)  # Stocker le dict complet en string
            ))
        
        self.conn.commit()
        print(f"[OK] {len(results)} resultats de scanner sauvegardes")
    
    def save_historical_data(self, symbol: str, df: pd.DataFrame, source: str = "unknown") -> bool:
        """
        Sauvegarde les données historiques dans la DB (cache).
        
        Args:
            symbol: Symbole de l'action
            df: DataFrame avec colonnes: date, open, high, low, close, volume
            source: Source des données ("ib", "yfinance", etc.)
            
        Returns:
            True si sauvegarde réussie, False sinon
        """
        self.connect()
        if self.conn is None:
            raise RuntimeError("La connexion à la base de données a échoué")
        
        df_copy = df.copy()
        
        # Si l'index est un DatetimeIndex, le convertir en colonne
        if isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy['date'] = df_copy.index.strftime('%Y-%m-%d')
            df_copy = df_copy.reset_index(drop=True)
        
        df_copy['symbol'] = symbol
        df_copy['source'] = source
        
        # Normaliser les noms de colonnes (yfinance utilise des majuscules)
        df_copy.columns = df_copy.columns.str.lower()
        
        # Convertir date en string si nécessaire
        if 'date' in df_copy.columns:
            df_copy['date'] = pd.to_datetime(df_copy['date']).dt.strftime('%Y-%m-%d') # type: ignore
        
        # Colonnes à garder
        base_columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'source']
        optional_columns = ['dividends', 'stock splits']
        
        # Garder seulement les colonnes qui existent
        columns_to_keep = base_columns.copy()
        for col in optional_columns:
            if col in df_copy.columns:
                columns_to_keep.append(col)
        
        df_copy = df_copy[[c for c in columns_to_keep if c in df_copy.columns]]
        
        # Renommer stock splits si présent
        if 'stock splits' in df_copy.columns:
            df_copy = df_copy.rename(columns={'stock splits': 'stock_splits'}) # type: ignore
        
        # Reset index pour éviter les conflits
        df_copy = df_copy.reset_index(drop=True)
        
        try:
            df_copy.to_sql(
                'historical_data', 
                self.conn, 
                if_exists='append', 
                index=False,
                dtype={
                    'symbol': 'TEXT',
                    'date': 'DATE',
                    'open': 'REAL',
                    'high': 'REAL',
                    'low': 'REAL',
                    'close': 'REAL',
                    'volume': 'INTEGER',
                    'dividends': 'REAL',
                    'stock_splits': 'REAL',
                    'source': 'TEXT'
                }
            )
            self.conn.commit()
            print(f"[OK] Donnees historiques pour {symbol} sauvegardees ({len(df)} barres)")
            return True
        except sqlite3.IntegrityError as e:
            # Données déjà existantes (conflit UNIQUE)
            print(f"[INFO] Donnees pour {symbol} deja en cache ({str(e)[:50]})")
            return True  # Considérer comme succès si déjà présent
        except Exception as e:
            print(f"[ERROR] Erreur sauvegarde {symbol}: {e}")
            return False
    
    def get_historical_data(
        self, 
        symbol: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        source: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Récupère les données historiques depuis la DB.
        
        Args:
            symbol: Symbole de l'action
            start_date: Date de début (optionnelle)
            end_date: Date de fin (optionnelle)
            source: Source spécifique (optionnelle)
            
        Returns:
            DataFrame ou None si pas de données
        """
        self.connect()
        if self.conn is None:
            raise RuntimeError("La connexion à la base de données a échoué")
        
        query = "SELECT * FROM historical_data WHERE symbol = ?"
        params: List[Any] = [symbol]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date.strftime('%Y-%m-%d'))
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date.strftime('%Y-%m-%d'))
        
        if source:
            query += " AND source = ?"
            params.append(source)
        
        query += " ORDER BY date ASC"
        
        df = pd.read_sql_query(query, self.conn, params=params)
        
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
        """
        Sauvegarde un signal de trading.
        
        Args:
            symbol: Symbole
            signal_type: "BUY", "SELL", "HOLD"
            strategy: Nom de la stratégie
            price: Prix au moment du signal
            confidence: Niveau de confiance (0-1)
            metadata: Informations supplémentaires
        """
        self.connect()
        if self.conn is None:
            raise RuntimeError("La connexion à la base de données a échoué")
        
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO trading_signals 
            (symbol, signal_type, strategy, price, confidence, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (symbol, signal_type, strategy, price, confidence, str(metadata or {})))
        
        self.conn.commit()
        print(f"[OK] Signal {signal_type} pour {symbol} sauvegarde")
    
    def get_trading_signals(
        self, 
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Récupère les signaux de trading.
        
        Args:
            symbol: Filtrer par symbole (optionnel)
            strategy: Filtrer par stratégie (optionnel)
            limit: Nombre maximum de résultats
            
        Returns:
            DataFrame des signaux
        """
        self.connect()
        if self.conn is None:
            raise RuntimeError("La connexion à la base de données a échoué")
        
        query = "SELECT * FROM trading_signals WHERE 1=1"
        params: List[Any] = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        return pd.read_sql_query(query, self.conn, params=params)
    
    def add_to_watchlist(self, symbol: str, reason: str = ""):
        """Ajoute un symbole à la watchlist."""
        self.connect()
        if self.conn is None:
            raise RuntimeError("La connexion à la base de données a échoué")
        
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO watchlist (symbol, reason)
                VALUES (?, ?)
            """, (symbol, reason))
            self.conn.commit()
            print(f"[OK] {symbol} ajoute a la watchlist")
        except sqlite3.IntegrityError:
            print(f"[INFO] {symbol} deja dans la watchlist")
    
    def get_watchlist(self, active_only: bool = True) -> List[str]:
        """Récupère la liste des symboles en watchlist."""
        self.connect()
        if self.conn is None:
            raise RuntimeError("La connexion à la base de données a échoué")
        
        query = "SELECT symbol FROM watchlist"
        if active_only:
            query += " WHERE active = 1"
        query += " ORDER BY added_date DESC"
        
        cursor = self.conn.cursor()
        cursor.execute(query)
        
        return [row[0] for row in cursor.fetchall()]
    
    def export_to_csv(self, table_name: str, output_path: str):
        """
        Exporte une table vers un fichier CSV.
        
        Args:
            table_name: Nom de la table
            output_path: Chemin du fichier CSV de sortie
        """
        self.connect()
        if self.conn is None:
            raise RuntimeError("La connexion à la base de données a échoué")
        
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", self.conn)
        df.to_csv(output_path, index=False)
        print(f"[OK] Table {table_name} exportee vers {output_path}")
    
    def get_all_symbols(self) -> List[str]:
        """
        Récupère tous les symboles uniques ayant des données historiques.
        
        Returns:
            Liste des symboles
        """
        self.connect()
        if self.conn is None:
            raise RuntimeError("La connexion à la base de données a échoué")
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM historical_data ORDER BY symbol")
        
        return [row[0] for row in cursor.fetchall()]
    
    def get_all_signals(self) -> List[Dict]:
        """
        Récupère tous les signaux de trading.
        
        Returns:
            Liste de dictionnaires avec les signaux
        """
        self.connect()
        if self.conn is None:
            raise RuntimeError("La connexion à la base de données a échoué")
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, symbol, signal_type, entry_price, entry_date, 
                   reason, confidence, metadata, timestamp
            FROM trading_signals
            ORDER BY entry_date DESC
        """)
        
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
