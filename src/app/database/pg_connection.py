"""
Module central de connexion PostgreSQL.
Remplace tous les sqlite3.connect() dans le projet.

Usage:
    from src.app.database.pg_connection import get_conn, get_engine, dict_cursor
    from src.app.database.pg_connection import get_conn_ca, get_engine_ca  # données canadiennes
"""
import psycopg2
import psycopg2.extras
from sqlalchemy import create_engine, text
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.database.pg_config import PG_CONFIG, PG_CONFIG_CA

_engine = None
_engine_ca = None


def get_conn(autocommit: bool = False):
    """
    Connexion psycopg2 vers la base PostgreSQL (données US).
    Équivalent de sqlite3.connect() avec row_factory=sqlite3.Row.
    """
    conn = psycopg2.connect(**PG_CONFIG)
    conn.autocommit = autocommit
    return conn


def get_conn_ca(autocommit: bool = False):
    """
    Connexion vers la base PostgreSQL des données canadiennes (stockca).
    """
    conn = psycopg2.connect(**PG_CONFIG_CA)
    conn.autocommit = autocommit
    return conn


def dict_cursor(conn):
    """
    Curseur qui retourne des lignes en tant que dict.
    Équivalent de conn.row_factory = sqlite3.Row.
    """
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


def get_engine():
    """
    Moteur SQLAlchemy pour pandas (to_sql / read_sql_query) — données US.
    Pool de 10 connexions persistantes (évite open/close sur réseau).
    """
    global _engine
    if _engine is None:
        c = PG_CONFIG
        url = (
            f"postgresql+psycopg2://{c['user']}:{c['password']}"
            f"@{c['host']}:{c['port']}/{c['database']}"
        )
        _engine = create_engine(
            url,
            pool_pre_ping=True,   # vérifie la connexion avant usage
            pool_size=10,         # 10 connexions persistantes
            max_overflow=5,       # 5 connexions supplémentaires si besoin
            pool_timeout=30,      # timeout d'attente d'une connexion libre
            pool_recycle=1800,    # renouvelle les connexions toutes les 30min
            connect_args={
                "sslmode": "disable",
                "keepalives": 1,           # activer TCP keepalive
                "keepalives_idle": 30,     # envoyer keepalive après 30s d'inactivité
                "keepalives_interval": 10, # relancer toutes les 10s si pas de réponse
                "keepalives_count": 5,     # 5 tentatives avant de déclarer mort
            },
        )
    return _engine


def get_engine_ca():
    """
    Moteur SQLAlchemy pour pandas — base stockca (données canadiennes).
    Pool de 10 connexions persistantes.
    """
    global _engine_ca
    if _engine_ca is None:
        c = PG_CONFIG_CA
        url = (
            f"postgresql+psycopg2://{c['user']}:{c['password']}"
            f"@{c['host']}:{c['port']}/{c['database']}"
        )
        _engine_ca = create_engine(
            url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=5,
            pool_timeout=30,
            pool_recycle=1800,
            connect_args={
                "sslmode": "disable",
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            },
        )
    return _engine_ca


def init_ca_schema():
    """No-op — stockca est une base independante, pas un schema."""
    pass


def read_sql(query: str, params=None) -> pd.DataFrame:
    """
    Exécute une requête SELECT et retourne un DataFrame pandas (base US).
    Utilise le pool de connexions SQLAlchemy pour éviter d'ouvrir/fermer
    une connexion par requête (essentiel sur connexion réseau).

    Usage:
        from src.app.database.pg_connection import read_sql
        df = read_sql("SELECT * FROM historical_data WHERE symbol = %s", (symbol,))
    """
    with get_engine().raw_connection() as conn:
        cur = conn.cursor()
        try:
            cur.execute(query, params or [])
        except Exception:
            # Ne jamais rendre au pool une connexion en transaction avortée
            # (sinon la requête suivante échoue avec "current transaction
            # is aborted")
            conn.rollback()
            raise
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=cols)


def read_sql_ca(query: str, params=None) -> pd.DataFrame:
    """
    Identique à read_sql() mais sur la base canadienne (stockca).
    Utilise le pool de connexions SQLAlchemy.
    """
    with get_engine_ca().raw_connection() as conn:
        cur = conn.cursor()
        try:
            cur.execute(query, params or [])
        except Exception:
            # Idem read_sql : rollback avant de rendre la connexion au pool
            conn.rollback()
            raise
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=cols)
