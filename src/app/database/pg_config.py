PG_CONFIG = {
    "host":     "192.168.0.108",
    "port":     5432,
    "user":     "drako",
    "password": "AndresyKely24!",
    "database": "stockus",
    "sslmode":  "disable",
}

PG_CONFIG_CA = {
    "host":     "192.168.0.108",
    "port":     5432,
    "user":     "drako",
    "password": "AndresyKely24!",
    "database": "stockca",
    "sslmode":  "disable",
}

def get_pg_dsn() -> str:
    """Retourne la DSN psycopg2 sous forme de chaîne."""
    c = PG_CONFIG
    return (
        f"host={c['host']} port={c['port']} "
        f"dbname={c['database']} user={c['user']} password={c['password']}"
    )
