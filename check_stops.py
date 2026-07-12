import sys; sys.path.insert(0, '.')
from src.app.database.pg_connection import get_conn, dict_cursor

conn = get_conn()
cur = dict_cursor(conn)
cur.execute("""
    SELECT symbol, entry_price, qty_remaining, stop_level, high_water_mark,
           profit_level, protection_type, active, last_checked
    FROM position_stops
    WHERE active = 1
    ORDER BY symbol
""")
rows = cur.fetchall()
print(f"{len(rows)} stop(s) actif(s) en PostgreSQL:\n")
for r in rows:
    r = dict(r)
    pnl_stop = (r["stop_level"] - r["entry_price"]) / r["entry_price"] * 100
    checked = str(r["last_checked"])[:16] if r["last_checked"] else "jamais"
    print(f'  {r["symbol"]:6} entree={r["entry_price"]:>8.2f}  qty={r["qty_remaining"]:>4}  '
          f'stop={r["stop_level"]:>8.2f} ({pnl_stop:+.1f}%)  '
          f'HWM={r["high_water_mark"]:>8.2f}  profit={r["profit_level"]:>8.2f}  '
          f'verifie={checked}')
cur.close()
conn.close()
