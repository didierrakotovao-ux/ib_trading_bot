import sys; sys.path.insert(0, '.')
from src.app.database.pg_connection import get_conn, dict_cursor
from datetime import date, timedelta

conn = get_conn()
cur = dict_cursor(conn)

# Derniers trades enregistrés
cur.execute("""
    SELECT trade_mode, DATE(date_entree)::text AS jour, COUNT(*) AS n
    FROM trades
    WHERE date_entree >= NOW() - INTERVAL '7 days'
    GROUP BY trade_mode, DATE(date_entree)
    ORDER BY jour DESC, trade_mode
""")
rows = cur.fetchall()
print("Trades des 7 derniers jours :")
if rows:
    for r in rows:
        print(f"  {r['jour']}  {r['trade_mode']:8}  {r['n']} trade(s)")
else:
    print("  Aucun trade")

print()

# Positions ouvertes (quantite_restante > 0)
cur.execute("""
    SELECT symbol, trade_mode, DATE(date_entree)::text AS entree,
           prix_entree, quantite_restante
    FROM trades
    WHERE quantite_restante > 0
    ORDER BY date_entree DESC
""")
open_trades = cur.fetchall()
print(f"Positions ouvertes ({len(open_trades)}) :")
for r in open_trades:
    print(f"  {r['symbol']:6} {r['trade_mode']:6}  entre {r['entree']}  prix={r['prix_entree']:.2f}  restant={r['quantite_restante']}")

cur.close(); conn.close()
