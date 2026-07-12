import sys; sys.path.insert(0, '.')
from src.app.database.pg_connection import get_conn, get_conn_ca
from datetime import date

# US data
conn = get_conn()
cur = conn.cursor()
cur.execute("SELECT MAX(date)::text FROM historical_data")
last_us = cur.fetchone()[0]
cur.execute("SELECT COUNT(DISTINCT symbol) FROM historical_data WHERE date = (SELECT MAX(date) FROM historical_data)")
n_us = cur.fetchone()[0]
cur.close(); conn.close()

# CA data
conn_ca = get_conn_ca()
cur_ca = conn_ca.cursor()
cur_ca.execute("SELECT MAX(date)::text FROM historical_data")
last_ca = cur_ca.fetchone()[0]
cur_ca.execute("SELECT COUNT(DISTINCT symbol) FROM historical_data WHERE date = (SELECT MAX(date) FROM historical_data)")
n_ca = cur_ca.fetchone()[0]
cur_ca.close(); conn_ca.close()

today = date.today()
print(f"Aujourd'hui         : {today}")
print(f"US  derniere date   : {last_us}  ({n_us} symboles)")
print(f"CA  derniere date   : {last_ca}  ({n_ca} symboles)")
print()
if str(last_us) == str(today):
    print("US  : a jour")
else:
    print(f"US  : EN RETARD (manque {last_us} -> {today})")
if str(last_ca) == str(today):
    print("CA  : a jour")
else:
    print(f"CA  : EN RETARD (manque {last_ca} -> {today})")
