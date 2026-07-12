"""
Consolidation du journal IB (CSV TWS) vers la base de données.
Insère les trades et exits manquants identifiés dans DUE674885.TRANSACTIONS.csv
"""
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, PROJECT_ROOT)
from src.app.database.pg_connection import get_conn, dict_cursor

# BUYs absents de la DB (date, symbol, qty, prix_moyen, commission)
MISSING_BUYS = [
    ('2026-02-12', 'CIFR',  1228,  15.8966,  -6.1400),
    ('2026-02-12', 'NBIS',   200,  86.9440,  -1.0000),
    ('2026-03-30', 'AAOI',   235,  83.8860,  -1.1750),
    ('2026-03-30', 'AXTI',   180,  54.0722,  -1.0000),
    ('2026-03-30', 'IBRX',  2886,   6.7395, -14.4300),
    ('2026-03-30', 'ONDS',  2403,   8.1125, -12.0150),
    ('2026-03-30', 'RKLB',   340,  57.7518,  -1.7000),
    ('2026-04-13', 'IREN',   100,  38.9550,  -1.0000),
    ('2026-04-13', 'ONDS',   500,   8.9560,  -2.5000),
    ('2026-04-13', 'WULF',   200,  18.6050,  -1.0000),
    ('2026-04-16', 'IREN',   420,  46.6762,  -2.1000),
    ('2026-04-20', 'IREN',   280,  48.1514,  -1.4000),
    ('2026-04-22', 'KYTX',  1000,   9.3170,  -5.0000),
    ('2026-04-22', 'ONDS',   700,  10.8986,  -3.5000),
    ('2026-04-27', 'RKLB',   251,  79.4000,  -1.2550),
]

# SELLs absents de trade_exits (date, symbol, qty, prix_moyen, commission)
MISSING_SELLS = [
    ('2026-01-30', 'INTC',  455,  47.0900,  -2.3637),
    ('2026-01-30', 'ONDS', 1629,  10.3600,  -8.4627),
    ('2026-01-30', 'RGTI',  908,  18.1500,  -4.7171),
    ('2026-02-02', 'MARA', 1928,   9.0600, -10.0160),
    ('2026-02-03', 'APLD',  590,  34.8200,  -3.0651),
    ('2026-02-03', 'CIFR', 1253,  15.7100,  -6.5093),
    ('2026-02-03', 'IREN',  376,  54.5500,  -1.9533),
    ('2026-02-03', 'ONDS', 1930,  10.7200, -10.0263),
    ('2026-02-03', 'SGMT', 3609,   6.6797, -18.7488),
    ('2026-02-04', 'APLD',  544,  30.1800,  -2.8261),
    ('2026-02-04', 'CIFR', 1230,  13.1400,  -6.3899),
    ('2026-02-04', 'HOOD',  229,  78.4500,  -1.1897),
    ('2026-02-04', 'IREN',  367,  46.3200,  -1.9066),
    ('2026-02-04', 'ONDS', 1757,   9.2500,  -9.1276),
    ('2026-02-06', 'APLD',  634,  34.7100,  -3.2936),
    ('2026-02-06', 'CIFR', 1403,  14.6000,  -7.2886),
    ('2026-02-06', 'IREN',  445,  43.4100,  -2.3118),
    ('2026-02-06', 'ONDS', 2066,   9.5800, -10.7329),
    ('2026-02-06', 'WULF', 1440,  14.2300,  -7.4808),
    ('2026-02-19', 'CIFR', 1228,  15.3668,  -6.3795),
    ('2026-02-19', 'HOOD',  231,  75.1617,  -1.2000),
    ('2026-02-19', 'IREN',  433,  42.3930,  -2.2494),
    ('2026-02-19', 'MU',    52,  416.2081,  -1.0101),
    ('2026-02-19', 'ONDS', 1934,  11.3105, -10.0471),
    ('2026-02-20', 'NBIS',  120, 100.2067,  -2.0234),
    ('2026-02-23', 'NBIS',   80,  99.5050,  -1.0156),
    ('2026-02-24', 'ONDS', 1755,  10.1810,  -9.1172),
    ('2026-02-24', 'RKLB',  282,  69.8279,  -1.4650),
    ('2026-02-25', 'ASTS',   10,  85.4000,  -1.0019),
    ('2026-02-25', 'CIFR', 1265,  17.1258,  -6.5717),
    ('2026-02-25', 'IREN',   60,  46.1083,  -1.0117),
    ('2026-02-25', 'WULF', 1200,  18.1875,  -6.2340),
    ('2026-02-26', 'WULF',   92,  17.8404,  -1.0179),
    ('2026-03-02', 'ASTS',   30,  84.3300,  -1.0058),
    ('2026-03-02', 'ONDS', 1923,  10.6600,  -9.9900),
    ('2026-03-30', 'AXTI',  180,  52.2333,  -1.0351),
    ('2026-03-30', 'SNDK',   10, 608.4900,  -1.0019),
    ('2026-04-02', 'CIFR', 1204,  12.8208,  -6.5728),
    ('2026-04-02', 'IREN',  454,  34.7157,  -2.6832),
    ('2026-04-07', 'AAOI',  235, 113.1700,  -1.7687),
    ('2026-04-07', 'IBRX', 2886,   6.8227, -15.3984),
    ('2026-04-07', 'ONDS', 2403,   9.4055, -12.9492),
    ('2026-04-07', 'RKLB',  340,  65.7694,  -3.6269),
    ('2026-04-07', 'SNDK',   21, 695.0300,  -1.3048),
    ('2026-04-14', 'ASTS',   30,  92.8800,  -1.0632),
    ('2026-04-14', 'IREN',  100,  46.3300,  -1.1149),
    ('2026-04-14', 'ONDS',  240,   9.4942,  -1.2937),
    ('2026-04-17', 'IREN',  420,  48.0352,  -2.5975),
    ('2026-04-17', 'WULF',  200,  20.0260,  -1.1215),
    ('2026-04-20', 'ASTS',   50,  79.0760,  -1.0912),
    ('2026-04-20', 'RKLB',   20,  87.6200,  -1.0400),
    ('2026-04-21', 'IREN',  280,  46.0000,  -1.7199),
    ('2026-04-21', 'ONDS',  260,  10.9000,  -1.4091),
    ('2026-04-23', 'KYTX',  600,   9.6267,  -3.2360),
    ('2026-04-27', 'KYTX',  400,   8.9300,  -2.1516),
    ('2026-04-27', 'ONDS',  700,  10.8700,  -3.7932),
    ('2026-04-27', 'RKLB',  251,  82.1500,  -1.7287),
]


def find_trade(cur, sym, sell_date, qty_sold):
    """Cherche le trade parent pour un SELL donné."""
    # 1. Trade ouvert (restante >= qty), entré avant sell_date (FIFO)
    cur.execute('''
        SELECT id, prix_entree, quantite, quantite_restante
        FROM trades
        WHERE symbol=%s AND trade_mode='paper'
          AND DATE(date_entree) <= %s AND quantite_restante >= %s
        ORDER BY date_entree ASC LIMIT 1
    ''', (sym, sell_date, qty_sold))
    row = cur.fetchone()
    if row:
        return dict(row), True

    # 2. Trade fermé sans exit pour ce qty
    cur.execute('''
        SELECT t.id, t.prix_entree, t.quantite, t.quantite_restante
        FROM trades t
        WHERE t.symbol=%s AND t.trade_mode='paper'
          AND DATE(t.date_entree) <= %s
          AND t.quantite_restante = 0
          AND NOT EXISTS (
              SELECT 1 FROM trade_exits te
              WHERE te.trade_id=t.id AND DATE(te.date_sortie)=%s AND te.quantite_vendue=%s
          )
        ORDER BY t.date_entree DESC LIMIT 1
    ''', (sym, sell_date, sell_date, qty_sold))
    row = cur.fetchone()
    if row:
        return dict(row), False

    return None, False


def main():
    conn = get_conn()
    cur = dict_cursor(conn)
    inserted_trades = inserted_exits = updated_restante = 0

    # --- Phase 1 : BUYs manquants ---
    print('=== Phase 1 : Insertion des BUYs manquants ===')
    for (date, sym, qty, px, comm) in MISSING_BUYS:
        cur.execute(
            'SELECT id FROM trades WHERE symbol=%s AND trade_mode=%s AND DATE(date_entree)=%s',
            (sym, 'paper', date)
        )
        existing = cur.fetchone()
        if existing:
            print(f'  [SKIP] {date} {sym} déjà en DB (id={existing["id"]})')
            continue
        cur.execute('''
            INSERT INTO trades (trade_mode, strategy_name, symbol, date_entree, prix_entree,
                quantite, quantite_restante, created_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,NOW())
            RETURNING id
        ''', ('paper', 'smooth_momentum', sym, date + ' 09:30:00', px, qty, qty))
        tid = cur.fetchone()['id']
        print(f'  [OK] {date} {sym} qty={qty} @ {px:.4f}  trade_id={tid}')
        inserted_trades += 1
    conn.commit()

    # --- Phase 2 : SELLs manquants ---
    print()
    print('=== Phase 2 : Insertion des SELLs manquants ===')
    for (date, sym, qty, px, comm) in MISSING_SELLS:
        trade, is_open = find_trade(cur, sym, date, qty)
        if trade is None:
            print(f'  [WARN] {date} {sym} SELL {qty}: aucun trade trouvé')
            continue
        cur.execute(
            'SELECT id FROM trade_exits WHERE trade_id=%s AND DATE(date_sortie)=%s AND quantite_vendue=%s',
            (trade['id'], date, qty)
        )
        if cur.fetchone():
            print(f'  [SKIP] {date} {sym} exit déjà enregistré')
            continue
        entry_px = trade['prix_entree']
        pnl_brut = round((px - entry_px) * qty, 2)
        pnl_net  = round(pnl_brut + comm, 2)
        cur.execute('''
            INSERT INTO trade_exits
                (trade_id, symbol, date_sortie, prix_sortie, quantite_vendue,
                 cause_sortie, pnl_brut, commission, pnl_net)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ''', (trade['id'], sym, date + ' 16:00:00', px, qty,
              'CSV_CLOSE', pnl_brut, comm, pnl_net))
        inserted_exits += 1
        if is_open:
            new_r = max(0, trade['quantite_restante'] - qty)
            cur.execute('UPDATE trades SET quantite_restante=%s WHERE id=%s',
                        (new_r, trade['id']))
            updated_restante += 1
        sign = '+' if pnl_net >= 0 else ''
        print(f'  [OK] {date} {sym} qty={qty} @ {px:.4f} PnL={sign}{pnl_net:.2f}  trade_id={trade["id"]}')
    conn.commit()

    # --- Phase 3 : Résumé ---
    print()
    print('=== Résumé ===')
    print(f'  Trades insérés    : {inserted_trades}')
    print(f'  Exits insérés     : {inserted_exits}')
    print(f'  Restante mis à jour: {updated_restante}')

    cur.execute("SELECT COUNT(*) FROM trade_exits te JOIN trades t ON t.id=te.trade_id WHERE t.trade_mode='paper'")
    total_exits = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM trades WHERE trade_mode='paper'")
    total_trades = cur.fetchone()[0]
    cur.execute("SELECT SUM(pnl_net) FROM trade_exits te JOIN trades t ON t.id=te.trade_id WHERE t.trade_mode='paper'")
    pnl = cur.fetchone()[0] or 0
    print(f'  Trades paper total: {total_trades}')
    print(f'  Exits paper total : {total_exits}')
    print(f'  PnL net total     : {pnl:+.2f}$')

    cur.close()
    conn.close()


if __name__ == '__main__':
    main()
