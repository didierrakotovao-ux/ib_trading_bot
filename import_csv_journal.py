"""
Import des transactions IB depuis un CSV Flex Query vers le journal de trading.
Usage unique: python import_csv_journal.py
"""
import csv
import sys
import os
from datetime import datetime
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.app.database.trade_journal import TradeJournal, TradeMode


def parse_csv(filepath):
    """Parse le CSV IB et retourne les transactions brutes."""
    transactions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 12 or row[0] != 'Transaction History' or row[1] != 'Data':
                continue
            # Ignorer les lignes sans symbole (intérêts, etc.)
            symbol = row[6].strip()
            if not symbol or symbol == '-':
                continue

            side = row[5].strip()  # Buy ou Sell
            if side not in ('Buy', 'Sell'):
                continue

            qty = abs(float(row[7]))
            price = float(row[8])
            # row[9] = Price Currency (USD), row[10] = Gross Amount, row[11] = Commission, row[12] = Net Amount
            comm_str = row[11].strip().replace(',', '')
            commission = abs(float(comm_str)) if comm_str and comm_str != '-' else 0.0

            transactions.append({
                'date': row[2].strip(),
                'symbol': symbol,
                'side': side,
                'qty': qty,
                'price': price,
                'commission': commission,
            })

    return transactions


def aggregate_fills(transactions):
    """Agrège les fills partiels par (symbol, date, side)."""
    groups = defaultdict(lambda: {'qty': 0, 'value': 0, 'commission': 0})

    for tx in transactions:
        key = (tx['symbol'], tx['date'], tx['side'])
        groups[key]['qty'] += tx['qty']
        groups[key]['value'] += tx['qty'] * tx['price']
        groups[key]['commission'] += tx['commission']

    result = []
    for (symbol, date, side), data in groups.items():
        avg_price = data['value'] / data['qty'] if data['qty'] > 0 else 0
        result.append({
            'symbol': symbol,
            'date': date,
            'side': side,
            'qty': data['qty'],
            'avg_price': round(avg_price, 4),
            'commission': round(data['commission'], 4),
        })

    # Trier par date, puis BUY avant SELL dans chaque date
    result.sort(key=lambda x: (x['date'], 0 if x['side'] == 'Buy' else 1))
    return result


def import_to_journal(aggregated, trade_mode=TradeMode.PAPER):
    """Insère les trades agrégés dans le journal."""
    journal = TradeJournal()
    journal.connect()
    cursor = journal.conn.cursor()

    stats = {'buys': 0, 'sells_matched': 0, 'sells_orphan': 0}

    for trade in aggregated:
        symbol = trade['symbol']
        date = datetime.strptime(trade['date'], '%Y-%m-%d')
        qty = trade['qty']
        price = trade['avg_price']
        commission = trade['commission']

        if trade['side'] == 'Buy':
            # Vérifier si déjà dans le journal
            cursor.execute('''
                SELECT id FROM trades
                WHERE symbol = ? AND trade_mode = ? AND date(date_entree) = ?
                AND ABS(quantite - ?) < 10
            ''', (symbol, trade_mode.value, trade['date'], qty))

            if cursor.fetchone():
                print(f"  [=] Déjà existant: BUY {symbol} {qty:.0f} @ {price:.2f} ({trade['date']})")
                continue

            trade_id = journal.log_trade(
                trade_mode=trade_mode,
                strategy_name="Momentum",
                symbol=symbol,
                date_entree=date,
                prix_entree=price,
                quantite=int(qty),
            )
            stats['buys'] += 1
            print(f"  [+] BUY  {symbol:6} {qty:7.0f} @ {price:8.2f}  ({trade['date']}) -> ID {trade_id}")

        else:  # Sell
            # Chercher une entrée ouverte pour ce symbole
            cursor.execute('''
                SELECT id, prix_entree, quantite FROM trades
                WHERE symbol = ? AND trade_mode = ? AND date_sortie IS NULL
                ORDER BY date_entree DESC LIMIT 1
            ''', (symbol, trade_mode.value))

            entry = cursor.fetchone()

            if entry:
                trade_id, entry_price, entry_qty = entry
                pnl_brut = (price - entry_price) * entry_qty
                # Utiliser la commission réelle du CSV (buy + sell)
                pnl_net = pnl_brut - commission

                cursor.execute('''
                    UPDATE trades SET
                        date_sortie = ?,
                        prix_sortie = ?,
                        cause_sortie = ?,
                        pnl_brut = ?,
                        commission = ?,
                        pnl_net = ?
                    WHERE id = ?
                ''', (
                    date.strftime('%Y-%m-%d %H:%M:%S'),
                    price,
                    'TRAILING_STOP',
                    round(pnl_brut, 2),
                    round(commission, 2),
                    round(pnl_net, 2),
                    trade_id
                ))
                journal.conn.commit()
                stats['sells_matched'] += 1
                print(f"  [+] SELL {symbol:6} {qty:7.0f} @ {price:8.2f}  ({trade['date']}) PnL: {pnl_net:+.2f}$ (ID {trade_id})")
            else:
                stats['sells_orphan'] += 1
                print(f"  [!] SELL orphelin: {symbol:6} {qty:7.0f} @ {price:8.2f}  ({trade['date']}) - pas d'entrée ouverte")

    journal.close()
    return stats


def clear_paper_trades():
    """Efface tous les trades paper avant réimport."""
    journal = TradeJournal()
    journal.connect()
    cursor = journal.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM trades WHERE trade_mode = 'paper'")
    count = cursor.fetchone()[0]
    cursor.execute("DELETE FROM trades WHERE trade_mode = 'paper'")
    journal.conn.commit()
    journal.close()
    print(f"[CLEAN] {count} trades paper supprimés")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file', nargs='?', default='Transaction_ib.csv')
    parser.add_argument('--clean', action='store_true', help='Nettoyer les trades paper avant import')
    args = parser.parse_args()

    csv_path = os.path.join(os.path.dirname(__file__), args.csv_file)

    if not os.path.exists(csv_path):
        print(f"[ERREUR] Fichier non trouvé: {csv_path}")
        return

    if args.clean:
        clear_paper_trades()

    print(f"[IMPORT] Lecture de {csv_path}...")
    transactions = parse_csv(csv_path)
    print(f"[IMPORT] {len(transactions)} transactions brutes lues")

    print(f"\n[IMPORT] Agrégation des fills partiels...")
    aggregated = aggregate_fills(transactions)
    print(f"[IMPORT] {len(aggregated)} trades agrégés")

    print(f"\n{'='*70}")
    print("TRADES AGRÉGÉS:")
    print(f"{'='*70}")
    for t in aggregated:
        print(f"  {t['date']} | {t['side']:4} {t['symbol']:6} {t['qty']:7.0f} @ {t['avg_price']:8.2f} | comm: {t['commission']:.2f}")

    print(f"\n{'='*70}")
    print("IMPORT DANS LE JOURNAL:")
    print(f"{'='*70}")
    stats = import_to_journal(aggregated)

    print(f"\n{'='*70}")
    print(f"RÉSUMÉ: {stats['buys']} entrées, {stats['sells_matched']} sorties matchées, {stats['sells_orphan']} SELL orphelins")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
