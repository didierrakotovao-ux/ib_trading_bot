"""
Journal de trading pour Paper/Live trading.
Peut être exécuté indépendamment pour synchroniser les trades depuis IB.
Usage: python live_journal.py [sync|show|summary]
"""
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.app.database.trade_journal import TradeJournal, TradeMode
from src.app.screener.providers.market_data_provider import MarketDataProvider


class LiveJournal:
    """
    Gère la journalisation des trades paper/live.
    Peut synchroniser avec IB et afficher les résultats.
    """

    def __init__(self, port=7497, client_id=None):
        """
        Initialise le journal.

        Args:
            port: Port IB (7497=paper TWS, 7496=live TWS, 4002=paper Gateway)
            client_id: ID client IB (None = aléatoire dans 90-99 pour éviter les conflits)
        """
        import random
        self.port = port
        self.client_id = client_id if client_id is not None else random.randint(150, 199)
        self.trade_mode = TradeMode.PAPER if port == 7497 else TradeMode.LIVE
        self.trade_journal = TradeJournal()
        self.market_data_provider = None

        print(f"[JOURNAL] Mode: {self.trade_mode.value}")

    def connect_ib(self) -> bool:
        """Connecte à IB pour récupérer les exécutions."""
        print(f"[JOURNAL] Connexion à IB (port {self.port}, client_id {self.client_id})...")
        self.market_data_provider = MarketDataProvider(port=self.port, client_id=self.client_id)
        connected = self.market_data_provider.connect()
        if connected:
            print("[JOURNAL] Connecté à IB")
        else:
            print("[JOURNAL] Échec de connexion à IB")
        return connected

    def disconnect_ib(self):
        """Déconnecte d'IB."""
        if self.market_data_provider and self.market_data_provider.is_connected():
            self.market_data_provider.disconnect()
            print("[JOURNAL] Déconnecté d'IB")

    def sync_from_ib(self, since_days: int = 7):
        """
        Synchronise les exécutions depuis IB vers le journal.
        Agrège les fills partiels automatiquement.

        Args:
            since_days: Nombre de jours en arrière pour récupérer les exécutions
        """
        if not self.market_data_provider or not self.market_data_provider.is_connected():
            if not self.connect_ib():
                return

        print(f"[SYNC] Récupération des exécutions depuis IB (derniers {since_days} jours)...")
        executions = self.market_data_provider.req_executions(since_days=since_days)

        if not executions:
            print("[SYNC] Aucune exécution trouvée dans cette session IB")
            return

        print(f"[SYNC] {len(executions)} exécutions brutes trouvées")

        # Agréger les fills du MÊME jour et du MÊME sens seulement
        # Les sells de jours différents restent séparés (sorties partielles sur plusieurs jours)
        aggregated = {}
        for ex in executions:
            symbol = ex['symbol']
            side = ex['side']
            try:
                date_part = ex['time'][:8]
            except Exception:
                date_part = "00000000"
            key = f"{symbol}_{side}_{date_part}"

            if key not in aggregated:
                aggregated[key] = {
                    'symbol': symbol,
                    'side': side,
                    'total_shares': 0,
                    'total_value': 0,
                    'time': ex['time'],
                    'executions': []
                }

            aggregated[key]['total_shares'] += ex['shares']
            aggregated[key]['total_value'] += ex['shares'] * ex['price']
            aggregated[key]['executions'].append(ex)

        # Convertir en trades agrégés
        by_symbol = {}
        for key, data in aggregated.items():
            symbol = data['symbol']
            avg_price = data['total_value'] / data['total_shares'] if data['total_shares'] > 0 else 0

            if symbol not in by_symbol:
                by_symbol[symbol] = {'buys': [], 'sells': []}

            agg_execution = {
                'symbol': symbol,
                'side': data['side'],
                'shares': data['total_shares'],
                'price': avg_price,
                'time': data['time']
            }

            if data['side'] == 'BOT':
                by_symbol[symbol]['buys'].append(agg_execution)
            else:
                by_symbol[symbol]['sells'].append(agg_execution)

        print(f"[SYNC] {len(aggregated)} trades agrégés")

        # Traiter chaque symbole
        self.trade_journal.connect()
        cursor = self.trade_journal.conn.cursor()

        for symbol, trades in by_symbol.items():
            print(f"[SYNC] {symbol}: {len(trades['buys'])} BUY, {len(trades['sells'])} SELL")

            # Traiter les BUY (entrées)
            for buy in trades['buys']:
                try:
                    fill_time = datetime.strptime(buy['time'], "%Y%m%d %H:%M:%S")
                except:
                    fill_time = datetime.now()

                date_str = fill_time.strftime('%Y-%m-%d')

                # Vérifier si déjà dans le journal
                cursor.execute('''
                    SELECT id FROM trades
                    WHERE symbol = %s AND trade_mode = %s AND DATE(date_entree) = %s
                    AND ABS(quantite - %s) < 10
                ''', (symbol, self.trade_mode.value, date_str, buy['shares']))

                existing = cursor.fetchone()

                if not existing:
                    from src.app.strategies.regime_selector import current_strategy_name
                    trade_id = self.trade_journal.log_trade(
                        trade_mode=self.trade_mode,
                        # Attribution au régime en vigueur à la date du fill
                        strategy_name=current_strategy_name(as_of=fill_time),
                        symbol=symbol,
                        date_entree=fill_time,
                        prix_entree=buy['price'],
                        quantite=buy['shares']
                    )
                    print(f"  [+] Entrée ajoutée: {symbol} {buy['shares']} @ {buy['price']:.2f} (ID: {trade_id})")
                else:
                    print(f"  [=] Entrée existante: {symbol} (ID: {existing[0]})")

            # Traiter les SELL (sorties partielles ou totales)
            for sell in trades['sells']:
                try:
                    fill_time = datetime.strptime(sell['time'], "%Y%m%d %H:%M:%S")
                except Exception:
                    fill_time = datetime.now()

                # Chercher une entrée ouverte (quantite_restante > 0)
                cursor.execute('''
                    SELECT id, prix_entree, quantite_restante FROM trades
                    WHERE symbol = %s AND trade_mode = %s AND (date_sortie IS NULL OR quantite_restante > 0)
                    ORDER BY date_entree DESC LIMIT 1
                ''', (symbol, self.trade_mode.value))

                entry = cursor.fetchone()

                if entry:
                    trade_id, entry_price, qty_restante = entry
                    qty_vendue = min(int(sell['shares']), qty_restante)
                    self.trade_journal.log_partial_exit(
                        trade_id=trade_id,
                        symbol=symbol,
                        date_sortie=fill_time,
                        prix_sortie=sell['price'],
                        quantite_vendue=qty_vendue,
                        prix_entree=entry_price,
                        cause_sortie='TRAILING_STOP'
                    )
                else:
                    print(f"  [!] SELL orphelin: {symbol} @ {sell['price']:.2f} (pas d'entrée ouverte)")

        print("[SYNC] Synchronisation terminée")

    def show_trades(self, days: int = 7):
        """Affiche les trades des X derniers jours."""
        self.trade_journal.connect()
        cursor = self.trade_journal.conn.cursor()

        date_limit = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        cursor.execute('''
            SELECT id, symbol, date_entree, prix_entree, quantite,
                   date_sortie, prix_sortie, cause_sortie, pnl_net
            FROM trades
            WHERE trade_mode = %s AND date_entree >= %s
            ORDER BY date_entree DESC
        ''', (self.trade_mode.value, date_limit))

        rows = cursor.fetchall()

        print()
        print(f"{'='*70}")
        print(f"TRADES {self.trade_mode.value.upper()} - {days} derniers jours")
        print(f"{'='*70}")

        if not rows:
            print("Aucun trade trouvé")
            return

        total_pnl = 0
        open_trades = 0
        closed_trades = 0

        for row in rows:
            id, symbol, date_e, prix_e, qty, date_s, prix_s, cause, pnl = row

            status = "OUVERT" if not date_s else "FERMÉ"
            pnl_str = f"PnL: {pnl:+.2f}$" if pnl else ""

            # date_e peut être str (SQLite) ou datetime (PostgreSQL/psycopg2)
            date_str = date_e.strftime('%Y-%m-%d %H:%M') if hasattr(date_e, 'strftime') else str(date_e)[:16]
            print(f"{symbol:6} | {date_str} | {qty:5} @ {prix_e:8.2f} | {status:6} {pnl_str}")

            if date_s:
                closed_trades += 1
                total_pnl += pnl if pnl else 0
            else:
                open_trades += 1

        print(f"{'='*70}")
        print(f"Trades fermés: {closed_trades} | Trades ouverts: {open_trades} | PnL Total: {total_pnl:+.2f}$")

    def show_summary(self):
        """Affiche le résumé de performance."""
        summary = self.trade_journal.get_performance_summary(
            trade_mode=self.trade_mode
        )

        print()
        print("=" * 60)
        print(f"RÉSUMÉ PERFORMANCE - {self.trade_mode.value.upper()}")
        print("=" * 60)

        if "error" in summary:
            print(summary["error"])
        else:
            print(f"Trades: {summary['total_trades']} (W:{summary['winning_trades']} / L:{summary['losing_trades']})")
            print(f"Win Rate: {summary['win_rate']:.1f}%")
            print(f"PnL Net Total: {summary['total_pnl_net']:,.2f}$")
            print(f"PnL Net Moyen: {summary['avg_pnl_net']:,.2f}$")
            print(f"Max Win: {summary['max_win']:,.2f}$ | Max Loss: {summary['max_loss']:,.2f}$")
            print(f"Commissions: {summary['total_commission']:,.2f}$")

        print("=" * 60)

    def add_manual_entry(self, symbol: str, qty: int, price: float, date: datetime = None):
        """Ajoute manuellement une entrée de trade."""
        trade_id = self.trade_journal.log_trade(
            trade_mode=self.trade_mode,
            strategy_name="Manual",
            symbol=symbol.upper(),
            date_entree=date or datetime.now(),
            prix_entree=price,
            quantite=qty
        )
        print(f"[JOURNAL] Entrée manuelle ajoutée: {symbol} {qty} @ {price} (ID: {trade_id})")
        return trade_id

    def add_manual_exit(self, symbol: str, price: float, cause: str = "MANUAL", date: datetime = None):
        """Ajoute manuellement une sortie de trade."""
        self.trade_journal.connect()
        cursor = self.trade_journal.conn.cursor()

        # Trouver l'entrée ouverte
        cursor.execute('''
            SELECT id, prix_entree, quantite FROM trades
            WHERE symbol = %s AND trade_mode = %s AND date_sortie IS NULL
            ORDER BY date_entree DESC LIMIT 1
        ''', (symbol.upper(), self.trade_mode.value))

        entry = cursor.fetchone()

        if not entry:
            print(f"[JOURNAL] Aucune entrée ouverte trouvée pour {symbol}")
            return None

        trade_id, entry_price, qty = entry
        exit_time = date or datetime.now()
        pnl_brut = (price - entry_price) * qty
        commission = (entry_price * qty + price * qty) * 0.001
        pnl_net = pnl_brut - commission

        cursor.execute('''
            UPDATE trades SET
                date_sortie = %s,
                prix_sortie = %s,
                cause_sortie = %s,
                pnl_brut = %s,
                commission = %s,
                pnl_net = %s
            WHERE id = %s
        ''', (
            exit_time.strftime('%Y-%m-%d %H:%M:%S'),
            price,
            cause,
            round(pnl_brut, 2),
            round(commission, 2),
            round(pnl_net, 2),
            trade_id
        ))
        self.trade_journal.conn.commit()
        print(f"[JOURNAL] Sortie ajoutée: {symbol} @ {price}, PnL: {pnl_net:.2f}$ (ID: {trade_id})")
        return trade_id

    def reset_paper_trades(self, since_days: int = 7):
        """
        Efface les trades paper des derniers N jours et relance un sync.
        Ne touche PAS aux trades plus anciens que la fenêtre since_days.

        Args:
            since_days: Nombre de jours en arrière pour le reset + re-sync
        """
        self.trade_journal.connect()
        cursor = self.trade_journal.conn.cursor()

        date_limit = (datetime.now() - timedelta(days=since_days)).strftime('%Y-%m-%d')

        # Compter les trades paper dans la fenêtre
        cursor.execute(
            "SELECT COUNT(*) FROM trades WHERE trade_mode = %s AND date_entree >= %s",
            (self.trade_mode.value, date_limit)
        )
        count = cursor.fetchone()[0]

        # Supprimer uniquement les trades dans la fenêtre since_days
        cursor.execute(
            "DELETE FROM trades WHERE trade_mode = %s AND date_entree >= %s",
            (self.trade_mode.value, date_limit)
        )
        self.trade_journal.conn.commit()

        print(f"[RESET] {count} trades {self.trade_mode.value} supprimés (depuis {date_limit})")
        print(f"[RESET] Re-synchronisation depuis IB (derniers {since_days} jours)...")

        # Re-sync complet
        self.sync_from_ib(since_days=since_days)

    def close(self):
        """Ferme les connexions."""
        self.disconnect_ib()
        self.trade_journal.close()


def main():
    """Point d'entrée principal."""
    import argparse

    parser = argparse.ArgumentParser(description="Journal de trading Paper/Live")
    parser.add_argument('action', nargs='?', default='show',
                        choices=['sync', 'show', 'summary', 'all', 'reset'],
                        help='Action à effectuer (default: show)')
    parser.add_argument('--port', type=int, default=7497,
                        help='Port IB (7497=paper, 7496=live)')
    parser.add_argument('--days', type=int, default=7,
                        help='Nombre de jours à afficher')
    parser.add_argument('--since', type=int, default=7,
                        help='Jours en arrière pour récupérer les exécutions IB (default: 7)')

    args = parser.parse_args()

    journal = LiveJournal(port=args.port)

    try:
        if args.action == 'sync':
            journal.sync_from_ib(since_days=args.since)
            journal.show_summary()
        elif args.action == 'show':
            journal.show_trades(days=args.days)
        elif args.action == 'summary':
            journal.show_summary()
        elif args.action == 'all':
            journal.sync_from_ib(since_days=args.since)
            journal.show_trades(days=args.days)
            journal.show_summary()
        elif args.action == 'reset':
            journal.reset_paper_trades(since_days=args.since)
            journal.show_trades(days=args.since)
            journal.show_summary()

    except KeyboardInterrupt:
        print("\nInterrompu par l'utilisateur")

    finally:
        journal.close()


if __name__ == "__main__":
    main()
