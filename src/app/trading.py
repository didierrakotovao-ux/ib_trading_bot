from datetime import datetime, timedelta
from strategies.momentum import MomentumStrategy
from screener.providers.market_data_provider import MarketDataProvider
from strategies.addivergence import AdDivergenceStrategy
from position_manager import PositionManager
from ibapi.tag_value import TagValue

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.app.database.trade_journal import TradeJournal, TradeMode


class Trading:
    """
        Classe centrale de l'application, orchestre les éléments de trading via un provider de données de marché.
        1-recevoir les critères de recherche de la stratégie
        2-execute les screeners via le market data provider
        3-transmets les données de marché à la stratégie pour décision
        4-place les ordres pour les symboles sélectionnés
        5-gère les positions ouvertes et le suivi des ordres
        6-écrit le journal de performance
    """
    def __init__(self, port=7497, client_id=1, total_capital=100000, max_positions=5,
                 use_trailing_stop=False, trailing_percent=5.0):
        self.port = port
        self.total_capital = total_capital
        self.max_positions = max_positions
        self.capital_per_position = total_capital / max_positions  # Max 1/5 du capital par position
        self.use_trailing_stop = use_trailing_stop
        self.trailing_percent = trailing_percent
        self.market_data_provider = MarketDataProvider(port=port, client_id=client_id)
        self.strategies = [MomentumStrategy(
            self.market_data_provider,
            capital=total_capital,
            max_stocks=max_positions,
            use_trailing_stop=use_trailing_stop,
            trailing_percent=trailing_percent,
            scoring_type="smooth_ml"
        )]
        self.orders = []
        self.position_manager = PositionManager()
        self.order_callbacks = []  # Liste de callbacks à appeler sur exécution d'ordre
        self.pending_orders = set()  # Symboles avec ordres en attente (pas encore exécutés)

        # Journal de trading en BD
        self.trade_journal = TradeJournal()
        # Détermine le mode: port 7497 = paper, sinon live
        self.trade_mode = TradeMode.PAPER if port == 7497 else TradeMode.LIVE
        self.trade_ids = {}  # Mapping symbol -> trade_id pour mettre à jour à la sortie

        print(f"[TRADING] Capital total: {total_capital:,.0f}$, Max positions: {max_positions}, Capital/position: {self.capital_per_position:,.0f}$")
        print(f"[TRADING] Trailing stop: {'Activé (' + str(trailing_percent) + '%)' if use_trailing_stop else 'Désactivé (gestion manuelle)'}")
        print(f"[JOURNAL] Mode de trading: {self.trade_mode.value}")

    def setup_order_callbacks(self):
        """
        Enregistre le callback pour recevoir les notifications d'exécution d'ordres.
        À appeler après la connexion au MarketDataProvider.
        """
        self.market_data_provider.register_order_callback(self.on_order_executed)
        print("[TRADING] Callback de journalisation enregistré")

    def register_order_callback(self, callback):
        """
        Permet d'enregistrer une fonction callback à appeler lors de l'exécution d'un ordre.
        La fonction doit accepter (symbol, order, status, fill_price, fill_qty, fill_time)
        """
        self.order_callbacks.append(callback)

    def on_order_executed(self, symbol, action, status, fill_price, fill_qty, fill_time, strategy_name="Momentum"):
        """
        Callback appelé par MarketDataProvider lors de l'exécution d'un ordre.
        Gère les BUY (entrées) et SELL (sorties) pour la journalisation.
        """
        print(f"[TRADING] Order executed: {symbol} {action} {fill_qty} @ {fill_price}")

        # Retirer des ordres en attente
        if symbol in self.pending_orders:
            self.pending_orders.discard(symbol)
            print(f"[Trading] {symbol} retiré des ordres en attente")

        if action == "BUY" and status.lower() == "filled":
            # Ajout de la position dans le PositionManager
            self.position_manager.open_position(
                symbol=symbol,
                qty=fill_qty,
                entry_price=fill_price,
                entry_time=fill_time
            )
            # Journaliser l'entrée en BD
            trade_id = self.trade_journal.log_trade(
                trade_mode=self.trade_mode,
                strategy_name=strategy_name,
                symbol=symbol,
                date_entree=fill_time or datetime.now(),
                prix_entree=fill_price,
                quantite=fill_qty
            )
            self.trade_ids[symbol] = trade_id
            print(f"[JOURNAL] Entrée enregistrée: {symbol} @ {fill_price} (ID: {trade_id})")

        elif action == "SELL" and status.lower() == "filled":
            # Fermer la position et mettre à jour le journal
            if self.position_manager.has_open_position(symbol):
                self.close_position(symbol, fill_price, fill_time, cause_sortie="TRAILING_STOP")
            else:
                print(f"[WARNING] SELL reçu pour {symbol} mais aucune position ouverte")

        # Notifier les autres callbacks enregistrés
        for cb in self.order_callbacks:
            cb(symbol, action, status, fill_price, fill_qty, fill_time)

    def get_capital_used(self):
        """
        Calcule le capital actuellement utilisé par les positions ouvertes.
        """
        capital_used = 0
        for pos in self.position_manager.get_open_positions():
            capital_used += pos.entry_price * pos.qty
        return capital_used

    def get_available_capital(self):
        """
        Retourne le capital disponible pour de nouvelles positions.
        """
        return self.total_capital - self.get_capital_used()

    def can_open_position(self, price, qty):
        """
        Vérifie si on peut ouvrir une nouvelle position.
        - Ne dépasse pas le max de positions
        - Le capital est suffisant
        """
        open_positions = len(self.position_manager.get_open_positions())
        if open_positions >= self.max_positions:
            return False, f"Max positions atteint ({self.max_positions})"

        position_cost = price * qty
        available = self.get_available_capital()
        if position_cost > available:
            return False, f"Capital insuffisant ({position_cost:,.0f}$ requis, {available:,.0f}$ disponible)"

        return True, "OK"

    def place_order(self, contract, order, parent_order_id=None):
        """
        Reçoit un contrat et un ordre généré par la stratégie et les transmet au provider via placeOrder.
        Vérifie le capital disponible avant de placer un ordre d'achat.
        Retourne l'orderId assigné par le provider.
        """
        symbol = getattr(contract, 'symbol', None)

        if order.action == "BUY":
            # Vérifier si position déjà ouverte
            if symbol and self.position_manager.has_open_position(symbol):
                print(f"[Trading] Position déjà ouverte sur {symbol}, ordre ignoré.")
                return None

            # Vérifier si un ordre est déjà en attente pour ce symbole
            if symbol and symbol in self.pending_orders:
                print(f"[Trading] Ordre déjà en attente pour {symbol}, ordre ignoré.")
                return None

            # Estimer le coût de la position (utiliser le dernier prix connu ou une estimation)
            lmt_price = getattr(order, 'lmtPrice', None)

            # Vérifier que lmtPrice est un prix valide (pas UNSET_DOUBLE de ibapi = 1.7976931348623157e+308)
            # Un prix d'action réaliste est < 100,000$
            if lmt_price and 0 < lmt_price < 100000:
                estimated_price = lmt_price
            else:
                # Pour les ordres market, estimer à partir du capital par position
                estimated_price = self.capital_per_position / order.totalQuantity

            can_open, reason = self.can_open_position(estimated_price, order.totalQuantity)
            if not can_open:
                print(f"[Trading] Ordre refusé pour {symbol}: {reason}")
                return None

        print(f"[IB ORDER] Placing order for {symbol} {order.action} {order.totalQuantity} @ {order.orderType}")
        order_id = self.market_data_provider.placeOrder(contract, order, parent_order_id=parent_order_id)
        self.orders.append((contract, order))

        # Marquer le symbole comme ayant un ordre en attente (pour les BUY)
        if order.action == "BUY" and symbol:
            self.pending_orders.add(symbol)
            print(f"[Trading] {symbol} ajouté aux ordres en attente")

        print(f"[IB ORDER] orderId={order_id}")
        return order_id

    def update_orders(self):
        """
        Met à jour le statut des ordres (remplis, annulés, etc.)
        """
        pass

    def get_positions(self):
        """
        Retourne les positions courantes (ouvertes)
        """
        return self.position_manager.get_open_positions()

    def close_position(self, symbol, exit_price, exit_time=None, cause_sortie="MANUAL"):
        """
        Ferme une position existante, met à jour le P&L et journalise la sortie.
        """
        position = self.position_manager.close_position(symbol, exit_price, exit_time)

        if position and symbol in self.trade_ids:
            # Calculer les commissions et PnL
            commission_rate = 0.001  # 0.1%
            commission = (position.entry_price * position.qty + exit_price * position.qty) * commission_rate
            pnl_brut = position.pnl
            pnl_net = pnl_brut - commission

            # Mettre à jour le trade en BD avec les infos de sortie
            self.trade_journal.connect()
            cursor = self.trade_journal.conn.cursor()
            cursor.execute("""
                UPDATE trades SET
                    date_sortie = ?,
                    prix_sortie = ?,
                    cause_sortie = ?,
                    pnl_brut = ?,
                    commission = ?,
                    pnl_net = ?
                WHERE id = ?
            """, (
                (exit_time or datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
                exit_price,
                cause_sortie,
                round(pnl_brut, 2),
                round(commission, 2),
                round(pnl_net, 2),
                self.trade_ids[symbol]
            ))
            self.trade_journal.conn.commit()
            print(f"[JOURNAL] Sortie enregistrée: {symbol} @ {exit_price}, PnL: {pnl_net:.2f}$ ({cause_sortie})")
            del self.trade_ids[symbol]

        return position

    def init_trade(self):
        """
        1- Obtient le critere de scanner de la stratégie
        2- Exécute le scanner via le market data provider
        3- Retourne la liste des symboles trouvés et passe à la stratégie pour décision de trading.
        4- Ouvre les positions dans le PositionManager uniquement lors de l'exécution effective de l'ordre d'entrée (voir on_order_executed).
        """
        try:
            trade_date = datetime.now()

            symbolList: list = []
            symbolToTrade: list = []
            # Connexion gérée par le main, pas de double connect ici
            for strategy in self.strategies:
                print(f"  Strategie: {strategy.name}...")
                scan_sub = strategy.scanner_filters()
                symbols = self.market_data_provider.get_scanner_results(scan_sub, max_results=200)
                strategy.set_symbols_to_analyse(symbols)
                
                for symbol in strategy.get_symbols(trade_date):
                    symbolToTrade.append(symbol)
                for orders in strategy.get_order_params():
                    symbol = orders['symbol']
                    contrat = self.market_data_provider.create_contract(symbol)
                    entry_order = orders['entry_order']

                    # Utiliser l'algorithme VWAP d'IB pour une meilleure exécution
                    # VWAP exécute l'ordre sur une période en suivant le volume du marché
                    entry_order.orderType = "LMT"  # VWAP nécessite un prix limite
                    entry_order.algoStrategy = "Vwap"
                    entry_order.algoParams = []
                    # Paramètres VWAP :
                    # - maxPctVol : % max du volume du marché (0.1 = 10%)
                    # - startTime/endTime : période d'exécution (vide = jusqu'à la clôture)
                    # - allowPastEndTime : continuer après endTime si non rempli
                    # - noTakeLiq : ne pas prendre de liquidité (plus passif)
                    entry_order.algoParams.append(TagValue("maxPctVol", "0.2"))  # Max 20% du volume
                    entry_order.algoParams.append(TagValue("noTakeLiq", "0"))    # Peut prendre liquidité
                    entry_order.algoParams.append(TagValue("allowPastEndTime", "1"))  # Continuer si besoin

                    print(f"[ORDER ALGO] {symbol}: VWAP avec limite {entry_order.lmtPrice:.2f}$ (maxPctVol=20%)")

                    # Placer l'ordre d'entrée
                    entry_order_id = self.place_order(contrat, entry_order)
                    # Vérifier que l'entrée a été acceptée (dans self.orders)
                    if (contrat, entry_order) in self.orders:
                        # Placer le stop lié au parent si configuré
                        if 'stop_order' in orders:
                            self.place_order(contrat, orders['stop_order'],
                                             parent_order_id=entry_order_id)

            for symbol in symbolToTrade:
                print(f"  Stocks a trader: {symbol}")

            return symbolList

        except Exception as e:
            # Disconnect géré par le main dans finally
            print(f"Erreur lors du trading: {e}")
            import traceback
            traceback.print_exc()

    def sync_positions_with_ib(self):
        """
        Synchronise le PositionManager avec les positions réelles récupérées via l'API IB.
        """
        # Vérification stricte de la connexion
        if not self.market_data_provider.is_connected():
            print("[DEBUG] MarketDataProvider non connecté, tentative de reconnexion...")
            # Recréation d'une nouvelle instance pour une reconnexion propre
            from screener.providers.market_data_provider import MarketDataProvider
            self.market_data_provider = MarketDataProvider(port=self.market_data_provider.port, client_id=self.market_data_provider.client_id)
            if not self.market_data_provider.connect():
                print("[DEBUG] Impossible de se connecter à IB, synchronisation annulée.")
                return
            # Attendre que l'API soit prête après reconnexion
            if not self.market_data_provider.wait_until_ready(timeout=15):
                print("[DEBUG] IB API non prête après reconnexion, synchronisation annulée.")
                return
            # Mettre à jour la référence dans les stratégies
            for strategy in self.strategies:
                strategy.market_data = self.market_data_provider
            # Ré-enregistrer le callback de journalisation
            self.setup_order_callbacks()
        ib_positions = self.market_data_provider.get_ib_positions()
        print(f"Positions IB récupérées: {ib_positions}")
        for pos in ib_positions:
            if not self.position_manager.has_open_position(pos['symbol']):
                self.position_manager.open_position(
                    symbol=pos['symbol'],
                    qty=pos['qty'],
                    entry_price=pos['avg_cost'],
                    entry_time=None,
                    position_type="long" if pos['qty'] > 0 else "short"
                )
        print(f"Positions synchronisées depuis IB: {[p.symbol for p in self.position_manager.get_open_positions()]}" )

    def sync_executions_from_ib(self):
        """
        Synchronise les exécutions de la session IB avec le journal de trading.
        Récupère les trades exécutés et les ajoute au journal s'ils n'y sont pas déjà.
        Agrège les fills partiels d'un même ordre.
        """
        print("[SYNC] Récupération des exécutions depuis IB...")
        executions = self.market_data_provider.req_executions()

        if not executions:
            print("[SYNC] Aucune exécution trouvée")
            return

        print(f"[SYNC] {len(executions)} exécutions brutes trouvées")

        # Agréger les fills partiels par symbole et side
        # Utiliser une fenêtre de temps de 60 secondes pour regrouper
        aggregated = {}
        for ex in executions:
            symbol = ex['symbol']
            side = ex['side']
            key = f"{symbol}_{side}"

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

        # Calculer le prix moyen et convertir en liste
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
        for symbol, trades in by_symbol.items():
            print(f"[SYNC] {symbol}: {len(trades['buys'])} BUY, {len(trades['sells'])} SELL")

            # Vérifier si déjà dans le journal
            self.trade_journal.connect()
            cursor = self.trade_journal.conn.cursor()

            for buy in trades['buys']:
                # Vérifier si cette entrée existe déjà (par symbole et date approximative)
                try:
                    fill_time = datetime.strptime(buy['time'], "%Y%m%d %H:%M:%S")
                except:
                    fill_time = datetime.now()

                date_str = fill_time.strftime('%Y-%m-%d')

                cursor.execute('''
                    SELECT id FROM trades
                    WHERE symbol = ? AND trade_mode = ? AND date(date_entree) = ?
                ''', (symbol, self.trade_mode.value, date_str))

                existing = cursor.fetchone()

                if not existing:
                    # Ajouter l'entrée au journal
                    trade_id = self.trade_journal.log_trade(
                        trade_mode=self.trade_mode,
                        strategy_name="Momentum",
                        symbol=symbol,
                        date_entree=fill_time,
                        prix_entree=buy['price'],
                        quantite=buy['shares']
                    )
                    self.trade_ids[symbol] = trade_id
                    print(f"[SYNC] Entrée ajoutée: {symbol} @ {buy['price']} (ID: {trade_id})")

                    # Ajouter à position_manager si pas déjà présent
                    if not self.position_manager.has_open_position(symbol):
                        self.position_manager.open_position(
                            symbol=symbol,
                            qty=buy['shares'],
                            entry_price=buy['price'],
                            entry_time=fill_time
                        )
                else:
                    self.trade_ids[symbol] = existing[0]
                    print(f"[SYNC] Entrée déjà existante: {symbol} (ID: {existing[0]})")

            # Traiter les SELL (sorties)
            for sell in trades['sells']:
                try:
                    fill_time = datetime.strptime(sell['time'], "%Y%m%d %H:%M:%S")
                except:
                    fill_time = datetime.now()

                # Chercher une entrée ouverte (sans date_sortie) pour ce symbole
                cursor.execute('''
                    SELECT id, prix_entree, quantite FROM trades
                    WHERE symbol = ? AND trade_mode = ? AND date_sortie IS NULL
                    ORDER BY date_entree DESC LIMIT 1
                ''', (symbol, self.trade_mode.value))

                entry = cursor.fetchone()

                if entry:
                    trade_id, entry_price, qty = entry
                    pnl_brut = (sell['price'] - entry_price) * qty
                    commission = (entry_price * qty + sell['price'] * qty) * 0.001
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
                        fill_time.strftime('%Y-%m-%d %H:%M:%S'),
                        sell['price'],
                        'TRAILING_STOP',
                        round(pnl_brut, 2),
                        round(commission, 2),
                        round(pnl_net, 2),
                        trade_id
                    ))
                    self.trade_journal.conn.commit()
                    print(f"[SYNC] Sortie ajoutée: {symbol} @ {sell['price']}, PnL: {pnl_net:.2f}$")

                    # Fermer la position dans position_manager
                    if self.position_manager.has_open_position(symbol):
                        self.position_manager.close_position(symbol, sell['price'], fill_time)

                    # Retirer du mapping
                    if symbol in self.trade_ids:
                        del self.trade_ids[symbol]
                else:
                    print(f"[SYNC] SELL orphelin ignoré: {symbol} @ {sell['price']} (pas d'entrée correspondante)")

        print("[SYNC] Synchronisation terminée")

    def show_performance_summary(self, strategy_name: str = None):
        """
        Affiche le résumé de performance pour le mode actuel.
        """
        summary = self.trade_journal.get_performance_summary(
            trade_mode=self.trade_mode,
            strategy_name=strategy_name
        )

        print("\n" + "=" * 60)
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

        print("=" * 60 + "\n")

    def close(self):
        """
        Déconnecte proprement le MarketDataProvider (IB), ferme le journal et arrête le thread associé.
        """
        # Afficher le résumé avant de fermer
        self.show_performance_summary()

        # Fermer le journal
        self.trade_journal.close()

        if self.market_data_provider.is_connected():
            self.market_data_provider.disconnect()
        print("Déconnexion propre IB terminée.")



if __name__ == "__main__":
    # Configuration
    # Port 7497 = Paper Trading (TWS), Port 7496 = Live Trading (TWS)
    # Port 4002 = Paper Trading (Gateway), Port 4001 = Live Trading (Gateway)
    PAPER_PORT = 7497
    LIVE_PORT = 7496
    CLIENT_ID = 11

    # Capital et gestion des positions
    TOTAL_CAPITAL = 100000  # Capital total disponible
    MAX_POSITIONS = 5       # Max 5 positions (chaque position = 1/5 du capital)

    # Utiliser paper trading par défaut
    trading = Trading(
        port=PAPER_PORT,
        client_id=CLIENT_ID,
        total_capital=TOTAL_CAPITAL,
        max_positions=MAX_POSITIONS
    )

    try:
        print("[TRADING] Connexion à IB...")
        connected = trading.market_data_provider.connect()
        print(f"[TRADING] Connecté: {connected}")
        if not connected:
            print("[TRADING] Échec de connexion, arrêt du script.")
            exit(1)

        # Attendre que l'API IB soit prête (nextValidId reçu)
        print("[TRADING] Attente de l'initialisation IB...")
        if not trading.market_data_provider.wait_until_ready(timeout=15):
            print("[TRADING] Timeout en attendant IB, arrêt du script.")
            exit(1)
        print("[TRADING] IB prêt")

        # Enregistrer le callback pour la journalisation des ordres
        trading.setup_order_callbacks()
        # Synchroniser les exécutions depuis IB (trades déjà exécutés)
        trading.sync_executions_from_ib()
        trading.sync_positions_with_ib()
        trading.init_trade()

    except KeyboardInterrupt:
        print("\nArrêt demandé par l'utilisateur (Ctrl+C)")

    finally:
        print("[TRADING] Fermeture...")
        trading.close()
