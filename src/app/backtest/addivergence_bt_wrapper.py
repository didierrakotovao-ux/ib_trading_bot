import backtrader as bt
from datetime import date, datetime

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.strategies.addivergence import AdDivergenceStrategy
from market_data_mock import MarketDataMock
from order_translator import OrderTranslator

class AdDivergenceBTWrapper(bt.Strategy):
    params = dict(
        capital=100000,
        max_stocks=5,
        stop_loss=0.05,   # Trailing stop à 5%
        take_profit=0.10,  # Take profit à 10%
        dataframes=None  # DataFrames pré-chargés depuis SQLite
    )

    def __init__(self, start_date, end_date, dataframes=None):
        self.start_date = start_date
        self.end_date = end_date

        # MarketData mock avec cache des DataFrames
        self.market_data = MarketDataMock(self.datas)

        # Injecter les DataFrames pré-chargés dans le cache
        if self.p.dataframes:
            self._preload_cache(self.p.dataframes)

        # Ta stratégie métier
        self.strategy = AdDivergenceStrategy(
            market_data=self.market_data,
            capital=self.p.capital,
            max_stocks=self.p.max_stocks
        )

        self.orders_by_symbol = {}
        self.pending_order_bundles = {}
        self.last_entry_prices = {}
        self.pending_stops = {}

        # Répertoire de sortie
        self._resultats_dir = os.path.join(os.path.dirname(__file__), 'resultats')
        os.makedirs(self._resultats_dir, exist_ok=True)

        # Journal de trading détaillé
        self.trade_entries = {}  # Stocke les infos d'entrée par symbole
        self.journal_filename = None
        self._init_trading_journal()

        # Préparation du fichier de diagnostic
        diag_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.diag_filename = os.path.join(self._resultats_dir, f"diagnostique_backtests_{diag_date}.txt")
        with open(self.diag_filename, "w") as f:
            f.write(f"--- Début du diagnostic Backtest ({diag_date}) ---\n")

    def _init_trading_journal(self):
        """Initialise le fichier journal de trading."""
        import csv
        journal_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.journal_filename = os.path.join(self._resultats_dir, f"trading_journal_{journal_date}.csv")
        with open(self.journal_filename, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'symbole', 'date_entree', 'quantite', 'prix_entree',
                'date_sortie', 'prix_sortie', 'cause_sortie',
                'pnl_brut', 'commission', 'pnl_net', 'scoring'
            ])
        print(f"[JOURNAL] Fichier créé: {self.journal_filename}")

    def _log_trade_journal(self, symbol, entry_date, qty, entry_price, exit_date, exit_price, cause, pnl_brut):
        """Écrit une ligne dans le journal de trading."""
        import csv
        scoring_name = self.strategy.scoring.name if hasattr(self.strategy, 'scoring') else 'N/A'

        # Calculer les commissions (0.1% à l'achat + 0.1% à la vente)
        commission_rate = 0.001  # 0.1%
        commission_entree = entry_price * qty * commission_rate
        commission_sortie = exit_price * qty * commission_rate
        total_commission = commission_entree + commission_sortie
        pnl_net = pnl_brut - total_commission

        with open(self.journal_filename, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                symbol, entry_date, qty, round(entry_price, 2),
                exit_date, round(exit_price, 2), cause,
                round(pnl_brut, 2), round(total_commission, 2), round(pnl_net, 2),
                scoring_name
            ])

    def _preload_cache(self, dataframes):
        """Pré-charge les DataFrames dans le cache du MarketDataMock."""
        import pandas as pd
        for symbol, df in dataframes.items():
            # Convertir le DataFrame indexé par date en format standard
            df_reset = df.reset_index()
            df_reset.columns = [col.lower() for col in df_reset.columns]
            if 'date' not in df_reset.columns and 'index' in df_reset.columns:
                df_reset = df_reset.rename(columns={'index': 'date'})
            df_reset['date'] = pd.to_datetime(df_reset['date'])
            self.market_data._data_cache[symbol] = df_reset
        self.market_data._preloaded = True
        print(f"[CACHE] {len(dataframes)} symboles pré-chargés dans le cache")

    def log_diag(self, message):
        with open(self.diag_filename, "a") as f:
            f.write(message + "\n")

    def log_trade_to_csv(self, dt, symbol, order_type, exec_type, price, size, pnl=None):
        import csv
        import os
        filename = os.path.join(self._resultats_dir, 'trades_log.csv')
        file_exists = os.path.isfile(filename)
        with open(filename, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(['datetime', 'symbol', 'order_type', 'exec_type', 'price', 'size', 'pnl'])
            writer.writerow([dt, symbol, order_type, exec_type, price, size, pnl])

    def notify_order(self, order):
        symbol = order.data._name if hasattr(order, 'data') and order.data else 'N/A'
        self.log_diag(f"[notify_order] status={order.getstatusname()}, symbol={symbol}, isbuy={order.isbuy()}, exectype={order.exectype}, price={getattr(order.executed, 'price', None)}, size={getattr(order.executed, 'size', None)}")
        bundle = self.pending_order_bundles.get(symbol)
        if order.status in [order.Completed]:
            if order.isbuy():
                order_type = 'ENTRY'
                self.last_entry_prices[symbol] = order.executed.price
                # Stocker les infos d'entrée pour le journal
                self.trade_entries[symbol] = {
                    'date_entree': order.data.datetime.datetime(0),
                    'prix_entree': order.executed.price,
                    'quantite': abs(order.executed.size)
                }
                if bundle:
                    bundle["entry_order"] = None
                    data = self._get_data(symbol)
                    # strategy.sell() soumet déjà l'ordre au broker — pas de broker.submit() supplémentaire
                    stop = OrderTranslator.stop(self, data, bundle["stop_order"], order)
                    tp = OrderTranslator.take_profit(self, data, bundle["take_profit_order"], order)
                    self.pending_stops[symbol] = {
                        "stop_order": stop,
                        "take_profit_order": tp
                    }
                    if stop is not None:
                        self.log_diag(f"[BT] Stop order submitted for {symbol} (post-entry)")
                    if tp is not None:
                        self.log_diag(f"[BT] Take profit order submitted for {symbol} (post-entry)")
            elif order.issell():
                self.orders_by_symbol[symbol] = None
                if symbol not in self.last_entry_prices:
                    self.log_diag(f"[WARNING] SL/TP orphelin pour {symbol} : aucune entrée trouvée. Log ignoré.")
                    if symbol in self.pending_stops:
                        self.pending_stops[symbol] = None
                    return
                last_entry_price = self.last_entry_prices[symbol]
                pnl = -1 * (order.executed.price - last_entry_price) * order.executed.size
                # Identifier TP vs SL via exectype (pas le PnL : un trailing stop peut déclencher
                # légèrement en profit si le trail a suivi la hausse, faussant la détection par signe)
                if order.exectype == bt.Order.Limit:
                    order_type = 'TP'
                    if symbol in self.pending_stops and self.pending_stops[symbol] is not None and self.pending_stops[symbol].get("stop_order") is not None:
                        self.broker.cancel(self.pending_stops[symbol]["stop_order"])
                else:
                    order_type = 'SL'
                    if symbol in self.pending_stops and self.pending_stops[symbol] is not None and self.pending_stops[symbol].get("take_profit_order") is not None:
                        self.broker.cancel(self.pending_stops[symbol]["take_profit_order"])

                # Écrire dans le journal de trading
                if symbol in self.trade_entries:
                    entry_info = self.trade_entries[symbol]
                    self._log_trade_journal(
                        symbol=symbol,
                        entry_date=entry_info['date_entree'],
                        qty=entry_info['quantite'],
                        entry_price=entry_info['prix_entree'],
                        exit_date=order.data.datetime.datetime(0),
                        exit_price=order.executed.price,
                        cause=order_type,
                        pnl_brut=pnl
                    )
                    del self.trade_entries[symbol]

                self.pending_stops[symbol] = None
                del self.last_entry_prices[symbol]
                self.log_diag(f"[CLEANUP] pending_stops et last_entry_price nettoyés pour {symbol} après {order_type}")
            self.pending_order_bundles[symbol] = None
            self.log_trade_to_csv(order.data.datetime.datetime(0), symbol, order_type, order.exectype, order.executed.price, order.executed.size, pnl if 'pnl' in locals() else '')
            self.log_diag(f"[notify_order] Order {order.getstatusname()} for {symbol}")

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        # Ignorer les barres hors période (données chargées plus tôt pour le warmup des indicateurs)
        if current_date < self.start_date.date() or current_date > self.end_date.date():
            return
        self.log_diag(f"Analyse pour la date {current_date}... Cash={self.broker.get_cash():.0f}")
        symbols = [d._name for d in self.datas]
        self.strategy.set_symbols_to_analyse(symbols)
        self.log_diag(f"2. Sélection + scoring")
        symbols_to_trade = self.strategy.get_symbols(current_date)
        self.log_diag(f"{current_date} Les symboles selectionnés {symbols_to_trade}")
        order_bundles = self.strategy.get_order_params()
        for bundle in order_bundles:
            symbol = bundle["symbol"]
            data = self._get_data(symbol)
            if not data:
                continue
            pos = self.getposition(data)
            self.log_diag(f"[DIAG] Avant entrée: {symbol} | Position size: {pos.size}")
            if pos.size > 0:
                self.log_diag(f"[CONTROL] Position déjà ouverte sur {symbol} (size={pos.size}), pas de nouvelle entrée.")
                continue
            if pos.size < 0:
                self.log_diag(f"[ERROR] Position short détectée sur {symbol} (size={pos.size}) : aucune entrée ni sortie autorisée. Intervention manuelle requise.")
                continue
            if pos.size == 0 and self.orders_by_symbol.get(symbol) is None:
                if symbol in self.pending_stops and self.pending_stops[symbol] is not None:
                    for order_key in ["stop_order", "take_profit_order"]:
                        pending_order = self.pending_stops[symbol].get(order_key)
                        if pending_order is not None and pending_order.status not in [pending_order.Completed, pending_order.Canceled]:
                            self.broker.cancel(pending_order)
                            self.log_diag(f"[CANCEL] Annulation {order_key} orphelin pour {symbol} avant nouvelle entrée")
                    self.pending_stops[symbol] = None
                # OrderTranslator.entry() appelle strategy.buy() qui soumet déjà au broker
                entry = OrderTranslator.entry(self, data, bundle["entry_order"])
                entry_price = entry.created.price if entry is not None else 'N/A'
                self.log_diag(f"{current_date} Placing order for {symbol}: Entry at {entry_price}")
                if entry is None:
                    self.log_diag(f"[WARNING] Entry order for {symbol} was not created!")
                self.orders_by_symbol[symbol] = {
                    "entry": entry
                }
                self.pending_order_bundles[symbol] = bundle

    def _get_data(self, symbol):
        for d in self.datas:
            if d._name == symbol:
                return d
        return None

    def notify_trade(self, trade):
        if trade.isclosed:
            symbol = trade.data._name
            self.orders_by_symbol.pop(symbol, None)

    def get_dates_in_range(self, start_date, end_date):
        import pandas as pd
        all_days = pd.date_range(start=start_date, end=end_date, freq='B')
        return [d.date() for d in all_days]