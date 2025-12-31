import backtrader as bt
from datetime import date

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
        take_profit=0.10  # Take profit à 10%
    )

    def __init__(self,start_date,end_date):
        self.start_date = start_date
        self.end_date = end_date
        # MarketData mock basé sur Backtrader
        self.market_data = MarketDataMock(self.datas)

        # Ta stratégie métier
        self.strategy = AdDivergenceStrategy(
            market_data=self.market_data,
            capital=self.p.capital,
            max_stocks=self.p.max_stocks
        )

        self.orders_by_symbol = {}

    def log_trade_to_csv(self, dt, symbol, order_type, price, size):
        import csv
        import os
        filename = 'trades_log.csv'
        file_exists = os.path.isfile(filename)
        with open(filename, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(['datetime', 'symbol', 'order_type', 'price', 'size'])
            writer.writerow([dt, symbol, order_type, price, size])

    def notify_order(self, order):
        #  dt = self.data.datetime.datetime(0)
        symbol = order.data._name if hasattr(order, 'data') and order.data else 'N/A'
        print(f"[notify_order] status={order.getstatusname()}, symbol={symbol}, isbuy={order.isbuy()}, exectype={order.exectype}, price={getattr(order.executed, 'price', None)}, size={getattr(order.executed, 'size', None)}")
        if order.status in [order.Completed]:
            order_type = 'UNKNOWN'
            if order.isbuy():
                order_type = 'ENTRY'
                bundle = getattr(self, 'pending_order_bundles', {}).get(symbol)
                if bundle:
                    data = self._get_data(symbol)
                    stop = OrderTranslator.stop(self, data, bundle["stop_order"], order)
                    tp = OrderTranslator.take_profit(self, data, bundle["take_profit_order"], order)
                    if stop is not None:
                        self.broker.submit(stop)
                        print(f"[BT] Stop order submitted for {symbol} (post-entry)")
                    if tp is not None:
                        self.broker.submit(tp)
                        print(f"[BT] Take profit order submitted for {symbol} (post-entry)")
            elif order.issell():
                # On distingue TP/SL par le type d'ordre
                self.orders_by_symbol[symbol] = None
                if order.exectype == bt.Order.Limit:
                    order_type = 'TP'
                elif order.exectype in [bt.Order.Stop, bt.Order.StopTrail]:
                    order_type = 'SL'
                else:
                    order_type = 'EXIT'
            self.log_trade_to_csv(order.data.datetime.datetime(0), symbol, order_type, order.executed.price, order.executed.size)
        # Optionnel : log sur la console
        print(f"Order executed: {order.info}")
    # -------------------------------------------------
    def next(self):
        current_date = self.datas[0].datetime.date(0)
        print(f"Analyse pour la date {current_date}...")
        # Diagnostic global
        print(f"[DIAG] Cash disponible: {self.broker.get_cash()} | Value: {self.broker.get_value()}")
        for d in self.datas:
            pos = self.getposition(d)
            print(f"[DIAG] {d._name} | Position size: {pos.size} | Price: {d.close[0]}")
        # 1️⃣ Fournir la liste des symboles à analyser
        symbols = [d._name for d in self.datas]
        self.strategy.set_symbols_to_analyse(symbols)

        # 2️⃣ Sélection + scoring
        print(f"2️⃣ Sélection + scoring")
        symbols_to_trade = self.strategy.get_symbols(current_date)

        print(f"{current_date} Les symboles selectionnés {symbols_to_trade}")

        # 3️⃣ Génération des ordres IBKR
        order_bundles = self.strategy.get_order_params()

        # 4️⃣ Traduction + exécution
        # On prépare les bundles d'ordres pour usage différé dans notify_order
        if not hasattr(self, 'pending_order_bundles'):
            self.pending_order_bundles = {}
        for bundle in order_bundles:
            symbol = bundle["symbol"]
            data = self._get_data(symbol)

            if not data:
                continue

            pos = self.getposition(data)
            
            print(f"[DIAG] Avant entrée: {symbol} | Position size: {pos.size}")
            """ Contrôle des positions existantes """
            if pos.size > 0:
                print(f"[CONTROL] Position déjà ouverte sur {symbol} (size={pos.size}), pas de nouvelle entrée.")
                continue
            if pos.size < 0:
                print(f"[ERROR] Position short détectée sur {symbol} (size={pos.size}) : aucune entrée ni sortie autorisée. Intervention manuelle requise.")
                continue
            # Entrée uniquement si flat
            if pos.size == 0 and self.orders_by_symbol.get(symbol) is None:
                entry = OrderTranslator.entry(self, data, bundle["entry_order"])
                if entry is not None:
                    self.broker.submit(entry)
                    print(f"[BT] Entry order submitted for {symbol}")
                entry_price = entry.created.price if entry is not None else 'N/A'
                print(f"{current_date} Placing order for {symbol}: Entry at {entry_price}")
                if entry is None:
                    print(f"[WARNING] Entry order for {symbol} was not created!")
                self.orders_by_symbol[symbol] = {
                    "entry": entry
                }
                # On stocke le bundle pour usage différé
                self.pending_order_bundles[symbol] = bundle

    # -------------------------------------------------
    def _get_data(self, symbol):
        for d in self.datas:
            if d._name == symbol:
                return d
        return None

    # -------------------------------------------------
    def notify_trade(self, trade):
        if trade.isclosed:
            symbol = trade.data._name
            self.orders_by_symbol.pop(symbol, None)

    def get_dates_in_range(self, start_date, end_date):
        """Retourne la liste des jours ouvrables (lundi à vendredi) entre start_date et end_date inclus."""
        import pandas as pd
        all_days = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' = business day
        return [d.date() for d in all_days]

