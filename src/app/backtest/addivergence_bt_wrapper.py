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
        max_stocks=5
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

    # -------------------------------------------------
    def next(self):
        current_date = self.datas[0].datetime.date(0)
        print(f"Analyse pour la date {current_date}...")
        # 1️⃣ Fournir la liste des symboles à analyser
        symbols = [d._name for d in self.datas]
        self.strategy.set_symbols_to_analyse(symbols)

        # 2️⃣ Sélection + scoring
        print(f"2️⃣ Sélection + scoring")
        symbols_to_trade = self.strategy.get_symbols(current_date)

        # 3️⃣ Génération des ordres IBKR
        order_bundles = self.strategy.get_order_params()

        # 4️⃣ Traduction + exécution
        for bundle in order_bundles:
            symbol = bundle["symbol"]
            data = self._get_data(symbol)

            if not data or self.getposition(data):
                continue

            entry = OrderTranslator.entry(self, data, bundle["entry_order"])
            stop = OrderTranslator.stop(self, data, bundle["stop_order"])
            tp = OrderTranslator.take_profit(self, data, bundle["take_profit_order"])

            self.orders_by_symbol[symbol] = {
                "entry": entry,
                "stop": stop,
                "tp": tp
            }

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

