"""
Wrapper Backtrader avec uniquement Trailing Stop (pas de Take Profit).
La position reste ouverte tant que le trailing stop n'est pas touché.
"""
import backtrader as bt
from datetime import date, datetime

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.strategies.addivergence import AdDivergenceStrategy
from src.app.database.trade_journal import TradeJournal, TradeMode
from market_data_mock import MarketDataMock


class TrailingOnlyBTWrapper(bt.Strategy):
    """
    Stratégie avec trailing stop uniquement (pas de TP).
    Laisse courir les gains tant que le stop n'est pas touché.
    """
    params = dict(
        capital=100000,
        max_stocks=5,
        trailing_percent=5.0,  # Trailing stop à 5%
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

        # Stratégie métier
        self.strategy = AdDivergenceStrategy(
            market_data=self.market_data,
            capital=self.p.capital,
            max_stocks=self.p.max_stocks
        )

        self.orders_by_symbol = {}
        self.pending_order_bundles = {}
        self.last_entry_prices = {}
        self.pending_stops = {}

        # Journal de trading en BD
        self.trade_entries = {}
        self.strategy_name = "TrailingOnly"
        self.trade_journal = TradeJournal()
        # Effacer les trades backtest existants pour cette stratégie
        self.trade_journal.clear_backtest_trades(self.strategy_name)

        # Préparation du fichier de diagnostic
        diag_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.diag_filename = f"diagnostique_trailing_only_{diag_date}.txt"
        with open(self.diag_filename, "w") as f:
            f.write(f"--- Début du diagnostic TrailingOnly ({diag_date}) ---\n")

    def _log_trade_journal(self, symbol, entry_date, qty, entry_price, exit_date, exit_price, cause, pnl_brut, bars_held=0):
        """Enregistre un trade dans le journal BD."""
        # Calculer les commissions (0.1% à l'achat + 0.1% à la vente)
        commission_rate = 0.001
        commission_entree = entry_price * qty * commission_rate
        commission_sortie = exit_price * qty * commission_rate
        total_commission = commission_entree + commission_sortie
        pnl_net = pnl_brut - total_commission

        self.trade_journal.log_trade(
            trade_mode=TradeMode.BACKTEST,
            strategy_name=self.strategy_name,
            symbol=symbol,
            date_entree=entry_date,
            prix_entree=entry_price,
            quantite=qty,
            date_sortie=exit_date,
            prix_sortie=exit_price,
            cause_sortie=cause,
            pnl_brut=round(pnl_brut, 2),
            commission=round(total_commission, 2),
            pnl_net=round(pnl_net, 2),
            bars_held=bars_held,
            backtest_start_date=self.start_date,
            backtest_end_date=self.end_date
        )

    def _preload_cache(self, dataframes):
        """Pré-charge les DataFrames dans le cache du MarketDataMock."""
        import pandas as pd
        for symbol, df in dataframes.items():
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

    def notify_order(self, order):
        symbol = order.data._name if hasattr(order, 'data') and order.data else 'N/A'
        self.log_diag(f"[notify_order] status={order.getstatusname()}, symbol={symbol}, isbuy={order.isbuy()}")

        if order.status in [order.Completed]:
            if order.isbuy():
                order_type = 'ENTRY'
                self.last_entry_prices[symbol] = order.executed.price
                # Stocker les infos d'entrée pour le journal
                self.trade_entries[symbol] = {
                    'date_entree': order.data.datetime.datetime(0),
                    'prix_entree': order.executed.price,
                    'quantite': abs(order.executed.size),
                    'bar_entree': len(self)
                }

                # Placer uniquement le trailing stop (PAS de TP)
                data = self._get_data(symbol)
                if data is not None:
                    stop_order = self.sell(
                        data=data,
                        size=abs(order.executed.size),
                        exectype=bt.Order.StopTrail,
                        trailpercent=self.p.trailing_percent / 100.0
                    )
                    self.pending_stops[symbol] = {"stop_order": stop_order}
                    self.log_diag(f"[BT] Trailing stop {self.p.trailing_percent}% placé pour {symbol}")

            elif order.issell():
                self.orders_by_symbol[symbol] = None
                if symbol not in self.last_entry_prices:
                    self.log_diag(f"[WARNING] Stop orphelin pour {symbol}")
                    if symbol in self.pending_stops:
                        self.pending_stops[symbol] = None
                    return

                last_entry_price = self.last_entry_prices[symbol]
                pnl = (order.executed.price - last_entry_price) * abs(order.executed.size)
                order_type = 'TRAILING_STOP'

                # Calculer bars_held et écrire dans le journal
                bars_held = 0
                if symbol in self.trade_entries:
                    entry_info = self.trade_entries[symbol]
                    bars_held = len(self) - entry_info.get('bar_entree', len(self))
                    self._log_trade_journal(
                        symbol=symbol,
                        entry_date=entry_info['date_entree'],
                        qty=entry_info['quantite'],
                        entry_price=entry_info['prix_entree'],
                        exit_date=order.data.datetime.datetime(0),
                        exit_price=order.executed.price,
                        cause=order_type,
                        pnl_brut=pnl,
                        bars_held=bars_held
                    )
                    del self.trade_entries[symbol]

                self.pending_stops[symbol] = None
                del self.last_entry_prices[symbol]
                self.log_diag(f"[CLEANUP] Position fermée pour {symbol} via {order_type}, PnL: {pnl:.2f}, Bars: {bars_held}")

            self.pending_order_bundles[symbol] = None

    def next(self):
        current_date = self.datas[0].datetime.date(0)

        # Ne trader que dans la période spécifiée
        if current_date < self.start_date.date() or current_date > self.end_date.date():
            return

        self.log_diag(f"Analyse pour la date {current_date}...")
        self.log_diag(f"[DIAG] Cash: {self.broker.get_cash():.2f} | Value: {self.broker.get_value():.2f}")

        symbols = [d._name for d in self.datas]
        self.strategy.set_symbols_to_analyse(symbols)
        symbols_to_trade = self.strategy.get_symbols(current_date)
        self.log_diag(f"{current_date} Symboles sélectionnés: {symbols_to_trade}")

        order_bundles = self.strategy.get_order_params()

        for bundle in order_bundles:
            symbol = bundle["symbol"]
            data = self._get_data(symbol)
            if not data:
                continue

            pos = self.getposition(data)

            if pos.size > 0:
                self.log_diag(f"[CONTROL] Position déjà ouverte sur {symbol}")
                continue
            if pos.size < 0:
                self.log_diag(f"[ERROR] Position short sur {symbol}")
                continue

            if pos.size == 0 and self.orders_by_symbol.get(symbol) is None:
                # Annuler les stops orphelins
                if symbol in self.pending_stops and self.pending_stops[symbol] is not None:
                    stop = self.pending_stops[symbol].get("stop_order")
                    if stop is not None and stop.status not in [stop.Completed, stop.Canceled]:
                        self.broker.cancel(stop)
                    self.pending_stops[symbol] = None

                # Ordre d'entrée simple (market order)
                entry_order = self.buy(data=data, size=bundle["entry_order"].totalQuantity)

                if entry_order is not None:
                    self.log_diag(f"[BT] Entry order pour {symbol}")
                    self.orders_by_symbol[symbol] = {"entry": entry_order}
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

    def stop(self):
        """Appelé à la fin du backtest - affiche le résumé."""
        print("\n" + "=" * 60)
        print("RÉSUMÉ BD - TrailingOnly")
        print("=" * 60)
        summary = self.trade_journal.get_performance_summary(
            trade_mode=TradeMode.BACKTEST,
            strategy_name=self.strategy_name
        )
        if "error" not in summary:
            print(f"Trades: {summary['total_trades']} (W:{summary['winning_trades']} / L:{summary['losing_trades']})")
            print(f"Win Rate: {summary['win_rate']:.1f}%")
            print(f"PnL Net Total: {summary['total_pnl_net']:,.2f}$")
            print(f"PnL Net Moyen: {summary['avg_pnl_net']:,.2f}$")
            print(f"Max Win: {summary['max_win']:,.2f}$ | Max Loss: {summary['max_loss']:,.2f}$")
        else:
            print(summary.get("error", "Erreur inconnue"))

        print("=" * 60)

        # Fermer la connexion BD
        self.trade_journal.close()
