"""
Wrapper Backtrader utilisant la configuration de stops depuis stop_config.json.

Supporte 4 combinaisons :
  - Protection : fixed (Stop Ã  prix fixe) ou trailing (StopTrail)
  - Profit     : fixed (Limit Ã  prix fixe) ou dynamic (Limit Ã  entry + NÃ—ATR)

Stop et TP sont liÃ©s en OCA (One Cancels All) : quand l'un se dÃ©clenche,
l'autre est annulÃ© automatiquement par Backtrader.

Usage :
    python test_stop_config.py
"""
import os
import sys
import backtrader as bt
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.app.strategies.momentum import MomentumStrategy
from src.app.strategies.wyckoff_ml import WyckoffMLStrategy
from src.app.database.trade_journal import TradeJournal, TradeMode
from src.app.stop_manager import StopConfig

# Import local (mÃªme rÃ©pertoire backtest)
sys.path.insert(0, os.path.dirname(__file__))
from market_data_mock import MarketDataMock


class StopConfigBTWrapper(bt.Strategy):
    """
    Wrapper Backtrader avec stops configurables depuis stop_config.json.
    Chaque entrÃ©e gÃ©nÃ¨re un couple Stop + TP en groupe OCA.
    """
    params = dict(
        capital=100000,
        max_stocks=5,
        config=None,           # StopConfig chargÃ© avant le lancement
        dataframes=None,
        scoring_type="smooth_ml",
        smooth_model_path=None,
        wyckoff_model_path=None,
        use_sue_filter=False,
        sue_threshold=0.0,
        db_path=None
    )

    def __init__(self, start_date, end_date, dataframes=None):
        self.start_date = start_date
        self.end_date = end_date

        # Config stops
        self.stop_cfg: StopConfig = self.p.config
        if self.stop_cfg is None:
            raise ValueError("Passer un StopConfig via le paramÃ¨tre 'config'")

        # MarketData mock
        self.market_data = MarketDataMock(self.datas)
        if self.p.dataframes:
            self._preload_cache(self.p.dataframes)

        # ATR par symbole (pour TP dynamique)
        self.atr = {}
        for d in self.datas:
            self.atr[d._name] = bt.indicators.ATR(
                d, period=self.stop_cfg.profit_atr_period
            )

        # RÃ©soudre db_path
        db_path = "trading_data.db"
        if self.p.db_path:
            db_path = os.path.basename(self.p.db_path)

        # StratÃ©gie mÃ©tier
        if self.p.scoring_type == "wyckoff_ml":
            self.strategy = WyckoffMLStrategy(
                market_data=self.market_data,
                capital=self.p.capital,
                max_stocks=self.p.max_stocks,
                use_trailing_stop=False,  # gÃ©rÃ© ici, pas dans la stratÃ©gie
                scoring_type=self.p.scoring_type,
                wyckoff_model_path=self.p.wyckoff_model_path,
                db_path=db_path
            )
        else:
            self.strategy = MomentumStrategy(
                market_data=self.market_data,
                capital=self.p.capital,
                max_stocks=self.p.max_stocks,
                use_trailing_stop=False,  # gÃ©rÃ© ici, pas dans la stratÃ©gie
                scoring_type=self.p.scoring_type,
                smooth_model_path=self.p.smooth_model_path,
                use_sue_filter=self.p.use_sue_filter,
                sue_threshold=self.p.sue_threshold,
                db_path=db_path
            )

        # Trackers par symbole
        self.orders_by_symbol = {}     # symbol â†’ {'entry': order}
        self.pending_stops = {}        # symbol â†’ {'stop': order, 'tp': order}
        self.trade_entries = {}        # symbol â†’ infos d'entrÃ©e
        self.last_entry_prices = {}

        # Journal
        prot = f"{self.stop_cfg.protection_type}{self.stop_cfg.protection_pct:.0f}"
        prof = (f"fixed{self.stop_cfg.profit_fixed_pct:.0f}"
                if self.stop_cfg.profit_type == "fixed"
                else f"atr{self.stop_cfg.profit_atr_mult:.1f}")
        model_tag = "Wyckoff" if self.p.scoring_type == "wyckoff_ml" else "Smooth"
        self.strategy_name = f"StopConfig_{model_tag}_{prot}_{prof}"

        self.trade_journal = TradeJournal("backtest_journal.db")
        # Efface uniquement les trades de la même période (les autres fenêtres
        # de backtest sont conservées)
        self.trade_journal.clear_backtest_trades(
            self.strategy_name,
            backtest_start_date=self.start_date,
            backtest_end_date=self.end_date)

        # Fichier de diagnostic
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        resultats_dir = os.path.join(os.path.dirname(__file__), 'resultats')
        os.makedirs(resultats_dir, exist_ok=True)
        self.diag_filename = os.path.join(resultats_dir, f"diagnostique_stop_config_{ts}.txt")
        with open(self.diag_filename, "w") as f:
            f.write(f"--- StopConfigBTWrapper ({ts}) ---\n")
            f.write(f"Protection : {self.stop_cfg.protection_type} -{self.stop_cfg.protection_pct}%\n")
            profit_desc = (f"fixed +{self.stop_cfg.profit_fixed_pct}%"
                           if self.stop_cfg.profit_type == "fixed"
                           else f"dynamic {self.stop_cfg.profit_atr_mult}Ã—ATR({self.stop_cfg.profit_atr_period})")
            f.write(f"Profit     : {profit_desc}\n\n")

        print(f"[BT] StratÃ©gie : {self.strategy_name}")
        print(f"[BT] Protection : {self.stop_cfg.protection_type} -{self.stop_cfg.protection_pct}%")
        print(f"[BT] Profit     : {profit_desc}")

    # ------------------------------------------------------------------
    # Calcul des prix
    # ------------------------------------------------------------------

    def _calc_stop_price(self, entry_price: float) -> float:
        return round(entry_price * (1 - self.stop_cfg.protection_pct / 100), 4)

    def _calc_profit_price(self, entry_price: float, symbol: str) -> float:
        if self.stop_cfg.profit_type == "fixed":
            return round(entry_price * (1 + self.stop_cfg.profit_fixed_pct / 100), 4)
        else:
            atr_val = self.atr[symbol][0]
            if atr_val <= 0:
                # Fallback fixed si ATR non dispo
                return round(entry_price * (1 + self.stop_cfg.profit_fixed_pct / 100), 4)
            return round(entry_price + self.stop_cfg.profit_atr_mult * atr_val, 4)

    # ------------------------------------------------------------------
    # Callbacks Backtrader
    # ------------------------------------------------------------------

    def notify_order(self, order):
        symbol = order.data._name if hasattr(order, 'data') and order.data else 'N/A'

        if order.status not in [order.Completed]:
            return

        if order.isbuy():
            entry_price = order.executed.price
            qty = abs(order.executed.size)
            data = self._get_data(symbol)

            self.last_entry_prices[symbol] = entry_price
            self.trade_entries[symbol] = {
                'date_entree': order.data.datetime.datetime(0),
                'prix_entree': entry_price,
                'quantite': qty,
                'bar_entree': len(self)
            }

            # Groupe OCA unique pour ce couple stop/TP
            oca_group = f"{symbol}_{len(self)}"

            # --- Stop de protection ---
            if self.stop_cfg.protection_type == "fixed":
                stop_price = self._calc_stop_price(entry_price)
                stop_order = self.sell(
                    data=data, size=qty,
                    exectype=bt.Order.Stop,
                    price=stop_price,
                    ocagroup=oca_group, ocatype=1
                )
                self.log_diag(f"[BT] {symbol}: Stop fixe @ {stop_price:.2f} "
                              f"(-{self.stop_cfg.protection_pct}%)")
            else:
                stop_order = self.sell(
                    data=data, size=qty,
                    exectype=bt.Order.StopTrail,
                    trailpercent=self.stop_cfg.protection_pct / 100.0,
                    ocagroup=oca_group, ocatype=1
                )
                self.log_diag(f"[BT] {symbol}: Trailing stop -{self.stop_cfg.protection_pct}%")

            # --- Prise de bÃ©nÃ©fice ---
            tp_price = self._calc_profit_price(entry_price, symbol)
            tp_order = self.sell(
                data=data, size=qty,
                exectype=bt.Order.Limit,
                price=tp_price,
                ocagroup=oca_group, ocatype=1
            )
            self.log_diag(f"[BT] {symbol}: TP @ {tp_price:.2f} "
                          f"(+{((tp_price/entry_price)-1)*100:.1f}%)")

            self.pending_stops[symbol] = {
                'stop': stop_order,
                'tp': tp_order,
                'oca': oca_group
            }

        elif order.issell():
            self.orders_by_symbol[symbol] = None

            if symbol not in self.last_entry_prices:
                self.log_diag(f"[WARNING] SELL orphelin pour {symbol}")
                return

            entry_price = self.last_entry_prices[symbol]
            exit_price = order.executed.price
            qty = abs(order.executed.size)
            pnl = (exit_price - entry_price) * qty

            # DÃ©terminer la cause de sortie
            stops = self.pending_stops.get(symbol, {})
            if stops and order == stops.get('tp'):
                cause = f"TAKE_PROFIT_{self.stop_cfg.profit_type.upper()}"
            elif self.stop_cfg.protection_type == "fixed":
                cause = "FIXED_STOP"
            else:
                cause = "TRAILING_STOP"

            # Journal
            if symbol in self.trade_entries:
                entry_info = self.trade_entries[symbol]
                bars_held = len(self) - entry_info.get('bar_entree', len(self))
                self._log_trade_journal(
                    symbol=symbol,
                    entry_date=entry_info['date_entree'],
                    qty=entry_info['quantite'],
                    entry_price=entry_info['prix_entree'],
                    exit_date=order.data.datetime.datetime(0),
                    exit_price=exit_price,
                    cause=cause,
                    pnl_brut=pnl,
                    bars_held=bars_held
                )
                self.log_diag(f"[BT] {symbol}: Sortie {cause} @ {exit_price:.2f}, "
                              f"PnL={pnl:+.2f}, Bars={bars_held}")
                del self.trade_entries[symbol]

            # Nettoyage
            self.pending_stops.pop(symbol, None)
            self.last_entry_prices.pop(symbol, None)
            self.orders_by_symbol.pop(symbol, None)

    def prenext(self):
        self.next()

    def next(self):
        current_date = self.datas[0].datetime.date(0)

        if current_date < self.start_date.date() or current_date > self.end_date.date():
            return

        current_positions = sum(1 for d in self.datas if self.getposition(d).size != 0)
        if current_positions >= self.p.max_stocks:
            return

        symbols = [d._name for d in self.datas]
        self.strategy.set_symbols_to_analyse(symbols)
        symbols_to_trade = self.strategy.get_symbols(current_date)

        # Filtrer symboles dÃ©jÃ  en position ou avec ordres en attente
        symbols_available = []
        for symbol in symbols_to_trade:
            data = self._get_data(symbol)
            if not data:
                continue
            pos = self.getposition(data)
            has_position = pos.size != 0
            has_pending = (symbol in self.orders_by_symbol and
                           self.orders_by_symbol[symbol] is not None)
            has_stop = (symbol in self.pending_stops and
                        self.pending_stops[symbol] is not None)
            if not (has_position or has_pending or has_stop):
                symbols_available.append(symbol)

        if not symbols_available:
            return

        self.strategy.symbolsToTrade = symbols_available
        order_bundles = self.strategy.get_order_params()

        # Sizing dynamique base sur le cash disponible pour limiter les rejets Margin.
        reserved_cash = float(self.broker.get_cash())
        planned_entries = 0
        cash_buffer_mult = 1.002

        for bundle in order_bundles:
            slots_left = self.p.max_stocks - current_positions - planned_entries
            if slots_left <= 0:
                break

            symbol = bundle["symbol"]
            data = self._get_data(symbol)
            if not data:
                continue
            if self.getposition(data).size != 0:
                continue

            requested_qty = int(bundle["entry_order"].totalQuantity)
            entry_price_ref = float(getattr(bundle["entry_order"], "lmtPrice", data.close[0] * 1.005))
            budget_per_slot = reserved_cash / max(slots_left, 1)
            affordable_qty = int(budget_per_slot / max(entry_price_ref * cash_buffer_mult, 0.01))
            qty = min(requested_qty, affordable_qty)

            if qty <= 0:
                self.log_diag(
                    f"[CASH] {symbol}: entrée ignorée (cash réservé={reserved_cash:.2f}, "
                    f"slots_left={slots_left}, prix_ref={entry_price_ref:.2f})"
                )
                continue

            entry_order = self.buy(data=data, size=qty)
            if entry_order:
                self.orders_by_symbol[symbol] = {"entry": entry_order}
                self.log_diag(f"[BT] {symbol}: EntrÃ©e qty={qty}")
                reserved_cash -= qty * entry_price_ref * cash_buffer_mult
                planned_entries += 1

    def stop(self):
        """Ferme les positions ouvertes en fin de backtest."""
        for data in self.datas:
            symbol = data._name
            pos = self.getposition(data)
            if pos.size == 0:
                continue

            exit_price = data.close[0]
            exit_date = data.datetime.datetime(0)
            qty = abs(pos.size)

            # Annuler stop/TP en attente
            stops = self.pending_stops.get(symbol, {})
            for key in ('stop', 'tp'):
                o = stops.get(key)
                if o and o.status not in [bt.Order.Completed, bt.Order.Canceled]:
                    self.cancel(o)

            if symbol in self.trade_entries:
                entry_info = self.trade_entries[symbol]
                pnl = (exit_price - entry_info['prix_entree']) * entry_info['quantite']
                bars_held = len(self) - entry_info.get('bar_entree', len(self))
                self._log_trade_journal(
                    symbol=symbol,
                    entry_date=entry_info['date_entree'],
                    qty=entry_info['quantite'],
                    entry_price=entry_info['prix_entree'],
                    exit_date=exit_date,
                    exit_price=exit_price,
                    cause='END_OF_BACKTEST',
                    pnl_brut=pnl,
                    bars_held=bars_held
                )
                del self.trade_entries[symbol]

        # RÃ©sumÃ©
        print("\n" + "=" * 60)
        print(f"RÃ‰SUMÃ‰ BD â€” {self.strategy_name}")
        print("=" * 60)
        summary = self.trade_journal.get_performance_summary(
            trade_mode=TradeMode.BACKTEST,
            strategy_name=self.strategy_name
        )
        if "error" not in summary:
            print(f"Trades    : {summary['total_trades']} "
                  f"(W:{summary['winning_trades']} / L:{summary['losing_trades']})")
            print(f"Win Rate  : {summary['win_rate']:.1f}%")
            print(f"PnL Net   : {summary['total_pnl_net']:,.2f}$")
            print(f"PnL Moyen : {summary['avg_pnl_net']:,.2f}$")
            print(f"Max Win   : {summary['max_win']:,.2f}$ | "
                  f"Max Loss  : {summary['max_loss']:,.2f}$")
        else:
            print(summary.get("error"))
        print("=" * 60)
        self.trade_journal.close()

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def _log_trade_journal(self, symbol, entry_date, qty, entry_price,
                           exit_date, exit_price, cause, pnl_brut, bars_held=0):
        """Enregistre un trade fermÃ© dans le journal BD."""
        commission_rate = 0.001
        commission = (entry_price + exit_price) * qty * commission_rate
        pnl_net = pnl_brut - commission

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
            commission=round(commission, 2),
            pnl_net=round(pnl_net, 2),
            bars_held=bars_held,
            backtest_start_date=self.start_date,
            backtest_end_date=self.end_date
        )

    def _get_data(self, symbol):
        for d in self.datas:
            if d._name == symbol:
                return d
        return None

    def _preload_cache(self, dataframes):
        import pandas as pd
        for symbol, df in dataframes.items():
            df_reset = df.reset_index()
            df_reset.columns = [col.lower() for col in df_reset.columns]
            if 'date' not in df_reset.columns and 'index' in df_reset.columns:
                df_reset = df_reset.rename(columns={'index': 'date'})
            df_reset['date'] = pd.to_datetime(df_reset['date'])
            self.market_data._data_cache[symbol] = df_reset
        self.market_data._preloaded = True

    def log_diag(self, message):
        with open(self.diag_filename, "a") as f:
            f.write(message + "\n")

    def notify_trade(self, trade):
        if trade.isclosed:
            self.orders_by_symbol.pop(trade.data._name, None)

