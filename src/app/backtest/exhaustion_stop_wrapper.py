"""
Wrapper Backtrader avec stop activé uniquement après détection d'essoufflement du mouvement.
L'idée: laisser courir les profits tant que le momentum est fort, puis activer un trailing stop
dès que des signes d'essoufflement sont détectés.
"""
import backtrader as bt
from datetime import date, datetime, timedelta
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.strategies.addivergence import AdDivergenceStrategy
from src.app.database.trade_journal import TradeJournal, TradeMode
from market_data_mock import MarketDataMock


class ExhaustionStopBTWrapper(bt.Strategy):
    """
    Stratégie avec stop activé uniquement après détection d'essoufflement.

    Indicateurs d'essoufflement utilisés:
    - MACD histogram décroissant sur N jours
    - RSI en zone extrême (>70 pour long)
    - Volume ratio décroissant
    - ADX décroissant (tendance qui faiblit)

    Le stop n'est placé que lorsque plusieurs de ces signaux convergent.
    """
    params = dict(
        capital=100000,
        max_stocks=5,
        trailing_percent=5.0,           # Trailing stop une fois activé
        exhaustion_threshold=2,         # Nombre de signaux requis pour confirmer l'essoufflement
        macd_lookback=3,                # Jours pour évaluer la tendance MACD
        rsi_extreme=70,                 # Seuil RSI surachat
        volume_lookback=5,              # Jours pour comparer le volume
        adx_lookback=3,                 # Jours pour évaluer la tendance ADX
        min_bars_before_check=3,        # Bars minimum avant de vérifier l'essoufflement
        dataframes=None
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

        # Tracking pour la détection d'essoufflement
        self.entry_bar = {}              # Bar d'entrée pour chaque symbole
        self.stop_activated = {}         # Si le stop a été activé
        self.exhaustion_detected = {}    # Si l'essoufflement a été détecté

        # Journal de trading en BD
        self.trade_entries = {}
        self.strategy_name = "ExhaustionStop"
        self.trade_journal = TradeJournal()
        # Effacer les trades backtest existants pour cette stratégie
        self.trade_journal.clear_backtest_trades(self.strategy_name)

        # Stats d'essoufflement
        self.exhaustion_stats = {
            'total_positions': 0,
            'exhaustion_detected': 0,
            'exhaustion_signals': []
        }

        # Préparation du fichier de diagnostic
        diag_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.diag_filename = f"diagnostique_exhaustion_{diag_date}.txt"
        with open(self.diag_filename, "w") as f:
            f.write(f"--- Début du diagnostic Exhaustion Stop ({diag_date}) ---\n")
            f.write(f"Paramètres:\n")
            f.write(f"  - trailing_percent: {self.p.trailing_percent}%\n")
            f.write(f"  - exhaustion_threshold: {self.p.exhaustion_threshold} signaux\n")
            f.write(f"  - rsi_extreme: {self.p.rsi_extreme}\n")
            f.write(f"  - min_bars_before_check: {self.p.min_bars_before_check}\n")

    def _log_trade_journal(self, symbol, entry_date, qty, entry_price, exit_date, exit_price, cause, pnl_brut, bars_held=0, exhaustion_signals=""):
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
            exhaustion_signals=exhaustion_signals,
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

    def _detect_exhaustion(self, symbol, df):
        """
        Détecte l'essoufflement du mouvement haussier.

        Returns:
            tuple: (is_exhausted: bool, signals: list of str)
        """
        if len(df) < 20:
            self.log_diag(f"[DETECT] {symbol}: Pas assez de données ({len(df)} < 20)")
            return False, []

        signals = []
        debug_info = []

        # 1. MACD histogram décroissant
        try:
            import pandas_ta as ta
            macd_df = ta.macd(df['close'], fast=12, slow=26)
            macd_hist = macd_df['MACDh_12_26_9']
        except:
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            macd = ema12 - ema26
            macd_signal = macd.ewm(span=9).mean()
            macd_hist = macd - macd_signal

        if len(macd_hist.dropna()) >= self.p.macd_lookback:
            recent_macd = macd_hist.iloc[-self.p.macd_lookback:]
            macd_diff_mean = recent_macd.diff().mean()
            debug_info.append(f"MACD_diff_mean={macd_diff_mean:.4f}")
            if macd_diff_mean < 0:
                signals.append("MACD_DECLINING")

        # 2. RSI en zone extrême (abaissé à 65 pour plus de sensibilité)
        try:
            import pandas_ta as ta
            rsi = ta.rsi(df['close'], length=14)
        except:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-6)
            rsi = 100 - (100 / (1 + rs))

        if len(rsi.dropna()) > 0:
            rsi_val = rsi.iloc[-1]
            debug_info.append(f"RSI={rsi_val:.1f}")
            if rsi_val > self.p.rsi_extreme:
                signals.append(f"RSI_EXTREME_{rsi_val:.0f}")

        # 3. Volume ratio décroissant
        sma20_vol = df['volume'].rolling(20).mean()
        vol_ratio = df['volume'] / (sma20_vol + 1e-6)

        if len(vol_ratio.dropna()) >= self.p.volume_lookback:
            recent_vol = vol_ratio.iloc[-self.p.volume_lookback:]
            avg_recent = recent_vol.iloc[-1]
            avg_prior = recent_vol.iloc[:-1].mean()
            debug_info.append(f"VOL_ratio={avg_recent:.2f}/{avg_prior:.2f}")
            if avg_recent < avg_prior * 0.9:  # Ajusté: 10% plus bas (était 20%)
                signals.append("VOLUME_DECLINING")

        # 4. ADX décroissant (tendance qui faiblit)
        try:
            import pandas_ta as ta
            adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
            adx = adx_df['ADX_14']
        except:
            adx = None

        if adx is not None and len(adx.dropna()) >= self.p.adx_lookback:
            recent_adx = adx.iloc[-self.p.adx_lookback:]
            adx_diff_mean = recent_adx.diff().mean()
            debug_info.append(f"ADX_diff_mean={adx_diff_mean:.4f}")
            if adx_diff_mean < 0:
                signals.append("ADX_DECLINING")

        # 5. Rendement qui ralentit (ajusté: return_5d < return_10d * 0.7)
        if len(df) >= 10:
            return_5d = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1)
            return_10d = (df['close'].iloc[-1] / df['close'].iloc[-10] - 1)
            debug_info.append(f"R5d={return_5d:.3f}/R10d={return_10d:.3f}")
            if return_5d < return_10d * 0.7 and return_10d > 0:  # Ajusté (était /2)
                signals.append("MOMENTUM_SLOWING")

        # 6. NOUVEAU: Prix proche du haut récent mais momentum faible
        if len(df) >= 20:
            high_20d = df['high'].iloc[-20:].max()
            current_close = df['close'].iloc[-1]
            pct_from_high = (high_20d - current_close) / high_20d
            debug_info.append(f"PctFromHigh={pct_from_high:.3f}")
            if pct_from_high > 0.02 and pct_from_high < 0.08:  # Entre -2% et -8% du haut
                signals.append("NEAR_HIGH_WEAKENING")

        is_exhausted = len(signals) >= self.p.exhaustion_threshold

        # Log détaillé pour debugging
        self.log_diag(f"[DETECT] {symbol}: {' | '.join(debug_info)}")
        self.log_diag(f"[DETECT] {symbol}: Signaux={signals} ({len(signals)}/{self.p.exhaustion_threshold})")

        return is_exhausted, signals

    def notify_order(self, order):
        symbol = order.data._name if hasattr(order, 'data') and order.data else 'N/A'
        self.log_diag(f"[notify_order] status={order.getstatusname()}, symbol={symbol}, isbuy={order.isbuy()}")

        if order.status in [order.Completed]:
            if order.isbuy():
                order_type = 'ENTRY'
                self.last_entry_prices[symbol] = order.executed.price # Symbole a analyser  pour l'essouflement
                self.entry_bar[symbol] = len(self)  # Stocker le bar d'entrée
                self.stop_activated[symbol] = False
                self.exhaustion_detected[symbol] = False
                self.exhaustion_stats['total_positions'] += 1

                # Stocker les infos d'entrée pour le journal
                self.trade_entries[symbol] = {
                    'date_entree': order.data.datetime.datetime(0),
                    'prix_entree': order.executed.price,
                    'quantite': abs(order.executed.size),
                    'bar_entree': len(self),
                    'exhaustion_signals': []
                }

                self.log_diag(f"[BT] Position ouverte {symbol} @ {order.executed.price:.2f} - Attente essoufflement...")

            elif order.issell():
                self.orders_by_symbol[symbol] = None
                if symbol not in self.last_entry_prices:
                    self.log_diag(f"[WARNING] Stop orphelin pour {symbol}")
                    if symbol in self.pending_stops:
                        self.pending_stops[symbol] = None
                    return

                last_entry_price = self.last_entry_prices[symbol]
                pnl = (order.executed.price - last_entry_price) * abs(order.executed.size)
                order_type = 'EXHAUSTION_STOP' if self.exhaustion_detected.get(symbol, False) else 'TRAILING_STOP'

                # Calculer le nombre de bars
                bars_held = 0
                exhaustion_signals_str = ""
                if symbol in self.trade_entries:
                    entry_info = self.trade_entries[symbol]
                    bars_held = len(self) - entry_info.get('bar_entree', len(self))
                    exhaustion_signals_str = "|".join(entry_info.get('exhaustion_signals', []))

                    self._log_trade_journal(
                        symbol=symbol,
                        entry_date=entry_info['date_entree'],
                        qty=entry_info['quantite'],
                        entry_price=entry_info['prix_entree'],
                        exit_date=order.data.datetime.datetime(0),
                        exit_price=order.executed.price,
                        cause=order_type,
                        pnl_brut=pnl,
                        bars_held=bars_held,
                        exhaustion_signals=exhaustion_signals_str
                    )
                    del self.trade_entries[symbol]

                self.pending_stops[symbol] = None
                del self.last_entry_prices[symbol]
                self.entry_bar.pop(symbol, None)
                self.stop_activated.pop(symbol, None)
                self.exhaustion_detected.pop(symbol, None)

                self.log_diag(f"[CLEANUP] Position fermée pour {symbol} via {order_type}, PnL: {pnl:.2f}, Bars: {bars_held}")

            self.pending_order_bundles[symbol] = None

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        current_bar = len(self)

        # Ne trader que dans la période spécifiée
        if current_date < self.start_date.date() or current_date > self.end_date.date():
            return
        
        # Vérifier l'essoufflement pour les positions ouvertes
        for symbol in self.last_entry_prices.keys():
            data = self._get_data(symbol)
            if data is None:
                continue

            # Vérifier si assez de bars sont passés
            entry_bar = self.entry_bar.get(symbol, 0)
            if current_bar - entry_bar < self.p.min_bars_before_check:
                continue

            # Récupérer le DataFrame historique (45 jours calendaires = ~30 jours de bourse)
            start_date_hist = current_date - timedelta(days=45)
            df = self.market_data.get_historical_data(
                symbol,
                start_date_hist,
                current_date
            )
            if df is None or len(df) < 20:
                continue

            # Détecter l'essoufflement
            is_exhausted, signals = self._detect_exhaustion(symbol, df)
            if is_exhausted and not self.stop_activated.get(symbol, False):
                self.log_diag(f"[EXHAUSTION] Essoufflement détecté pour {symbol} avec signaux: {signals}")

                # Placer le trailing stop
                stop_order = self.sell(
                    data=data,
                    size=abs(self.getposition(data).size),
                    exectype=bt.Order.StopTrail,
                    trailpercent=self.p.trailing_percent / 100.0
                )
                self.pending_stops[symbol] = {"stop_order": stop_order}
                self.stop_activated[symbol] = True
                self.exhaustion_detected[symbol] = True
                self.exhaustion_stats['exhaustion_detected'] += 1
                self.exhaustion_stats['exhaustion_signals'].append({
                    'symbol': symbol,
                    'signals': signals
                })

                # Mettre à jour le journal d'entrée
                if symbol in self.trade_entries:
                    self.trade_entries[symbol]['exhaustion_signals'] = signals
        
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
        """Appelé à la fin du backtest - affiche les statistiques."""
        print("\n" + "=" * 60)
        print("STATISTIQUES ESSOUFFLEMENT")
        print("=" * 60)
        print(f"Total positions: {self.exhaustion_stats['total_positions']}")
        print(f"Essoufflements détectés: {self.exhaustion_stats['exhaustion_detected']}")

        if self.exhaustion_stats['total_positions'] > 0:
            pct = (self.exhaustion_stats['exhaustion_detected'] /
                   self.exhaustion_stats['total_positions'] * 100)
            print(f"Taux de détection: {pct:.1f}%")

        # Compter les types de signaux
        signal_counts = {}
        for item in self.exhaustion_stats['exhaustion_signals']:
            for sig in item['signals']:
                sig_type = sig.split('_')[0]  # Prendre le premier mot
                signal_counts[sig_type] = signal_counts.get(sig_type, 0) + 1

        if signal_counts:
            print("\nSignaux les plus fréquents:")
            for sig, count in sorted(signal_counts.items(), key=lambda x: -x[1]):
                print(f"  - {sig}: {count}")

        # Afficher le résumé depuis la BD
        print("\n" + "-" * 60)
        print("RÉSUMÉ BD")
        print("-" * 60)
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

        print("=" * 60)

        # Fermer la connexion BD
        self.trade_journal.close()
