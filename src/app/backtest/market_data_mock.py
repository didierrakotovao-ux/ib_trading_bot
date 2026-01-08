import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.app.screener.providers.market_data_provider import MarketDataProvider


class MarketDataMock(MarketDataProvider):
    """
    Mock du MarketDataProvider pour le backtest.
    Utilise les données déjà chargées dans Backtrader au lieu de télécharger via yfinance.
    """

    def __init__(self, datas):
        self.datas = datas
        self._next_req_id = 100000
        self._data_cache = {}  # Cache des DataFrames par symbole
        self._preloaded = False

    def preload_all_data(self):
        """
        Pré-charge toutes les données Backtrader en DataFrames.
        Appelé une seule fois au début du backtest.
        """
        if self._preloaded:
            return

        for data in self.datas:
            symbol = data._name
            # Extraire toutes les données de ce feed Backtrader
            dates = []
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []

            # Parcourir toutes les barres du feed
            for i in range(-len(data) + 1, 1):
                try:
                    dt = data.datetime.date(i)
                    dates.append(dt)
                    opens.append(data.open[i])
                    highs.append(data.high[i])
                    lows.append(data.low[i])
                    closes.append(data.close[i])
                    volumes.append(data.volume[i])
                except IndexError:
                    break

            if dates:
                df = pd.DataFrame({
                    'date': dates,
                    'open': opens,
                    'high': highs,
                    'low': lows,
                    'close': closes,
                    'volume': volumes
                })
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                self._data_cache[symbol] = df

        self._preloaded = True
        print(f"[CACHE] Pré-chargement terminé: {len(self._data_cache)} symboles en cache")

    def get_historical_data(self, symbol, start_date, end_date, interval="1d"):
        """
        Retourne les données historiques depuis le cache.
        Beaucoup plus rapide que de télécharger via yfinance.
        """
        # S'assurer que les données sont pré-chargées
        if not self._preloaded:
            self.preload_all_data()

        if symbol not in self._data_cache:
            return None

        df = self._data_cache[symbol].copy()

        # Convertir les dates de filtrage
        if hasattr(start_date, 'date'):
            start_date = start_date.date()
        if hasattr(end_date, 'date'):
            end_date = end_date.date()

        # Filtrer par date
        df['date_only'] = pd.to_datetime(df['date']).dt.date
        mask = (df['date_only'] >= start_date) & (df['date_only'] <= end_date)
        df = df[mask].drop(columns=['date_only'])

        if df.empty:
            return None

        return df.reset_index(drop=True)

    def get_current_data_for_symbol(self, symbol, current_date, lookback_days=350):
        """
        Méthode optimisée pour obtenir les données jusqu'à la date courante.
        Utilisée par la stratégie pendant le backtest.
        """
        if not self._preloaded:
            self.preload_all_data()

        if symbol not in self._data_cache:
            return None

        df = self._data_cache[symbol].copy()

        # Convertir current_date
        if hasattr(current_date, 'date'):
            current_date = current_date.date()

        # Filtrer: données jusqu'à current_date
        df['date_only'] = pd.to_datetime(df['date']).dt.date
        df = df[df['date_only'] <= current_date].drop(columns=['date_only'])

        # Garder les derniers lookback_days
        if len(df) > lookback_days:
            df = df.tail(lookback_days)

        if df.empty:
            return None

        return df.reset_index(drop=True)
