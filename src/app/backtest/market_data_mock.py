import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.app.screener.providers.market_data_provider import MarketDataProvider

class MarketDataMock(MarketDataProvider):
    def __init__(self, datas):
        self.datas = datas
        self._next_req_id = 100000  # requis par ta stratégie

    def get_historical_data(self, symbol, start_date, end_date, interval="1d"):
        return super().get_historical_data(symbol, start_date, end_date, interval)
