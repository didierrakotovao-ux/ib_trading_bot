from ast import List
import datetime
from typing import Optional

from pyparsing import Dict
from app.screener.providers.market_data_provider import MarketDataProvider
import pandas as pd
import threading


class MarketDataMock(MarketDataProvider):
    def __init__(self):
        self._next_req_id = 1

    # --- Gestion des positions IB ---
    def req_ib_positions(self):
        pass

    def position(self, account, contract, position, avgCost):
        pass
    def positionEnd(self):
        pass

    def get_ib_positions(self):
        pass

    def connect(self) -> bool:
        pass 
    
    def disconnect(self):
        pass
    
    def is_connected(self) -> bool:
        pass
    
    def nextValidId(self, orderId: int):
        pass 
    
    def wait_until_ready(self, timeout=10):
        pass
    def get_scanner_results(
        self,
        scan_sub: any, 
        max_results: int = 50
    ) -> List[Dict]:
        pass
        
    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        return super().get_historical_data(
            symbol,
            start_date,
            end_date,
            interval
        )   
    
    def error(self, reqId, errorCode, errorString):
        pass

    def placeOrder(self, contract, order):
        pass

    def create_contract(self, ticker: str, sec_type: str = 'STK', exchange: str = 'SMART', currency: str = 'USD') -> Contract:
        pass
    
    def force_close(self):
        pass

    def scannerData(self, reqId, rank, contract, distance, benchmark, projection, legsStr):
        pass

    def scannerDataEnd(self, reqId):
        pass
