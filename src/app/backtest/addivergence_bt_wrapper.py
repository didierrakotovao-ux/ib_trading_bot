import backtrader as bt
from app.backtest.market_data_mock import MarketDataMock
from app.ml.addivergencescoring import AdDivergenceScoring
from strategies.addivergence import AdDivergenceStrategy

class AdDivergenceBTWrapper(bt.Strategy):
    params = dict(
        capital=100000,
        max_stocks=5,
        score_threshold=60,
        lookback_days=350,
        stop_loss=0.10,  # 10% stop loss
        take_profit=0.20 # 20% take profit
    )

    def __init__(self, start_date, end_date):
        # Initialiser la stratégie maison avec un mock MarketDataProvider
        # Ici, self.datas[0] est le DataFeed Backtrader (pandas)
        self.strategy = AdDivergenceStrategy(
            market_data=MarketDataMock(),  # À adapter si besoin
            capital=self.p.capital,
            max_stocks=self.p.max_stocks
        )
        self.strategy.scoring = AdDivergenceScoring()
        self.ordered = False
        self.entry_price = None
        self.highest = None
        self.start_date = start_date
        self.end_date = end_date    
        self.stocks_to_trade = [
            # Mega caps
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'UNH',
            'XOM', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV',
            # Large caps tech
            'AVGO', 'PEP', 'COST', 'ADBE', 'CRM', 'CSCO', 'ACN', 'NFLX', 'TMO', 'AMD',
            'INTC', 'QCOM', 'TXN', 'INTU', 'AMAT', 'MU', 'ADI', 'LRCX', 'KLAC', 'SNPS',
            # Finance
            'BAC', 'WFC', 'MS', 'GS', 'C', 'SCHW', 'AXP', 'BLK', 'SPGI', 'CB',
            'MMC', 'PGR', 'TFC', 'USB', 'PNC', 'CME', 'MCO', 'AON', 'ICE', 'AFL',
            # Healthcare
            'LLY', 'UNH', 'JNJ', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR', 'AMGN',
            'BMY', 'GILD', 'VRTX', 'CVS', 'ISRG', 'CI', 'MDT', 'REGN', 'HUM', 'ZTS',
            # Consumer
            'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'CVS', 'BKNG', 'LULU',
            'CMG', 'MAR', 'ABNB', 'ORLY', 'AZO', 'YUM', 'GM', 'F', 'TSLA', 'ROST',
            # Industrial
            'CAT', 'BA', 'HON', 'UNP', 'RTX', 'GE', 'LMT', 'MMM', 'UPS', 'DE',
            'EMR', 'ETN', 'NOC', 'ITW', 'WM', 'PH', 'GD', 'NSC', 'FDX', 'PCAR',
            # Energy
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HES',
            # Utilities & Real Estate
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'PLD', 'AMT', 'CCI',
            # Materials
            'LIN', 'APD', 'SHW', 'ECL', 'NEM', 'FCX', 'NUE', 'DOW', 'DD', 'PPG'
        ]
        self.datas = []
        if not hasattr(self, 'bt_orders'):
            self.bt_orders = dict()


    def next(self):
        for trade_date in self.get_dates_in_range(self.start_date, self.end_date):
            self.strategy.set_symbols_to_analyse(self.stocks_to_trade)
            for symbol in self.strategy.get_symbols(trade_date):
                if symbol not in self.datas:
                    self.datas.append(symbol)
            for orders in self.strategy.get_order_params():
                symbol = orders['symbol']
                entryOrder = orders['entry_order']
                slOrder = orders['stop_order']
                tpOrder = orders['take_profit_order']
                # Stocker les ordres pour gestion Backtrader (exemple avec un dict)
                if symbol not in self.bt_orders:
                    self.bt_orders[symbol] = {
                        'entry': entryOrder,
                        'stop': slOrder,
                        'take_profit': tpOrder
                    }

            for data in self.datas:
                symbol = data._name if hasattr(data, '_name') else str(data)
                orders = self.bt_orders.get(symbol)
                pos = self.getposition(data)
                
                if not pos:
                    # Entrée si pas de position
                    pos.price =  getattr(orders['entry'], 'lmtPrice', None)
                    pos.qty = getattr(orders['entry'], 'totalQuantity', None)
                    self.orders[data] = self.buy(data=data)
                else:
                    # Initialisation à l'entrée
                    if symbol not in self.highest_dict:
                        self.highest_dict[symbol] = pos.price
                    # Mise à jour du plus haut
                    if data.close[0] > self.highest_dict[symbol]:
                        self.highest_dict[symbol] = data.close[0]
                    highest = self.highest_dict[symbol]          
                    trailing_stop = highest * (1 - orders['stop'].trailingPercent / 100) if orders and hasattr(orders['stop'], 'trailingPercent') else None          
                    # Sortie sur stop loss ou take profit
                    if data.close[0] >= orders['take_profit'].lmtPrice:
                        self.close(data=data)
                    elif trailing_stop is not None and data.close[0] <= trailing_stop:
                        self.close(data=data)

    def get_dates_in_range(self, start_date, end_date):
        """Retourne la liste des jours ouvrables (lundi à vendredi) entre start_date et end_date inclus."""
        import pandas as pd
        all_days = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' = business day
        return [d.date() for d in all_days]
