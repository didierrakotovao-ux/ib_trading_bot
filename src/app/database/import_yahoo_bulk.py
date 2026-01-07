import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from db_manager import DatabaseManager

# Paramètres
symbols = [
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
start_date = (datetime.now() - timedelta(days=365*12)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

db = DatabaseManager("trading_data.db")

for i, symbol in enumerate(symbols, 1):
    print(f"[{i}/{len(symbols)}] Téléchargement {symbol} ...")
    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if df.empty:
            print(f"[WARN] Pas de données pour {symbol}")
            continue
        # Si MultiIndex, on garde juste le premier niveau
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # yfinance retourne parfois 'Adj Close' au lieu de 'adjusted_close'
        if 'Adj Close' in df.columns:
            df = df.rename(columns={'Adj Close': 'adjusted_close'})
        db.save_historical_data(symbol, df, source="yfinance")    
    except Exception as e:
        print(f"[ERROR] {symbol}: {e}")

print("[OK] Import terminé.")
