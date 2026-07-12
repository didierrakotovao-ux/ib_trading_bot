"""
Calcul des features basées sur les earnings (Novy-Marx).

SUE (Standardized Unexpected Earnings):
    - Surprise des résultats par rapport aux attentes
    - Disponible directement via yfinance (Surprise%)

CAR3 (Cumulative Abnormal Return 3 jours):
    - Rendement anormal autour de l'annonce des résultats
    - CAR3 = Rendement_stock(J-1 à J+1) - Rendement_SPY(J-1 à J+1)

Usage:
    from earnings_features import EarningsFeatures
    ef = EarningsFeatures()
    sue, car3 = ef.get_latest_earnings_features('AAPL')
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
from pathlib import Path
import sys as _sys, os as _os
_sys.path.append(_os.path.abspath(_os.path.join(_os.path.dirname(__file__), '../../..')))
from src.app.database.pg_connection import get_conn, read_sql


class EarningsFeatures:
    """Calcul des features basées sur les earnings."""

    def __init__(self, db_path=None):  # db_path ignoré — connexion via pg_config.py
        self._spy_cache = None  # Cache des prix SPY

    def get_earnings_dates(self, symbol: str, limit: int = 12) -> pd.DataFrame:
        """
        Récupère les dates d'earnings et surprises via yfinance.

        Args:
            symbol: Symbole de l'action
            limit: Nombre de quarters à récupérer

        Returns:
            DataFrame avec colonnes: date, eps_estimate, eps_reported, surprise_pct
        """
        try:
            ticker = yf.Ticker(symbol)
            ed = ticker.earnings_dates

            if ed is None or ed.empty:
                return pd.DataFrame()

            # Nettoyer et renommer
            df = ed.reset_index()
            df.columns = ['date', 'eps_estimate', 'eps_reported', 'surprise_pct']

            # Filtrer les dates passées avec des résultats
            df = df[df['eps_reported'].notna()].head(limit)

            # Convertir la date en datetime naive (sans timezone)
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

            return df

        except Exception as e:
            print(f"[WARN] Impossible de récupérer les earnings pour {symbol}: {e}")
            return pd.DataFrame()

    def get_price_data(self, symbol: str, start_date: datetime,
                       end_date: datetime) -> pd.DataFrame:
        """Récupère les prix historiques depuis la BD ou yfinance."""
        try:
            # Essayer d'abord la BD
            query = """
                SELECT date, close FROM historical_data
                WHERE symbol = %s AND date BETWEEN %s AND %s
                ORDER BY date
            """
            df = read_sql(query, (
                symbol,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            ))

            if len(df) >= 3:
                df['date'] = pd.to_datetime(df['date'])
                return df

            # Sinon yfinance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date + timedelta(days=5))

            if hist.empty:
                return pd.DataFrame()

            df = hist[['Close']].reset_index()
            df.columns = ['date', 'close']
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

            return df

        except Exception as e:
            print(f"[WARN] Erreur prix {symbol}: {e}")
            return pd.DataFrame()

    def get_spy_prices(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Récupère les prix SPY (benchmark) avec cache."""
        # Utiliser le cache si disponible et couvre la période
        if self._spy_cache is not None:
            cache_start = self._spy_cache['date'].min()
            cache_end = self._spy_cache['date'].max()
            if cache_start <= start_date and cache_end >= end_date:
                mask = (self._spy_cache['date'] >= start_date) & \
                       (self._spy_cache['date'] <= end_date)
                return self._spy_cache[mask].copy()

        # Sinon télécharger
        df = self.get_price_data('SPY', start_date - timedelta(days=10),
                                 end_date + timedelta(days=10))

        if not df.empty:
            self._spy_cache = df.copy()

        return df

    def calc_car3(self, symbol: str, earnings_date: datetime) -> Optional[float]:
        """
        Calcule le CAR3 (Cumulative Abnormal Return 3 jours) autour d'une date d'earnings.

        CAR3 = Rendement_stock(J-1 à J+1) - Rendement_SPY(J-1 à J+1)

        Args:
            symbol: Symbole de l'action
            earnings_date: Date de l'annonce des résultats

        Returns:
            CAR3 en décimal (ex: 0.05 = +5%), ou None si impossible
        """
        try:
            # Récupérer les prix (J-5 à J+5 pour avoir de la marge)
            start = earnings_date - timedelta(days=10)
            end = earnings_date + timedelta(days=10)

            stock_prices = self.get_price_data(symbol, start, end)
            spy_prices = self.get_spy_prices(start, end)

            if stock_prices.empty or spy_prices.empty:
                return None

            # Trouver les jours de trading autour de earnings_date
            # J-1, J, J+1 (ou les jours de trading les plus proches)
            stock_prices = stock_prices.sort_values('date')
            spy_prices = spy_prices.sort_values('date')

            # Trouver J-1 (dernier jour avant ou égal à earnings_date - 1)
            j_minus_1_stock = stock_prices[stock_prices['date'] < earnings_date].tail(1)
            # Trouver J+1 (premier jour après earnings_date)
            j_plus_1_stock = stock_prices[stock_prices['date'] > earnings_date].head(1)

            if j_minus_1_stock.empty or j_plus_1_stock.empty:
                return None

            # Prix stock
            price_before = j_minus_1_stock['close'].values[0]
            price_after = j_plus_1_stock['close'].values[0]
            stock_return = (price_after - price_before) / price_before

            # Mêmes dates pour SPY
            date_before = j_minus_1_stock['date'].values[0]
            date_after = j_plus_1_stock['date'].values[0]

            spy_before = spy_prices[spy_prices['date'] <= pd.Timestamp(date_before)].tail(1)
            spy_after = spy_prices[spy_prices['date'] >= pd.Timestamp(date_after)].head(1)

            if spy_before.empty or spy_after.empty:
                return None

            spy_price_before = spy_before['close'].values[0]
            spy_price_after = spy_after['close'].values[0]
            spy_return = (spy_price_after - spy_price_before) / spy_price_before

            # CAR3 = rendement anormal
            car3 = stock_return - spy_return

            return car3

        except Exception as e:
            print(f"[WARN] Erreur CAR3 {symbol}: {e}")
            return None

    def get_latest_earnings_features(self, symbol: str) -> Dict[str, Optional[float]]:
        """
        Récupère les features earnings les plus récentes pour un symbole.

        Args:
            symbol: Symbole de l'action

        Returns:
            Dict avec:
                - sue: Surprise % du dernier quarter (standardized)
                - sue_avg4: Moyenne SUE des 4 derniers quarters
                - car3: CAR3 du dernier quarter
                - car3_avg4: Moyenne CAR3 des 4 derniers quarters
                - earnings_momentum: Tendance des surprises (positif = amélioration)
        """
        result = {
            'sue': None,
            'sue_avg4': None,
            'car3': None,
            'car3_avg4': None,
            'earnings_momentum': None
        }

        # Récupérer les earnings
        earnings_df = self.get_earnings_dates(symbol, limit=8)

        if earnings_df.empty:
            return result

        # SUE = Surprise%
        surprises = earnings_df['surprise_pct'].dropna()

        if len(surprises) >= 1:
            result['sue'] = surprises.iloc[0]

        if len(surprises) >= 4:
            result['sue_avg4'] = surprises.iloc[:4].mean()

            # Earnings momentum = différence entre les 2 derniers SUE
            result['earnings_momentum'] = surprises.iloc[0] - surprises.iloc[1]

        # CAR3 pour les quarters disponibles
        car3_values = []
        for _, row in earnings_df.head(4).iterrows():
            car3 = self.calc_car3(symbol, row['date'])
            if car3 is not None:
                car3_values.append(car3)

        if len(car3_values) >= 1:
            result['car3'] = car3_values[0]

        if len(car3_values) >= 4:
            result['car3_avg4'] = np.mean(car3_values)

        return result

    def get_earnings_features_batch(self, symbols: list) -> pd.DataFrame:
        """
        Récupère les features earnings pour une liste de symboles.

        Args:
            symbols: Liste de symboles

        Returns:
            DataFrame avec une ligne par symbole
        """
        rows = []

        for symbol in symbols:
            features = self.get_latest_earnings_features(symbol)
            features['symbol'] = symbol
            rows.append(features)

        return pd.DataFrame(rows)


def main():
    """Test des features earnings."""
    ef = EarningsFeatures()

    # Tester avec quelques symboles
    test_symbols = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'META']

    print("=" * 70)
    print("TEST EARNINGS FEATURES (SUE / CAR3)")
    print("=" * 70)

    for symbol in test_symbols:
        print(f"\n{symbol}:")

        # Earnings dates
        ed = ef.get_earnings_dates(symbol, limit=4)
        if not ed.empty:
            print(f"  Derniers earnings:")
            for _, row in ed.iterrows():
                print(f"    {row['date'].strftime('%Y-%m-%d')}: "
                      f"Est={row['eps_estimate']:.2f}, "
                      f"Real={row['eps_reported']:.2f}, "
                      f"Surprise={row['surprise_pct']:+.2f}%")

        # Features
        features = ef.get_latest_earnings_features(symbol)
        print(f"  Features:")
        print(f"    SUE (dernier): {features['sue']:.2f}%" if features['sue'] else "    SUE: N/A")
        print(f"    SUE (avg 4Q): {features['sue_avg4']:.2f}%" if features['sue_avg4'] else "    SUE avg: N/A")
        print(f"    CAR3 (dernier): {features['car3']*100:.2f}%" if features['car3'] else "    CAR3: N/A")
        print(f"    Earnings Mom: {features['earnings_momentum']:.2f}" if features['earnings_momentum'] else "    E-Mom: N/A")


if __name__ == "__main__":
    main()
