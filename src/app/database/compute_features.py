"""
Script pour calculer et sauvegarder les features ML dans la base de données.
À exécuter après import_yahoo_bulk.py ou update_historical_data.py

Usage:
    python compute_features.py              # Calcule pour tous les symboles
    python compute_features.py AAPL MSFT    # Calcule pour des symboles spécifiques
    python compute_features.py --incremental # Calcule seulement les nouvelles dates
"""
import sys
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.database.db_manager import TradingDatabase


class FeatureComputer:
    """Calcule les features ML et les sauvegarde en base de données."""

    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = db_path
        self.db = TradingDatabase(db_path)

    def _calc_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcule le RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _calc_macd(self, prices: pd.Series) -> tuple:
        """Calcule MACD, Signal et Histogramme."""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        hist = macd - signal
        return macd, signal, hist

    def _calc_adx(self, high: pd.Series, low: pd.Series, close: pd.Series,
                  period: int = 14) -> pd.Series:
        """Calcule l'ADX simplifié."""
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()

        # Direction
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
        minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

        plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=period).mean()

        return adx

    def _calc_bollinger(self, prices: pd.Series, period: int = 20) -> tuple:
        """Calcule les bandes de Bollinger."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        return upper, lower

    def _calc_smoothness(self, prices: pd.Series, window: int) -> pd.Series:
        """Calcule le R² sur une fenêtre glissante (smoothness)."""
        x = np.arange(window, dtype=np.float64)
        sum_x = x.sum()
        sum_x2 = (x ** 2).sum()
        n = window
        denom_x = n * sum_x2 - sum_x ** 2

        def calc_r2(y):
            if len(y) < window or np.isnan(y).any():
                return np.nan
            sum_y = y.sum()
            sum_y2 = (y ** 2).sum()
            sum_xy = (x * y).sum()
            denom_y = n * sum_y2 - sum_y ** 2
            if denom_y <= 0 or denom_x <= 0:
                return np.nan
            r = (n * sum_xy - sum_x * sum_y) / (np.sqrt(denom_x * denom_y) + 1e-10)
            return r ** 2

        return prices.rolling(window=window).apply(calc_r2, raw=True)

    def _calc_momentum_12_1(self, close: pd.Series) -> pd.Series:
        """Calcule le momentum 12-1 mois (Gray & Vogel)."""
        # Rendement entre M-12 et M-1 (~252 - 21 jours)
        price_12m = close.shift(252)
        price_1m = close.shift(21)
        return (price_1m - price_12m) / (price_12m + 1e-10)

    def _calc_fip(self, close: pd.Series, flat_threshold: float = 0.005) -> pd.Series:
        """Calcule le Frog-in-the-Pan (Gray & Vogel)."""
        def calc_fip_window(window_data):
            if len(window_data) < 252:
                return np.nan
            daily_returns = np.diff(window_data) / window_data[:-1]
            n_days = len(daily_returns)
            pos_days = np.sum(daily_returns > flat_threshold)
            neg_days = np.sum(daily_returns < -flat_threshold)
            flat_days = n_days - pos_days - neg_days
            pct_pos = pos_days / n_days
            pct_neg = neg_days / n_days
            pct_flat = flat_days / n_days
            return pct_flat * (pct_neg - pct_pos)

        return close.rolling(window=252).apply(calc_fip_window, raw=True)

    def compute_features_for_symbol(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule toutes les features pour un symbole.

        Args:
            symbol: Symbole de l'action
            df: DataFrame avec OHLCV

        Returns:
            DataFrame avec les features calculées
        """
        if len(df) < 260:  # Besoin de ~252 jours pour les features
            return pd.DataFrame()

        df = df.copy()
        df.columns = df.columns.str.lower()

        # S'assurer que les colonnes existent
        required = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(c in df.columns for c in required):
            print(f"[WARN] {symbol}: Colonnes manquantes")
            return pd.DataFrame()

        # --- Calcul des features ---

        # SMA20 Volume
        df['sma20_volume'] = df['volume'].rolling(20).mean()

        # HL et OC normalisés par volume
        df['hl_sma20vol'] = (df['high'] - df['low']) / (df['sma20_volume'] + 1e-10)
        df['oc_sma20vol'] = (df['open'] - df['close']) / (df['sma20_volume'] + 1e-10)

        # RSI
        df['rsi'] = self._calc_rsi(df['close'])

        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calc_macd(df['close'])

        # ADX
        df['adx'] = self._calc_adx(df['high'], df['low'], df['close'])

        # Bollinger Bands
        df['bb_high'], df['bb_low'] = self._calc_bollinger(df['close'])
        bb_range = df['bb_high'] - df['bb_low']
        df['bb_position'] = np.where(
            bb_range > 0,
            (df['close'] - df['bb_low']) / bb_range,
            0.5
        )

        # RSI momentum
        df['rsi_momentum'] = df['rsi'] - 50

        # Volume ratio
        df['volume_ratio'] = df['volume'] / (df['sma20_volume'] + 1e-10)

        # Trend strength
        df['trend_strength'] = df['adx'] * np.sign(df['macd'])

        # Returns
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        df['return_20d'] = df['close'].pct_change(20)
        df['pct_close'] = df['close'].pct_change()

        # Volatility
        df['volatility_10d'] = df['close'].pct_change().rolling(10).std()

        # Position vs 52-week high
        high_52w = df['high'].rolling(252, min_periods=50).max()
        df['high_52w_pct'] = df['close'] / (high_52w + 1e-10)

        # Smoothness (R²)
        df['smoothness_20d'] = self._calc_smoothness(df['close'], 20)
        df['smoothness_50d'] = self._calc_smoothness(df['close'], 50)

        # Seasonality (month encoding)
        df['date'] = pd.to_datetime(df['date'])
        month = df['date'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)

        # Gray & Vogel filters
        df['momentum_12_1'] = self._calc_momentum_12_1(df['close'])
        df['fip'] = self._calc_fip(df['close'])

        # Ajouter le symbole
        df['symbol'] = symbol

        # Sélectionner les colonnes pertinentes
        feature_columns = [
            'symbol', 'date', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'adx',
            'bb_high', 'bb_low', 'bb_position', 'rsi_momentum',
            'volume_ratio', 'trend_strength', 'return_5d', 'return_10d',
            'return_20d', 'volatility_10d', 'high_52w_pct', 'hl_sma20vol',
            'oc_sma20vol', 'pct_close', 'smoothness_20d', 'smoothness_50d',
            'month_sin', 'month_cos', 'momentum_12_1', 'fip'
        ]

        result = df[feature_columns].copy()
        result['date'] = result['date'].dt.strftime('%Y-%m-%d')

        # Supprimer les lignes avec trop de NaN
        result = result.dropna(subset=['rsi', 'macd', 'momentum_12_1'])

        return result

    def compute_all(self, symbols: List[str] = None, incremental: bool = False):
        """
        Calcule les features pour tous les symboles.

        Args:
            symbols: Liste de symboles (optionnel, sinon tous)
            incremental: Si True, calcule seulement les nouvelles dates
        """
        if symbols is None:
            symbols = self.db.get_all_symbols()

        print(f"[COMPUTE] Calcul des features pour {len(symbols)} symboles")

        total_rows = 0
        errors = 0

        for i, symbol in enumerate(symbols):
            try:
                # Récupérer les données historiques
                df = self.db.get_historical_data(symbol)

                if df is None or len(df) < 260:
                    continue

                # Si incremental, filtrer les dates déjà calculées
                if incremental:
                    latest_date = self.db.get_latest_features_date(symbol)
                    if latest_date:
                        df = df[df['date'] > latest_date]
                        if len(df) < 260:
                            continue

                # Calculer les features
                features_df = self.compute_features_for_symbol(symbol, df)

                if len(features_df) > 0:
                    # Sauvegarder en base
                    conn = sqlite3.connect(self.db_path)
                    features_df.to_sql('computed_features', conn,
                                       if_exists='append', index=False)
                    conn.commit()
                    conn.close()
                    total_rows += len(features_df)

                if (i + 1) % 50 == 0:
                    print(f"[COMPUTE] {i + 1}/{len(symbols)} symboles traités, "
                          f"{total_rows} lignes insérées")

            except Exception as e:
                errors += 1
                if errors < 10:
                    print(f"[ERROR] {symbol}: {e}")

        print(f"[COMPUTE] Terminé: {total_rows} lignes insérées, {errors} erreurs")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Calcul des features ML")
    parser.add_argument('symbols', nargs='*', help='Symboles spécifiques (optionnel)')
    parser.add_argument('--incremental', '-i', action='store_true',
                        help='Calcule seulement les nouvelles dates')
    parser.add_argument('--db', default='trading_data.db', help='Chemin BD')

    args = parser.parse_args()

    computer = FeatureComputer(args.db)

    symbols = args.symbols if args.symbols else None
    computer.compute_all(symbols=symbols, incremental=args.incremental)


if __name__ == "__main__":
    main()
