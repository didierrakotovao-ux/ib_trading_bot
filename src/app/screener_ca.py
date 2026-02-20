"""
Screener quotidien pour actions canadiennes (Disnat).
Charge les données depuis trading_data_ca.db et applique le pipeline momentum ML.

Usage:
    python src/app/screener_ca.py                        # Défaut: top 10, smooth_ml
    python src/app/screener_ca.py --top 15               # Top 15
    python src/app/screener_ca.py --scoring ml            # Modèle basique
    python src/app/screener_ca.py --min-volume 50000      # Ajuster volume min
    python src/app/screener_ca.py --date 2026-02-05       # Date spécifique (backtest)
"""
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.app.strategies.momentum_filters import MomentumFilters
from src.app.ml.ml_smooth_scoring import SmoothMLScoring
from src.app.ml.ml_scoring import MLScoring


class CanadianScreener:
    """Screener quotidien pour actions canadiennes."""

    def __init__(self, db_path: str = "trading_data_ca.db",
                 scoring_type: str = "smooth_ml",
                 min_price: float = 2.0,
                 min_volume: int = 100000,
                 top_n: int = 10):
        self.db_path = self._resolve_db_path(db_path)
        self.scoring_type = scoring_type
        self.min_price = min_price
        self.min_volume = min_volume
        self.top_n = top_n

        # Pipeline stats
        self.stats = {}

        # Charger le scorer ML
        self.scorer = self._init_scorer()

    def _resolve_db_path(self, db_path: str) -> str:
        """Résout le chemin de la DB par rapport à la racine du projet."""
        if os.path.isabs(db_path):
            return db_path
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        return os.path.join(project_root, db_path)

    def _init_scorer(self):
        """Initialise le scorer ML selon le type choisi."""
        if self.scoring_type == "smooth_ml":
            return SmoothMLScoring(
                model_path="models/smooth_momentum_model.pkl",
                db_path=os.path.basename(self.db_path)
            )
        elif self.scoring_type == "ml":
            return MLScoring(model_path="models/momentum_model.pkl")
        else:
            raise ValueError(f"Type de scoring inconnu: {self.scoring_type}")

    def _load_all_data(self, trade_date: datetime) -> dict:
        """
        Charge toutes les données depuis la DB en bulk.

        Args:
            trade_date: Date du screener

        Returns:
            Dict {symbol: DataFrame} avec les données OHLCV
        """
        conn = sqlite3.connect(self.db_path)

        # Charger 400 jours calendaires (~252 jours de trading + marge)
        data_start = (trade_date - timedelta(days=400)).strftime('%Y-%m-%d')
        data_end = trade_date.strftime('%Y-%m-%d')

        # Requête 1: Symboles normaux (filtre volume)
        query_normal = """
            SELECT symbol, date, open, high, low, close, volume
            FROM historical_data
            WHERE date BETWEEN ? AND ?
            AND close >= ?
            AND symbol NOT LIKE '%.NE'
            AND volume > 0
            ORDER BY symbol, date
        """
        df_normal = pd.read_sql_query(query_normal, conn,
                                       params=(data_start, data_end, self.min_price))

        # Requête 2: CDR (.NE) - pas de filtre volume (liquides par design)
        query_cdr = """
            SELECT symbol, date, open, high, low, close, volume
            FROM historical_data
            WHERE date BETWEEN ? AND ?
            AND close >= ?
            AND symbol LIKE '%.NE'
            AND volume > 0
            ORDER BY symbol, date
        """
        df_cdr = pd.read_sql_query(query_cdr, conn,
                                    params=(data_start, data_end, self.min_price))

        conn.close()

        # Combiner
        df_all = pd.concat([df_normal, df_cdr], ignore_index=True)

        if df_all.empty:
            print("[WARN] Aucune donnée trouvée en base.")
            return {}

        # Filtrer par volume moyen 20j pour les symboles normaux
        dataframes = {}
        for symbol, group in df_all.groupby('symbol'):
            group = group.copy()
            group['date'] = pd.to_datetime(group['date'])
            group = group.sort_values('date').reset_index(drop=True)

            # Filtre volume: CDR exemptés, les autres doivent avoir volume moyen >= min_volume
            if not symbol.endswith('.NE'):
                vol_avg = group['volume'].tail(20).mean()
                if vol_avg < self.min_volume:
                    continue

            # Minimum 252 barres pour le momentum 12-1
            if len(group) < 252:
                continue

            dataframes[symbol] = group

        self.stats['total_loaded'] = df_all['symbol'].nunique()
        self.stats['after_filters'] = len(dataframes)

        print(f"[DATA] {self.stats['total_loaded']} symboles chargés, "
              f"{len(dataframes)} après filtres (price >= ${self.min_price}, "
              f"vol >= {self.min_volume:,}, >= 252 barres)")

        return dataframes

    def _filter_momentum(self, dataframes: dict) -> list:
        """
        Étape 1: Calcule le momentum 12-1 et garde le top 30%.

        Returns:
            Liste de tuples (symbol, momentum, df)
        """
        candidates = []
        for symbol, df in dataframes.items():
            mom = MomentumFilters.calc_momentum_12_1(df)
            if not np.isnan(mom) and mom > 0:  # Momentum positif seulement
                candidates.append((symbol, mom, df))

        # Top 30%
        filtered = MomentumFilters.filter_top_momentum(candidates, top_pct=0.3)

        self.stats['after_momentum'] = len(filtered)
        print(f"[MOM]  {len(candidates)} momentum positif -> {len(filtered)} (top 30%)")

        return filtered

    def _filter_fip(self, candidates: list) -> list:
        """
        Étape 2: Filtre FIP (Frog-in-the-Pan). Garde FIP < 0.

        Returns:
            Liste de tuples (symbol, momentum, fip, df)
        """
        filtered = []
        for symbol, mom, df in candidates:
            fip = MomentumFilters.calc_fip(df)
            if not np.isnan(fip) and fip < 0:
                filtered.append((symbol, mom, fip, df))

        self.stats['after_fip'] = len(filtered)
        print(f"[FIP]  {len(filtered)} avec FIP négatif (momentum smooth)")

        return filtered

    def _score_candidates(self, candidates: list, trade_date: datetime) -> list:
        """
        Étape 3: Score ML pour chaque candidat.

        Returns:
            Liste de tuples (symbol, score, momentum, fip, close, vol20d)
        """
        month = trade_date.month
        threshold = MomentumFilters.get_score_threshold(month)

        scored = []
        for symbol, mom, fip, df in candidates:
            if self.scoring_type == "smooth_ml":
                score = self.scorer.score(df, symbol=symbol)
            else:
                score = self.scorer.score(df)

            if score >= threshold:
                close = df['close'].iloc[-1]
                vol20d = df['volume'].tail(20).mean()
                scored.append((symbol, score, mom, fip, close, vol20d))

        self.stats['after_score'] = len(scored)
        self.stats['threshold'] = threshold
        print(f"[ML]   {len(scored)} avec score >= {threshold} (seuil mois {month})")

        return scored

    def run(self, trade_date: datetime = None) -> pd.DataFrame:
        """
        Exécute le pipeline complet du screener.

        Args:
            trade_date: Date du screener (défaut: aujourd'hui)

        Returns:
            DataFrame avec les recommandations triées par score
        """
        if trade_date is None:
            trade_date = datetime.now()

        print(f"\n{'='*60}")
        print(f"SCREENER CANADIEN - {trade_date.strftime('%Y-%m-%d')}")
        print(f"Modèle: {self.scoring_type} | Top {self.top_n}")
        print(f"{'='*60}\n")

        # Pipeline
        dataframes = self._load_all_data(trade_date)
        if not dataframes:
            print("[STOP] Aucune donnée disponible.")
            return pd.DataFrame()

        momentum_candidates = self._filter_momentum(dataframes)
        if not momentum_candidates:
            print("[STOP] Aucun candidat momentum.")
            return pd.DataFrame()

        fip_candidates = self._filter_fip(momentum_candidates)
        if not fip_candidates:
            print("[STOP] Aucun candidat FIP.")
            return pd.DataFrame()

        scored = self._score_candidates(fip_candidates, trade_date)
        if not scored:
            print("[STOP] Aucun candidat au-dessus du seuil ML.")
            return pd.DataFrame()

        # Trier par score décroissant et prendre le top N
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:self.top_n]

        # Construire le DataFrame résultat
        results = pd.DataFrame(top, columns=[
            'symbol', 'score', 'momentum_12_1', 'fip', 'close', 'vol_20d'
        ])
        results['rank'] = range(1, len(results) + 1)

        # Afficher les résultats
        self._display_results(results, trade_date)

        # Sauvegarder en CSV
        csv_path = self._save_csv(results, trade_date)

        # Résumé du pipeline
        print(f"\n{'='*60}")
        print(f"Pipeline: {self.stats.get('after_filters', 0)} -> "
              f"{self.stats.get('after_momentum', 0)} (mom) -> "
              f"{self.stats.get('after_fip', 0)} (fip) -> "
              f"{self.stats.get('after_score', 0)} (score>={self.stats.get('threshold', 0)}) -> "
              f"{len(top)} (top)")
        print(f"CSV: {csv_path}")
        print(f"{'='*60}\n")

        return results

    def _display_results(self, results: pd.DataFrame, trade_date: datetime):
        """Affiche les résultats en format tableau."""
        threshold = self.stats.get('threshold', 0)
        print(f"\n{'='*60}")
        print(f"Seuil: {threshold} | Top {self.top_n}")
        print(f"{'='*60}")
        print(f"{'Rk':>3}  {'Symbole':<12} {'Score':>5}  {'Mom12-1':>8}  "
              f"{'FIP':>7}  {'Close':>8}  {'Vol20d':>8}")
        print(f"{'-'*60}")

        for _, row in results.iterrows():
            vol_str = self._format_volume(row['vol_20d'])
            print(f"{row['rank']:>3}  {row['symbol']:<12} {row['score']:>5}  "
                  f"{row['momentum_12_1']:>+7.1%}  "
                  f"{row['fip']:>7.3f}  "
                  f"${row['close']:>7.2f}  {vol_str:>8}")

    @staticmethod
    def _format_volume(vol: float) -> str:
        """Formate le volume en K/M."""
        if vol >= 1_000_000:
            return f"{vol/1_000_000:.1f}M"
        elif vol >= 1_000:
            return f"{vol/1_000:.0f}K"
        return str(int(vol))

    def _save_csv(self, results: pd.DataFrame, trade_date: datetime) -> str:
        """Sauvegarde les résultats en CSV."""
        date_str = trade_date.strftime('%Y-%m-%d')
        filename = f"screener_ca_{date_str}.csv"
        results.to_csv(filename, index=False, float_format='%.4f')
        return filename


def main():
    parser = argparse.ArgumentParser(
        description="Screener quotidien pour actions canadiennes (Disnat)"
    )
    parser.add_argument('--top', type=int, default=10,
                        help='Nombre de recommandations (default: 10)')
    parser.add_argument('--scoring', choices=['smooth_ml', 'ml'], default='smooth_ml',
                        help='Type de scoring ML (default: smooth_ml)')
    parser.add_argument('--min-volume', type=int, default=100000,
                        help='Volume moyen minimum (default: 100000)')
    parser.add_argument('--min-price', type=float, default=2.0,
                        help='Prix minimum (default: 2.0)')
    parser.add_argument('--date', type=str, default=None,
                        help='Date spécifique YYYY-MM-DD (default: aujourd\'hui)')
    parser.add_argument('--db', default='trading_data_ca.db',
                        help='Chemin de la base de données (default: trading_data_ca.db)')

    args = parser.parse_args()

    trade_date = None
    if args.date:
        trade_date = datetime.strptime(args.date, '%Y-%m-%d')

    screener = CanadianScreener(
        db_path=args.db,
        scoring_type=args.scoring,
        min_volume=args.min_volume,
        min_price=args.min_price,
        top_n=args.top,
    )
    screener.run(trade_date=trade_date)


if __name__ == "__main__":
    main()
