"""
Value Strategy - Stratégie d'investissement basée sur l'analyse fondamentale.
Sélectionne les actions sous-évaluées pour un rebalancement mensuel.
"""
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.ml.value_scoring import ValueScoring
from src.app.strategies.strategy import Strategy
from ibapi.order import Order
from typing import List, Optional


class ValueStrategy(Strategy):
    """
    Stratégie Value Investing pour rebalancement mensuel.

    Utilise les données Yahoo Finance pour calculer la fair value
    et sélectionner les actions les plus sous-évaluées.
    """
    name = "ValueStrategy"
    symbolsToAnalyse = []
    symbolsToTrade = []

    def __init__(self, capital: float = 100000, max_stocks: int = 10,
                 morningstar_csv: Optional[str] = None,
                 min_score: int = 60, min_margin_of_safety: float = 10.0):
        """
        Initialise la stratégie Value.

        Args:
            capital: Capital total dédié à la stratégie
            max_stocks: Nombre maximum de positions
            morningstar_csv: Chemin vers CSV avec étoiles Morningstar
            min_score: Score minimum pour sélection (0-100)
            min_margin_of_safety: Marge de sécurité minimum (%)
        """
        self.scoring = ValueScoring(morningstar_csv_path=morningstar_csv)
        self.capital = capital
        self.max_stocks = max_stocks
        self.min_score = min_score
        self.min_margin_of_safety = min_margin_of_safety
        self.analyses = {}

    def set_symbols_to_analyse(self, symbols: List[str]):
        """Définit la liste des symboles à analyser."""
        self.symbolsToAnalyse = [s.upper() for s in symbols]

    def get_symbols(self, trade_date: datetime = None) -> List[str]:
        """
        Retourne la liste des symboles à trader.

        Filtre par:
        - Score minimum
        - Marge de sécurité minimum
        - Limite au max_stocks meilleurs

        Args:
            trade_date: Date de trading (ignoré pour Value, utilise données temps réel)

        Returns:
            Liste des symboles sélectionnés
        """
        scored_symbols = []

        print(f"\n[VALUE] Analyse de {len(self.symbolsToAnalyse)} symboles...")

        for symbol in self.symbolsToAnalyse:
            try:
                analysis = self.scoring.get_full_analysis(symbol)

                if 'error' in analysis:
                    print(f"[VALUE] {symbol}: Erreur - {analysis['error']}")
                    continue

                score = analysis.get('score', 0)
                margin = analysis.get('average_margin_of_safety', -100)

                # Filtrer par critères
                if score < self.min_score:
                    print(f"[VALUE] {symbol}: Score {score} < {self.min_score} - ignoré")
                    continue

                if margin < self.min_margin_of_safety:
                    print(f"[VALUE] {symbol}: Marge {margin:.1f}% < {self.min_margin_of_safety}% - ignoré")
                    continue

                scored_symbols.append((symbol, score, margin, analysis))
                print(f"[VALUE] {symbol}: Score={score}, Marge={margin:+.1f}%, "
                      f"Signal={analysis.get('interpretation')}")

            except Exception as e:
                print(f"[VALUE] {symbol}: Exception - {e}")

        # Trier par score décroissant
        scored_symbols.sort(key=lambda x: x[1], reverse=True)

        # Garder les meilleurs
        selected = scored_symbols[:self.max_stocks]

        self.symbolsToTrade = [s[0] for s in selected]
        self.analyses = {s[0]: s[3] for s in selected}

        print(f"\n[VALUE] {len(self.symbolsToTrade)} symboles sélectionnés")

        return self.symbolsToTrade

    def get_order_params(self) -> List[dict]:
        """
        Génère les paramètres d'ordre pour le portefeuille Value.

        Pour une stratégie Value mensuelle, on utilise des ordres Market
        car le timing précis est moins important que l'évaluation.

        Returns:
            Liste de dicts avec paramètres d'ordres
        """
        if not self.symbolsToTrade:
            return []

        n = len(self.symbolsToTrade)
        capital_per_stock = self.capital / n

        order_params = []

        for symbol in self.symbolsToTrade:
            analysis = self.analyses.get(symbol, {})
            current_price = analysis.get('current_price')

            if not current_price:
                print(f"[VALUE] {symbol}: Prix non disponible - ignoré")
                continue

            qty = int(capital_per_stock / current_price)

            if qty <= 0:
                print(f"[VALUE] {symbol}: Quantité 0 - capital insuffisant")
                continue

            # Ordre Market (pour stratégie long terme)
            entry_order = Order()
            entry_order.action = "BUY"
            entry_order.orderType = "MKT"
            entry_order.totalQuantity = qty
            entry_order.tif = "DAY"
            entry_order.eTradeOnly = False
            entry_order.firmQuoteOnly = False

            order_params.append({
                'symbol': symbol,
                'entry_order': entry_order,
                'analysis': analysis,
            })

            print(f"[VALUE] Ordre préparé: {symbol} x{qty} @ ~${current_price:.2f}")

        return order_params

    def get_portfolio_summary(self) -> str:
        """
        Retourne un résumé du portefeuille Value.

        Returns:
            Résumé formaté
        """
        if not self.analyses:
            return "Aucune analyse disponible"

        lines = [
            "=" * 70,
            "PORTEFEUILLE VALUE - RÉSUMÉ",
            "=" * 70,
            f"Capital: ${self.capital:,.0f}",
            f"Positions: {len(self.symbolsToTrade)}/{self.max_stocks}",
            "-" * 70,
        ]

        total_margin = 0
        total_score = 0

        for symbol in self.symbolsToTrade:
            analysis = self.analyses.get(symbol, {})
            price = analysis.get('current_price', 0)
            score = analysis.get('score', 0)
            margin = analysis.get('average_margin_of_safety', 0)
            signal = analysis.get('interpretation', 'N/A')
            stars = analysis.get('morningstar_stars')

            stars_str = f"{'★' * stars}" if stars else "-"

            lines.append(f"{symbol:6} | ${price:8.2f} | Score: {score:3d} | "
                         f"Marge: {margin:+6.1f}% | {signal:12} | {stars_str}")

            total_margin += margin
            total_score += score

        avg_margin = total_margin / len(self.symbolsToTrade) if self.symbolsToTrade else 0
        avg_score = total_score / len(self.symbolsToTrade) if self.symbolsToTrade else 0

        lines.extend([
            "-" * 70,
            f"Marge moyenne: {avg_margin:+.1f}%",
            f"Score moyen: {avg_score:.0f}/100",
            "=" * 70,
        ])

        return "\n".join(lines)

    def get_rebalance_recommendations(self, current_positions: dict) -> dict:
        """
        Compare le portefeuille actuel avec les recommandations et suggère des ajustements.

        Args:
            current_positions: Dict {symbol: quantity} des positions actuelles

        Returns:
            Dict avec recommendations: {'buy': [...], 'sell': [...], 'hold': [...]}
        """
        recommendations = {
            'buy': [],
            'sell': [],
            'hold': [],
        }

        current_symbols = set(current_positions.keys())
        target_symbols = set(self.symbolsToTrade)

        # Symboles à acheter (dans target mais pas dans current)
        to_buy = target_symbols - current_symbols
        for symbol in to_buy:
            analysis = self.analyses.get(symbol, {})
            recommendations['buy'].append({
                'symbol': symbol,
                'score': analysis.get('score'),
                'margin': analysis.get('average_margin_of_safety'),
                'reason': 'Nouvelle opportunité value'
            })

        # Symboles à vendre (dans current mais plus dans target)
        to_sell = current_symbols - target_symbols
        for symbol in to_sell:
            recommendations['sell'].append({
                'symbol': symbol,
                'reason': 'Ne répond plus aux critères value'
            })

        # Symboles à conserver
        to_hold = current_symbols & target_symbols
        for symbol in to_hold:
            analysis = self.analyses.get(symbol, {})
            recommendations['hold'].append({
                'symbol': symbol,
                'score': analysis.get('score'),
                'margin': analysis.get('average_margin_of_safety'),
            })

        return recommendations


# Test standalone
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TEST ValueStrategy")
    print("=" * 70)

    # Liste de symboles à tester
    test_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JNJ', 'JPM', 'XOM', 'PG',
        'KO', 'PEP', 'WMT', 'HD', 'V', 'MA', 'UNH', 'ABBV'
    ]

    strategy = ValueStrategy(
        capital=100000,
        max_stocks=10,
        min_score=50,  # Score minimum bas pour le test
        min_margin_of_safety=-20,  # Marge négative ok pour le test
    )

    strategy.set_symbols_to_analyse(test_symbols)
    selected = strategy.get_symbols()

    print("\n" + strategy.get_portfolio_summary())
