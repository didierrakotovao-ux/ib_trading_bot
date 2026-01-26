"""
Value Scoring - Évaluation fondamentale basée sur les données Yahoo Finance.
Calcule plusieurs estimations de fair value et compare avec le prix actuel.
Supporte l'import optionnel des étoiles Morningstar depuis un fichier CSV.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, Dict, List
from pathlib import Path
import csv
from datetime import datetime


class ValueScoring:
    """
    Scoring basé sur l'analyse fondamentale (Value Investing).
    Utilise les données gratuites de Yahoo Finance pour calculer la fair value.
    """
    name = "ValueScoring"

    # Taux sans risque et prime de risque pour DCF
    RISK_FREE_RATE = 0.045  # 4.5% (taux obligataire ~2024-2025)
    EQUITY_RISK_PREMIUM = 0.05  # 5%
    TERMINAL_GROWTH_RATE = 0.025  # 2.5% croissance perpétuelle

    def __init__(self, morningstar_csv_path: Optional[str] = None):
        """
        Initialise le scoring Value.

        Args:
            morningstar_csv_path: Chemin optionnel vers CSV avec étoiles Morningstar
                                  Format: symbol,stars (ex: AAPL,5)
        """
        self.morningstar_stars = {}
        self.cache = {}  # Cache pour les données Yahoo Finance

        if morningstar_csv_path:
            self.load_morningstar_csv(morningstar_csv_path)

    def load_morningstar_csv(self, filepath: str) -> int:
        """
        Charge les étoiles Morningstar depuis un fichier CSV.

        Format CSV attendu:
        symbol,stars
        AAPL,5
        MSFT,4
        ...

        Args:
            filepath: Chemin vers le fichier CSV

        Returns:
            Nombre de symboles chargés
        """
        try:
            path = Path(filepath)
            if not path.exists():
                print(f"[WARN] Fichier Morningstar non trouvé: {filepath}")
                return 0

            with open(path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    symbol = row.get('symbol', '').upper().strip()
                    stars = row.get('stars', '3')
                    if symbol:
                        self.morningstar_stars[symbol] = int(stars)

            print(f"[OK] {len(self.morningstar_stars)} étoiles Morningstar chargées")
            return len(self.morningstar_stars)

        except Exception as e:
            print(f"[ERROR] Erreur chargement Morningstar CSV: {e}")
            return 0

    def get_yahoo_data(self, symbol: str, use_cache: bool = True) -> Optional[Dict]:
        """
        Récupère les données fondamentales depuis Yahoo Finance.

        Args:
            symbol: Symbole du stock
            use_cache: Utiliser le cache (recommandé pour éviter rate limiting)

        Returns:
            Dictionnaire avec les données fondamentales ou None
        """
        if use_cache and symbol in self.cache:
            return self.cache[symbol]

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Extraire les données clés
            data = {
                'symbol': symbol,
                'current_price': info.get('currentPrice') or info.get('regularMarketPrice'),
                'trailing_eps': info.get('trailingEps'),
                'forward_eps': info.get('forwardEps'),
                'trailing_pe': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'book_value': info.get('bookValue'),
                'price_to_book': info.get('priceToBook'),
                'dividend_yield': info.get('dividendYield'),
                'profit_margins': info.get('profitMargins'),
                'return_on_equity': info.get('returnOnEquity'),
                'debt_to_equity': info.get('debtToEquity'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'free_cashflow': info.get('freeCashflow'),
                'operating_cashflow': info.get('operatingCashflow'),
                'total_cash': info.get('totalCash'),
                'total_debt': info.get('totalDebt'),
                'shares_outstanding': info.get('sharesOutstanding'),
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'target_mean_price': info.get('targetMeanPrice'),
                'target_high_price': info.get('targetHighPrice'),
                'target_low_price': info.get('targetLowPrice'),
                'recommendation_mean': info.get('recommendationMean'),
                'recommendation_key': info.get('recommendationKey'),
                'number_of_analyst_opinions': info.get('numberOfAnalystOpinions'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'company_name': info.get('shortName') or info.get('longName'),
            }

            # Nettoyer les None et NaN
            data = {k: v for k, v in data.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}

            if use_cache:
                self.cache[symbol] = data

            return data

        except Exception as e:
            print(f"[ERROR] Erreur récupération Yahoo Finance pour {symbol}: {e}")
            return None

    def calculate_graham_number(self, data: Dict) -> Optional[float]:
        """
        Calcule le Graham Number (formule de Benjamin Graham).

        Formula: √(22.5 × EPS × Book Value)

        Le Graham Number représente le prix maximum qu'un investisseur
        défensif devrait payer pour une action.

        Args:
            data: Données Yahoo Finance

        Returns:
            Graham Number ou None si données insuffisantes
        """
        eps = data.get('trailing_eps')
        book_value = data.get('book_value')

        if eps is None or book_value is None:
            return None
        if eps <= 0 or book_value <= 0:
            return None

        graham = np.sqrt(22.5 * eps * book_value)
        return round(graham, 2)

    def calculate_dcf_simple(self, data: Dict, years: int = 5) -> Optional[float]:
        """
        Calcule une fair value simplifiée par DCF (Discounted Cash Flow).

        Utilise le Free Cash Flow et un taux de croissance estimé.

        Args:
            data: Données Yahoo Finance
            years: Nombre d'années de projection

        Returns:
            Fair value DCF par action ou None
        """
        fcf = data.get('free_cashflow')
        shares = data.get('shares_outstanding')
        growth = data.get('earnings_growth') or data.get('revenue_growth') or 0.05

        if fcf is None or shares is None or shares == 0:
            return None
        if fcf <= 0:
            return None

        # Limiter le taux de croissance à des valeurs raisonnables
        growth = max(0, min(growth, 0.25))  # Cap à 25%

        # Taux d'actualisation (WACC simplifié)
        discount_rate = self.RISK_FREE_RATE + self.EQUITY_RISK_PREMIUM

        # Projeter les FCF futurs
        total_pv = 0
        projected_fcf = fcf
        for year in range(1, years + 1):
            projected_fcf *= (1 + growth)
            pv = projected_fcf / ((1 + discount_rate) ** year)
            total_pv += pv

        # Valeur terminale (Gordon Growth Model)
        terminal_fcf = projected_fcf * (1 + self.TERMINAL_GROWTH_RATE)
        terminal_value = terminal_fcf / (discount_rate - self.TERMINAL_GROWTH_RATE)
        terminal_pv = terminal_value / ((1 + discount_rate) ** years)

        # Valeur totale
        enterprise_value = total_pv + terminal_pv

        # Ajuster pour la dette et le cash
        total_debt = data.get('total_debt', 0) or 0
        total_cash = data.get('total_cash', 0) or 0
        equity_value = enterprise_value - total_debt + total_cash

        fair_value_per_share = equity_value / shares
        return round(fair_value_per_share, 2)

    def calculate_peg_fair_value(self, data: Dict) -> Optional[float]:
        """
        Calcule la fair value basée sur le ratio PEG.

        Un PEG de 1 est considéré comme juste valeur.
        Fair Value = EPS × Growth Rate × 100

        Args:
            data: Données Yahoo Finance

        Returns:
            Fair value PEG ou None
        """
        eps = data.get('trailing_eps')
        growth = data.get('earnings_growth')

        if eps is None or growth is None:
            return None
        if eps <= 0 or growth <= 0:
            return None

        # PEG = 1 implique P/E = Growth × 100
        # Donc Fair Value = EPS × Growth × 100
        fair_pe = growth * 100
        fair_value = eps * fair_pe

        return round(fair_value, 2)

    def calculate_relative_pe_value(self, data: Dict, sector_pe: float = 20) -> Optional[float]:
        """
        Calcule la fair value basée sur un P/E de référence.

        Args:
            data: Données Yahoo Finance
            sector_pe: P/E moyen du secteur (défaut 20)

        Returns:
            Fair value relative ou None
        """
        eps = data.get('trailing_eps')

        if eps is None or eps <= 0:
            return None

        fair_value = eps * sector_pe
        return round(fair_value, 2)

    def get_analyst_fair_value(self, data: Dict) -> Optional[float]:
        """
        Retourne la fair value moyenne des analystes (target price).

        Args:
            data: Données Yahoo Finance

        Returns:
            Target price moyen ou None
        """
        return data.get('target_mean_price')

    def calculate_margin_of_safety(self, current_price: float, fair_value: float) -> float:
        """
        Calcule la marge de sécurité.

        Margin of Safety = (Fair Value - Current Price) / Fair Value × 100

        Positif = sous-évalué (opportunité d'achat)
        Négatif = surévalué

        Args:
            current_price: Prix actuel
            fair_value: Fair value estimée

        Returns:
            Marge de sécurité en pourcentage
        """
        if fair_value <= 0:
            return 0
        return round((fair_value - current_price) / fair_value * 100, 1)

    def score(self, symbol: str) -> int:
        """
        Calcule un score Value de 0 à 100 pour un symbole.

        Le score est basé sur:
        - Marge de sécurité moyenne des différentes méthodes
        - Qualité fondamentale (ROE, dette, croissance)
        - Étoiles Morningstar (si disponibles)

        Args:
            symbol: Symbole du stock

        Returns:
            Score de 0 à 100
        """
        analysis = self.get_full_analysis(symbol)

        if 'error' in analysis:
            return 0

        score = 50  # Base neutre

        # --- 1. Marge de sécurité (max +/- 30 points) ---
        avg_margin = analysis.get('average_margin_of_safety', 0)
        margin_contribution = min(max(avg_margin, -30), 30)
        score += margin_contribution

        # --- 2. Qualité fondamentale (max +20 points) ---
        quality = analysis.get('quality_metrics', {})

        # ROE > 15% = bon
        roe = quality.get('return_on_equity', 0) or 0
        if roe > 0.20:
            score += 10
        elif roe > 0.15:
            score += 5
        elif roe < 0.05:
            score -= 5

        # Debt/Equity < 1 = bon
        de_ratio = quality.get('debt_to_equity', 100) or 100
        if de_ratio < 0.5:
            score += 5
        elif de_ratio < 1.0:
            score += 2
        elif de_ratio > 2.0:
            score -= 5

        # Croissance positive
        growth = quality.get('earnings_growth', 0) or 0
        if growth > 0.15:
            score += 5
        elif growth > 0.05:
            score += 2
        elif growth < 0:
            score -= 5

        # --- 3. Étoiles Morningstar (max +15 points) ---
        stars = self.morningstar_stars.get(symbol.upper(), 3)
        star_contribution = (stars - 3) * 5  # -10 à +10
        score += star_contribution

        # --- 4. Consensus analystes (max +5 points) ---
        recommendation = analysis.get('analyst_recommendation', 3)
        if recommendation <= 2:  # Strong Buy / Buy
            score += 5
        elif recommendation >= 4:  # Sell / Strong Sell
            score -= 5

        return max(0, min(100, int(score)))

    def get_full_analysis(self, symbol: str) -> Dict:
        """
        Retourne une analyse complète pour un symbole.

        Args:
            symbol: Symbole du stock

        Returns:
            Dictionnaire avec toutes les analyses
        """
        data = self.get_yahoo_data(symbol)

        if data is None:
            return {'error': f'Impossible de récupérer les données pour {symbol}'}

        current_price = data.get('current_price')
        if current_price is None:
            return {'error': f'Prix actuel non disponible pour {symbol}'}

        # Calculer les fair values
        fair_values = {}
        margins = {}

        # Graham Number
        graham = self.calculate_graham_number(data)
        if graham:
            fair_values['graham_number'] = graham
            margins['graham_number'] = self.calculate_margin_of_safety(current_price, graham)

        # DCF Simple
        dcf = self.calculate_dcf_simple(data)
        if dcf:
            fair_values['dcf_simple'] = dcf
            margins['dcf_simple'] = self.calculate_margin_of_safety(current_price, dcf)

        # PEG Fair Value
        peg = self.calculate_peg_fair_value(data)
        if peg:
            fair_values['peg_value'] = peg
            margins['peg_value'] = self.calculate_margin_of_safety(current_price, peg)

        # Relative P/E (P/E = 20)
        rel_pe = self.calculate_relative_pe_value(data, sector_pe=20)
        if rel_pe:
            fair_values['relative_pe_20'] = rel_pe
            margins['relative_pe_20'] = self.calculate_margin_of_safety(current_price, rel_pe)

        # Analyst Target
        analyst = self.get_analyst_fair_value(data)
        if analyst:
            fair_values['analyst_target'] = analyst
            margins['analyst_target'] = self.calculate_margin_of_safety(current_price, analyst)

        # Moyenne des marges de sécurité
        avg_margin = np.mean(list(margins.values())) if margins else 0

        # Interprétation
        if avg_margin >= 20:
            interpretation = "STRONG_BUY"
            interpretation_fr = "Fortement sous-évalué - Excellente opportunité"
        elif avg_margin >= 10:
            interpretation = "BUY"
            interpretation_fr = "Sous-évalué - Bonne opportunité"
        elif avg_margin >= -10:
            interpretation = "HOLD"
            interpretation_fr = "Correctement évalué"
        elif avg_margin >= -20:
            interpretation = "CAUTION"
            interpretation_fr = "Légèrement surévalué - Prudence"
        else:
            interpretation = "AVOID"
            interpretation_fr = "Surévalué - À éviter"

        return {
            'symbol': symbol,
            'company_name': data.get('company_name'),
            'sector': data.get('sector'),
            'current_price': current_price,
            'fair_values': fair_values,
            'margins_of_safety': margins,
            'average_margin_of_safety': round(avg_margin, 1),
            'interpretation': interpretation,
            'interpretation_fr': interpretation_fr,
            'morningstar_stars': self.morningstar_stars.get(symbol.upper(), None),
            'analyst_recommendation': data.get('recommendation_mean'),
            'analyst_target_price': data.get('target_mean_price'),
            'quality_metrics': {
                'trailing_pe': data.get('trailing_pe'),
                'forward_pe': data.get('forward_pe'),
                'peg_ratio': data.get('peg_ratio'),
                'price_to_book': data.get('price_to_book'),
                'return_on_equity': data.get('return_on_equity'),
                'debt_to_equity': data.get('debt_to_equity'),
                'profit_margins': data.get('profit_margins'),
                'earnings_growth': data.get('earnings_growth'),
                'revenue_growth': data.get('revenue_growth'),
                'dividend_yield': data.get('dividend_yield'),
            },
            'score': self.score(symbol),
        }

    def screen_symbols(self, symbols: List[str], min_score: int = 60) -> List[Dict]:
        """
        Filtre une liste de symboles par score minimum.

        Args:
            symbols: Liste de symboles à analyser
            min_score: Score minimum pour inclusion

        Returns:
            Liste des analyses triées par score décroissant
        """
        results = []

        for symbol in symbols:
            try:
                analysis = self.get_full_analysis(symbol)
                if 'error' not in analysis and analysis.get('score', 0) >= min_score:
                    results.append(analysis)
            except Exception as e:
                print(f"[WARN] Erreur pour {symbol}: {e}")

        # Trier par score décroissant
        results.sort(key=lambda x: x.get('score', 0), reverse=True)

        return results

    def get_summary_table(self, symbols: List[str]) -> pd.DataFrame:
        """
        Génère un tableau récapitulatif pour une liste de symboles.

        Args:
            symbols: Liste de symboles

        Returns:
            DataFrame avec les métriques clés
        """
        rows = []

        for symbol in symbols:
            analysis = self.get_full_analysis(symbol)
            if 'error' in analysis:
                continue

            rows.append({
                'Symbol': symbol,
                'Price': analysis.get('current_price'),
                'Graham': analysis.get('fair_values', {}).get('graham_number'),
                'DCF': analysis.get('fair_values', {}).get('dcf_simple'),
                'Analyst': analysis.get('fair_values', {}).get('analyst_target'),
                'Avg Margin': analysis.get('average_margin_of_safety'),
                'Stars': analysis.get('morningstar_stars', '-'),
                'Score': analysis.get('score'),
                'Signal': analysis.get('interpretation'),
            })

        return pd.DataFrame(rows)


# Test standalone
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TEST ValueScoring - Analyse Fondamentale Yahoo Finance")
    print("=" * 70)

    scoring = ValueScoring()

    # Tester quelques symboles
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'JNJ', 'JPM', 'XOM']

    for symbol in test_symbols:
        print(f"\n{'='*50}")
        print(f"Analyse: {symbol}")
        print('='*50)

        analysis = scoring.get_full_analysis(symbol)

        if 'error' in analysis:
            print(f"  Erreur: {analysis['error']}")
            continue

        print(f"  Entreprise: {analysis.get('company_name')}")
        print(f"  Secteur: {analysis.get('sector')}")
        print(f"  Prix actuel: ${analysis.get('current_price')}")

        print(f"\n  Fair Values:")
        for method, value in analysis.get('fair_values', {}).items():
            margin = analysis.get('margins_of_safety', {}).get(method, 0)
            print(f"    - {method}: ${value} (marge: {margin:+.1f}%)")

        print(f"\n  Marge moyenne: {analysis.get('average_margin_of_safety'):+.1f}%")
        print(f"  Interprétation: {analysis.get('interpretation')} - {analysis.get('interpretation_fr')}")
        print(f"  Score Value: {analysis.get('score')}/100")

        if analysis.get('morningstar_stars'):
            print(f"  Étoiles Morningstar: {'★' * analysis['morningstar_stars']}")

    print("\n" + "=" * 70)
    print("Tableau récapitulatif:")
    print("=" * 70)

    df = scoring.get_summary_table(test_symbols)
    print(df.to_string(index=False))
