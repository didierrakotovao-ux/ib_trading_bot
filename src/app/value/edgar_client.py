"""
Client SEC EDGAR pour les données fondamentales XBRL.

API utilisée (gratuite, sans clé) :
  - https://www.sec.gov/files/company_tickers.json   -> ticker -> CIK
  - https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json -> tous les faits XBRL

Avantages vs yfinance :
  - ~10 000 entreprises US (tous les filers SEC)
  - Historique depuis 2009 (vs 5 ans yfinance)
  - Date de dépôt exacte ('filed') -> anti-lookahead parfait, pas d'estimation de lag
  - Annuel (10-K) ET trimestriel (10-Q)

Taux limite SEC : 10 req/sec -> on force 120ms entre requêtes (~8 req/sec).

Usage :
    from src.app.value.edgar_client import get_fundamentals
    data = get_fundamentals('AAPL')   # dict compatible fundamental_cache.save_symbol()
"""
import requests
import json
import os
import time
from datetime import date
from typing import Optional, Dict, Tuple
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EDGAR_DATA_BASE = "https://data.sec.gov"
EDGAR_WWW_BASE  = "https://www.sec.gov"

# Requis par les CGU SEC : "Nom Entreprise contact@domaine.com"
USER_AGENT = "QuantTradingBot research didier.rakotovao@gmail.com"

_MIN_GAP_S = 0.12   # 120ms entre requêtes -> ~8 req/sec

_last_req_ts = 0.0

# Cache CIK local (refresh hebdomadaire)
_CIK_CACHE_FILE = os.path.join(os.path.dirname(__file__), '_edgar_tickers_cache.json')
_cik_map: dict = {}


# ---------------------------------------------------------------------------
# Concepts XBRL -> colonnes fundamental_cache
# L'ordre dans la liste = priorité (premier trouvé avec valeur non-nulle)
# Préfixe 'dei/' pour les concepts du namespace DEI (non us-gaap)
# ---------------------------------------------------------------------------

CONCEPT_MAP: Dict[str, list] = {
    'ebit': [
        'OperatingIncomeLoss',
    ],
    'total_revenue': [
        'Revenues',
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'SalesRevenueNet',
        'RevenueFromContractWithCustomerIncludingAssessedTax',
        'SalesRevenueGoodsNet',
    ],
    'gross_profit': [
        'GrossProfit',
    ],
    'net_income': [
        'NetIncomeLoss',
        'NetIncomeLossAvailableToCommonStockholdersBasic',
    ],
    'pretax_income': [
        'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
        'IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments',
    ],
    'interest_expense': [
        'InterestExpense',
        'InterestAndDebtExpense',
        'InterestExpenseDebt',
    ],
    'total_assets': [
        'Assets',
    ],
    'current_assets': [
        'AssetsCurrent',
    ],
    'current_liabilities': [
        'LiabilitiesCurrent',
    ],
    'long_term_debt': [
        'LongTermDebtAndCapitalLeaseObligations',
        'LongTermDebt',
        'LongTermNotesPayable',
    ],
    'cash_equivalents': [
        'CashAndCashEquivalentsAtCarryingValue',
        'CashCashEquivalentsAndShortTermInvestments',
        'CashEquivalentsAtCarryingValue',
    ],
    'shares_outstanding': [
        'dei/EntityCommonStockSharesOutstanding',
        'CommonStockSharesOutstanding',
        'CommonStockSharesIssuedNet',
    ],
    'stockholders_equity': [
        'StockholdersEquity',
        'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
    ],
    'total_debt': [
        'DebtAndCapitalLeaseObligations',
        'LongTermDebtAndCapitalLeaseObligations',
        'LongTermDebt',
    ],
    'operating_cash_flow': [
        'NetCashProvidedByUsedInOperatingActivities',
    ],
    'capex': [
        'PaymentsToAcquirePropertyPlantAndEquipment',
        'CapitalExpenditures',
    ],
    # Pour EBITDA = EBIT + D&A (calculé dans save_edgar_symbol)
    '_depreciation': [
        'DepreciationDepletionAndAmortization',
        'DepreciationAndAmortization',
        'Depreciation',
    ],
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _http_get(url: str, retries: int = 3) -> dict:
    """GET JSON avec rate limiting et retry automatique."""
    global _last_req_ts
    gap = time.time() - _last_req_ts
    if gap < _MIN_GAP_S:
        time.sleep(_MIN_GAP_S - gap)
    _last_req_ts = time.time()

    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            if r.status_code == 429:
                wait = 60 * attempt
                print(f"[EDGAR] Rate limit 429, attente {wait}s...")
                time.sleep(wait)
                continue
            if r.status_code == 404:
                return {}
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(3 * attempt)
    return {}


# ---------------------------------------------------------------------------
# Mapping ticker -> CIK
# ---------------------------------------------------------------------------

def load_cik_map(force_refresh: bool = False) -> dict:
    """
    Charge le mapping ticker -> CIK (10 chiffres zero-padded).
    Cache local refreshé hebdomadairement.
    """
    global _cik_map
    if _cik_map and not force_refresh:
        return _cik_map

    # Vérifier le cache local
    if os.path.exists(_CIK_CACHE_FILE) and not force_refresh:
        age_days = (time.time() - os.path.getmtime(_CIK_CACHE_FILE)) / 86400
        if age_days < 7:
            with open(_CIK_CACHE_FILE, encoding='utf-8') as f:
                _cik_map = json.load(f)
            return _cik_map

    print("[EDGAR] Telechargement du mapping ticker -> CIK depuis SEC...")
    url  = f"{EDGAR_WWW_BASE}/files/company_tickers.json"
    data = _http_get(url)
    if not data:
        raise RuntimeError("[EDGAR] Impossible de télécharger company_tickers.json")

    # Format SEC : {"0": {"cik_str": "320193", "ticker": "AAPL", "title": "Apple Inc."}}
    _cik_map = {
        v['ticker'].upper(): str(v['cik_str']).zfill(10)
        for v in data.values()
        if 'ticker' in v and 'cik_str' in v
    }

    with open(_CIK_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(_cik_map, f)

    print(f"[EDGAR] {len(_cik_map):,} tickers mappes vers CIK")
    return _cik_map


def get_cik(symbol: str) -> Optional[str]:
    """Retourne le CIK (10 chiffres) pour un ticker, ou None si non trouvé."""
    cmap = load_cik_map()
    # Essayer le symbole brut, puis sans suffixe .TO etc.
    sym = symbol.upper()
    if sym in cmap:
        return cmap[sym]
    # Enlever suffixe pays (ex: RY.TO -> RY)
    base = sym.split('.')[0]
    return cmap.get(base)


# ---------------------------------------------------------------------------
# Extraction des périodes depuis company facts
# ---------------------------------------------------------------------------

def _extract_concept(facts: dict, concept: str,
                     form_types: tuple) -> Dict[str, Tuple[float, str]]:
    """
    Extrait les valeurs d'un concept XBRL pour les types de formulaires donnés.

    Args:
        facts      : company facts JSON complet
        concept    : ex: 'OperatingIncomeLoss' ou 'dei/EntityCommonStockSharesOutstanding'
        form_types : ex: ('10-K',) ou ('10-Q',)

    Returns:
        {end_date_str: (value, filed_date_str)}
        En cas de doublons pour la même end_date, garde le filing le plus récent.
    """
    namespace = 'us-gaap'
    if concept.startswith('dei/'):
        namespace = 'dei'
        concept   = concept[4:]

    try:
        units_dict = (facts.get('facts', {})
                           .get(namespace, {})
                           .get(concept, {})
                           .get('units', {}))
    except Exception:
        return {}

    result: Dict[str, Tuple[float, str]] = {}

    for _unit, entries in units_dict.items():
        for e in entries:
            if e.get('form') not in form_types:
                continue
            end   = e.get('end')
            filed = e.get('filed')
            val   = e.get('val')
            if end is None or filed is None or val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            # Déduplication : garder le filing le plus récent pour cette date
            if end not in result or filed > result[end][1]:
                result[end] = (fval, filed)

    return result


def _pick_concept(facts: dict, concepts: list,
                  form_types: tuple) -> Dict[str, Tuple[float, str]]:
    """Essaie les concepts dans l'ordre, retourne le premier non-vide."""
    for c in concepts:
        data = _extract_concept(facts, c, form_types)
        if data:
            return data
    return {}


# ---------------------------------------------------------------------------
# Conversion en dict compatible fundamental_cache.save_edgar_symbol()
# ---------------------------------------------------------------------------

def _build_period_records(facts: dict, form_types: tuple,
                          period_type: str) -> list:
    """
    Construit la liste de records à insérer en base pour un type de période.

    Returns:
        Liste de dicts avec clés = colonnes fundamental_cache +
        'fiscal_year_end' (str), 'report_available_date' (str = filed date)
    """
    # Extraire toutes les métriques
    extracted: Dict[str, Dict[str, Tuple[float, str]]] = {}
    for col, concepts in CONCEPT_MAP.items():
        extracted[col] = _pick_concept(facts, concepts, form_types)

    # Union de toutes les end_dates disponibles
    all_dates: set = set()
    for col_data in extracted.values():
        all_dates.update(col_data.keys())

    if not all_dates:
        return []

    records = []
    for end_date in sorted(all_dates):
        rec = {'fiscal_year_end': end_date}

        # La report_available_date = la date de dépôt la plus ancienne parmi
        # toutes les métriques pour cette période (quand le rapport est d'abord dispo)
        filed_dates = [
            v[1] for v in [col_data.get(end_date) for col_data in extracted.values()]
            if v is not None
        ]
        if not filed_dates:
            continue
        rec['report_available_date'] = min(filed_dates)  # plus tôt = plus prudent anti-lookahead
        rec['period_type'] = period_type

        # Métriques financières
        def _val(col):
            v = extracted.get(col, {}).get(end_date)
            return v[0] if v is not None else None

        rec['ebit']               = _val('ebit')
        rec['total_revenue']      = _val('total_revenue')
        rec['gross_profit']       = _val('gross_profit')
        rec['net_income']         = _val('net_income')
        rec['pretax_income']      = _val('pretax_income')
        rec['interest_expense']   = _val('interest_expense')
        rec['total_assets']       = _val('total_assets')
        rec['current_assets']     = _val('current_assets')
        rec['current_liabilities']= _val('current_liabilities')
        rec['long_term_debt']     = _val('long_term_debt')
        rec['cash_equivalents']   = _val('cash_equivalents')
        rec['shares_outstanding'] = _val('shares_outstanding')
        rec['stockholders_equity']= _val('stockholders_equity')
        rec['total_debt']         = _val('total_debt')
        rec['operating_cash_flow']= _val('operating_cash_flow')
        rec['capex']              = _val('capex')

        # Net debt calculé
        ltd  = rec['long_term_debt']
        cash = rec['cash_equivalents']
        rec['net_debt'] = (ltd - cash) if (ltd is not None and cash is not None) else None

        # Free cash flow calculé
        ocf   = rec['operating_cash_flow']
        capex = rec['capex']
        rec['free_cash_flow'] = (ocf - abs(capex)) if (ocf is not None and capex is not None) else None

        # EBITDA = EBIT + D&A
        ebit = rec['ebit']
        da   = _val('_depreciation')
        rec['normalized_ebitda'] = (ebit + da) if (ebit is not None and da is not None) else None

        records.append(rec)

    return records


# ---------------------------------------------------------------------------
# API publique principale
# ---------------------------------------------------------------------------

def get_fundamentals(symbol: str) -> Optional[dict]:
    """
    Récupère les données fondamentales depuis SEC EDGAR pour un symbole.

    Returns:
        dict avec keys 'annual_records', 'quarterly_records', chacune
        étant une liste de dicts à insérer dans fundamental_cache,
        ou None si le CIK n'est pas trouvé.

        Format :
        {
            'annual_records':    [{fiscal_year_end, report_available_date,
                                   ebit, total_revenue, ...}, ...],
            'quarterly_records': [{...}, ...]
        }
    """
    cik = get_cik(symbol)
    if cik is None:
        return None

    url   = f"{EDGAR_DATA_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    facts = _http_get(url)
    if not facts:
        return None

    annual_recs    = _build_period_records(facts, ('10-K',),  'annual')
    quarterly_recs = _build_period_records(facts, ('10-Q',),  'quarterly')

    if not annual_recs and not quarterly_recs:
        return None

    return {
        'annual_records':    annual_recs,
        'quarterly_records': quarterly_recs,
    }


# ---------------------------------------------------------------------------
# Test rapide
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    print(f"Test EDGAR pour {sym}...")
    data = get_fundamentals(sym)
    if data is None:
        print(f"CIK non trouvé pour {sym}")
    else:
        print(f"  Annuel    : {len(data['annual_records'])} périodes")
        print(f"  Trimestriel: {len(data['quarterly_records'])} périodes")
        if data['annual_records']:
            r = data['annual_records'][-1]
            print(f"  Dernière période : {r['fiscal_year_end']} "
                  f"(déposé: {r['report_available_date']})")
            print(f"  EBIT: {r['ebit']:,.0f}" if r['ebit'] else "  EBIT: N/A")
            print(f"  Revenue: {r['total_revenue']:,.0f}" if r['total_revenue'] else "  Revenue: N/A")
