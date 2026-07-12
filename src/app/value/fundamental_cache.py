"""
Cache PostgreSQL pour les données fondamentales (états financiers annuels).

Évite de re-télécharger depuis yfinance à chaque backtest.
Retourne des DataFrames compatibles avec fundamental_filters.py.

Limitation yfinance : seulement ~5 ans d'historique annuel disponibles.
Les filtres fondamentaux ne sont utilisables qu'à partir de ~2020.

Anti-lookahead :
  report_available_date = fiscal_year_end + reporting_lag (90j par défaut)
  Un backtest au 2023-02-01 n'utilise que les rapports publiés avant cette date.

Usage :
    from src.app.value.fundamental_cache import load_symbol, save_symbol, create_table
    create_table()
    data = load_symbol('AAPL')    # → dict {'income': df, 'balance': df, 'cashflow': df}
    save_symbol('AAPL', yf_data)  # yf_data = dict retourné par fundamental_filters._fetch()
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.database.pg_connection import get_conn, read_sql


# ---------------------------------------------------------------------------
# Mapping DB → noms d'index yfinance (plusieurs alias par métrique)
# L'ordre compte : le 1er alias est le nom principal stocké dans le DF reconstruit,
# les suivants sont des alias que _get() cherchera aussi.
# ---------------------------------------------------------------------------

INCOME_MAP: Dict[str, list] = {
    'ebit':            ['EBIT', 'Operating Income', 'Total Operating Income As Reported'],
    'total_revenue':   ['Total Revenue', 'Operating Revenue'],
    'gross_profit':    ['Gross Profit'],
    'net_income':      ['Net Income', 'Net Income Common Stockholders',
                        'Net Income From Continuing Operation Net Minority Interest',
                        'Net Income Continuous Operations'],
    'normalized_ebitda': ['Normalized EBITDA', 'EBITDA'],
    'pretax_income':   ['Pretax Income'],
    'interest_expense':['Interest Expense', 'Interest Expense Non Operating'],
}

BALANCE_MAP: Dict[str, list] = {
    'total_assets':        ['Total Assets'],
    'current_assets':      ['Current Assets'],
    'current_liabilities': ['Current Liabilities'],
    'long_term_debt':      ['Long Term Debt And Capital Lease Obligation',
                            'Long Term Debt', 'Total Debt'],
    'cash_equivalents':    ['Cash Cash Equivalents And Short Term Investments',
                            'Cash And Cash Equivalents', 'Cash Equivalents'],
    'net_debt':            ['Net Debt'],
    'shares_outstanding':  ['Ordinary Shares Number', 'Share Issued'],
    'stockholders_equity': ['Stockholders Equity', 'Common Stock Equity'],
    'total_debt':          ['Total Debt', 'Long Term Debt And Capital Lease Obligation'],
}

CASHFLOW_MAP: Dict[str, list] = {
    'operating_cash_flow': ['Operating Cash Flow',
                            'Cash Flow From Continuing Operating Activities'],
    'capex':               ['Capital Expenditure'],
    'free_cash_flow':      ['Free Cash Flow'],
}


# ---------------------------------------------------------------------------
# Création de la table
# ---------------------------------------------------------------------------

def create_table():
    """Crée la table fundamental_cache si elle n'existe pas."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fundamental_cache (
            symbol                 TEXT    NOT NULL,
            fiscal_year_end        DATE    NOT NULL,
            report_available_date  DATE    NOT NULL,
            period_type            TEXT    NOT NULL DEFAULT 'annual',

            -- Compte de résultat
            ebit                   DOUBLE PRECISION,
            total_revenue          DOUBLE PRECISION,
            gross_profit           DOUBLE PRECISION,
            net_income             DOUBLE PRECISION,
            normalized_ebitda      DOUBLE PRECISION,
            pretax_income          DOUBLE PRECISION,
            interest_expense       DOUBLE PRECISION,

            -- Bilan
            total_assets           DOUBLE PRECISION,
            current_assets         DOUBLE PRECISION,
            current_liabilities    DOUBLE PRECISION,
            long_term_debt         DOUBLE PRECISION,
            cash_equivalents       DOUBLE PRECISION,
            net_debt               DOUBLE PRECISION,
            shares_outstanding     DOUBLE PRECISION,
            stockholders_equity    DOUBLE PRECISION,
            total_debt             DOUBLE PRECISION,

            -- Flux de trésorerie
            operating_cash_flow    DOUBLE PRECISION,
            capex                  DOUBLE PRECISION,
            free_cash_flow         DOUBLE PRECISION,

            -- Données live (info yfinance — valeur au moment du fetch, pas historique)
            market_cap_live        DOUBLE PRECISION,
            enterprise_value_live  DOUBLE PRECISION,
            ev_to_ebitda_live      DOUBLE PRECISION,

            last_fetched           TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),

            PRIMARY KEY (symbol, fiscal_year_end, period_type)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fc_symbol   ON fundamental_cache(symbol)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fc_available ON fundamental_cache(report_available_date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fc_fetched  ON fundamental_cache(last_fetched)")
    # Migration : ajouter les colonnes live si table existante sans elles
    for col in ['market_cap_live', 'enterprise_value_live', 'ev_to_ebitda_live']:
        try:
            cur.execute(f"ALTER TABLE fundamental_cache ADD COLUMN IF NOT EXISTS {col} DOUBLE PRECISION")
        except Exception:
            pass
    conn.commit()
    cur.close()
    conn.close()
    print("[FUND-CACHE] Table fundamental_cache prête.")


# ---------------------------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------------------------

def save_symbol(symbol: str, data: dict, reporting_lag: int = 90,
                period_type: str = 'annual') -> int:
    """
    Sauvegarde les données yfinance d'un symbole en base (upsert).

    Args:
        symbol        : ticker
        data          : dict avec keys 'income', 'balance', 'cashflow' (DataFrames yfinance)
        reporting_lag : jours après fin d'exercice avant publication (anti-lookahead)
                        Annuel : 90j par défaut | Trimestriel : 45j recommandé
        period_type   : 'annual' ou 'quarterly'

    Returns:
        Nombre d'exercices sauvegardés
    """
    income   = data.get('income')
    balance  = data.get('balance')
    cashflow = data.get('cashflow')

    # Union des dates d'exercice disponibles
    all_dates = set()
    for df in [income, balance, cashflow]:
        if df is not None and not df.empty:
            all_dates.update(df.columns.tolist())

    if not all_dates:
        return 0

    conn = get_conn()
    cur  = conn.cursor()
    inserted = 0

    for fiscal_date in sorted(all_dates):
        ts          = pd.Timestamp(fiscal_date)
        report_date = (ts + pd.Timedelta(days=reporting_lag)).date()

        def _get(df, *names):
            """Extrait la valeur d'une colonne (date fiscale) pour un ou plusieurs noms d'index."""
            if df is None or df.empty or fiscal_date not in df.columns:
                return None
            col = df[fiscal_date]
            for name in names:
                if name in col.index:
                    v = col[name]
                    try:
                        f = float(v)
                        if not np.isnan(f):
                            return f
                    except (TypeError, ValueError):
                        pass
            return None

        # --- Income Statement ---
        ebit         = _get(income, 'EBIT', 'Operating Income', 'Total Operating Income As Reported')
        total_rev    = _get(income, 'Total Revenue', 'Operating Revenue')
        gross_profit = _get(income, 'Gross Profit')
        net_income   = _get(income, 'Net Income', 'Net Income Common Stockholders',
                            'Net Income From Continuing Operation Net Minority Interest',
                            'Net Income Continuous Operations')
        norm_ebitda  = _get(income, 'Normalized EBITDA', 'EBITDA')
        pretax       = _get(income, 'Pretax Income')
        int_exp      = _get(income, 'Interest Expense', 'Interest Expense Non Operating')

        # --- Balance Sheet ---
        total_assets = _get(balance, 'Total Assets')
        curr_assets  = _get(balance, 'Current Assets')
        curr_liab    = _get(balance, 'Current Liabilities')
        ltd          = _get(balance, 'Long Term Debt And Capital Lease Obligation',
                            'Long Term Debt', 'Total Debt')
        cash         = _get(balance, 'Cash Cash Equivalents And Short Term Investments',
                            'Cash And Cash Equivalents', 'Cash Equivalents')
        net_debt     = _get(balance, 'Net Debt')
        shares       = _get(balance, 'Ordinary Shares Number', 'Share Issued')
        equity       = _get(balance, 'Stockholders Equity', 'Common Stock Equity')
        total_debt   = _get(balance, 'Total Debt', 'Long Term Debt And Capital Lease Obligation')

        # --- Cash Flow ---
        cfo          = _get(cashflow, 'Operating Cash Flow',
                            'Cash Flow From Continuing Operating Activities')
        capex        = _get(cashflow, 'Capital Expenditure')
        fcf          = _get(cashflow, 'Free Cash Flow')

        # Calculer net_debt si absent
        if net_debt is None and ltd is not None and cash is not None:
            net_debt = ltd - cash

        # Données live (extraites de info, stockées uniquement sur la rangée la plus récente)
        info       = data.get('info') or {}
        mktcap_l   = info.get('marketCap')
        ev_l       = info.get('enterpriseValue')
        ev_ebitda_l= info.get('enterpriseToEbitda')
        # Convertir en float ou None
        def _f(v):
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None
        mktcap_l    = _f(mktcap_l)
        ev_l        = _f(ev_l)
        ev_ebitda_l = _f(ev_ebitda_l)

        try:
            cur.execute("""
                INSERT INTO fundamental_cache
                  (symbol, fiscal_year_end, report_available_date, period_type,
                   ebit, total_revenue, gross_profit, net_income, normalized_ebitda,
                   pretax_income, interest_expense,
                   total_assets, current_assets, current_liabilities,
                   long_term_debt, cash_equivalents, net_debt, shares_outstanding,
                   stockholders_equity, total_debt,
                   operating_cash_flow, capex, free_cash_flow,
                   market_cap_live, enterprise_value_live, ev_to_ebitda_live,
                   last_fetched)
                VALUES (%s,%s,%s,%s,
                        %s,%s,%s,%s,%s,%s,%s,
                        %s,%s,%s,%s,%s,%s,%s,%s,%s,
                        %s,%s,%s,
                        %s,%s,%s,
                        NOW())
                ON CONFLICT (symbol, fiscal_year_end, period_type) DO UPDATE SET
                   ebit                  = EXCLUDED.ebit,
                   total_revenue         = EXCLUDED.total_revenue,
                   gross_profit          = EXCLUDED.gross_profit,
                   net_income            = EXCLUDED.net_income,
                   normalized_ebitda     = EXCLUDED.normalized_ebitda,
                   pretax_income         = EXCLUDED.pretax_income,
                   interest_expense      = EXCLUDED.interest_expense,
                   total_assets          = EXCLUDED.total_assets,
                   current_assets        = EXCLUDED.current_assets,
                   current_liabilities   = EXCLUDED.current_liabilities,
                   long_term_debt        = EXCLUDED.long_term_debt,
                   cash_equivalents      = EXCLUDED.cash_equivalents,
                   net_debt              = EXCLUDED.net_debt,
                   shares_outstanding    = EXCLUDED.shares_outstanding,
                   stockholders_equity   = EXCLUDED.stockholders_equity,
                   total_debt            = EXCLUDED.total_debt,
                   operating_cash_flow   = EXCLUDED.operating_cash_flow,
                   capex                 = EXCLUDED.capex,
                   free_cash_flow        = EXCLUDED.free_cash_flow,
                   market_cap_live       = EXCLUDED.market_cap_live,
                   enterprise_value_live = EXCLUDED.enterprise_value_live,
                   ev_to_ebitda_live     = EXCLUDED.ev_to_ebitda_live,
                   last_fetched          = NOW()
            """, (
                symbol, ts.date(), report_date, period_type,
                ebit, total_rev, gross_profit, net_income, norm_ebitda, pretax, int_exp,
                total_assets, curr_assets, curr_liab, ltd, cash, net_debt, shares, equity, total_debt,
                cfo, capex, fcf,
                mktcap_l, ev_l, ev_ebitda_l,
            ))
            inserted += 1
        except Exception as e:
            print(f"[FUND-CACHE] {symbol} {ts.date()}: erreur save ({e})")

    conn.commit()
    cur.close()
    conn.close()
    return inserted


# ---------------------------------------------------------------------------
# Chargement et reconstruction des DataFrames yfinance-compatibles
# ---------------------------------------------------------------------------

def _build_df(rows_df: pd.DataFrame, col_map: Dict[str, list]) -> Optional[pd.DataFrame]:
    """
    Reconstruit un DataFrame compatible yfinance à partir des lignes DB.

    Résultat :
        index   = noms des métriques (ex: 'EBIT', 'Total Revenue', ...)
        columns = pd.Timestamp des fins d'exercice (ex: Timestamp('2024-09-30'))
    """
    data: Dict[str, Dict] = {}
    for _, row in rows_df.iterrows():
        ts = pd.Timestamp(row['fiscal_year_end'])
        for db_col, yf_names in col_map.items():
            raw = row.get(db_col)
            val = None if (raw is None or (isinstance(raw, float) and np.isnan(raw))) else float(raw)
            for name in yf_names:
                if name not in data:
                    data[name] = {}
                data[name][ts] = val

    if not data:
        return None

    df = pd.DataFrame(data).T   # shape: (nb_metrics, nb_fiscal_dates)
    return df


def load_symbol(symbol: str, prefer: str = 'quarterly') -> Optional[dict]:
    """
    Charge les données fondamentales depuis PostgreSQL en fusionnant
    les données trimestrielles ET annuelles.

    Stratégie de fusion :
      - Toutes les dates des deux sources sont conservées dans le DataFrame
      - Pour chaque (date, métrique) : trimestriel en priorité, annual en complément
        si la valeur trimestrielle est NaN (ex: EBIT en fin d'exercice = dans le 10-K)
      - L'annuel fournit aussi l'historique long (10+ ans) absent du trimestriel

    Args:
        symbol : ticker
        prefer : 'quarterly' (défaut) ou 'annual' — uniquement pour le fallback
                 si aucun des deux n'est disponible.

    Returns:
        dict avec keys 'info', 'income', 'balance', 'cashflow'
        compatible avec fundamental_filters.FundamentalFilters._fetch()
        ou None si aucune donnée disponible.
    """
    # Charger les deux sources
    rows_q = read_sql("""
        SELECT * FROM fundamental_cache
        WHERE symbol = %s AND period_type = 'quarterly'
        ORDER BY fiscal_year_end DESC
    """, (symbol,))

    rows_a = read_sql("""
        SELECT * FROM fundamental_cache
        WHERE symbol = %s AND period_type = 'annual'
        ORDER BY fiscal_year_end DESC
    """, (symbol,))

    if rows_q.empty and rows_a.empty:
        return None

    # --- Fusion des deux sources ---
    # Construire les DataFrames séparément puis fusionner colonne par colonne
    # (la fusion combine les dates des deux, en complétant les NaN par l'autre source)
    def _merge_dfs(col_map: Dict[str, list]) -> Optional[pd.DataFrame]:
        df_q = _build_df(rows_q, col_map) if not rows_q.empty else None
        df_a = _build_df(rows_a, col_map) if not rows_a.empty else None

        if df_q is None and df_a is None:
            return None
        if df_q is None:
            return df_a
        if df_a is None:
            return df_q

        # Les deux existent : fusionner
        # 1. Union de toutes les colonnes (dates)
        all_cols = df_q.columns.union(df_a.columns)
        df_q = df_q.reindex(columns=all_cols)
        df_a = df_a.reindex(columns=all_cols)

        # 2. Union de toutes les lignes (métriques)
        all_idx = df_q.index.union(df_a.index)
        df_q = df_q.reindex(index=all_idx)
        df_a = df_a.reindex(index=all_idx)

        # 3. Pour chaque cellule : trimestriel en priorité, annuel si NaN
        merged = df_q.combine_first(df_a)
        return merged

    income   = _merge_dfs(INCOME_MAP)
    balance  = _merge_dfs(BALANCE_MAP)
    cashflow = _merge_dfs(CASHFLOW_MAP)

    if income is None and balance is None and cashflow is None:
        return None

    # Reconstruire info avec les valeurs live depuis la rangée la plus récente
    latest = rows_q.iloc[0] if not rows_q.empty else rows_a.iloc[0]
    def _fv(col):
        v = latest.get(col)
        return float(v) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else None

    info = {}
    if _fv('market_cap_live')       is not None: info['marketCap']            = _fv('market_cap_live')
    if _fv('enterprise_value_live') is not None: info['enterpriseValue']       = _fv('enterprise_value_live')
    if _fv('ev_to_ebitda_live')     is not None: info['enterpriseToEbitda']    = _fv('ev_to_ebitda_live')

    return {
        'info':     info,
        'income':   income,
        'balance':  balance,
        'cashflow': cashflow,
    }


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def get_cached_symbols(max_age_days: int = 30) -> set:
    """Retourne les symboles déjà en cache et pas trop anciens."""
    rows = read_sql(f"""
        SELECT DISTINCT symbol
        FROM fundamental_cache
        WHERE last_fetched > NOW() - INTERVAL '{max_age_days} days'
    """)
    return set(rows['symbol'].tolist()) if not rows.empty else set()


def get_cache_stats() -> dict:
    """Statistiques du cache."""
    stats = read_sql("""
        SELECT
            COUNT(DISTINCT symbol)          AS nb_symbols,
            COUNT(*)                         AS nb_rows,
            MIN(fiscal_year_end)             AS oldest_fiscal,
            MAX(fiscal_year_end)             AS newest_fiscal,
            MIN(last_fetched)                AS oldest_fetch,
            MAX(last_fetched)                AS newest_fetch
        FROM fundamental_cache
    """)
    return stats.iloc[0].to_dict() if not stats.empty else {}


if __name__ == '__main__':
    create_table()
    stats = get_cache_stats()
    if stats.get('nb_symbols'):
        print(f"Cache: {stats['nb_symbols']:,} symboles | "
              f"{stats['nb_rows']:,} lignes | "
              f"exercices: {stats['oldest_fiscal']} → {stats['newest_fiscal']}")
    else:
        print("Cache vide.")
