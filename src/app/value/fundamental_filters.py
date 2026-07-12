"""
Filtres fondamentaux : EBIT/TEV (valeur) et Piotroski F-Score (qualité).

Anti-lookahead bias :
  Toutes les méthodes acceptent un paramètre `trade_date`.
  Quand trade_date est fourni, seuls les exercices fiscaux dont le rapport
  annuel était publié AVANT trade_date sont utilisés.
  Délai de publication par défaut : 90 jours après la fin de l'exercice.

  Exemple : exercice clos le 2022-12-31 → rapport disponible ~2023-03-31.
            Un backtest au 2023-02-01 n'utilisera PAS ces chiffres.

Usage :
  # Live trading (pas de trade_date = données les plus récentes)
  filters = FundamentalFilters()
  ratio = filters.calc_ebit_tev("AAPL")

  # Backtest (trade_date fourni = données disponibles à cette date)
  ratio = filters.calc_ebit_tev("AAPL", trade_date=date(2023, 1, 15))
  score = filters.calc_piotroski("AAPL", trade_date=date(2023, 1, 15))
"""
import time
import yfinance as yf
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


class FundamentalFilters:
    """Filtres fondamentaux avec cache journalier, fetch parallèle et contrôle du lookahead."""

    DEFAULT_REPORTING_LAG = 90   # jours après fin d'exercice avant publication du rapport

    def __init__(self, cache_ttl_hours: int = 24, max_workers: int = 5,
                 reporting_lag_days: int = DEFAULT_REPORTING_LAG,
                 backtest_mode: bool = False):
        self._cache: dict = {}           # {symbol: (timestamp, data)}
        self._ttl = cache_ttl_hours * 3600
        self._max_workers = max_workers
        self._lag = reporting_lag_days   # délai de publication (anti-lookahead)
        self._backtest_mode = backtest_mode  # True = pas de fallback yfinance (évite lookahead)

    # ------------------------------------------------------------------
    # Fetch & cache
    # ------------------------------------------------------------------

    def _fetch(self, symbol: str) -> dict | None:
        """
        Récupère les données fondamentales : cache RAM → PostgreSQL → yfinance.

        Ordre de priorité :
          1. Cache RAM  (TTL = cache_ttl_hours, intra-session, le plus rapide)
          2. Cache PostgreSQL  (persistant entre sessions, peuplé par populate_fundamental_cache.py)
          3. yfinance  (réseau, lent, sauvegarde automatiquement en PostgreSQL)
        """
        now = time.time()

        # 1. Cache RAM
        if symbol in self._cache:
            ts, data = self._cache[symbol]
            if now - ts < self._ttl:
                return data

        # 2. Cache PostgreSQL
        try:
            from src.app.value.fundamental_cache import load_symbol, save_symbol as _save
            db_data = load_symbol(symbol)
            if db_data is not None:
                # Enrichir avec les dates d'annonce 8-K (anti-lookahead précis)
                db_data['announcement_dates'] = self._load_announcement_dates(symbol)
                self._cache[symbol] = (now, db_data)
                return db_data
            # Symbole absent de la DB : en mode backtest, ne pas aller chercher yfinance
            # (les données yfinance actuelles créeraient un lookahead biais)
            if self._backtest_mode:
                self._cache[symbol] = (now, None)  # mettre en cache le "non trouvé"
                return None
        except Exception:
            _save = None   # DB non accessible, continuer vers yfinance
            if self._backtest_mode:
                return None

        # 3. yfinance (fallback réseau — live trading uniquement)
        try:
            t = yf.Ticker(symbol)
            info = t.info or {}

            try:
                income = t.income_stmt
            except Exception:
                income = t.financials

            try:
                balance = t.balance_sheet
            except Exception:
                balance = t.balance_sheet

            try:
                cashflow = t.cash_flow
            except Exception:
                cashflow = t.cashflow

            data = {
                'info':     info,
                'income':   income   if (income   is not None and not income.empty)   else None,
                'balance':  balance  if (balance  is not None and not balance.empty)  else None,
                'cashflow': cashflow if (cashflow is not None and not cashflow.empty) else None,
            }

            # Sauvegarder en PostgreSQL pour les prochaines sessions
            try:
                from src.app.value.fundamental_cache import save_symbol
                save_symbol(symbol, data)
            except Exception:
                pass

            # Enrichir avec les dates d'annonce 8-K
            data['announcement_dates'] = self._load_announcement_dates(symbol)
            self._cache[symbol] = (now, data)
            return data
        except Exception as e:
            print(f"[FUND] {symbol}: erreur fetch ({e})")
            return None

    def _load_announcement_dates(self, symbol: str) -> dict:
        """
        Charge les dates d'annonce 8-K depuis earnings_dates.

        Returns:
            dict {period_end_str: announcement_date_str}
            ex: {'2024-09-30': '2024-10-31', '2024-06-30': '2024-08-01', ...}
            Vide si la table n'existe pas ou si le symbole n'y est pas.
        """
        try:
            from src.app.value.earnings_dates import load_symbol as load_ann
            ann_df = load_ann(symbol)
            if ann_df.empty:
                return {}
            return {
                str(row['period_end'])[:10]: str(row['announcement_date'])[:10]
                for _, row in ann_df.iterrows()
                if row['period_end'] is not None and row['announcement_date'] is not None
            }
        except Exception:
            return {}

    def _prefetch_parallel(self, symbols: list):
        """Pré-charge les données de plusieurs symboles en parallèle."""
        missing = [s for s in symbols if s not in self._cache or
                   (time.time() - self._cache[s][0]) >= self._ttl]
        if not missing:
            return
        print(f"[FUND] Chargement fondamentaux: {len(missing)} symboles...")
        with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
            futs = {ex.submit(self._fetch, s): s for s in missing}
            for fut in as_completed(futs):
                pass

    # ------------------------------------------------------------------
    # Helpers date-aware (anti-lookahead)
    # ------------------------------------------------------------------

    def _available_cols(self, df: pd.DataFrame, trade_date,
                        announcement_dates: dict = None) -> list:
        """
        Retourne les colonnes d'un DataFrame financier (exercices fiscaux)
        dont le rapport était disponible avant trade_date.

        Priorité pour la date de disponibilité :
          1. Date d'annonce 8-K depuis earnings_dates (la plus précise)
          2. Fallback : fiscal_year_end + lag fixe (défaut 90 jours)

        Si trade_date est None, retourne toutes les colonnes (live trading).

        Args:
            df                 : DataFrame financier (colonnes = dates fiscales)
            trade_date         : date de décision pour le backtest
            announcement_dates : dict {period_end_str: ann_date_str} du symbole
        """
        if df is None or df.empty:
            return []
        if trade_date is None:
            return sorted(df.columns, reverse=True)   # toutes, plus récentes d'abord

        cutoff   = pd.Timestamp(trade_date)
        lag      = pd.Timedelta(days=self._lag)
        ann_map  = announcement_dates or {}
        available = []

        for col in df.columns:
            col_ts  = pd.Timestamp(col)
            col_str = col_ts.strftime('%Y-%m-%d')

            # 1. Date d'annonce 8-K exacte
            if col_str in ann_map:
                avail_date = pd.Timestamp(ann_map[col_str])
            else:
                # Recherche approchée : ±15 jours (trimestres non-standard)
                avail_date = None
                for period_str, ann_str in ann_map.items():
                    if abs((pd.Timestamp(period_str) - col_ts).days) <= 15:
                        avail_date = pd.Timestamp(ann_str)
                        break
                # 2. Fallback : lag fixe
                if avail_date is None:
                    avail_date = col_ts + lag

            if avail_date <= cutoff:
                available.append(col)

        return sorted(available, reverse=True)         # plus récentes d'abord

    def _get(self, df: pd.DataFrame, cols: list, *names) -> float | None:
        """Valeur la plus récente pour un champ, parmi une liste de colonnes disponibles."""
        if df is None or not cols:
            return None
        for name in names:
            if name in df.index:
                for col in cols:
                    try:
                        v = df.loc[name, col]
                        if v is not None and not (isinstance(v, float) and np.isnan(v)):
                            return float(v)
                    except Exception:
                        continue
        return None

    def _get_pair(self, df: pd.DataFrame, cols: list, *names,
                  min_gap_days: int = 300) -> tuple:
        """
        Retourne (valeur_t, valeur_t-1) pour calculer des variations annuelles.

        valeur_t   = valeur la plus récente non-NaN (parmi toutes les colonnes)
        valeur_t-1 = valeur non-NaN d'une colonne ≥ min_gap_days avant valeur_t

        Cherche le MÊME nom de champ pour les deux valeurs (cohérence).
        Gère les DataFrames EDGAR sparse où cols[0]/cols[1] peuvent être NaN.

        Args:
            min_gap_days : écart minimum entre t et t-1 (défaut 300j ≈ ~annuel)
        """
        if df is None or not cols:
            return None, None

        for name in names:
            if name not in df.index:
                continue

            val_t      = None
            col_t_date = None

            # Chercher val_t : première valeur non-NaN dans les colonnes disponibles
            for col in cols:
                try:
                    v = df.loc[name, col]
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        val_t      = float(v)
                        col_t_date = pd.Timestamp(col)
                        break
                except Exception:
                    continue

            if val_t is None:
                continue   # pas de val_t pour ce nom → essayer le suivant

            # Chercher val_t-1 : première non-NaN ≥ min_gap_days avant col_t
            val_t1 = None
            for col in cols:
                col_date = pd.Timestamp(col)
                if (col_t_date - col_date).days < min_gap_days:
                    continue   # trop récent
                try:
                    v = df.loc[name, col]
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        val_t1 = float(v)
                        break
                except Exception:
                    continue

            # Retourner dès qu'on a les deux valeurs pour le même champ
            if val_t is not None:
                return val_t, val_t1   # val_t1 peut être None si pas assez d'historique

        return None, None

    def _market_cap_at_date(self, data: dict, trade_date,
                             current_price: float | None = None) -> float | None:
        """
        Capitalisation boursière à trade_date :
        - Si current_price fourni (depuis prix historiques) + shares de la balance sheet
        - Sinon fallback sur info['marketCap'] (live seulement, pas de backtest)
        """
        if current_price is not None:
            cols_b = self._available_cols(data['balance'], trade_date)
            shares = self._get(data['balance'], cols_b,
                               'Ordinary Shares Number', 'Share Issued',
                               'Common Stock Equity')
            if shares and shares > 0:
                return current_price * shares
        # Fallback live (pas de trade_date ou pas de prix)
        if trade_date is None:
            return data['info'].get('marketCap')
        return None

    # ------------------------------------------------------------------
    # EBIT / TEV
    # ------------------------------------------------------------------

    def calc_ebit_tev(self, symbol: str, trade_date=None,
                      current_price: float | None = None) -> float | None:
        """
        Calcule EBIT/TEV (rendement des bénéfices sur la valeur d'entreprise).

        Args:
            symbol       : ticker
            trade_date   : date de la décision (None = live, données les + récentes)
            current_price: prix à trade_date pour calculer la market cap historique

        Returns:
            ratio EBIT/TEV, ou None si données insuffisantes

        Interprétation :
            > 15% = très bon marché (type "Magic Formula")
            8-15% = correct
            < 5%  = potentiellement surévalué
        """
        data = self._fetch(symbol)
        if not data:
            return None

        info = data['info']

        # --- Live trading : shortcuts depuis .info ---
        if trade_date is None:
            ev_ebitda = info.get('enterpriseToEbitda')
            if ev_ebitda and isinstance(ev_ebitda, (int, float)) and ev_ebitda > 0:
                return 1.0 / ev_ebitda
            ev    = info.get('enterpriseValue')
            ebitda = info.get('ebitda')
            if ev and ev > 0 and ebitda and isinstance(ebitda, (int, float)):
                return ebitda / ev

        # --- Calcul depuis les états financiers (live ou backtest) ---
        ann = data.get('announcement_dates', {})
        cols_i = self._available_cols(data['income'],  trade_date, ann)
        cols_b = self._available_cols(data['balance'], trade_date, ann)

        ebit = self._get(data['income'], cols_i,
                         'EBIT', 'Operating Income',
                         'Total Operating Income As Reported')
        if ebit is None:
            return None

        # TEV = Market Cap + Net Debt  (Net Debt = Total Debt - Cash, déjà calculé par yfinance)
        mktcap = self._market_cap_at_date(data, trade_date, current_price)
        # Essayer Net Debt d'abord (plus fiable), sinon calculer manuellement
        net_debt = self._get(data['balance'], cols_b, 'Net Debt')
        if net_debt is None:
            debt = self._get(data['balance'], cols_b,
                             'Long Term Debt And Capital Lease Obligation',
                             'Long Term Debt', 'Total Debt')
            cash = self._get(data['balance'], cols_b,
                             'Cash Cash Equivalents And Short Term Investments',
                             'Cash And Cash Equivalents')
            net_debt = (debt or 0) - (cash or 0)

        if mktcap is None or mktcap <= 0:
            return None
        tev = mktcap + net_debt
        if tev <= 0:
            return None

        return ebit / tev

    # ------------------------------------------------------------------
    # Piotroski F-Score
    # ------------------------------------------------------------------

    def calc_piotroski(self, symbol: str, trade_date=None) -> int | None:
        """
        Calcule le Piotroski F-Score (0-9).

        9 signaux binaires (1 point chacun) :
          Rentabilité  : F1 ROA>0, F2 CFO>0, F3 ΔROA>0, F4 CFO>NI
          Endettement  : F5 ΔLevier<0, F6 ΔLiquidité>0, F7 Pas dilution
          Efficacité   : F8 ΔMarge brute>0, F9 ΔRotation actifs>0

        Args:
            symbol     : ticker
            trade_date : date de la décision (None = live)

        Returns:
            score 0-9, ou None si données insuffisantes
        """
        data = self._fetch(symbol)
        if not data:
            return None

        info     = data['info']
        income   = data['income']
        balance  = data['balance']
        cashflow = data['cashflow']

        # Colonnes disponibles à trade_date (anti-lookahead)
        ann    = data.get('announcement_dates', {})
        cols_i = self._available_cols(income,   trade_date, ann)
        cols_b = self._available_cols(balance,  trade_date, ann)
        cols_c = self._available_cols(cashflow, trade_date, ann)

        if not cols_i or not cols_b:
            return None   # pas assez de données historiques

        score = 0

        # -- Valeurs de base (exercice t et t-1) --
        ni_t,  ni_t1  = self._get_pair(income,  cols_i,
                                        'Net Income',
                                        'Net Income Common Stockholders',
                                        'Net Income From Continuing Operation Net Minority Interest',
                                        'Net Income Continuous Operations')
        rev_t, rev_t1 = self._get_pair(income,  cols_i,
                                        'Total Revenue', 'Operating Revenue')
        gp_t,  gp_t1  = self._get_pair(income,  cols_i, 'Gross Profit')
        ta_t,  ta_t1  = self._get_pair(balance, cols_b, 'Total Assets')
        ca_t,  ca_t1  = self._get_pair(balance, cols_b, 'Current Assets')
        cl_t,  cl_t1  = self._get_pair(balance, cols_b, 'Current Liabilities')
        ltd_t, ltd_t1 = self._get_pair(balance, cols_b,
                                        'Long Term Debt And Capital Lease Obligation',
                                        'Long Term Debt', 'Total Debt')
        sh_t,  sh_t1  = self._get_pair(balance, cols_b,
                                        'Ordinary Shares Number', 'Share Issued')
        cfo_t, _      = self._get_pair(cashflow, cols_c,
                                        'Operating Cash Flow',
                                        'Cash Flow From Continuing Operating Activities')

        # Fallback CFO depuis .info (live seulement, sans trade_date)
        if cfo_t is None and trade_date is None:
            cfo_t = info.get('operatingCashflow')

        # ---- F1 : ROA > 0 ----
        roa_t = None
        if ni_t is not None and ta_t and ta_t > 0:
            roa_t = ni_t / ta_t
        elif trade_date is None:
            roa_live = info.get('returnOnAssets')
            if roa_live is not None:
                roa_t = roa_live
        f1 = 1 if (roa_t is not None and roa_t > 0) else 0
        score += f1

        # ---- F2 : CFO > 0 ----
        f2 = 1 if (cfo_t is not None and cfo_t > 0) else 0
        score += f2

        # ---- F3 : ΔROA > 0 ----
        f3 = 0
        if ni_t is not None and ni_t1 is not None and \
           ta_t and ta_t > 0 and ta_t1 and ta_t1 > 0:
            roa_t1 = ni_t1 / ta_t1
            if roa_t is not None:
                f3 = 1 if roa_t > roa_t1 else 0
        score += f3

        # ---- F4 : CFO > NI (qualité des bénéfices) ----
        f4 = 0
        if cfo_t is not None and ni_t is not None:
            f4 = 1 if cfo_t > ni_t else 0
        score += f4

        # ---- F5 : Δ Levier < 0 (dette LT / actifs a diminué) ----
        f5 = 0
        if ltd_t is not None and ltd_t1 is not None and \
           ta_t and ta_t > 0 and ta_t1 and ta_t1 > 0:
            lev_t  = ltd_t  / ta_t
            lev_t1 = ltd_t1 / ta_t1
            f5 = 1 if lev_t < lev_t1 else 0
        elif trade_date is None and info.get('debtToEquity') is not None:
            f5 = 1 if info['debtToEquity'] < 100 else 0
        score += f5

        # ---- F6 : Δ Ratio courant > 0 ----
        f6 = 0
        if ca_t and cl_t and cl_t > 0 and ca_t1 and cl_t1 and cl_t1 > 0:
            cr_t  = ca_t  / cl_t
            cr_t1 = ca_t1 / cl_t1
            f6 = 1 if cr_t > cr_t1 else 0
        elif trade_date is None and info.get('currentRatio') is not None:
            f6 = 1 if info['currentRatio'] > 1.0 else 0
        score += f6

        # ---- F7 : Pas de dilution ----
        f7 = 0
        if sh_t is not None and sh_t1 is not None and sh_t1 > 0:
            f7 = 1 if sh_t <= sh_t1 * 1.01 else 0
        elif trade_date is None:
            f7 = 1   # info seule ne permet pas de vérifier → neutre
        score += f7

        # ---- F8 : Δ Marge brute > 0 ----
        f8 = 0
        if gp_t and rev_t and rev_t > 0 and gp_t1 and rev_t1 and rev_t1 > 0:
            gm_t  = gp_t  / rev_t
            gm_t1 = gp_t1 / rev_t1
            f8 = 1 if gm_t > gm_t1 else 0
        elif trade_date is None and info.get('grossMargins') is not None:
            f8 = 1 if info['grossMargins'] > 0.20 else 0
        score += f8

        # ---- F9 : Δ Rotation des actifs > 0 ----
        f9 = 0
        if rev_t and ta_t and ta_t > 0 and rev_t1 and ta_t1 and ta_t1 > 0:
            at_t  = rev_t  / ta_t
            at_t1 = rev_t1 / ta_t1
            f9 = 1 if at_t > at_t1 else 0
        score += f9

        return score

    # ------------------------------------------------------------------
    # Filtres de liste
    # ------------------------------------------------------------------

    def filter_ebit_tev(self, symbols: list, min_ratio: float = 0.05,
                        trade_date=None,
                        prices: dict | None = None) -> tuple:
        """
        Filtre les symboles avec EBIT/TEV >= min_ratio.

        Args:
            symbols    : liste de symboles
            min_ratio  : seuil (défaut 5%)
            trade_date : date pour backtest (None = live)
            prices     : dict {symbol: prix} pour calcul TEV historique

        Returns:
            (filtered_symbols, ratios_dict)
        """
        self._prefetch_parallel(symbols)
        filtered, ratios = [], {}
        for sym in symbols:
            price = (prices or {}).get(sym)
            ratio = self.calc_ebit_tev(sym, trade_date=trade_date, current_price=price)
            ratios[sym] = ratio
            if ratio is None:
                print(f"  [EBIT/TEV] {sym}: donnees indisponibles [?]")
                filtered.append(sym)   # pas de pénalité si données absentes
            elif ratio >= min_ratio:
                print(f"  [EBIT/TEV] {sym}: {ratio:.2%} >= {min_ratio:.0%} [OK]")
                filtered.append(sym)
            else:
                print(f"  [EBIT/TEV] {sym}: {ratio:.2%} < {min_ratio:.0%} [X]")
        return filtered, ratios

    def filter_piotroski(self, symbols: list, min_score: int = 6,
                         trade_date=None) -> tuple:
        """
        Filtre les symboles avec Piotroski F-Score >= min_score.

        Args:
            symbols    : liste de symboles
            min_score  : seuil (défaut 6/9)
            trade_date : date pour backtest (None = live)

        Returns:
            (filtered_symbols, scores_dict)
        """
        self._prefetch_parallel(symbols)
        filtered, scores = [], {}
        n_ok = n_fail = n_nodata = 0
        for sym in symbols:
            fscore = self.calc_piotroski(sym, trade_date=trade_date)
            scores[sym] = fscore
            if fscore is None:
                n_nodata += 1
                # En mode backtest : exclure les symboles sans données (pas de pass-through)
                if not self._backtest_mode:
                    filtered.append(sym)
            elif fscore >= min_score:
                n_ok += 1
                filtered.append(sym)
            else:
                n_fail += 1
        print(f"  [PIOTROSKI] {len(symbols)} symboles -> {n_ok} OK (>={min_score}), "
              f"{n_fail} exclus, {n_nodata} sans données"
              + (" (exclus backtest)" if self._backtest_mode else " (gardés live)"))
        return filtered, scores
