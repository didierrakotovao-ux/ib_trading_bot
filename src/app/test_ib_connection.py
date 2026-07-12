"""
Test de connectivité IB + données historiques yfinance — ne place aucun ordre.
Usage:
    python test_ib_connection.py              # paper TWS  (port 7497)
    python test_ib_connection.py --live       # live TWS   (port 7496)
    python test_ib_connection.py --gateway    # paper Gateway (port 4002)
    python test_ib_connection.py --gateway --live  # live Gateway (port 4001)
    python test_ib_connection.py --data-only  # test yfinance uniquement (sans TWS)
    python test_ib_connection.py --l2         # + test données L2 (market depth)
    python test_ib_connection.py --l2 --symbol MSFT  # L2 sur un symbole précis
"""
import sys
import os
import argparse
import time
from datetime import datetime, timedelta
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.app.screener.providers.market_data_provider import MarketDataProvider


def test_historical_data(symbol: str = "AAPL"):
    """
    Teste la récupération de données historiques via yfinance.
    Le bot utilise yfinance pour les historiques (pas l'API IB),
    donc aucune souscription IB n'est requise pour cette partie.
    """
    print(f"\n{'='*50}")
    print(f"  TEST DONNÉES HISTORIQUES — {symbol}")
    print(f"  Source : yfinance (pas l'API IB)")
    print(f"{'='*50}")

    import yfinance as yf
    end_date   = datetime.now()
    start_date = end_date - timedelta(days=30)

    print(f"\n[1/2] Récupération des 30 derniers jours pour {symbol}...")
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval="1d", auto_adjust=False)
        if df is None or df.empty:
            print(f"  ❌ Aucune donnée retournée pour {symbol}")
            return False
        print(f"  ✅ {len(df)} barres reçues")
    except Exception as e:
        print(f"  ❌ Erreur yfinance : {e}")
        return False

    print(f"\n[2/2] Dernières barres {symbol} :")
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    for _, row in df.tail(5).iterrows():
        d = str(row['date'])[:10]
        print(f"  {d}  O={row['open']:8.2f}  H={row['high']:8.2f}  "
              f"L={row['low']:8.2f}  C={row['close']:8.2f}  Vol={int(row['volume']):>12,}")

    last_close = df['close'].iloc[-1]
    print(f"\n  ✅ Dernier close {symbol} : {last_close:.2f} $")
    print(f"\n  ℹ️  Les données historiques fonctionnent — aucune souscription IB requise.")
    return True


def test_l2_data(port: int, symbol: str = "AAPL", client_id: int = 98):
    """
    Teste la disponibilité des données L2 (market depth) avec le compte paper.
    Utilise d'abord reqMktDepthExchanges() pour identifier les exchanges valides,
    puis tente reqMktDepth() dessus.
    Codes d'erreur clés :
      354  → souscription manquante
      310  → cancel sur une souscription jamais créée (= rejet silencieux du reqMktDepth)
      10092→ exchange non supporté pour ce type de contrat
    """
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    import threading

    print(f"\n{'='*50}")
    print(f"  TEST DONNÉES L2 (MARKET DEPTH) — {symbol}")
    print(f"  Port : {port}  |  ClientID : {client_id}")
    print(f"{'='*50}")

    REQ_EXCHANGES = 3000
    REQ_DEPTH_BASE = 2001

    class L2Tester(EWrapper, EClient):
        def __init__(self):
            EWrapper.__init__(self)
            EClient.__init__(self, self)
            self.ready = threading.Event()
            # L2 exchanges disponibles
            self.depth_exchanges = []
            self.depth_exchanges_done = threading.Event()
            # Résultat par req_id
            self.results = {}   # req_id -> {'rows': [], 'error_code': None, 'error_msg': None}
            self.events  = {}   # req_id -> threading.Event

        def nextValidId(self, orderId):
            self.ready.set()

        def mktDepthExchanges(self, depthMktDataDescriptions):
            for d in depthMktDataDescriptions:
                self.depth_exchanges.append({
                    'exchange':   d.exchange,
                    'secType':    d.secType,
                    'listingExch':getattr(d, 'listingExch', ''),
                    'serviceDataType': getattr(d, 'serviceDataType', ''),
                })
            self.depth_exchanges_done.set()

        def updateMktDepth(self, reqId, position, operation, side, price, size):
            if reqId in self.results:
                self.results[reqId]['rows'].append(
                    {'pos': position, 'side': 'BID' if side == 1 else 'ASK',
                     'price': price, 'size': size})
                self.events[reqId].set()

        def updateMktDepthL2(self, reqId, position, marketMaker, operation, side, price, size, isSmartDepth):
            if reqId in self.results:
                self.results[reqId]['rows'].append(
                    {'pos': position, 'mm': marketMaker,
                     'side': 'BID' if side == 1 else 'ASK',
                     'price': price, 'size': size})
                self.events[reqId].set()

        def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
            if reqId in self.results:
                self.results[reqId]['error_code'] = errorCode
                self.results[reqId]['error_msg']  = errorString
                self.events[reqId].set()

    tester = L2Tester()

    print("\n[1/4] Connexion TCP...")
    tester.connect("127.0.0.1", port, client_id)
    thread = threading.Thread(target=tester.run, daemon=True)
    thread.start()

    if not tester.ready.wait(timeout=10):
        print("  ❌ Timeout — TWS non joignable sur ce port.")
        tester.disconnect()
        return False
    print("  ✅ Connexion OK")

    # ── Étape 2 : exchanges qui supportent le market depth ──────────────────
    print("\n[2/4] Interrogation des exchanges L2 disponibles (reqMktDepthExchanges)...")
    tester.reqMktDepthExchanges()
    tester.depth_exchanges_done.wait(timeout=5)

    stk_exchanges = [d['exchange'] for d in tester.depth_exchanges if d['secType'] == 'STK']
    if stk_exchanges:
        print(f"  ✅ Exchanges STK supportés : {', '.join(stk_exchanges)}")
    else:
        print("  ⚠️  Aucun exchange L2 retourné — le compte ne semble pas avoir de souscription depth.")
        print("     (en paper trading, les exchanges L2 n'apparaissent que si la souscription est active)")
        stk_exchanges = ["ISLAND", "NYSE", "ARCA"]   # fallback

    # ── Étape 3 : reqMktDepth sur chaque exchange ───────────────────────────
    print(f"\n[3/4] Tentative reqMktDepth sur {symbol}...")
    final_rows       = []
    final_error_code = None
    final_error_msg  = None
    success_exchange = None

    for offset, exchange in enumerate(stk_exchanges[:4]):
        req_id = REQ_DEPTH_BASE + offset
        tester.results[req_id] = {'rows': [], 'error_code': None, 'error_msg': None}
        tester.events[req_id]  = threading.Event()

        contract = Contract()
        contract.symbol   = symbol
        contract.secType  = "STK"
        contract.currency = "USD"
        contract.exchange = exchange

        tester.reqMktDepth(req_id, contract, 5, False, [])
        tester.events[req_id].wait(timeout=5)

        r = tester.results[req_id]

        # Annuler proprement — ignorer l'erreur 310 qui suit le cancel
        try:
            tester.cancelMktDepth(req_id, False)
        except Exception:
            pass
        time.sleep(0.2)

        if r['rows']:
            final_rows       = r['rows']
            success_exchange = exchange
            break
        elif r['error_code'] == 354:
            print(f"  ❌ {exchange} : souscription L2 manquante (erreur 354)")
            final_error_code = 354
            final_error_msg  = r['error_msg']
            break
        elif r['error_code'] in (310, None):
            # 310 = cancel sur une souscription inexistante → rejet silencieux du subscribe
            # None = timeout sans réponse → même diagnostic
            label = f"erreur {r['error_code']}" if r['error_code'] else "timeout"
            print(f"  ⚠️  {exchange} : pas de souscription créée ({label}), essai suivant...")
            final_error_code = r['error_code']
            final_error_msg  = r['error_msg']
        elif r['error_code'] == 10092:
            print(f"  ⚠️  {exchange} : non supporté pour STK (erreur 10092), essai suivant...")
        else:
            print(f"  ⚠️  {exchange} : erreur {r['error_code']} — {r['error_msg']}")

    tester.disconnect()
    time.sleep(0.3)

    # ── Étape 4 : diagnostic ────────────────────────────────────────────────
    print(f"\n[4/4] Diagnostic :")
    if final_rows:
        print(f"  ✅ L2 DISPONIBLE sur {success_exchange} — {len(final_rows)} niveau(x) reçus :")
        for r in final_rows[:5]:
            mm = f"  MM={r.get('mm','')}" if 'mm' in r else ''
            print(f"     [{r['side']}] pos={r['pos']}  prix={r['price']:.2f}  taille={r['size']}{mm}")
        print(f"\n  ✅ Les données L2 fonctionnent sur ce compte paper.")
        return True
    elif final_error_code == 354:
        print(f"  ❌ L2 NON SOUSCRIT — la souscription market depth est absente sur ce compte paper.")
        print("     Pour l'activer :")
        print("     TWS → Account Management → Market Data Subscriptions")
        print("     → 'NASDAQ TotalView' (ISLAND)  et/ou  'NYSE OpenBook'")
        print("     (souvent gratuit ou inclus en paper trading)")
        return False
    elif final_error_code == 310 or (not stk_exchanges and final_error_code is None):
        print(f"  ❌ L2 NON DISPONIBLE — reqMktDepth n'a créé aucune souscription active.")
        print("     Cause probable : souscription 'NASDAQ TotalView' / 'NYSE OpenBook' absente.")
        print("     → TWS → Account Management → Market Data Subscriptions")
        return False
    else:
        print(f"  ❌ L2 inaccessible — dernière erreur : {final_error_code} — {final_error_msg}")
        return False


def test_connection(port: int, client_id: int = 99):
    print(f"\n{'='*50}")
    print(f"  TEST CONNECTIVITÉ IB")
    print(f"  Host : 127.0.0.1")
    print(f"  Port : {port}")
    print(f"  ClientID : {client_id}")
    print(f"{'='*50}")

    mdp = MarketDataProvider(host="127.0.0.1", port=port, client_id=client_id)

    # 1. Connexion TCP
    print("\n[1/4] Connexion TCP...")
    connected = mdp.connect()
    if not connected:
        print("  ❌ Échec — TWS/Gateway n'est pas démarré ou le port est incorrect.")
        print("     Vérifie que TWS est ouvert et que l'API est activée :")
        print("     TWS → File → Global Configuration → API → Settings")
        print("     ✓ Enable ActiveX and Socket Clients")
        print(f"     ✓ Socket port = {port}")
        print("     ✓ Allow connections from localhost")
        return False
    print("  ✅ Connexion TCP établie")

    # 2. Attente nextValidId (preuve que l'API répond)
    print("\n[2/4] Attente initialisation API (nextValidId)...")
    ready = mdp.wait_until_ready(timeout=15)
    if not ready:
        print("  ❌ Timeout — TWS connecté mais API ne répond pas.")
        print("     Vérifie les permissions API dans TWS.")
        mdp.disconnect()
        return False
    print("  ✅ API prête")

    # 3. Récupération du cash disponible (vérification des droits de compte)
    print("\n[3/4] Récupération du solde du compte...")
    try:
        cash = mdp.req_account_cash()
        if cash >= 0:
            print(f"  ✅ Solde disponible : {cash:,.2f} $")
        else:
            print("  ⚠️  Solde non disponible (API connectée mais compte non accessible)")
    except Exception as e:
        print(f"  ⚠️  Erreur lors de la récupération du solde : {e}")

    # 4. Récupération des positions ouvertes
    print("\n[4/4] Récupération des positions ouvertes...")
    try:
        positions = mdp.get_ib_positions()
        if positions:
            print(f"  ✅ {len(positions)} position(s) ouverte(s) :")
            for pos in positions:
                print(f"     {pos['symbol']:10s}  qty={pos['qty']:6}  avg_cost={pos['avg_cost']:.2f}")
        else:
            print("  ✅ Aucune position ouverte")
    except Exception as e:
        print(f"  ⚠️  Erreur positions : {e}")

    # Déconnexion propre
    mdp.disconnect()
    time.sleep(0.5)

    print(f"\n{'='*50}")
    print("  ✅ CONNEXION IB OPÉRATIONNELLE")
    print(f"{'='*50}\n")
    return True


def test_fundamental_data(symbol: str = "AAPL"):
    """
    Teste la disponibilité des données fondamentales EDGAR dans PostgreSQL
    et le calcul des scores Piotroski et EBIT/TEV.
    """
    from datetime import date
    print(f"\n{'='*50}")
    print(f"  TEST DONNÉES FONDAMENTALES — {symbol}")
    print(f"  Source : PostgreSQL (fundamental_cache EDGAR)")
    print(f"{'='*50}")

    # 1. Connexion PostgreSQL
    print("\n[1/4] Connexion PostgreSQL...")
    try:
        from src.app.database.pg_connection import read_sql
        df_test = read_sql("SELECT COUNT(*) as n FROM fundamental_cache WHERE source = 'edgar'")
        n_total = int(df_test['n'].iloc[0])
        print(f"  ✅ PostgreSQL OK — {n_total:,} entrées EDGAR dans fundamental_cache")
    except Exception as e:
        print(f"  ❌ PostgreSQL inaccessible : {e}")
        return False

    # 2. Données disponibles pour le symbole
    print(f"\n[2/4] Données EDGAR pour {symbol}...")
    try:
        df_sym = read_sql(
            "SELECT period, fiscal_year, fiscal_period, updated_at "
            "FROM fundamental_cache WHERE symbol = %s AND source = 'edgar' "
            "ORDER BY updated_at DESC LIMIT 5",
            (symbol,)
        )
        if df_sym.empty:
            print(f"  ⚠️  Aucune donnée EDGAR pour {symbol} dans fundamental_cache")
        else:
            print(f"  ✅ {len(df_sym)} rapport(s) disponible(s) pour {symbol} :")
            for _, row in df_sym.iterrows():
                print(f"     period={str(row['period'])[:10]}  "
                      f"{row['fiscal_year']}Q{row['fiscal_period']}  "
                      f"mis_à_jour={str(row['updated_at'])[:10]}")
    except Exception as e:
        print(f"  ⚠️  Erreur lecture fundamental_cache : {e}")

    # 3. Score Piotroski
    print(f"\n[3/4] Calcul Piotroski F-Score pour {symbol}...")
    try:
        from src.app.value.fundamental_filters import FundamentalFilters
        ff = FundamentalFilters(backtest_mode=False, max_workers=1)
        trade_date = date.today()
        fscore = ff.calc_piotroski(symbol, trade_date=trade_date)
        if fscore is None:
            print(f"  ⚠️  Score Piotroski non calculable (données insuffisantes pour {symbol})")
        else:
            quality = "FORT" if fscore >= 7 else "MOYEN" if fscore >= 5 else "FAIBLE"
            print(f"  ✅ Piotroski F-Score : {fscore}/9 ({quality})")
    except Exception as e:
        print(f"  ⚠️  Erreur calcul Piotroski : {e}")

    # 4. Ratio EBIT/TEV
    print(f"\n[4/4] Calcul EBIT/TEV pour {symbol}...")
    try:
        import yfinance as yf
        price = yf.Ticker(symbol).fast_info.last_price or 0.0
        ratio = ff.calc_ebit_tev(symbol, trade_date=trade_date, current_price=price)
        if ratio is None:
            print(f"  ⚠️  EBIT/TEV non calculable (données insuffisantes pour {symbol})")
        else:
            print(f"  ✅ EBIT/TEV : {ratio*100:.2f}%  (prix utilisé : {price:.2f} $)")
    except Exception as e:
        print(f"  ⚠️  Erreur calcul EBIT/TEV : {e}")

    print(f"\n{'='*50}\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test de connectivité IB (aucun ordre placé)")
    parser.add_argument('--live',        action='store_true', help='Live trading (port 7496/4001)')
    parser.add_argument('--gateway',     action='store_true', help='IB Gateway (ports 4002/4001) au lieu de TWS (7497/7496)')
    parser.add_argument('--client',      type=int, default=99, help='Client ID (défaut: 99)')
    parser.add_argument('--data-only',   action='store_true', help='Tester yfinance + fondamentaux uniquement (sans TWS)')
    parser.add_argument('--symbol',      type=str, default='AAPL', help='Symbole pour les tests (défaut: AAPL)')
    parser.add_argument('--no-fundamental', action='store_true', help='Passer le test fondamental')
    parser.add_argument('--l2',            action='store_true', help='Tester la disponibilité des données L2 (market depth)')
    args = parser.parse_args()

    # Test données historiques yfinance (toujours)
    test_historical_data(args.symbol)

    # Test données fondamentales PostgreSQL
    if not args.no_fundamental:
        test_fundamental_data(args.symbol)

    # Test IB (sauf si --data-only)
    if not args.data_only:
        if args.gateway:
            port = 4001 if args.live else 4002
            mode = "Gateway LIVE" if args.live else "Gateway PAPER"
        else:
            port = 7496 if args.live else 7497
            mode = "TWS LIVE" if args.live else "TWS PAPER"

        print(f"\nMode IB : {mode}")
        test_connection(port=port, client_id=args.client)

        if args.l2:
            test_l2_data(port=port, symbol=args.symbol, client_id=args.client - 1)
