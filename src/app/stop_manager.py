"""
StopManager — gestion indépendante des stops de protection et des prises de bénéfice.

Principe anti stop-hunters : les niveaux ne sont JAMAIS placés dans le carnet IB.
Le StopManager surveille les prix en temps réel et envoie l'ordre seulement
quand le niveau est atteint (stop virtuel).

Configuration : stop_config.json à la racine du projet.

Usage autonome :
    python src/app/stop_manager.py               # boucle de surveillance
    python src/app/stop_manager.py --setup-only  # configure les stops sans boucle
    python src/app/stop_manager.py --show        # affiche les niveaux actifs
"""
import json
import os
import sqlite3
import sys
import time
import yfinance as yf
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ibapi.order import Order
from ibapi.tag_value import TagValue

from src.app.screener.providers.market_data_provider import MarketDataProvider
from src.app.database.trade_journal import TradeJournal, TradeMode


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class StopConfig:
    # Protection
    protection_type: str = "trailing"   # "fixed" | "trailing"
    protection_pct: float = 8.0

    # Profit
    profit_type: str = "fixed"          # "fixed" | "dynamic"
    profit_fixed_pct: float = 15.0
    profit_atr_mult: float = 2.0
    profit_atr_period: int = 14

    # Exécution
    use_darkice_for_profit: bool = True
    darkice_start: str = "09:35:00"
    darkice_end: str = "15:45:00"
    protection_order_type: str = "MKT"

    # Surveillance
    check_interval_sec: int = 300
    active_hours_start: str = "09:30"
    active_hours_end: str = "16:00"

    @classmethod
    def from_json(cls, path: str) -> "StopConfig":
        """Charge la config depuis stop_config.json."""
        if not os.path.exists(path):
            print(f"[CONFIG] {path} introuvable, utilisation des valeurs par défaut")
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            protection_type=data["protection"]["type"],
            protection_pct=data["protection"]["pct"],
            profit_type=data["profit"]["type"],
            profit_fixed_pct=data["profit"]["fixed_pct"],
            profit_atr_mult=data["profit"]["dynamic_atr_mult"],
            profit_atr_period=data["profit"]["dynamic_atr_period"],
            use_darkice_for_profit=data["execution"]["use_darkice_for_profit"],
            darkice_start=data["execution"]["darkice_start"],
            darkice_end=data["execution"]["darkice_end"],
            protection_order_type=data["execution"]["protection_order_type"],
            check_interval_sec=data["monitoring"]["check_interval_sec"],
            active_hours_start=data["monitoring"]["active_hours_start"],
            active_hours_end=data["monitoring"]["active_hours_end"],
        )

    def summary(self) -> str:
        lines = [
            f"  Protection : {self.protection_type.upper()} -{self.protection_pct}%"
            f"  (ordre {self.protection_order_type})",
            f"  Profit     : {self.profit_type.upper()} "
            + (f"+{self.profit_fixed_pct}%" if self.profit_type == "fixed"
               else f"{self.profit_atr_mult}×ATR({self.profit_atr_period})")
            + (f"  [DarkIce {self.darkice_start}-{self.darkice_end}]"
               if self.use_darkice_for_profit else "  [LMT simple]"),
            f"  Surveillance : toutes les {self.check_interval_sec}s"
            f"  ({self.active_hours_start}–{self.active_hours_end})",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# StopManager
# ---------------------------------------------------------------------------

class StopManager:
    """
    Gère les stops virtuels et prises de bénéfice pour toutes les positions ouvertes.

    Flux :
      1. setup_positions()  → calcule et stocke les niveaux en DB pour chaque position
      2. scan()             → toutes les N secondes, vérifie les prix et déclenche si atteint
    """

    def __init__(
        self,
        config: StopConfig,
        market_data: MarketDataProvider,
        db_path: str = "trading_data.db",
        trade_mode: TradeMode = TradeMode.PAPER,
    ):
        self.config = config
        self.market_data = market_data
        self.db_path = db_path
        self.trade_mode = trade_mode
        # Symboles pour lesquels un SELL a été placé mais pas encore confirmé dans IB.
        # Évite la race condition : stop déclenché → LMT en vol → scan suivant voit
        # encore la position dans IB → recrée un stop → 2e SELL → position short.
        self._pending_sells: set = set()
        self._create_table()
        print(f"[STOPS] StopManager initialisé\n{config.summary()}")

    # ------------------------------------------------------------------
    # DB
    # ------------------------------------------------------------------

    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_table(self):
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS position_stops (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol           TEXT    NOT NULL,
                trade_id         INTEGER,
                trade_mode       TEXT    NOT NULL,

                -- Prix d'entrée (référence)
                entry_price      REAL    NOT NULL,
                qty_remaining    INTEGER NOT NULL,

                -- Stop de protection
                protection_type  TEXT    NOT NULL,   -- fixed | trailing
                protection_pct   REAL    NOT NULL,
                stop_level       REAL    NOT NULL,
                high_water_mark  REAL    NOT NULL,   -- pour trailing

                -- Prise de bénéfice
                profit_type      TEXT    NOT NULL,   -- fixed | dynamic
                profit_pct       REAL,               -- pour fixed
                profit_atr_mult  REAL,               -- pour dynamic
                profit_level     REAL    NOT NULL,

                -- Etat
                active           INTEGER NOT NULL DEFAULT 1,
                triggered_type   TEXT,               -- 'stop' | 'profit' | NULL
                triggered_at     DATETIME,
                last_checked     DATETIME,
                created_at       DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_position_stops_active
            ON position_stops(active, trade_mode)
        """)
        # Index partiel : un seul stop actif par symbole/mode (remplace l'ancien UNIQUE sur active)
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_position_stops_one_active
            ON position_stops(symbol, trade_mode) WHERE active = 1
        """)
        # Migration DB existante : nettoyer les doublons inactifs anciens
        conn.execute("""
            DELETE FROM position_stops
            WHERE active = 0 AND id NOT IN (
                SELECT MAX(id) FROM position_stops
                WHERE active = 0
                GROUP BY symbol, trade_mode
            )
        """)
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Calcul des niveaux
    # ------------------------------------------------------------------

    def _calc_stop_level(self, entry_price: float) -> float:
        """Niveau initial de stop (identique pour fixed et trailing au départ)."""
        return round(entry_price * (1 - self.config.protection_pct / 100), 4)

    def _calc_profit_level(self, entry_price: float, atr: float = 0.0) -> float:
        """Niveau de prise de bénéfice."""
        if self.config.profit_type == "fixed":
            return round(entry_price * (1 + self.config.profit_fixed_pct / 100), 4)
        else:
            # dynamic : entry + N × ATR
            if atr <= 0:
                # Fallback sur fixed si ATR non dispo
                print(f"[STOPS] ATR non disponible, fallback sur fixed {self.config.profit_fixed_pct}%")
                return round(entry_price * (1 + self.config.profit_fixed_pct / 100), 4)
            return round(entry_price + self.config.profit_atr_mult * atr, 4)

    def _get_atr(self, symbol: str) -> float:
        """Calcule l'ATR(N) depuis la DB historique."""
        try:
            conn = self._connect()
            cursor = conn.cursor()
            n = self.config.profit_atr_period
            cursor.execute("""
                SELECT high, low, close FROM historical_data
                WHERE symbol = ? ORDER BY date DESC LIMIT ?
            """, (symbol, n + 1))
            rows = cursor.fetchall()
            conn.close()
            if len(rows) < 2:
                return 0.0
            # True Range simplifié
            trs = []
            for i in range(len(rows) - 1):
                h, l, prev_c = rows[i]["high"], rows[i]["low"], rows[i + 1]["close"]
                trs.append(max(h - l, abs(h - prev_c), abs(l - prev_c)))
            return sum(trs) / len(trs) if trs else 0.0
        except Exception as e:
            print(f"[STOPS] Erreur ATR pour {symbol}: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # Configuration des stops pour une position
    # ------------------------------------------------------------------

    def setup_position(self, symbol: str, entry_price: float, qty: int,
                       trade_id: int = None) -> bool:
        """
        Calcule et stocke les niveaux de stop/profit pour une position.
        Appelé après chaque entrée en position.
        """
        atr = self._get_atr(symbol) if self.config.profit_type == "dynamic" else 0.0
        stop_level = self._calc_stop_level(entry_price)
        profit_level = self._calc_profit_level(entry_price, atr)

        conn = self._connect()
        try:
            # Supprimer l'historique inactif puis désactiver l'ancienne config
            conn.execute("""
                DELETE FROM position_stops
                WHERE symbol = ? AND trade_mode = ? AND active = 0
            """, (symbol, self.trade_mode.value))
            conn.execute("""
                UPDATE position_stops SET active = 0
                WHERE symbol = ? AND trade_mode = ? AND active = 1
            """, (symbol, self.trade_mode.value))

            conn.execute("""
                INSERT INTO position_stops (
                    symbol, trade_id, trade_mode,
                    entry_price, qty_remaining,
                    protection_type, protection_pct, stop_level, high_water_mark,
                    profit_type, profit_pct, profit_atr_mult, profit_level,
                    active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (
                symbol, trade_id, self.trade_mode.value,
                entry_price, qty,
                self.config.protection_type,
                self.config.protection_pct,
                stop_level,
                entry_price,           # high_water_mark = entry au départ
                self.config.profit_type,
                self.config.profit_fixed_pct if self.config.profit_type == "fixed" else None,
                self.config.profit_atr_mult if self.config.profit_type == "dynamic" else None,
                profit_level,
            ))
            conn.commit()
            print(f"[STOPS] {symbol}: stop={stop_level:.2f} "
                  f"({'trailing' if self.config.protection_type == 'trailing' else 'fixe'} "
                  f"-{self.config.protection_pct}%), "
                  f"profit={profit_level:.2f} "
                  f"({'ATR×' + str(self.config.profit_atr_mult) if self.config.profit_type == 'dynamic' else '+' + str(self.config.profit_fixed_pct) + '%'})")
            return True
        except Exception as e:
            print(f"[STOPS] Erreur setup_position {symbol}: {e}")
            return False
        finally:
            conn.close()

    def setup_all_open_positions(self):
        """
        Configure les stops pour toutes les positions ouvertes sans config active.
        TWS est la source de vérité pour les positions réelles ;
        la DB fournit le prix d'entrée. Fallback sur la DB si TWS ne répond pas.
        """
        ib_positions = self.market_data.req_ib_positions()
        ib_open = {p["symbol"]: int(abs(p["qty"])) for p in ib_positions if p["qty"] != 0}

        if not ib_open:
            # Fallback DB si TWS ne renvoie rien
            print("[STOPS] Positions TWS non disponibles — fallback sur la DB")
            conn = self._connect()
            cursor = conn.execute("""
                SELECT t.id, t.symbol, t.prix_entree, t.quantite_restante
                FROM trades t
                LEFT JOIN position_stops ps
                    ON ps.symbol = t.symbol AND ps.trade_mode = t.trade_mode AND ps.active = 1
                WHERE t.trade_mode = ? AND t.date_sortie IS NULL AND t.quantite_restante > 0
                  AND ps.id IS NULL
            """, (self.trade_mode.value,))
            rows = cursor.fetchall()
            conn.close()
            if not rows:
                print("[STOPS] Toutes les positions ont déjà une config de stop active")
                return
            print(f"[STOPS] Configuration des stops pour {len(rows)} position(s) sans stop actif...")
            for row in rows:
                self.setup_position(
                    symbol=row["symbol"],
                    entry_price=row["prix_entree"],
                    qty=row["quantite_restante"],
                    trade_id=row["id"]
                )
            return

        # Pour chaque position TWS sans stop actif, chercher le prix d'entrée en DB
        conn = self._connect()
        to_setup = []
        for symbol, ib_qty in ib_open.items():
            existing = conn.execute("""
                SELECT id FROM position_stops
                WHERE symbol = ? AND trade_mode = ? AND active = 1
            """, (symbol, self.trade_mode.value)).fetchone()
            if existing:
                continue

            row = conn.execute("""
                SELECT id, prix_entree FROM trades
                WHERE symbol = ? AND trade_mode = ? AND quantite_restante > 0
                ORDER BY date_entree DESC LIMIT 1
            """, (symbol, self.trade_mode.value)).fetchone()

            if row:
                to_setup.append((symbol, row["prix_entree"], ib_qty, row["id"]))
            else:
                print(f"[STOPS] {symbol}: position TWS ({ib_qty} actions) sans entrée en DB — stop non configuré")
        conn.close()

        if not to_setup:
            print("[STOPS] Toutes les positions ont déjà une config de stop active")
            return

        print(f"[STOPS] Configuration des stops pour {len(to_setup)} position(s) depuis TWS...")
        for symbol, entry_price, qty, trade_id in to_setup:
            self.setup_position(symbol=symbol, entry_price=entry_price, qty=qty, trade_id=trade_id)

    # ------------------------------------------------------------------
    # Mise à jour trailing
    # ------------------------------------------------------------------

    def _update_trailing(self, stop_id: int, symbol: str,
                         current_price: float, entry_price: float,
                         protection_pct: float) -> float:
        """
        Met à jour le high_water_mark et recalcule le stop si trailing.
        Retourne le nouveau stop_level.
        """
        conn = self._connect()
        cursor = conn.execute(
            "SELECT high_water_mark FROM position_stops WHERE id = ?", (stop_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return entry_price * (1 - protection_pct / 100)

        hwm = row["high_water_mark"]
        new_hwm = max(hwm, current_price)
        new_stop = round(new_hwm * (1 - protection_pct / 100), 4)

        if new_hwm > hwm:
            conn = self._connect()
            conn.execute("""
                UPDATE position_stops
                SET high_water_mark = ?, stop_level = ?, last_checked = ?
                WHERE id = ?
            """, (new_hwm, new_stop, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), stop_id))
            conn.commit()
            conn.close()
            print(f"[STOPS] {symbol}: trailing HWM {hwm:.2f}→{new_hwm:.2f}, "
                  f"stop ajusté → {new_stop:.2f}")

        return new_stop

    # ------------------------------------------------------------------
    # Exécution des ordres
    # ------------------------------------------------------------------

    def _place_protection_order(self, symbol: str, qty: int):
        """Ordre MKT de protection — exécution immédiate."""
        contract = self.market_data.create_contract(symbol)
        order = Order()
        order.action = "SELL"
        order.orderType = self.config.protection_order_type
        order.totalQuantity = qty
        order.eTradeOnly = False
        order.firmQuoteOnly = False
        order.tif = "DAY"
        order_id = self.market_data.placeOrder(contract, order)
        print(f"[STOPS] {symbol}: STOP déclenché → {self.config.protection_order_type} "
              f"SELL {qty} (orderId={order_id})")
        return order_id

    def _place_profit_order(self, symbol: str, qty: int, target_price: float):
        """Ordre LMT de prise de bénéfice, avec DarkIce si configuré, fallback LMT simple."""
        contract = self.market_data.create_contract(symbol)
        order = Order()
        order.action = "SELL"
        order.orderType = "LMT"
        order.lmtPrice = round(target_price, 2)
        order.totalQuantity = qty
        order.eTradeOnly = False
        order.firmQuoteOnly = False
        order.tif = "DAY"

        if self.config.use_darkice_for_profit:
            order.algoStrategy = "DarkIce"
            order.algoParams = [
                TagValue("displaySize", "0"),
                TagValue("startTime", self.config.darkice_start),
                TagValue("endTime", self.config.darkice_end),
            ]
            print(f"[STOPS] {symbol}: PROFIT target={target_price:.2f} → LMT DarkIce SELL {qty}")
        else:
            print(f"[STOPS] {symbol}: PROFIT target={target_price:.2f} → LMT SELL {qty}")

        # Réinitialiser avant de placer pour éviter un résidu d'un ordre précédent
        if hasattr(self.market_data, '_last_order_error'):
            self.market_data._last_order_error = None
        order_id = self.market_data.placeOrder(contract, order)

        # Vérifier si IB a retourné une erreur sur cet ordre (attente courte)
        import time
        time.sleep(2)
        if hasattr(self.market_data, '_last_order_error') and \
                self.market_data._last_order_error and \
                self.market_data._last_order_error.get('req_id') == order_id:
            error_msg = self.market_data._last_order_error.get('msg', '')
            print(f"[STOPS] {symbol}: DarkIce refusé ({error_msg}), fallback LMT simple")
            self.market_data._last_order_error = None
            # Fallback : LMT sans algo
            fallback = Order()
            fallback.action = "SELL"
            fallback.orderType = "LMT"
            fallback.lmtPrice = round(target_price, 2)
            fallback.totalQuantity = qty
            fallback.eTradeOnly = False
            fallback.firmQuoteOnly = False
            fallback.tif = "DAY"
            order_id = self.market_data.placeOrder(contract, fallback)
            print(f"[STOPS] {symbol}: PROFIT → LMT fallback SELL {qty} (orderId={order_id})")

        return order_id

    def _mark_triggered(self, stop_id: int, trigger_type: str):
        conn = self._connect()
        conn.execute("""
            DELETE FROM position_stops
            WHERE active = 0 AND (symbol, trade_mode) IN (
                SELECT symbol, trade_mode FROM position_stops WHERE id = ?
            )
        """, (stop_id,))
        conn.execute("""
            UPDATE position_stops
            SET active = 0, triggered_type = ?, triggered_at = ?
            WHERE id = ?
        """, (trigger_type, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), stop_id))
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Scan principal
    # ------------------------------------------------------------------

    def scan(self):
        """
        Vérifie tous les stops actifs.
        Commence par récupérer les positions réelles IB — désactive les stops
        orphelins (symboles fermés côté IB mais encore dans la DB).
        """
        # 1. Positions réelles dans IB
        ib_positions = self.market_data.req_ib_positions()
        ib_symbols = {p["symbol"] for p in ib_positions if p["qty"] != 0}
        # Source de vérité pour la quantité réelle (long uniquement)
        ib_qty_map = {p["symbol"]: int(abs(p["qty"])) for p in ib_positions if p["qty"] > 0}

        conn = self._connect()
        cursor = conn.execute("""
            SELECT * FROM position_stops
            WHERE active = 1 AND trade_mode = ?
        """, (self.trade_mode.value,))
        stops = cursor.fetchall()

        # 2. Désactiver les stops orphelins — seulement si IB a renvoyé des positions
        #    (si ib_symbols est vide à cause d'un échec IB, on ne touche rien)
        if ib_symbols:
            orphans = [s for s in stops if s["symbol"] not in ib_symbols]
            if orphans:
                for s in orphans:
                    conn.execute(
                        "DELETE FROM position_stops WHERE symbol = ? AND trade_mode = ? AND active = 0",
                        (s["symbol"], self.trade_mode.value)
                    )
                    conn.execute(
                        "UPDATE position_stops SET active = 0 WHERE id = ?",
                        (s["id"],)
                    )
                    # Position confirmée fermée → retirer du suivi des sells en cours
                    self._pending_sells.discard(s["symbol"])
                    print(f"[STOPS] {s['symbol']}: position fermée dans IB → stop désactivé en DB")
                conn.commit()
                stops = [s for s in stops if s["symbol"] in ib_symbols]

            # 3. Détecter les nouvelles positions TWS sans stop actif
            active_symbols = {s["symbol"] for s in stops}
            new_symbols = ib_symbols - active_symbols
            # Exclure les symboles avec un SELL en cours (ordre en vol, position encore visible dans IB)
            pending_new = new_symbols & self._pending_sells
            if pending_new:
                print(f"[STOPS] Ordre SELL en cours pour {pending_new} — setup ignoré (attente exécution IB)")
            new_symbols -= self._pending_sells
            if new_symbols:
                print(f"[STOPS] Nouvelles positions sans stop détectées: {new_symbols}")
                for symbol in new_symbols:
                    ib_qty = next(int(abs(p["qty"])) for p in ib_positions if p["symbol"] == symbol)
                    row = conn.execute("""
                        SELECT id, prix_entree FROM trades
                        WHERE symbol = ? AND trade_mode = ? AND quantite_restante > 0
                        ORDER BY date_entree DESC LIMIT 1
                    """, (symbol, self.trade_mode.value)).fetchone()
                    if row:
                        self.setup_position(symbol, row["prix_entree"], ib_qty, trade_id=row["id"])
                    else:
                        print(f"[STOPS] {symbol}: nouvelle position TWS sans entrée DB — stop non configuré")
        else:
            print("[STOPS] Positions IB non disponibles — vérification des orphelins ignorée")

        conn.close()

        if not stops:
            return

        # Récupérer les ordres SELL déjà actifs dans IB pour éviter les doublons
        # (un LMT DAY non exécuté la veille ne doit pas déclencher un 2e ordre)
        open_sell_symbols: set = set()
        try:
            open_orders = self.market_data.req_open_orders()
            open_sell_symbols = {
                o["symbol"] for o in open_orders
                if o["action"] == "SELL" and o.get("status") not in ("Cancelled", "Filled", "Inactive")
            }
            if open_sell_symbols:
                print(f"[STOPS] Ordres SELL actifs dans IB: {open_sell_symbols}")
        except Exception as e:
            print(f"[STOPS] Impossible de récupérer les ordres ouverts: {e}")

        print(f"[STOPS] Scan de {len(stops)} position(s)...")

        for s in stops:
            symbol = s["symbol"]
            stop_id = s["id"]
            qty = s["qty_remaining"]
            # Quantité réelle dans IB (source de vérité — évite de vendre plus que détenu)
            actual_qty = ib_qty_map.get(symbol, qty) if ib_qty_map else qty
            if actual_qty <= 0:
                print(f"[STOPS] {symbol}: qty IB=0 — position fermée, stop désactivé")
                self._mark_triggered(stop_id, "no_position")
                continue

            # Récupération du prix :
            #   - PAPER : yfinance en source principale (pas de subscription IB requise)
            #   - LIVE  : IB en source principale, yfinance en fallback
            current_price = 0.0
            if self.trade_mode == TradeMode.LIVE:
                current_price = self.market_data.get_current_price(symbol)

            if current_price <= 0:
                try:
                    ticker = yf.Ticker(symbol)
                    current_price = ticker.fast_info.last_price or 0.0
                    if current_price > 0:
                        src = "yfinance" if self.trade_mode == TradeMode.PAPER else "yfinance (fallback)"
                        print(f"[STOPS] {symbol}: {src} prix={current_price:.2f}")
                except Exception:
                    pass

            if current_price <= 0:
                print(f"[STOPS] {symbol}: prix indisponible, scan ignoré")
                continue

            # --- Mise à jour trailing ---
            if s["protection_type"] == "trailing":
                stop_level = self._update_trailing(
                    stop_id, symbol, current_price,
                    s["entry_price"], s["protection_pct"]
                )
            else:
                stop_level = s["stop_level"]

            profit_level = s["profit_level"]

            # Vérifier qu'aucun ordre SELL n'est déjà en cours dans IB
            if symbol in open_sell_symbols:
                print(f"[STOPS] {symbol}: ordre SELL déjà actif dans IB — scan ignoré")
                continue

            # --- Vérification stop de protection ---
            if current_price <= stop_level:
                print(f"[STOPS] {symbol}: STOP DECLENCHE "
                      f"prix={current_price:.2f} <= stop={stop_level:.2f} "
                      f"(qty IB={actual_qty})")
                self._place_protection_order(symbol, actual_qty)
                self._pending_sells.add(symbol)
                self._mark_triggered(stop_id, "stop")
                continue   # pas besoin de vérifier profit

            # --- Vérification prise de bénéfice ---
            if current_price >= profit_level:
                print(f"[STOPS] {symbol}: PROFIT DECLENCHE "
                      f"prix={current_price:.2f} >= target={profit_level:.2f} "
                      f"(qty IB={actual_qty})")
                self._place_profit_order(symbol, actual_qty, profit_level)
                self._pending_sells.add(symbol)
                self._mark_triggered(stop_id, "profit")
                continue

            # Mise à jour last_checked
            conn = self._connect()
            conn.execute(
                "UPDATE position_stops SET last_checked = ? WHERE id = ?",
                (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), stop_id)
            )
            conn.commit()
            conn.close()

    # ------------------------------------------------------------------
    # Boucle de surveillance
    # ------------------------------------------------------------------

    def _is_market_open(self) -> bool:
        now = datetime.now()
        if now.weekday() >= 5:
            return False
        h_start, m_start = map(int, self.config.active_hours_start.split(":"))
        h_end, m_end = map(int, self.config.active_hours_end.split(":"))
        start = now.replace(hour=h_start, minute=m_start, second=0, microsecond=0)
        end = now.replace(hour=h_end, minute=m_end, second=0, microsecond=0)
        return start <= now <= end

    def run_loop(self):
        """Boucle de surveillance — tourne pendant les heures de marché puis s'arrête."""
        print(f"[STOPS] Boucle de surveillance démarrée "
              f"(intervalle: {self.config.check_interval_sec}s)")
        try:
            while True:
                now = datetime.now()
                if self._is_market_open():
                    self.scan()
                else:
                    h_end, m_end = map(int, self.config.active_hours_end.split(":"))
                    end = now.replace(hour=h_end, minute=m_end, second=0, microsecond=0)
                    if now > end:
                        print("[STOPS] Marché fermé — arrêt de la surveillance")
                        break
                    print(f"[STOPS] Marché pas encore ouvert — prochain scan dans "
                          f"{self.config.check_interval_sec}s")
                time.sleep(self.config.check_interval_sec)
        except KeyboardInterrupt:
            print("[STOPS] Surveillance arrêtée")

    # ------------------------------------------------------------------
    # Affichage
    # ------------------------------------------------------------------

    def show(self):
        """Affiche les stops actifs."""
        conn = self._connect()
        cursor = conn.execute("""
            SELECT symbol, protection_type, protection_pct, stop_level,
                   high_water_mark, profit_type, profit_level,
                   entry_price, qty_remaining, last_checked
            FROM position_stops
            WHERE active = 1 AND trade_mode = ?
            ORDER BY symbol
        """, (self.trade_mode.value,))
        rows = cursor.fetchall()
        conn.close()

        print()
        print(f"{'='*75}")
        print(f"STOPS ACTIFS — mode {self.trade_mode.value.upper()}")
        print(f"{'='*75}")
        if not rows:
            print("Aucun stop actif")
        for r in rows:
            pnl_stop = (r["stop_level"] - r["entry_price"]) / r["entry_price"] * 100
            pnl_profit = (r["profit_level"] - r["entry_price"]) / r["entry_price"] * 100
            print(
                f"{r['symbol']:6} | entrée={r['entry_price']:.2f} | qty={r['qty_remaining']}"
                f"\n       stop  [{r['protection_type']:8}] {r['stop_level']:.2f}"
                f" ({pnl_stop:+.1f}%)"
                + (f"  HWM={r['high_water_mark']:.2f}" if r["protection_type"] == "trailing" else "")
                + f"\n       profit[{r['profit_type']:8}] {r['profit_level']:.2f}"
                f" ({pnl_profit:+.1f}%)"
                f"\n       vérifié: {r['last_checked'] or 'jamais'}"
            )
        print(f"{'='*75}\n")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main():
    import argparse

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    CONFIG_PATH = os.path.join(PROJECT_ROOT, 'stop_config.json')
    DB_PATH = os.path.join(PROJECT_ROOT, 'trading_data.db')

    parser = argparse.ArgumentParser(description="StopManager — surveillance des stops virtuels")
    parser.add_argument('--port', type=int, default=7497,
                        help='Port IB (7497=paper TWS, 7496=live TWS)')
    parser.add_argument('--live', action='store_true', help='Mode live trading')
    parser.add_argument('--setup-only', action='store_true',
                        help='Configure les stops sans démarrer la boucle')
    parser.add_argument('--show', action='store_true',
                        help='Affiche les stops actifs et quitte')
    args = parser.parse_args()

    trade_mode = TradeMode.LIVE if args.live else TradeMode.PAPER
    config = StopConfig.from_json(CONFIG_PATH)

    # Connexion IB
    port = 7496 if args.live else args.port
    import random
    mdp = MarketDataProvider(port=port, client_id=random.randint(20, 29))
    if not args.show:
        print(f"[STOPS] Connexion IB (port {port})...")
        if not mdp.connect():
            print("[STOPS] Échec connexion IB")
            sys.exit(1)
        if not mdp.wait_until_ready(timeout=15):
            print("[STOPS] IB non prêt")
            sys.exit(1)

    manager = StopManager(config, mdp, db_path=DB_PATH, trade_mode=trade_mode)

    try:
        if args.show:
            manager.show()
        elif args.setup_only:
            manager.setup_all_open_positions()
            manager.show()
        else:
            manager.setup_all_open_positions()
            manager.show()
            manager.run_loop()
    finally:
        if mdp.is_connected():
            mdp.disconnect()


if __name__ == "__main__":
    main()
