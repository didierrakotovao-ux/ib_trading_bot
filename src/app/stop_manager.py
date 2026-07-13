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
import sys
import time
import numpy as np
import yfinance as yf
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.app.database.pg_connection import get_conn, dict_cursor
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
    protection_pct: float = 5.0

    # Profit
    profit_type: str = "fixed"          # "fixed" | "dynamic"
    profit_fixed_pct: float = 8.0
    profit_atr_mult: float = 2.0
    profit_atr_period: int = 14

    # Barrière de temps — alignée sur le label triple-barrière du modèle ML :
    # une position qui n'a touché ni le stop ni le profit après N jours de
    # bourse est fermée (libère le capital pour un meilleur candidat)
    max_holding_days: int = 20          # jours de bourse ; 0 = désactivé

    # Exécution
    use_darkice_for_profit: bool = True
    use_darkice_for_protection: bool = True
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

        protection = data.get("protection", {})
        profit = data.get("profit", {})
        time_cfg = data.get("time", {})
        execution = data.get("execution", {})
        monitoring = data.get("monitoring", {})
        defaults = cls()

        return cls(
            protection_type=protection.get("type", defaults.protection_type),
            protection_pct=float(protection.get("pct", defaults.protection_pct)),
            profit_type=profit.get("type", defaults.profit_type),
            profit_fixed_pct=float(profit.get("fixed_pct", defaults.profit_fixed_pct)),
            profit_atr_mult=float(profit.get("dynamic_atr_mult", defaults.profit_atr_mult)),
            profit_atr_period=int(profit.get("dynamic_atr_period", defaults.profit_atr_period)),
            max_holding_days=int(time_cfg.get("max_holding_days", defaults.max_holding_days)),
            use_darkice_for_profit=bool(execution.get("use_darkice_for_profit", defaults.use_darkice_for_profit)),
            use_darkice_for_protection=bool(execution.get("use_darkice_for_protection", defaults.use_darkice_for_protection)),
            darkice_start=execution.get("darkice_start", defaults.darkice_start),
            darkice_end=execution.get("darkice_end", defaults.darkice_end),
            protection_order_type=execution.get("protection_order_type", defaults.protection_order_type),
            check_interval_sec=int(monitoring.get("check_interval_sec", defaults.check_interval_sec)),
            active_hours_start=monitoring.get("active_hours_start", defaults.active_hours_start),
            active_hours_end=monitoring.get("active_hours_end", defaults.active_hours_end),
        )

    def summary(self) -> str:
        lines = [
            f"  Protection : {self.protection_type.upper()} -{self.protection_pct}%"
            f"  (ordre {self.protection_order_type})",
            f"  Profit     : "
            + ("DÉSACTIVÉ (trailing-only)" if self.profit_type == "none"
               else self.profit_type.upper() + " "
               + (f"+{self.profit_fixed_pct}%" if self.profit_type == "fixed"
                  else f"{self.profit_atr_mult}×ATR({self.profit_atr_period})")
               + (f"  [DarkIce {self.darkice_start}-{self.darkice_end}]"
                  if self.use_darkice_for_profit else "  [LMT simple]")),
            f"  Temps      : "
            + (f"sortie après {self.max_holding_days} jours de bourse"
               if self.max_holding_days > 0 else "désactivée"),
                f"  Trigger STOP: "
                + ("DarkIce (fallback MKT)" if self.use_darkice_for_protection else f"{self.protection_order_type} direct"),
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
        db_path=None,
        trade_mode: TradeMode = TradeMode.PAPER,
    ):
        self.config = config
        self.market_data = market_data
        self._ib_port = market_data.port  # conserve le port pour les reconnexions
        self.trade_mode = trade_mode
        self._run_lock_conn = None
        # Clé de lock distincte paper/live pour éviter deux loops concurrentes sur le même mode.
        self._run_lock_key = 91001 if self.trade_mode == TradeMode.PAPER else 91002
        # Symboles pour lesquels un SELL a été placé mais pas encore confirmé dans IB.
        # Évite la race condition : stop déclenché → LMT en vol → scan suivant voit
        # encore la position dans IB → recrée un stop → 2e SELL → position short.
        self._pending_sells: set = set()
        self._create_table()
        print(f"[STOPS] StopManager initialisé\n{config.summary()}")

    def _acquire_run_lock(self) -> bool:
        """Empêche plusieurs instances concurrentes du stop_manager sur le même mode."""
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("SELECT pg_try_advisory_lock(%s)", (self._run_lock_key,))
            ok = cur.fetchone()[0]
            cur.close()
            if ok:
                self._run_lock_conn = conn
                return True
            conn.close()
            return False
        except Exception as e:
            print(f"[STOPS][WARN] Impossible d'acquérir le lock d'instance: {e}")
            return False

    def _release_run_lock(self):
        """Libère le lock d'instance stop_manager."""
        if self._run_lock_conn is None:
            return
        try:
            cur = self._run_lock_conn.cursor()
            cur.execute("SELECT pg_advisory_unlock(%s)", (self._run_lock_key,))
            cur.close()
        except Exception:
            pass
        try:
            self._run_lock_conn.close()
        except Exception:
            pass
        self._run_lock_conn = None

    # ------------------------------------------------------------------
    # DB
    # ------------------------------------------------------------------

    def _connect(self):
        return get_conn()

    def _create_table(self):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS position_stops (
                id               SERIAL PRIMARY KEY,
                symbol           TEXT    NOT NULL,
                trade_id         INTEGER,
                trade_mode       TEXT    NOT NULL,
                entry_price      DOUBLE PRECISION NOT NULL,
                qty_remaining    INTEGER NOT NULL,
                protection_type  TEXT    NOT NULL,
                protection_pct   DOUBLE PRECISION NOT NULL,
                stop_level       DOUBLE PRECISION NOT NULL,
                high_water_mark  DOUBLE PRECISION NOT NULL,
                profit_type      TEXT    NOT NULL,
                profit_pct       DOUBLE PRECISION,
                profit_atr_mult  DOUBLE PRECISION,
                profit_level     DOUBLE PRECISION NOT NULL,
                active           INTEGER NOT NULL DEFAULT 1,
                triggered_type   TEXT,
                triggered_at     TIMESTAMP,
                last_checked     TIMESTAMP,
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_position_stops_active
            ON position_stops(active, trade_mode)
        """)
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_position_stops_one_active
            ON position_stops(symbol, trade_mode) WHERE active = 1
        """)
        # Nettoyer les doublons inactifs anciens (garder le plus récent par symbole/mode)
        cur.execute("""
            DELETE FROM position_stops
            WHERE active = 0 AND id NOT IN (
                SELECT MAX(id) FROM position_stops
                WHERE active = 0
                GROUP BY symbol, trade_mode
            )
        """)
        conn.commit()
        cur.close()
        conn.close()

    # ------------------------------------------------------------------
    # Calcul des niveaux
    # ------------------------------------------------------------------

    def _calc_stop_level(self, entry_price: float) -> float:
        """Niveau initial de stop (identique pour fixed et trailing au départ)."""
        return round(entry_price * (1 - self.config.protection_pct / 100), 4)

    # Sentinelle "jamais atteint" quand la prise de bénéfice est désactivée
    # (profit.type = "none") — la colonne profit_level est NOT NULL en DB
    PROFIT_DISABLED_LEVEL = 9_999_999_999.0

    def _calc_profit_level(self, entry_price: float, atr: float = 0.0) -> float:
        """Niveau de prise de bénéfice."""
        if self.config.profit_type == "none":
            # Trailing-only : sortie uniquement par stop suiveur / temps
            # (aligné sur les backtests TrailingOnly — les runners au-delà
            # de +8% portent la performance)
            return self.PROFIT_DISABLED_LEVEL
        if self.config.profit_type == "fixed":
            return round(entry_price * (1 + self.config.profit_fixed_pct / 100), 4)
        else:
            # dynamic : entry + N × ATR
            if atr <= 0:
                # Fallback sur fixed si ATR non dispo
                print(f"[STOPS] ATR non disponible, fallback sur fixed {self.config.profit_fixed_pct}%")
                return round(entry_price * (1 + self.config.profit_fixed_pct / 100), 4)
            return round(entry_price + self.config.profit_atr_mult * atr, 4)

    def _get_fresh_ib_qty(self, symbol: str) -> int:
        """
        Interroge IB pour la quantité longue ACTUELLE d'un symbole.
        Appelé juste avant chaque SELL pour éviter les shorts si la position
        a bougé depuis le snapshot ib_qty_map du début du scan.
        """
        try:
            positions = self.market_data.req_ib_positions()
            for p in positions:
                if p["symbol"] == symbol:
                    return int(p["qty"])
            return 0
        except Exception as e:
            print(f"[STOPS] {symbol}: impossible de vérifier la position fraîche ({e})")
            return -1  # -1 = incertain, bloquer le SELL par sécurité

    def _get_atr(self, symbol: str) -> float:
        """Calcule l'ATR(N) depuis la DB historique."""
        try:
            conn = self._connect()
            cur = dict_cursor(conn)
            n = self.config.profit_atr_period
            cur.execute("""
                SELECT high, low, close FROM historical_data
                WHERE symbol = %s ORDER BY date DESC LIMIT %s
            """, (symbol, n + 1))
            rows = cur.fetchall()
            cur.close()
            conn.close()
            if len(rows) < 2:
                return 0.0
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

    def refresh_active_stops_to_config(self):
        """
        Recalcule les stops actifs selon la config courante (utile après changement de règles).
        N'altère pas active/triggered, met uniquement à jour les niveaux et paramètres.
        """
        conn = self._connect()
        cur = dict_cursor(conn)
        cur.execute("""
            SELECT id, symbol, entry_price, high_water_mark
            FROM position_stops
            WHERE active = 1 AND trade_mode = %s
        """, (self.trade_mode.value,))
        rows = cur.fetchall()
        cur.close()

        if not rows:
            conn.close()
            print("[STOPS] Aucun stop actif à recalculer")
            return

        ucur = conn.cursor()
        updated = 0
        for row in rows:
            symbol = row["symbol"]
            entry = float(row["entry_price"])
            hwm = max(float(row["high_water_mark"]), entry)
            stop_level = round(hwm * (1 - self.config.protection_pct / 100), 4)

            atr = self._get_atr(symbol) if self.config.profit_type == "dynamic" else 0.0
            profit_level = self._calc_profit_level(entry, atr)

            ucur.execute("""
                UPDATE position_stops
                SET protection_type = %s,
                    protection_pct = %s,
                    stop_level = %s,
                    high_water_mark = %s,
                    profit_type = %s,
                    profit_pct = %s,
                    profit_atr_mult = %s,
                    profit_level = %s,
                    last_checked = %s
                WHERE id = %s
            """, (
                self.config.protection_type,
                self.config.protection_pct,
                stop_level,
                hwm,
                self.config.profit_type,
                self.config.profit_fixed_pct if self.config.profit_type == "fixed" else None,
                self.config.profit_atr_mult if self.config.profit_type == "dynamic" else None,
                profit_level,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                row["id"],
            ))
            updated += 1

        conn.commit()
        ucur.close()
        conn.close()
        print(f"[STOPS] Politique appliquée à {updated} stop(s) actif(s)")

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
        cur = conn.cursor()
        try:
            cur.execute("""
                DELETE FROM position_stops
                WHERE symbol = %s AND trade_mode = %s AND active = 0
            """, (symbol, self.trade_mode.value))
            cur.execute("""
                UPDATE position_stops SET active = 0
                WHERE symbol = %s AND trade_mode = %s AND active = 1
            """, (symbol, self.trade_mode.value))
            cur.execute("""
                INSERT INTO position_stops (
                    symbol, trade_id, trade_mode,
                    entry_price, qty_remaining,
                    protection_type, protection_pct, stop_level, high_water_mark,
                    profit_type, profit_pct, profit_atr_mult, profit_level,
                    active
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,1)
            """, (
                symbol, trade_id, self.trade_mode.value,
                entry_price, qty,
                self.config.protection_type, self.config.protection_pct,
                stop_level, entry_price,
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
            conn.rollback()
            print(f"[STOPS] Erreur setup_position {symbol}: {e}")
            return False
        finally:
            cur.close()
            conn.close()

    def _recently_closed(self, symbol: str, conn, minutes: int = 60) -> bool:
        """
        Retourne True si ce symbole a été fermé (quantite_restante=0) dans les N dernières minutes.
        Protège contre la latence IB : une position vendue reste visible dans TWS
        quelques secondes/minutes, ce qui provoquerait une nouvelle entrée journal et un doublon SELL.
        """
        cur = dict_cursor(conn)
        cur.execute("""
            SELECT id FROM trades
            WHERE symbol = %s AND trade_mode = %s
              AND quantite_restante = 0
              AND date_sortie >= NOW() - (%s * INTERVAL '1 minute')
        """, (symbol, self.trade_mode.value, minutes))
        row = cur.fetchone()
        cur.close()
        return row is not None

    def _recently_triggered(self, symbol: str, conn, minutes: int = 90) -> bool:
        """
        Retourne True si un stop a été déclenché pour ce symbole dans les N dernières minutes.
        Protège contre le redémarrage du stop_manager après un trigger partiel :
        _pending_sells se réinitialise en mémoire, mais triggered_at persiste en DB.
        Évite qu'une position partiellement vendue (user a annulé une partie) déclenche
        un deuxième SELL qui créerait une position short.
        """
        cur = dict_cursor(conn)
        cur.execute("""
            SELECT id FROM position_stops
            WHERE symbol = %s AND trade_mode = %s AND active = 0
              AND triggered_at >= NOW() - (%s * INTERVAL '1 minute')
        """, (symbol, self.trade_mode.value, minutes))
        row = cur.fetchone()
        cur.close()
        return row is not None

    def _create_journal_entry(self, symbol: str, avg_cost: float, qty: int) -> Optional[int]:
        """
        Crée une entrée minimale dans trades quand trading.py n'a pas journalisé le fill.
        Utilise avg_cost IB comme prix d'entrée.
        Refuse si le symbole a été fermé récemment (position IB en cours de règlement).
        """
        try:
            conn = self._connect()
            if self._recently_closed(symbol, conn):
                print(f"[STOPS] {symbol}: trade fermé récemment — position IB en règlement, entrée ignorée")
                conn.close()
                return None
            # Attribution au régime courant (la vraie date d'entrée est
            # inconnue ici — position découverte dans TWS sans trace journal)
            from src.app.strategies.regime_selector import current_strategy_name
            strategy_name = current_strategy_name()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO trades
                    (trade_mode, strategy_name, symbol, date_entree,
                     prix_entree, quantite, quantite_restante)
                VALUES (%s, %s, %s, NOW(), %s, %s, %s)
                RETURNING id
            """, (self.trade_mode.value, strategy_name, symbol, avg_cost, qty, qty))
            trade_id = cur.fetchone()[0]
            conn.commit()
            cur.close()
            conn.close()
            print(f"[STOPS] {symbol}: entrée journal créée automatiquement "
                  f"(ID={trade_id}, avg_cost IB={avg_cost:.4f}, qty={qty})")
            return trade_id
        except Exception as e:
            print(f"[STOPS] {symbol}: impossible de créer l'entrée journal: {e}")
            return None

    def setup_all_open_positions(self):
        """
        Configure les stops pour toutes les positions ouvertes sans config active.
        TWS est la source de vérité pour les positions réelles ;
        la DB fournit le prix d'entrée. Fallback sur la DB si TWS ne répond pas.
        """
        ib_positions = self.market_data.req_ib_positions()
        ib_open = {p["symbol"]: int(abs(p["qty"])) for p in ib_positions if p["qty"] != 0}
        ib_avg_cost = {p["symbol"]: p["avg_cost"] for p in ib_positions if p["qty"] != 0}

        if not ib_open:
            # Fallback DB si TWS ne renvoie rien
            print("[STOPS] Positions TWS non disponibles — fallback sur la DB")
            conn = self._connect()
            cur = dict_cursor(conn)
            cur.execute("""
                SELECT t.id, t.symbol, t.prix_entree, t.quantite_restante
                FROM trades t
                LEFT JOIN position_stops ps
                    ON ps.symbol = t.symbol AND ps.trade_mode = t.trade_mode AND ps.active = 1
                WHERE t.trade_mode = %s AND t.date_sortie IS NULL AND t.quantite_restante > 0
                  AND ps.id IS NULL
            """, (self.trade_mode.value,))
            rows = cur.fetchall()
            cur.close()
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
        cur = dict_cursor(conn)
        to_setup = []
        for symbol, ib_qty in ib_open.items():
            cur.execute("""
                SELECT id FROM position_stops
                WHERE symbol = %s AND trade_mode = %s AND active = 1
            """, (symbol, self.trade_mode.value))
            existing = cur.fetchone()
            if existing:
                continue

            cur.execute("""
                SELECT id, prix_entree FROM trades
                WHERE symbol = %s AND trade_mode = %s AND quantite_restante > 0
                ORDER BY date_entree DESC LIMIT 1
            """, (symbol, self.trade_mode.value))
            row = cur.fetchone()

            if self._recently_triggered(symbol, conn):
                print(f"[STOPS] {symbol}: stop déclenché récemment — setup ignoré au redémarrage")
                continue

            if row:
                to_setup.append((symbol, row["prix_entree"], ib_qty, row["id"]))
            else:
                avg_cost = ib_avg_cost.get(symbol, 0.0)
                if avg_cost > 0:
                    trade_id = self._create_journal_entry(symbol, avg_cost, ib_qty)
                    if trade_id:
                        to_setup.append((symbol, avg_cost, ib_qty, trade_id))
                else:
                    print(f"[STOPS] {symbol}: position TWS sans entrée DB ni avg_cost — stop non configuré")
        cur.close()
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
        cur = dict_cursor(conn)
        cur.execute("SELECT high_water_mark FROM position_stops WHERE id = %s", (stop_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            return entry_price * (1 - protection_pct / 100)

        hwm = row["high_water_mark"]
        new_hwm = max(hwm, current_price)
        new_stop = round(new_hwm * (1 - protection_pct / 100), 4)

        if new_hwm > hwm:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("""
                UPDATE position_stops
                SET high_water_mark = %s, stop_level = %s, last_checked = %s
                WHERE id = %s
            """, (new_hwm, new_stop, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), stop_id))
            conn.commit()
            cur.close()
            conn.close()
            print(f"[STOPS] {symbol}: trailing HWM {hwm:.2f}->{new_hwm:.2f}, "
                  f"stop ajuste -> {new_stop:.2f}")

        return new_stop

    # ------------------------------------------------------------------
    # Exécution des ordres
    # ------------------------------------------------------------------

    def _place_protection_order(self, symbol: str, qty: int, trigger_price: float,
                                label: str = "STOP"):
        """Ordre de protection: DarkIce LMT (si activé) puis fallback direct."""
        contract = self.market_data.create_contract(symbol)

        if self.config.use_darkice_for_protection:
            darkice = Order()
            darkice.action = "SELL"
            darkice.orderType = "LMT"
            # Léger buffer sous le prix courant pour améliorer le remplissage immédiat.
            darkice.lmtPrice = round(trigger_price * 0.998, 2)
            darkice.totalQuantity = qty
            darkice.eTradeOnly = False
            darkice.firmQuoteOnly = False
            darkice.tif = "DAY"

            start_tz = self.config.darkice_start if " " in self.config.darkice_start \
                       else f"{self.config.darkice_start} US/Eastern"
            end_tz = self.config.darkice_end if " " in self.config.darkice_end \
                     else f"{self.config.darkice_end} US/Eastern"
            darkice.algoStrategy = "DarkIce"
            darkice.algoParams = [
                TagValue("displaySize", "0"),
                TagValue("startTime", start_tz),
                TagValue("endTime", end_tz),
            ]

            if hasattr(self.market_data, '_last_order_error'):
                self.market_data._last_order_error = None

            order_id = self.market_data.placeOrder(contract, darkice)
            print(f"[STOPS] {symbol}: {label} declenche -> DarkIce LMT SELL {qty} @ {darkice.lmtPrice:.2f} "
                  f"(orderId={order_id})")

            time.sleep(2)
            if hasattr(self.market_data, '_last_order_error') and \
                    self.market_data._last_order_error and \
                    self.market_data._last_order_error.get('req_id') == order_id:
                error_msg = self.market_data._last_order_error.get('msg', '')
                print(f"[STOPS] {symbol}: DarkIce {label} refuse ({error_msg}), fallback {self.config.protection_order_type}")
                self.market_data._last_order_error = None
                try:
                    self.market_data.cancelOrder(order_id)
                    time.sleep(1)
                except Exception:
                    pass
            else:
                return order_id

        order = Order()
        order.action = "SELL"
        order.orderType = self.config.protection_order_type
        order.totalQuantity = qty
        order.eTradeOnly = False
        order.firmQuoteOnly = False
        order.tif = "DAY"
        if order.orderType == "LMT":
            order.lmtPrice = round(trigger_price * 0.998, 2)
        order_id = self.market_data.placeOrder(contract, order)
        print(f"[STOPS] {symbol}: {label} declenche -> {self.config.protection_order_type} "
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
            # Format avec timezone explicite pour eviter l'avertissement IB code 2174
            start_tz = self.config.darkice_start if " " in self.config.darkice_start \
                       else f"{self.config.darkice_start} US/Eastern"
            end_tz   = self.config.darkice_end   if " " in self.config.darkice_end \
                       else f"{self.config.darkice_end} US/Eastern"
            order.algoParams = [
                TagValue("displaySize", "0"),
                TagValue("startTime", start_tz),
                TagValue("endTime",   end_tz),
            ]
            print(f"[STOPS] {symbol}: PROFIT target={target_price:.2f} -> LMT DarkIce SELL {qty}")
        else:
            print(f"[STOPS] {symbol}: PROFIT target={target_price:.2f} -> LMT SELL {qty}")

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
            print(f"[STOPS] {symbol}: DarkIce refuse ({error_msg}), annulation + fallback LMT")
            self.market_data._last_order_error = None
            # ANNULER l'ordre DarkIce avant de placer le fallback
            # (sans annulation, les deux ordres s'executent → position short)
            self.market_data.cancelOrder(order_id)
            time.sleep(1)   # laisser le temps à IB de traiter l'annulation
            # Fallback : LMT simple sans algo
            fallback = Order()
            fallback.action = "SELL"
            fallback.orderType = "LMT"
            fallback.lmtPrice = round(target_price, 2)
            fallback.totalQuantity = qty
            fallback.eTradeOnly = False
            fallback.firmQuoteOnly = False
            fallback.tif = "DAY"
            order_id = self.market_data.placeOrder(contract, fallback)
            print(f"[STOPS] {symbol}: PROFIT -> LMT fallback SELL {qty} (orderId={order_id})")

        return order_id

    def _mark_triggered(self, stop_id: int, trigger_type: str):
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            DELETE FROM position_stops
            WHERE active = 0 AND (symbol, trade_mode) IN (
                SELECT symbol, trade_mode FROM position_stops WHERE id = %s
            )
        """, (stop_id,))
        cur.execute("""
            UPDATE position_stops
            SET active = 0, triggered_type = %s, triggered_at = %s
            WHERE id = %s
        """, (trigger_type, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), stop_id))
        conn.commit()
        cur.close()
        conn.close()

    def _claim_trigger(self, stop_id: int, trigger_type: str) -> bool:
        """
        Claim atomique d'un stop avant envoi d'ordre.
        Retourne False si un autre process/thread l'a déjà pris.
        """
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            UPDATE position_stops
            SET active = 2, triggered_type = %s, triggered_at = %s
            WHERE id = %s AND active = 1
        """, (trigger_type, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), stop_id))
        claimed = cur.rowcount == 1
        conn.commit()
        cur.close()
        conn.close()
        return claimed

    def _finalize_trigger(self, stop_id: int):
        """Finalise un trigger déjà claimé (active=2 -> active=0)."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            DELETE FROM position_stops
            WHERE active = 0 AND (symbol, trade_mode) IN (
                SELECT symbol, trade_mode FROM position_stops WHERE id = %s
            )
        """, (stop_id,))
        cur.execute("UPDATE position_stops SET active = 0 WHERE id = %s AND active = 2", (stop_id,))
        conn.commit()
        cur.close()
        conn.close()

    def _rollback_trigger(self, stop_id: int):
        """Annule le claim en cas d'échec d'envoi d'ordre (active=2 -> active=1)."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("""
            UPDATE position_stops
            SET active = 1, triggered_type = NULL, triggered_at = NULL
            WHERE id = %s AND active = 2
        """, (stop_id,))
        conn.commit()
        cur.close()
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
        cur = dict_cursor(conn)
        cur.execute("""
            SELECT * FROM position_stops
            WHERE active = 1 AND trade_mode = %s
        """, (self.trade_mode.value,))
        stops = cur.fetchall()
        cur.close()

        # 2. Désactiver les stops orphelins — seulement si IB a renvoyé des positions
        #    (si ib_symbols est vide à cause d'un échec IB, on ne touche rien)
        if ib_symbols:
            orphans = [s for s in stops if s["symbol"] not in ib_symbols]
            if orphans:
                try:
                    ocur = conn.cursor()
                    for s in orphans:
                        ocur.execute(
                            "DELETE FROM position_stops WHERE symbol = %s AND trade_mode = %s AND active = 0",
                            (s["symbol"], self.trade_mode.value)
                        )
                        ocur.execute(
                            "UPDATE position_stops SET active = 0 WHERE id = %s",
                            (s["id"],)
                        )
                        self._pending_sells.discard(s["symbol"])
                        print(f"[STOPS] {s['symbol']}: position fermee dans IB -> stop desactive en DB")
                    conn.commit()
                    ocur.close()
                    stops = [s for s in stops if s["symbol"] in ib_symbols]
                except Exception as db_err:
                    conn.rollback()
                    print(f"[STOPS][WARN] DB locked lors nettoyage orphelins ({db_err}) — sera reessaye au prochain scan")

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
                    try:
                        ib_pos  = next(p for p in ib_positions if p["symbol"] == symbol)
                        # Ignorer les positions SHORT (qty < 0) — ne jamais configurer un stop sur un short
                        if ib_pos["qty"] < 0:
                            print(f"[STOPS] {symbol}: position SHORT ({ib_pos['qty']:.0f}) — stop ignore")
                            continue
                        ib_qty  = int(abs(ib_pos["qty"]))
                        avg_cost = ib_pos.get("avg_cost", 0.0)
                        ncur = dict_cursor(conn)
                        ncur.execute("""
                            SELECT id, prix_entree FROM trades
                            WHERE symbol = %s AND trade_mode = %s AND quantite_restante > 0
                            ORDER BY date_entree DESC LIMIT 1
                        """, (symbol, self.trade_mode.value))
                        row = ncur.fetchone()
                        ncur.close()
                        if row:
                            self.setup_position(symbol, row["prix_entree"], ib_qty, trade_id=row["id"])
                        elif self._recently_triggered(symbol, conn):
                            print(f"[STOPS] {symbol}: stop déclenché récemment — position en cours de règlement, setup ignoré")
                        elif self._recently_closed(symbol, conn):
                            print(f"[STOPS] {symbol}: trade fermé récemment — position IB en règlement, stop ignoré")
                        elif avg_cost > 0:
                            trade_id = self._create_journal_entry(symbol, avg_cost, ib_qty)
                            if trade_id:
                                self.setup_position(symbol, avg_cost, ib_qty, trade_id=trade_id)
                        else:
                            print(f"[STOPS] {symbol}: nouvelle position TWS sans entrée DB ni avg_cost — stop non configuré")
                    except Exception as setup_err:
                        import traceback as _tb
                        print(f"[STOPS][WARN] {symbol}: erreur setup nouvelle position — scan continue ({setup_err})")
                        _tb.print_exc()
        else:
            print("[STOPS] Positions IB non disponibles — vérification des orphelins ignorée")

        conn.close()

        if not stops:
            print(f"[STOPS] Aucun stop actif [{datetime.now().strftime('%H:%M:%S')}]")
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

        # Dates d'entrée pour la barrière de temps (une seule requête pour tous les stops)
        entry_dates = {}
        if self.config.max_holding_days > 0:
            trade_ids = [s["trade_id"] for s in stops if s["trade_id"]]
            if trade_ids:
                try:
                    tconn = self._connect()
                    tcur = dict_cursor(tconn)
                    tcur.execute("SELECT id, date_entree FROM trades WHERE id = ANY(%s)",
                                 (trade_ids,))
                    entry_dates = {r["id"]: r["date_entree"] for r in tcur.fetchall()}
                    tcur.close()
                    tconn.close()
                except Exception as e:
                    print(f"[STOPS] Impossible de récupérer les dates d'entrée: {e}")

        print(f"[STOPS] Scan de {len(stops)} position(s)... [{datetime.now().strftime('%H:%M:%S')}]")

        for s in stops:
          try:
            symbol = s["symbol"]
            stop_id = s["id"]
            qty = s["qty_remaining"]
            # Quantité réelle dans IB (source de vérité — évite de vendre plus que détenu)
            # Si IB a répondu (ib_qty_map non vide) mais le symbole n'est pas dans les longs,
            # c'est que la position est déjà SHORT ou nulle → ne pas vendre davantage.
            if ib_qty_map and symbol not in ib_qty_map:
                print(f"[STOPS] {symbol}: absent des longs IB (position short ou fermée) — stop désactivé")
                self._mark_triggered(stop_id, "no_long_position")
                continue
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
                print(f"[STOPS] {symbol}: STOP DECLENCHE [{datetime.now().strftime('%H:%M:%S')}] "
                      f"prix={current_price:.2f} <= stop={stop_level:.2f} "
                      f"(qty IB={actual_qty})")
                if not self._claim_trigger(stop_id, "stop"):
                    print(f"[STOPS] {symbol}: trigger déjà pris par une autre instance")
                    continue
                try:
                    # Vérification fraîche juste avant le SELL — évite les shorts
                    # si la position a bougé depuis le snapshot du début du scan
                    fresh_qty = self._get_fresh_ib_qty(symbol)
                    if fresh_qty <= 0:
                        print(f"[STOPS] {symbol}: position IB={fresh_qty} au moment du trigger "
                              f"(position fermée ou short) — SELL annulé")
                        self._rollback_trigger(stop_id)
                        continue
                    actual_qty = fresh_qty
                    self._place_protection_order(symbol, actual_qty, current_price)
                    self._pending_sells.add(symbol)
                    self._finalize_trigger(stop_id)
                except Exception as e:
                    print(f"[STOPS] {symbol}: échec envoi ordre stop ({e}) — rollback trigger")
                    self._rollback_trigger(stop_id)
                continue   # pas besoin de vérifier profit

            # --- Vérification prise de bénéfice ---
            # (jamais déclenchée si profit.type = "none" : niveau sentinelle)
            if s["profit_type"] != "none" and current_price >= profit_level:
                print(f"[STOPS] {symbol}: PROFIT DECLENCHE [{datetime.now().strftime('%H:%M:%S')}] "
                      f"prix={current_price:.2f} >= target={profit_level:.2f} "
                      f"(qty IB={actual_qty})")
                if not self._claim_trigger(stop_id, "profit"):
                    print(f"[STOPS] {symbol}: trigger profit déjà pris par une autre instance")
                    continue
                try:
                    fresh_qty = self._get_fresh_ib_qty(symbol)
                    if fresh_qty <= 0:
                        print(f"[STOPS] {symbol}: position IB={fresh_qty} au moment du profit "
                              f"(position fermée ou short) — SELL annulé")
                        self._rollback_trigger(stop_id)
                        continue
                    actual_qty = fresh_qty
                    self._place_profit_order(symbol, actual_qty, profit_level)
                    self._pending_sells.add(symbol)
                    self._finalize_trigger(stop_id)
                except Exception as e:
                    print(f"[STOPS] {symbol}: échec envoi ordre profit ({e}) — rollback trigger")
                    self._rollback_trigger(stop_id)
                continue

            # --- Vérification barrière de temps ---
            # Alignée sur le label triple-barrière du modèle : une position qui
            # n'a touché ni stop ni profit après max_holding_days jours de
            # bourse est fermée pour libérer le capital.
            if self.config.max_holding_days > 0:
                entry_date = entry_dates.get(s["trade_id"]) or s["created_at"]
                if entry_date is not None:
                    start = entry_date.date() if hasattr(entry_date, "date") else entry_date
                    held = int(np.busday_count(start, datetime.now().date()))
                    if held >= self.config.max_holding_days:
                        print(f"[STOPS] {symbol}: BARRIERE DE TEMPS [{datetime.now().strftime('%H:%M:%S')}] "
                              f"{held} jours de bourse >= {self.config.max_holding_days} "
                              f"(prix={current_price:.2f}, qty IB={actual_qty})")
                        if not self._claim_trigger(stop_id, "time"):
                            print(f"[STOPS] {symbol}: trigger temps déjà pris par une autre instance")
                            continue
                        try:
                            self._place_protection_order(symbol, actual_qty, current_price, label="TEMPS")
                            self._pending_sells.add(symbol)
                            self._finalize_trigger(stop_id)
                        except Exception as e:
                            print(f"[STOPS] {symbol}: échec envoi ordre sortie temps ({e}) — rollback trigger")
                            self._rollback_trigger(stop_id)
                        continue

            # Mise à jour last_checked
            conn = self._connect()
            lcur = conn.cursor()
            lcur.execute(
                "UPDATE position_stops SET last_checked = %s WHERE id = %s",
                (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), stop_id)
            )
            conn.commit()
            lcur.close()
            conn.close()
          except Exception as sym_err:
            import traceback as _tb
            _sym = s.get("symbol", "?") if isinstance(s, dict) else "?"
            print(f"[STOPS][WARN] {_sym}: exception isolée — scan continue ({sym_err})")
            _tb.print_exc()

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

    def _reconnect(self, max_attempts: int = 5, wait_sec: int = 60) -> bool:
        """
        Tente de reconnecter à IB après une déconnexion.
        Crée un nouveau MarketDataProvider avec un client_id différent à chaque
        tentative pour éviter l'erreur 326 (client_id déjà utilisé).
        """
        import random
        for attempt in range(1, max_attempts + 1):
            new_cid = random.randint(20, 99)
            print(f"[STOPS] Reconnexion IB tentative {attempt}/{max_attempts} (client_id={new_cid})...")
            try:
                # Déconnecter proprement l'ancienne instance
                try:
                    self.market_data.disconnect()
                except Exception:
                    pass
                time.sleep(5)
                # Nouvelle instance avec client_id différent (évite erreur 326)
                new_mdp = MarketDataProvider(port=self._ib_port, client_id=new_cid)
                if new_mdp.connect() and new_mdp.wait_until_ready(timeout=15):
                    self.market_data = new_mdp   # remplacer la référence
                    print("[STOPS] Reconnexion IB reussie")
                    self.setup_all_open_positions()
                    self.show()
                    return True
            except Exception as e:
                print(f"[STOPS] Tentative {attempt} echouee: {e}")
            if attempt < max_attempts:
                print(f"[STOPS] Prochaine tentative dans {wait_sec}s...")
                time.sleep(wait_sec)
        print(f"[STOPS] Reconnexion impossible apres {max_attempts} tentatives — arret")
        return False

    def run_loop(self):
        """Boucle de surveillance — tourne pendant les heures de marché puis s'arrête."""
        import traceback
        if not self._acquire_run_lock():
            print(f"[STOPS] Une autre instance tourne déjà pour le mode {self.trade_mode.value} — arrêt")
            return
        print(f"[STOPS] Boucle de surveillance démarrée "
              f"(intervalle: {self.config.check_interval_sec}s)")
        try:
            while True:
                now = datetime.now()
                if self._is_market_open():
                    # Vérification proactive de la connexion IB avant chaque scan
                    if not self.market_data.is_connected():
                        print(f"[STOPS] Connexion IB perdue [{now.strftime('%H:%M:%S')}] — reconnexion...")
                        if not self._reconnect():
                            # Reconnexion impossible : on continue en mode dégradé
                            # (prix via yfinance, ordres IB suspendus jusqu'à rétablissement)
                            print(f"[STOPS][WARN] Reconnexion IB échouée — mode dégradé, prochain essai au prochain cycle")
                        else:
                            continue    # Reprendre la boucle après reconnexion réussie
                    try:
                        self.scan()
                    except Exception as e:
                        print(f"[STOPS][ERREUR] Exception dans scan() à {now.strftime('%H:%M:%S')} : {e}")
                        traceback.print_exc()
                        print(f"[STOPS] Reprise dans {self.config.check_interval_sec}s...")
                else:
                    h_end, m_end = map(int, self.config.active_hours_end.split(":"))
                    end = now.replace(hour=h_end, minute=m_end, second=0, microsecond=0)
                    if now > end:
                        print(f"[STOPS] Marché fermé — arrêt de la surveillance [{now.strftime('%H:%M:%S')}]")
                        break
                    print(f"[STOPS] Marché pas encore ouvert — prochain scan dans "
                          f"{self.config.check_interval_sec}s")
                time.sleep(self.config.check_interval_sec)
        except KeyboardInterrupt:
            print("[STOPS] Surveillance arrêtée")
        finally:
            self._release_run_lock()

    # ------------------------------------------------------------------
    # Affichage
    # ------------------------------------------------------------------

    def show(self):
        """Affiche les stops actifs."""
        conn = self._connect()
        cur = dict_cursor(conn)
        cur.execute("""
            SELECT symbol, protection_type, protection_pct, stop_level,
                   high_water_mark, profit_type, profit_level,
                   entry_price, qty_remaining, last_checked
            FROM position_stops
            WHERE active = 1 AND trade_mode = %s
            ORDER BY symbol
        """, (self.trade_mode.value,))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        print()
        print(f"{'='*75}")
        print(f"STOPS ACTIFS — mode {self.trade_mode.value.upper()}")
        print(f"{'='*75}")
        if not rows:
            print("Aucun stop actif")
        for r in rows:
            pnl_stop = (r["stop_level"] - r["entry_price"]) / r["entry_price"] * 100
            if r["profit_type"] == "none":
                profit_line = "\n       profit[désactivé] trailing-only"
            else:
                pnl_profit = (r["profit_level"] - r["entry_price"]) / r["entry_price"] * 100
                profit_line = (f"\n       profit[{r['profit_type']:8}] "
                               f"{r['profit_level']:.2f} ({pnl_profit:+.1f}%)")
            print(
                f"{r['symbol']:6} | entrée={r['entry_price']:.2f} | qty={r['qty_remaining']}"
                f"\n       stop  [{r['protection_type']:8}] {r['stop_level']:.2f}"
                f" ({pnl_stop:+.1f}%)"
                + (f"  HWM={r['high_water_mark']:.2f}" if r["protection_type"] == "trailing" else "")
                + profit_line
                + f"\n       vérifié: {r['last_checked'] or 'jamais'}"
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
    parser.add_argument('--refresh-config', action='store_true',
                        help='Recalcule tous les stops actifs selon la config courante puis quitte')
    parser.add_argument('--show', action='store_true',
                        help='Affiche les stops actifs et quitte')
    args = parser.parse_args()

    trade_mode = TradeMode.LIVE if args.live else TradeMode.PAPER
    config = StopConfig.from_json(CONFIG_PATH)

    # Connexion IB
    port = 7496 if args.live else args.port
    import random
    mdp = MarketDataProvider(port=port, client_id=random.randint(20, 99))
    if not (args.show or args.refresh_config):
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
        elif args.refresh_config:
            manager.refresh_active_stops_to_config()
            manager.show()
        elif args.setup_only:
            manager.setup_all_open_positions()
            manager.refresh_active_stops_to_config()
            manager.show()
        else:
            manager.setup_all_open_positions()
            manager.refresh_active_stops_to_config()
            manager.show()
            manager.run_loop()
    finally:
        if mdp.is_connected():
            mdp.disconnect()


if __name__ == "__main__":
    main()
