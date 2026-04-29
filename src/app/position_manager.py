from datetime import datetime

class Position:
    def __init__(self, symbol, qty, entry_price, entry_time=None, position_type="long"):
        self.symbol = symbol
        self.qty = qty            # quantité initiale à l'entrée
        self.qty_remaining = qty  # quantité encore détenue
        self.entry_price = entry_price
        self.entry_time = entry_time or datetime.now()
        self.exit_price = None
        self.exit_time = None
        self.position_type = position_type
        self.status = "open"
        self.orders = []  # Liste des ordres liés à la position
        self.pnl = 0

    def reduce(self, qty_sold: float, exit_price: float, exit_time=None) -> float:
        """
        Réduit la position d'une quantité vendue.
        Retourne le PnL brut de cette vente partielle.
        Si qty_remaining atteint 0, ferme la position.
        """
        qty_sold = min(qty_sold, self.qty_remaining)
        pnl = (exit_price - self.entry_price) * qty_sold if self.position_type == "long" \
              else (self.entry_price - exit_price) * qty_sold
        self.pnl += pnl
        self.qty_remaining -= qty_sold
        if self.qty_remaining <= 0:
            self.qty_remaining = 0
            self.exit_price = exit_price
            self.exit_time = exit_time or datetime.now()
            self.status = "closed"
        return pnl

    def close(self, exit_price, exit_time=None):
        """Ferme la totalité de la position restante."""
        self.reduce(self.qty_remaining, exit_price, exit_time)

class PositionManager:
    def __init__(self):
        self.positions = []

    def has_open_position(self, symbol):
        """Retourne True si une position ouverte existe pour ce symbole."""
        return any(p.symbol == symbol and p.status == "open" for p in self.positions)

    def open_position(self, symbol, qty, entry_price, entry_time=None, position_type="long"):
        # Vérifie s'il existe déjà une position ouverte sur ce symbole
        if any(p.symbol == symbol and p.status == "open" for p in self.positions):
            # Ne crée pas de doublon
            return None
        pos = Position(symbol, qty, entry_price, entry_time, position_type)
        self.positions.append(pos)
        return pos

    def close_position(self, symbol, exit_price, exit_time=None):
        """Ferme la totalité de la position restante."""
        for pos in self.positions:
            if pos.symbol == symbol and pos.status == "open":
                pos.close(exit_price, exit_time)
                return pos
        return None

    def partial_close_position(self, symbol, qty_sold, exit_price, exit_time=None):
        """
        Réduit la position d'une quantité vendue.
        Retourne (position, pnl_brut_partiel).
        La position reste "open" tant que qty_remaining > 0.
        """
        for pos in self.positions:
            if pos.symbol == symbol and pos.status == "open":
                pnl = pos.reduce(qty_sold, exit_price, exit_time)
                return pos, pnl
        return None, 0.0

    def get_open_positions(self):
        return [p for p in self.positions if p.status == "open"]

    def get_closed_positions(self):
        return [p for p in self.positions if p.status == "closed"]

    def get_position(self, symbol):
        for pos in self.positions:
            if pos.symbol == symbol:
                return pos
        return None

    def update_position(self, symbol, **kwargs):
        pos = self.get_position(symbol)
        if pos:
            for k, v in kwargs.items():
                setattr(pos, k, v)
            return pos
        return None
