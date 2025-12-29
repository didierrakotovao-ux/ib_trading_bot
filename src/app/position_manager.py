from datetime import datetime

class Position:
    def __init__(self, symbol, qty, entry_price, entry_time=None, position_type="long"):
        self.symbol = symbol
        self.qty = qty
        self.entry_price = entry_price
        self.entry_time = entry_time or datetime.now()
        self.exit_price = None
        self.exit_time = None
        self.position_type = position_type
        self.status = "open"
        self.orders = []  # Liste des ordres liés à la position
        self.pnl = 0

    def close(self, exit_price, exit_time=None):
        self.exit_price = exit_price
        self.exit_time = exit_time or datetime.now()
        self.status = "closed"
        self.pnl = (self.exit_price - self.entry_price) * self.qty if self.position_type == "long" else (self.entry_price - self.exit_price) * self.qty

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
        for pos in self.positions:
            if pos.symbol == symbol and pos.status == "open":
                pos.close(exit_price, exit_time)
                return pos
        return None

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
