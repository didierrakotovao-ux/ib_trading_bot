from datetime import datetime
from unittest.mock import patch

from src.app.database.trade_journal import TradeJournal


class FakeCursor:
    def __init__(self):
        self.inserted_params = None
        self.updated_params = None
        self.last_result = None

    def execute(self, query, params=None):
        query_text = query.lower()
        if "insert into trades" in query_text:
            self.inserted_params = params
            self.last_result = (1,)
        elif "insert into trade_exits" in query_text:
            self.last_result = (42,)
        elif "update trades set" in query_text and "date_sortie" in query_text:
            self.updated_params = params
        elif "select quantite_restante from trades" in query_text:
            self.last_result = (0,)
        elif "select sum(pnl_brut" in query_text:
            self.last_result = (100.0, 2.0, 98.0, datetime(2024, 1, 2, 15, 0))
        else:
            self.last_result = None

    def fetchone(self):
        return self.last_result

    def close(self):
        pass


class FakeConn:
    def __init__(self):
        self.closed = False
        self.cursor_obj = FakeCursor()

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        pass

    def close(self):
        self.closed = True


@patch("src.app.database.trade_journal.get_conn", return_value=FakeConn())
def test_trade_journal_persists_score_and_bars_held(_mock_conn):
    journal = TradeJournal()

    trade_id = journal.log_trade(
        trade_mode=None,
        strategy_name="Momentum",
        symbol="AAPL",
        date_entree=datetime(2024, 1, 1, 10, 0),
        prix_entree=100.0,
        quantite=10,
        score_entree=71.5,
    )

    assert trade_id == 1

    exit_id = journal.log_partial_exit(
        trade_id=trade_id,
        symbol="AAPL",
        date_sortie=datetime(2024, 1, 2, 15, 0),
        prix_sortie=110.0,
        quantite_vendue=10,
        prix_entree=100.0,
        cause_sortie="TRAILING_STOP",
        bars_held=8,
        score_entree=71.5,
    )

    assert exit_id == 42

    inserted_params = journal.conn.cursor().inserted_params
    assert any(value == 71.5 for value in inserted_params)

    updated_params = journal.conn.cursor().updated_params
    assert any(value == 8 for value in updated_params)
    assert any(value == 71.5 for value in updated_params)
