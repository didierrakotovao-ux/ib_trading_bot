import backtrader as bt

class OrderTranslator:

    @staticmethod
    def entry(strategy, data, ib_order):
        if ib_order.orderType == "STP LMT":
            return strategy.buy(
                data=data,
                exectype=bt.Order.Stop,
                price=ib_order.auxPrice,
                size=ib_order.totalQuantity
            )

        if ib_order.orderType == "LMT":
            return strategy.buy(
                data=data,
                exectype=bt.Order.Limit,
                price=ib_order.lmtPrice,
                size=ib_order.totalQuantity
            )

        if ib_order.orderType == "MKT":
            return strategy.buy(
                data=data,
                size=ib_order.totalQuantity
            )

        raise NotImplementedError(ib_order.orderType)

    @staticmethod
    def stop(strategy, data, ib_order):
        if hasattr(ib_order, "trailingPercent"):
            return strategy.sell(
                data=data,
                exectype=bt.Order.StopTrail,
                trailpercent=ib_order.trailingPercent / 100
            )

        return strategy.sell(
            data=data,
            exectype=bt.Order.Stop,
            price=ib_order.auxPrice
        )

    @staticmethod
    def take_profit(strategy, data, ib_order):
        return strategy.sell(
            data=data,
            exectype=bt.Order.Limit,
            price=ib_order.lmtPrice
        )
