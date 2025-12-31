import backtrader as bt

class OrderTranslator:

    @staticmethod
    def entry(strategy, data, ib_order):
        pos = strategy.getposition(data)
        qty = int(ib_order.totalQuantity)
        print(f"[OrderTranslator] ENTRY: {data._name} | Avant buy: position size={pos.size} | qty transmis={qty} (type={type(qty)})")
        if ib_order.orderType == "STP LMT":
            order = strategy.buy(
                data=data,
                exectype=bt.Order.Stop,
                price=ib_order.auxPrice,
                size=qty
            )
        elif ib_order.orderType == "LMT":
            order = strategy.buy(
                data=data,
                exectype=bt.Order.Limit,
                price=ib_order.lmtPrice,
                size=qty
            )
        elif ib_order.orderType == "MKT":
            order = strategy.buy(
                data=data,
                size=qty
            )
        else:
            raise NotImplementedError(ib_order.orderType)
        # Log après soumission (l'état changera au fill, mais on log l'intention)
        print(f"[OrderTranslator] ENTRY: {data._name} | Demande buy de {qty}")
        return order


    @staticmethod
    def stop(strategy, data, ib_order,parent_order=None):
        pos = strategy.getposition(data)
        print(f"[OrderTranslator] STOP: {data._name} | Avant sell: position size={pos.size}")
        if pos.size > 0:
            if hasattr(ib_order, "trailingPercent"):
                order = strategy.sell(
                    data=data,
                    exectype=bt.Order.StopTrail,
                    trailpercent=ib_order.trailingPercent / 100,
                    size=pos.size,
                    parent_order=parent_order
                )
            else:
                order = strategy.sell(
                    data=data,
                    exectype=bt.Order.Stop,
                    price=ib_order.auxPrice,
                    size=pos.size,
                    parent_order=parent_order
                )
            print(f"[OrderTranslator] STOP: {data._name} | Demande sell (stop) de {pos.size}")
            return order
        else:
            print(f"[OrderTranslator] STOP: {data._name} | Aucun sell exécuté (pas de position long)")
            return None

    @staticmethod
    def take_profit(strategy, data, ib_order,parent_order=None):
        pos = strategy.getposition(data)
        print(f"[OrderTranslator] TAKE_PROFIT: {data._name} | Avant sell: position size={pos.size}")
        if pos.size > 0:
            order = strategy.sell(
                data=data,
                exectype=bt.Order.Limit,
                price=ib_order.lmtPrice,
                size=pos.size
            )
            print(f"[OrderTranslator] TAKE_PROFIT: {data._name} | Demande sell (take profit) de {pos.size}")
            return order
        else:
            print(f"[OrderTranslator] TAKE_PROFIT: {data._name} | Aucun sell exécuté (pas de position long)")
            return None
