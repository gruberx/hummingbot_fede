import logging
from decimal import Decimal
from typing import Any, Dict

from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class SimpleArbitrage(ScriptStrategyBase):
    order_amount = Decimal("0.01")  # in base asset
    min_profitability = Decimal("0.002")  # in percentage
    trading_pair = "ETH-USDT"
    exchanges = ["binance_paper_trade", "kucoin_paper_trade"]

    markets = {exchanges[0]: {trading_pair},
               exchanges[1]: {trading_pair}}

    def on_tick(self):
        vwap_prices = self.get_vwap_prices_for_amount()
        proposal = self.check_profitability_and_create_proposal(vwap_prices)
        if len(proposal) > 0:
            proposal_adjusted: Dict[str, OrderCandidate] = self.adjust_proposal_to_budget(proposal)
            self.place_orders(proposal_adjusted)

    def get_vwap_prices_for_amount(self):
        bid_ex_0 = self.connectors[self.exchanges[0]].get_vwap_for_volume(self.trading_pair, False, self.order_amount)
        ask_ex_0 = self.connectors[self.exchanges[0]].get_vwap_for_volume(self.trading_pair, True, self.order_amount)
        bid_ex_1 = self.connectors[self.exchanges[1]].get_vwap_for_volume(self.trading_pair, False, self.order_amount)
        ask_ex_1 = self.connectors[self.exchanges[1]].get_vwap_for_volume(self.trading_pair, True, self.order_amount)
        vwap_prices = {
            self.exchanges[0]: {
                "bid": bid_ex_0.result_price,
                "ask": ask_ex_0.result_price
            },
            self.exchanges[1]: {
                "bid": bid_ex_1.result_price,
                "ask": ask_ex_1.result_price
            }
        }
        return vwap_prices

    def check_profitability_and_create_proposal(self, vwap_prices: Dict[str, Any]) -> Dict:
        proposal = {}
        if vwap_prices[self.exchanges[0]]["ask"] * (1 + self.min_profitability) < vwap_prices[self.exchanges[1]]["bid"]:
            # This means that the ask of the first exchange is lower than the bid of the second one
            proposal[self.exchanges[0]] = OrderCandidate(trading_pair=self.trading_pair, is_maker=False,
                                                         order_type=OrderType.MARKET,
                                                         order_side=TradeType.BUY, amount=self.order_amount,
                                                         price=vwap_prices[self.exchanges[0]]["ask"])
            proposal[self.exchanges[1]] = OrderCandidate(trading_pair=self.trading_pair, is_maker=False,
                                                         order_type=OrderType.MARKET,
                                                         order_side=TradeType.SELL, amount=Decimal(self.order_amount),
                                                         price=vwap_prices[self.exchanges[1]]["bid"])
        elif vwap_prices[self.exchanges[1]]["ask"] * (1 + self.min_profitability) < vwap_prices[self.exchanges[0]]["bid"]:
            # This means that the ask of the second exchange is lower than the bid of the first one
            proposal[self.exchanges[1]] = OrderCandidate(trading_pair=self.trading_pair, is_maker=False,
                                                         order_type=OrderType.MARKET,
                                                         order_side=TradeType.BUY, amount=self.order_amount,
                                                         price=vwap_prices[self.exchanges[1]]["ask"])
            proposal[self.exchanges[0]] = OrderCandidate(trading_pair=self.trading_pair, is_maker=False,
                                                         order_type=OrderType.MARKET,
                                                         order_side=TradeType.SELL, amount=Decimal(self.order_amount),
                                                         price=vwap_prices[self.exchanges[0]]["bid"])

        return proposal

    def adjust_proposal_to_budget(self, proposal: Dict[str, OrderCandidate]) -> Dict[str, OrderCandidate]:
        for connector, order in proposal.items():
            proposal[connector] = self.connectors[connector].budget_checker.adjust_candidate(order, all_or_none=True)
        return proposal

    def place_orders(self, proposal: Dict[str, OrderCandidate]) -> None:
        for connector, order in proposal.items():
            self.place_order(connector_name=connector, order=order)

    def place_order(self, connector_name: str, order: OrderCandidate):
        if order.order_side == TradeType.SELL:
            self.sell(connector_name=connector_name, trading_pair=order.trading_pair, amount=order.amount,
                      order_type=order.order_type, price=order.price)
        elif order.order_side == TradeType.BUY:
            self.buy(connector_name=connector_name, trading_pair=order.trading_pair, amount=order.amount,
                     order_type=order.order_type, price=order.price)

    def did_fill_order(self, event: OrderFilledEvent):
        msg = (
            f"{event.trade_type.name} {round(event.amount, 2)} {event.trading_pair} at {round(event.price, 2)}")
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)
