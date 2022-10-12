from collections import deque
from decimal import Decimal
from statistics import mean
from typing import Dict, List

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_book import OrderBook
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.event_forwarder import SourceInfoEventForwarder
from hummingbot.core.event.events import OrderBookEvent, OrderBookTradeEvent
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class DynamicPriceCeilingFloorPMM(ScriptStrategyBase):
    bid_spread = 0.08
    ask_spread = 0.08
    order_refresh_time = 15
    order_amount = 0.01
    create_timestamp = 0
    trading_pair = "ETH-USDT"
    exchange = "binance_paper_trade"
    # Here you can use for example the LastTrade price to use in your strategy
    price_source = PriceType.MidPrice

    # Price ceiling/floor configuration
    ceiling_pct = 0.05
    floor_pct = 0.04
    trades_buffer = 1500
    buy_trades_buffer = deque(maxlen=trades_buffer)
    sell_trades_buffer = deque(maxlen=trades_buffer)

    # Flag to trigger the initialization of the trades event listener
    trades_event_initialized = False

    markets = {exchange: {trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        """
        Initialising a new script strategy object.

        :param connectors: A dictionary of connector names and their corresponding connector.
        """
        super().__init__(connectors)
        self._public_trades_forwarder: SourceInfoEventForwarder = SourceInfoEventForwarder(self._process_public_trades)

    def on_tick(self):
        if not self.trades_event_initialized:
            for connector in self.connectors.values():
                for order_book in connector.order_books.values():
                    order_book.add_listener(OrderBookEvent.TradeEvent, self._public_trades_forwarder)
            self.trades_event_initialized = True
        if self.create_timestamp <= self.current_timestamp:
            self.cancel_all_orders()
            proposal: List[OrderCandidate] = self.create_proposal()
            if self.is_the_proposal_inside_the_bounds(proposal):
                proposal_adjusted: List[OrderCandidate] = self.adjust_proposal_to_budget(proposal)
                self.place_orders(proposal_adjusted)
            self.create_timestamp = self.order_refresh_time + self.current_timestamp

    def create_proposal(self) -> List[OrderCandidate]:
        ref_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
        buy_price = ref_price * Decimal(1 - self.bid_spread)
        sell_price = ref_price * Decimal(1 + self.ask_spread)

        buy_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                   order_side=TradeType.BUY, amount=Decimal(self.order_amount), price=buy_price)

        sell_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                    order_side=TradeType.SELL, amount=Decimal(self.order_amount), price=sell_price)

        return [buy_order, sell_order]

    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        proposal_adjusted = self.connectors[self.exchange].budget_checker.adjust_candidates(proposal, all_or_none=True)
        return proposal_adjusted

    def place_orders(self, proposal: List[OrderCandidate]) -> None:
        for order in proposal:
            self.place_order(connector_name=self.exchange, order=order)

    def place_order(self, connector_name: str, order: OrderCandidate):
        if order.order_side == TradeType.SELL:
            self.sell(connector_name=connector_name, trading_pair=order.trading_pair, amount=order.amount,
                      order_type=order.order_type, price=order.price)
        elif order.order_side == TradeType.BUY:
            self.buy(connector_name=connector_name, trading_pair=order.trading_pair, amount=order.amount,
                     order_type=order.order_type, price=order.price)

    def cancel_all_orders(self):
        for exchange in self.connectors.values():
            safe_ensure_future(exchange.cancel_all(timeout_seconds=5))

    def is_the_proposal_inside_the_bounds(self, proposal):
        if len(self.sell_trades_buffer) > 0 or len(self.buy_trades_buffer) > 0:
            inside_bounds = True
            sell_average_price = mean(self.sell_trades_buffer)
            buy_average_price = mean(self.buy_trades_buffer)
            for order in proposal:
                if order.order_side == TradeType.SELL:
                    if order.price < sell_average_price * (1 - self.floor_pct):
                        inside_bounds = False
                        break
                elif order.order_side == TradeType.BUY:
                    if order.price > buy_average_price * (1 + self.ceiling_pct):
                        inside_bounds = False
                        break
            return inside_bounds
        else:
            return False

    def _process_public_trades(self,
                               event_tag: int,
                               order_book: OrderBook,
                               event: OrderBookTradeEvent):
        if event.type == TradeType.SELL:
            self.sell_trades_buffer.append(event.price)
        elif event.type == TradeType.BUY:
            self.buy_trades_buffer.append(event.price)
