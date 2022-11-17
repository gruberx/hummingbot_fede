import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Union

import pandas as pd
from pydantic import BaseModel

from hummingbot.connector.derivative.position import PositionSide
from hummingbot.connector.exchange_base import ExchangeBase
from hummingbot.core.data_type.common import OrderType, PositionAction
from hummingbot.core.data_type.in_flight_order import InFlightOrder
from hummingbot.core.event.events import (
    BuyOrderCompletedEvent,
    BuyOrderCreatedEvent,
    OrderFilledEvent,
    SellOrderCompletedEvent,
    SellOrderCreatedEvent,
)
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class PositionConfig(BaseModel):
    stop_loss: Decimal
    take_profit: Decimal
    time_limit: int
    order_type: OrderType
    price: Decimal
    amount: Decimal
    side: PositionSide


class Signal(BaseModel):
    id: int
    timestamp: float
    value: float
    trading_pair: str
    exchange: str
    position_config: PositionConfig


class BotProfile(BaseModel):
    balance_limit: Decimal
    max_order_amount: Decimal
    take_profit_threshold: float
    stop_loss_threshold: float
    leverage: float


class SignalExecutorStatus(Enum):
    NOT_STARTED = 1
    ORDER_PLACED = 2
    ACTIVE_POSITION = 3
    CLOSE_PLACED = 4
    CLOSED = 5


class SignalExecutor:
    def __init__(self, signal: Signal, strategy: ScriptStrategyBase):
        self._signal = signal
        self._strategy = strategy
        self._status: SignalExecutorStatus = SignalExecutorStatus.NOT_STARTED
        self._open_order_json: Union[InFlightOrder, None] = None
        self._open_order_id: Union[str, None] = None
        self._exit_order_id: Union[str, None] = None
        self._take_profit_order_id: Union[str, None] = None
        self._filled_amount: Decimal = Decimal("0")
        self._average_price: Decimal = Decimal("0")

    @property
    def signal(self):
        return self._signal

    @property
    def status(self):
        return self._status

    @property
    def open_order_id(self):
        return self._open_order_id

    @property
    def exit_order_id(self):
        return self._exit_order_id

    @property
    def take_profit_order_id(self):
        return self._take_profit_order_id

    @property
    def connector(self) -> ExchangeBase:
        return self._strategy.connectors[self._signal.exchange]

    def set_open_order_json(self, order: InFlightOrder):
        order_json = order.to_json()
        order_json["average_executed_price"] = order.average_executed_price
        self._open_order_json = order_json

    def change_status(self, status: SignalExecutorStatus):
        self._status = status

    def get_order(self, order_id: str):
        orders = self.connector._client_order_tracker.all_orders
        return orders.get(order_id, None)

    @property
    def open_order(self):
        open_order = self.get_order(self._open_order_id)
        if open_order:
            self.set_open_order_json(open_order)
        return self._open_order_json

    @property
    def take_profit_order(self):
        return self.get_order(self._take_profit_order_id)

    @property
    def exit_profit_order(self):
        # TODO: split on sl order and tl order
        return self.get_order(self._exit_order_id)

    def get_filled_amount(self, order_id: str):
        order = self.get_order(order_id)
        if order:
            self._filled_amount = order.executed_amount_base
        return self._filled_amount

    def get_average_price(self, order_id: str):
        order = self.get_order(order_id)
        if order:
            self._average_price = order.average_executed_price
        return self._average_price

    def control_position(self):
        if self._status == SignalExecutorStatus.NOT_STARTED:
            self.control_open_order()
        elif self._status == SignalExecutorStatus.ORDER_PLACED:
            self.control_order_placed_time_limit()
        elif self._status == SignalExecutorStatus.ACTIVE_POSITION:
            self.control_stop_loss()
            self.control_take_profit()
            self.control_position_time_limit()
        elif self._status == SignalExecutorStatus.CLOSED:
            pass

    def control_open_order(self):
        if not self._open_order_id:
            order_id = self._strategy.place_order(
                connector_name=self._signal.exchange,
                trading_pair=self._signal.trading_pair,
                amount=self._signal.position_config.amount,
                price=self._signal.position_config.price,
                order_type=self._signal.position_config.order_type,
                position_action=PositionAction.OPEN,
                position_side=self._signal.position_config.side
            )
            self._open_order_id = order_id
            self._strategy.logger().info(f"Signal id {self._signal.id}: Placing open order")

        else:
            self.ask_order_status(self._open_order_id)

    def control_order_placed_time_limit(self):
        if self._signal.timestamp / 1000 + self._signal.position_config.time_limit >= self._strategy.current_timestamp:
            self._strategy.cancel(
                connector_name=self._signal.exchange,
                trading_pair=self._signal.trading_pair,
                order_id=self._open_order_id
            )
            self._strategy.logger().info(f"Signal id {self._signal.id}: Canceling limit order by time limit")

    def control_take_profit(self):
        price = self.get_average_price(self._open_order_id)
        if self._signal.position_config.side == PositionSide.LONG:
            tp_multiplier = 1 + self._signal.position_config.take_profit
        else:
            tp_multiplier = 1 - self._signal.position_config.take_profit
        if not self._take_profit_order_id:
            order_id = self._strategy.place_order(
                connector_name=self._signal.exchange,
                trading_pair=self._signal.trading_pair,
                amount=self.get_filled_amount(self._open_order_id),
                price=price * tp_multiplier,
                order_type=OrderType.LIMIT,
                position_action=PositionAction.CLOSE,
                position_side=PositionSide.LONG if self._signal.position_config.side == PositionSide.SHORT else PositionSide.SHORT
            )
            self._take_profit_order_id = order_id
            self._strategy.logger().info(f"Signal id {self._signal.id}: Placing take profit")
        else:
            take_profit_order: InFlightOrder = self.get_order(self._take_profit_order_id)
            if self.get_filled_amount(self._open_order_id) != take_profit_order.amount:
                # TODO: Check if it's canceled in the exchange
                self._strategy.cancel(
                    connector_name=self._signal.exchange,
                    trading_pair=self._signal.trading_pair,
                    order_id=self._take_profit_order_id
                )
                self._take_profit_order_id = None
                self._strategy.logger().info(f"Signal id {self._signal.id}: Needs to replace take profit")

    @property
    def stop_loss_price(self):
        entry_price = self.get_average_price(self._open_order_id)
        stop_loss_price = 0
        if entry_price:
            # TODO: extract logic to get stop loss price by using the order candidate
            if self._signal.position_config.side == PositionSide.LONG:
                stop_loss_price = entry_price * (1 - self._signal.position_config.stop_loss)
            else:
                stop_loss_price = entry_price * (1 + self._signal.position_config.stop_loss)
        return stop_loss_price

    def control_stop_loss(self):
        entry_price = self.get_average_price(self._open_order_id)
        current_price = self.connector.get_mid_price(self._signal.trading_pair)
        trigger_stop_loss = False
        # TODO: extract logic to get stop loss price by using the order candidate
        if self._signal.position_config.side == PositionSide.LONG:
            stop_loss_price = entry_price * (1 - self._signal.position_config.stop_loss)
            if current_price <= stop_loss_price:
                trigger_stop_loss = True
        else:
            stop_loss_price = entry_price * (1 + self._signal.position_config.stop_loss)
            if current_price >= stop_loss_price:
                trigger_stop_loss = True

        if trigger_stop_loss:
            if not self._exit_order_id:
                order_id = self._strategy.place_order(
                    connector_name=self._signal.exchange,
                    trading_pair=self._signal.trading_pair,
                    amount=self.get_filled_amount(self._open_order_id),
                    price=current_price,
                    order_type=OrderType.MARKET,
                    position_action=PositionAction.CLOSE,
                    position_side=PositionSide.LONG if self._signal.position_config.side == PositionSide.SHORT else PositionSide.SHORT
                )
                self._exit_order_id = order_id
                # TODO: Check if it's canceled in the exchange
                self._strategy.cancel(
                    connector_name=self._signal.exchange,
                    trading_pair=self._signal.trading_pair,
                    order_id=self._take_profit_order_id
                )
            else:
                self.ask_order_status(self._exit_order_id)

    def control_position_time_limit(self):
        position_expired = self._signal.timestamp / 1000 + self._signal.position_config.time_limit >= self._strategy.current_timestamp
        if position_expired:
            if not self._exit_order_id:
                price = self.connectors.get_mid_price(self._signal.trading_pair)
                order_id = self._strategy.place_order(
                    connector_name=self._signal.exchange,
                    trading_pair=self._signal.trading_pair,
                    amount=self.get_filled_amount(self._open_order_id),
                    price=price,
                    order_type=OrderType.MARKET,
                    position_action=PositionAction.CLOSE,
                    position_side=PositionSide.LONG if self._signal.position_config.side == PositionSide.SHORT else PositionSide.SHORT
                )
                self._exit_order_id = order_id
                self._strategy.logger().info(f"Signal id {self._signal.id}: Closing position by time limit")
            else:
                self.ask_order_status(self._exit_order_id)

    def ask_order_status(self, order_id):
        self._strategy.logger().info(f"Signal id {self._signal.id}: Checking order {order_id}")
        pass


class DirectionalStrategyPerpetuals(ScriptStrategyBase):
    bot_profile = BotProfile(
        balance_limit=Decimal(1000),
        max_order_amount=Decimal(20),
        take_profit_threshold=0.8,
        stop_loss_threshold=-0.8,
        leverage=10,
    )
    max_executors = 1
    signal_executors: List[SignalExecutor] = []
    markets = {"binance_perpetual_testnet": {"ETH-USDT"}}

    def get_active_executors(self):
        return [executor for executor in self.signal_executors if executor._status != SignalExecutorStatus.CLOSED]

    def get_active_positions_df(self):
        active_positions = []
        for connector_name, connector in self.connectors.items():
            for trading_pair, position in connector.account_positions.items():
                active_positions.append({
                    "exchange": connector_name,
                    "trading_pair": trading_pair,
                    "side": position.position_side,
                    "entry_price": position.entry_price,
                    "amount": position.amount,
                    "leverage": position.leverage,
                    "unrealized_pnl": position.unrealized_pnl
                })
        return pd.DataFrame(active_positions)

    def on_tick(self):
        if len(self.get_active_executors()) < self.max_executors:
            signal: Signal = self.get_signal()
            if signal.value > self.bot_profile.take_profit_threshold or signal.value < self.bot_profile.stop_loss_threshold:
                price = self.connectors[signal.exchange].get_mid_price(signal.trading_pair)
                signal.position_config.amount = (self.bot_profile.max_order_amount / price) * signal.position_config.amount
                self.signal_executors.append(SignalExecutor(
                    signal=signal,
                    strategy=self
                ))
        for executor in self.signal_executors:
            executor.control_position()

    def get_signal(self):
        return Signal(
            id=420,
            timestamp=1247812934,
            value=0.9,
            trading_pair="ETH-USDT",
            exchange="binance_perpetual_testnet",
            position_config=PositionConfig(
                stop_loss=Decimal(0.03),
                take_profit=Decimal(0.03),
                time_limit=300,
                order_type=OrderType.MARKET,
                price=Decimal(1400),
                amount=Decimal(1),
                side=PositionSide.SHORT,
            ),
        )

    def place_order(self,
                    connector_name: str,
                    trading_pair: str,
                    position_side: PositionSide,
                    amount: Decimal,
                    order_type: OrderType,
                    position_action: PositionAction,
                    price=Decimal("NaN"),
                    ):
        if position_side == PositionSide.LONG:
            return self.buy(connector_name, trading_pair, amount, order_type, price, position_action)
        else:
            return self.sell(connector_name, trading_pair, amount, order_type, price, position_action)

    def did_complete_buy_order(self, event: BuyOrderCompletedEvent):
        self.did_complete_order(event)

    def did_complete_sell_order(self, event: SellOrderCompletedEvent):
        self.did_complete_order(event)

    def did_complete_order(self, event: Union[BuyOrderCompletedEvent, SellOrderCompletedEvent]):
        for executor in self.signal_executors:
            if executor.open_order_id == event.order_id:
                executor.set_open_order_json(executor.open_order)
                executor.change_status(SignalExecutorStatus.ACTIVE_POSITION)
            elif executor.exit_order_id == event.order_id:
                executor.change_status(SignalExecutorStatus.CLOSED)
            elif executor.take_profit_order_id == event.order_id:
                executor.change_status(SignalExecutorStatus.CLOSED)

    def did_create_buy_order(self, event: BuyOrderCreatedEvent):
        self.did_create_order(event)

    def did_create_sell_order(self, event: SellOrderCreatedEvent):
        self.did_create_order(event)

    def did_create_order(self, event: Union[BuyOrderCreatedEvent, SellOrderCreatedEvent]):
        for executor in self.signal_executors:
            if executor.open_order_id == event.order_id:
                executor.set_open_order_json(executor.open_order)
                executor.change_status(SignalExecutorStatus.ORDER_PLACED)

    def did_fill_order(self, event: OrderFilledEvent):
        for executor in self.signal_executors:
            if executor.open_order_id == event.order_id:
                executor.change_status(SignalExecutorStatus.ACTIVE_POSITION)

    @property
    def format_status(self) -> str:
        """
        Returns status of the current strategy on user balances and current active orders. This function is called
        when status command is issued. Override this function to create custom status display output.
        """
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        lines = []
        warning_lines = []
        warning_lines.extend(self.network_warning(self.get_market_trading_pair_tuples()))

        balance_df = self.get_balance_df()
        lines.extend(["", "  Balances:"] + ["    " + line for line in balance_df.to_string(index=False).split("\n")])

        try:
            df = self.active_orders_df()
            lines.extend(["", "  Orders:"] + ["    " + line for line in df.to_string(index=False).split("\n")])
        except ValueError:
            lines.extend(["", "  No active maker orders."])

        # Show active positions
        positions_df = self.get_active_positions_df()
        if not positions_df.empty:
            lines.extend(["", "  Positions:"] + ["    " + line for line in positions_df.to_string(index=False).split("\n")])
        else:
            lines.extend(["", "  No active positions."])

        for executor in self.signal_executors:
            current_price = self.connectors[executor.signal.exchange].get_mid_price(executor.signal.trading_pair)
            lines.extend(["", f"  Signal: {executor.signal.id}"] +
                         [f"            - Value: {executor.signal.value}"] +
                         [f"            - Trading Pair: {executor.signal.trading_pair}"] +
                         [f"            - Side: {executor.signal.position_config.side}"] +
                         [f"            - Status: {executor.status}"] +
                         [f"            - Exchange: {executor.signal.exchange}"])
            if executor.open_order:
                lines.extend([f"        - Open Order: {executor.open_order_id}"] +
                             [f"                    - Entry price: {executor.open_order['average_executed_price']}"] +
                             [f"                    - Amount: {executor.open_order['amount']}"])

            lines.extend([f"        - Current price: {current_price}"])
            if executor.take_profit_order:
                lines.extend([f"        - Take Profit Order: {executor.take_profit_order_id}"] +
                             [f"                    - Price: {executor.take_profit_order.price:.2f}"] +
                             [f"                    - Amount: {executor.take_profit_order.amount}"] +
                             [f"                    - Entry price: {executor.take_profit_order.average_executed_price}"] +
                             [f"                    - Executed amount base: {executor.take_profit_order.executed_amount_base}"] +
                             [f"                    - Executed amount quote: {executor.take_profit_order.executed_amount_quote}"] +
                             [f"                    - Percentage: {100 * executor.signal.position_config.take_profit:.2f} %"] +
                             [f"                    - Distance from mid price: {100 * (executor.take_profit_order.price - current_price) / current_price:.2f} %"]
                             )
            if executor.stop_loss_price != 0:
                lines.extend(
                    ["        - Stop Loss:"] +
                    [f"                    - Price: {executor.stop_loss_price}"] +
                    [f"                    - Percentage: {100 * executor.signal.position_config.stop_loss} %"] +
                    [f"                    - Distance from mid price: {100 *(executor.stop_loss_price - current_price) / current_price:.2f} %"]
                )
            start_time = datetime.datetime.fromtimestamp(executor.signal.timestamp)
            duration = datetime.timedelta(seconds=executor.signal.position_config.time_limit)
            end_time = start_time + duration
            current_timestamp = datetime.datetime.fromtimestamp(self.current_timestamp)
            seconds_remaining = (end_time - current_timestamp)
            lines.extend(
                ["        - Time Limit:"] +
                [f"                    - Start time: {start_time}"] +
                [f"                    - Current time: {current_timestamp}"] +
                [f"                    - Duration: {duration}"] +
                [f"                    - Seconds remaining: {seconds_remaining}"]
            )

        warning_lines.extend(self.balance_warning(self.get_market_trading_pair_tuples()))
        if len(warning_lines) > 0:
            lines.extend(["", "*** WARNINGS ***"] + warning_lines)
        return "\n".join(lines)
