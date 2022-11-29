import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Union

import pandas as pd
from pydantic import BaseModel

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.connector.derivative.position import PositionSide
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
    long_threshold: float
    short_threshold: float
    leverage: float


class SignalExecutorStatus(Enum):
    NOT_STARTED = 1
    ORDER_PLACED = 2
    ACTIVE_POSITION = 3
    CLOSE_PLACED = 4
    CLOSED_BY_TIME_LIMIT = 5
    CLOSED_BY_STOP_LOSS = 6
    CLOSED_BY_TAKE_PROFIT = 7


class TrackedOrder:
    def __init__(self, order_id: str):
        self._order_id = order_id
        self._order = None

    @property
    def order_id(self):
        return self._order_id

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, order: InFlightOrder):
        self._order = order


class SignalExecutor:
    def __init__(self, signal: Signal, strategy: ScriptStrategyBase):
        self._signal = signal
        self._strategy = strategy
        self._status: SignalExecutorStatus = SignalExecutorStatus.NOT_STARTED
        self._open_order: Union[TrackedOrder, None] = None
        self._take_profit_order: Union[TrackedOrder, None] = None
        self._time_limit_order: Union[TrackedOrder, None] = None
        self._stop_loss_order: Union[TrackedOrder, None] = None

    @property
    def signal(self):
        return self._signal

    @property
    def status(self):
        return self._status

    @property
    def connector(self) -> ConnectorBase:
        return self._strategy.connectors[self._signal.exchange]

    def change_status(self, status: SignalExecutorStatus):
        self._status = status

    def get_order(self, order_id: str):
        order = self.connector._client_order_tracker.fetch_order(client_order_id=order_id)
        return order

    @property
    def open_order(self):
        return self._open_order

    @property
    def take_profit_order(self):
        return self._take_profit_order

    @property
    def stop_loss_order(self):
        return self._stop_loss_order

    @property
    def time_limit_order(self):
        return self._time_limit_order

    def control_position(self):
        if self.status == SignalExecutorStatus.NOT_STARTED:
            self.control_open_order()
        elif self.status == SignalExecutorStatus.ORDER_PLACED:
            self.control_order_placed_time_limit()
        elif self.status == SignalExecutorStatus.ACTIVE_POSITION:
            self.control_take_profit()
            self.control_stop_loss()
            self.control_position_time_limit()
        elif self.status == SignalExecutorStatus.CLOSE_PLACED:
            pass

    def remove_take_profit(self):
        self._strategy.cancel(
            connector_name=self._signal.exchange,
            trading_pair=self._signal.trading_pair,
            order_id=self._take_profit_order.order_id
        )
        self._strategy.logger().info("Removing take profit since the position is not longer available")

    def control_open_order(self):
        if not self._open_order:
            order_id = self._strategy.place_order(
                connector_name=self._signal.exchange,
                trading_pair=self._signal.trading_pair,
                amount=self._signal.position_config.amount,
                price=self._signal.position_config.price,
                order_type=self._signal.position_config.order_type,
                position_action=PositionAction.OPEN,
                position_side=self._signal.position_config.side
            )
            self._open_order = TrackedOrder(order_id)
            self._strategy.logger().info(f"Signal id {self._signal.id}: Placing open order")
        else:
            self.ask_order_status(self._open_order)

    def control_order_placed_time_limit(self):
        if self._signal.timestamp / 1000 + self._signal.position_config.time_limit >= self._strategy.current_timestamp:
            self._strategy.cancel(
                connector_name=self._signal.exchange,
                trading_pair=self._signal.trading_pair,
                order_id=self._open_order.order_id
            )
            self._strategy.logger().info(f"Signal id {self._signal.id}: Canceling limit order by time limit")

    def control_take_profit(self):
        if not self._take_profit_order:
            entry_price = self.open_order.order.average_executed_price
            if self._signal.position_config.side == PositionSide.LONG:
                tp_multiplier = 1 + self._signal.position_config.take_profit
            else:
                tp_multiplier = 1 - self._signal.position_config.take_profit
            order_id = self._strategy.place_order(
                connector_name=self._signal.exchange,
                trading_pair=self._signal.trading_pair,
                amount=self.open_order.order.executed_amount_base,
                price=entry_price * tp_multiplier,
                order_type=OrderType.LIMIT,
                position_action=PositionAction.CLOSE,
                position_side=PositionSide.LONG if self._signal.position_config.side == PositionSide.SHORT else PositionSide.SHORT
            )
            self._take_profit_order = TrackedOrder(order_id)
            self._strategy.logger().info(f"Signal id {self._signal.id}: Placing take profit")
            return
        else:
            self.ask_order_status(self._take_profit_order)

    @property
    def stop_loss_price(self):
        stop_loss_price = 0
        if self.open_order and self.open_order.order:
            if self._signal.position_config.side == PositionSide.LONG:
                stop_loss_price = self.open_order.order.average_executed_price * (1 - self._signal.position_config.stop_loss)
            else:
                stop_loss_price = self.open_order.order.average_executed_price * (1 + self._signal.position_config.stop_loss)
        return stop_loss_price

    def control_stop_loss(self):
        entry_price = self.open_order.order.average_executed_price
        current_price = self.connector.get_mid_price(self._signal.trading_pair)
        trigger_stop_loss = False
        if self._signal.position_config.side == PositionSide.LONG:
            stop_loss_price = entry_price * (1 - self._signal.position_config.stop_loss)
            if current_price <= stop_loss_price:
                trigger_stop_loss = True
        else:
            stop_loss_price = entry_price * (1 + self._signal.position_config.stop_loss)
            if current_price >= stop_loss_price:
                trigger_stop_loss = True

        if trigger_stop_loss:
            if not self._stop_loss_order:
                order_id = self._strategy.place_order(
                    connector_name=self._signal.exchange,
                    trading_pair=self._signal.trading_pair,
                    amount=self.open_order.order.executed_amount_base,
                    price=current_price,
                    order_type=OrderType.MARKET,
                    position_action=PositionAction.CLOSE,
                    position_side=PositionSide.LONG if self._signal.position_config.side == PositionSide.SHORT else PositionSide.SHORT
                )
                self._stop_loss_order = TrackedOrder(order_id)
                self._status = SignalExecutorStatus.CLOSE_PLACED
            else:
                self.ask_order_status(self._stop_loss_order)

    def control_position_time_limit(self):
        end_time = self._signal.timestamp + self._signal.position_config.time_limit
        position_expired = end_time < self._strategy.current_timestamp
        if position_expired:
            if not self._time_limit_order:
                price = self.connector.get_mid_price(self._signal.trading_pair)
                order_id = self._strategy.place_order(
                    connector_name=self._signal.exchange,
                    trading_pair=self._signal.trading_pair,
                    amount=self.open_order.order.executed_amount_base,
                    price=price,
                    order_type=OrderType.MARKET,
                    position_action=PositionAction.CLOSE,
                    position_side=PositionSide.LONG if self._signal.position_config.side == PositionSide.SHORT else PositionSide.SHORT
                )
                self._time_limit_order = TrackedOrder(order_id)
                self._status = SignalExecutorStatus.CLOSE_PLACED
                self._strategy.logger().info(f"Signal id {self._signal.id}: Closing position by time limit")
            else:
                self.ask_order_status(self._time_limit_order)

    def ask_order_status(self, order_id):
        pass


class DirectionalStrategyPerpetuals(ScriptStrategyBase):
    bot_profile = BotProfile(
        balance_limit=Decimal(1000),
        max_order_amount=Decimal(20),
        long_threshold=0.8,
        short_threshold=-0.8,
        leverage=10,
    )
    max_executors = 1
    signal_executors: List[SignalExecutor] = []
    markets = {"binance_perpetual_testnet": {"ETH-USDT"}}

    def get_active_executors(self):
        return [executor for executor in self.signal_executors if executor.status not in
                [SignalExecutorStatus.CLOSED_BY_TIME_LIMIT,
                 SignalExecutorStatus.CLOSED_BY_TAKE_PROFIT,
                 SignalExecutorStatus.CLOSED_BY_STOP_LOSS]
                ]

    def get_closed_executors(self):
        return [executor for executor in self.signal_executors if executor.status in
                [SignalExecutorStatus.CLOSED_BY_TIME_LIMIT,
                 SignalExecutorStatus.CLOSED_BY_TAKE_PROFIT,
                 SignalExecutorStatus.CLOSED_BY_STOP_LOSS]
                ]

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
            if signal.value > self.bot_profile.long_threshold or signal.value < self.bot_profile.short_threshold:
                price = self.connectors[signal.exchange].get_mid_price(signal.trading_pair)
                signal.position_config.amount = (self.bot_profile.max_order_amount / price) * signal.position_config.amount
                self.signal_executors.append(SignalExecutor(
                    signal=signal,
                    strategy=self
                ))
        for executor in self.get_active_executors():
            executor.control_position()

    def get_signal(self):
        return Signal(
            id=420,
            timestamp=datetime.datetime.now().timestamp(),
            value=0.9,
            trading_pair="ETH-USDT",
            exchange="binance_perpetual_testnet",
            position_config=PositionConfig(
                stop_loss=Decimal(0.03),
                take_profit=Decimal(0.03),
                time_limit=30,
                order_type=OrderType.MARKET,
                price=Decimal(1400),
                amount=Decimal(1),
                side=PositionSide.LONG,
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
            if executor.open_order.order_id == event.order_id:
                executor.change_status(SignalExecutorStatus.ACTIVE_POSITION)
            elif executor.stop_loss_order and executor.stop_loss_order.order_id == event.order_id:
                self.logger().info("Closed by Stop loss")
                executor.remove_take_profit()
                executor.change_status(SignalExecutorStatus.CLOSED_BY_STOP_LOSS)
            elif executor.time_limit_order and executor.time_limit_order.order_id == event.order_id:
                self.logger().info("Closed by Time Limit")
                executor.remove_take_profit()
                executor.change_status(SignalExecutorStatus.CLOSED_BY_TIME_LIMIT)
            elif executor.take_profit_order.order_id == event.order_id:
                self.logger().info("Closed by Take Profit")
                executor.change_status(SignalExecutorStatus.CLOSED_BY_TAKE_PROFIT)

    def did_create_buy_order(self, event: BuyOrderCreatedEvent):
        self.did_create_order(event)

    def did_create_sell_order(self, event: SellOrderCreatedEvent):
        self.did_create_order(event)

    def did_create_order(self, event: Union[BuyOrderCreatedEvent, SellOrderCreatedEvent]):
        for executor in self.signal_executors:
            if executor.open_order.order_id == event.order_id:
                executor.open_order.order = executor.get_order(event.order_id)
                executor.change_status(SignalExecutorStatus.ORDER_PLACED)
            elif executor.take_profit_order.order_id == event.order_id:
                executor.take_profit_order.order = executor.get_order(event.order_id)

    def did_fill_order(self, event: OrderFilledEvent):
        for executor in self.signal_executors:
            if executor.open_order.order_id == event.order_id:
                executor.change_status(SignalExecutorStatus.ACTIVE_POSITION)

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
            lines.extend(
                ["", "  Positions:"] + ["    " + line for line in positions_df.to_string(index=False).split("\n")])
        else:
            lines.extend(["", "  No active positions."])

        for executor in self.get_closed_executors():
            lines.extend(["\n-------------------------------||-------------------------------"])
            lines.extend(["", f"  Signal: {executor.signal.id}"] +
                         [f"            - Value: {executor.signal.value}"] +
                         [f"            - Trading Pair: {executor.signal.trading_pair}"] +
                         [f"            - Side: {executor.signal.position_config.side}"] +
                         [f"            - Exchange: {executor.signal.exchange}"] +
                         [f"            - Status: {executor.status}"]
                         )

        for executor in self.get_active_executors():
            lines.extend(["\n-------------------------------||-------------------------------"])
            current_price = self.connectors[executor.signal.exchange].get_mid_price(executor.signal.trading_pair)
            lines.extend([f"  Current price: {current_price}"])
            lines.extend(["", f"  Signal: {executor.signal.id}"] +
                         [f"            - Value: {executor.signal.value}"] +
                         [f"            - Trading Pair: {executor.signal.trading_pair}"] +
                         [f"            - Side: {executor.signal.position_config.side}"] +
                         [f"            - Exchange: {executor.signal.exchange}"] +
                         [f"            - Status: {executor.status}"]
                         )

            start_time = datetime.datetime.fromtimestamp(executor.signal.timestamp)
            duration = datetime.timedelta(seconds=executor.signal.position_config.time_limit)
            end_time = start_time + duration
            current_timestamp = datetime.datetime.fromtimestamp(self.current_timestamp)
            seconds_remaining = (end_time - current_timestamp)

            scale = 50
            time_progress = (duration.seconds - seconds_remaining.seconds) / duration.seconds
            time_bar = ['*' if i < scale * time_progress else '-' for i in range(scale)]
            lines.extend(["Time limit:\n"])
            lines.extend(["".join(time_bar)])

            if executor.stop_loss_price != 0 and executor.take_profit_order and executor.take_profit_order.order:
                progress = 0
                stop_loss_price = executor.stop_loss_price
                if executor.signal.position_config.side == PositionSide.LONG:
                    price_range = executor.take_profit_order.order.price - stop_loss_price
                    progress = (current_price - stop_loss_price) / price_range
                elif executor.signal.position_config.side == PositionSide.SHORT:
                    price_range = stop_loss_price - executor.take_profit_order.order.price
                    progress = (stop_loss_price - current_price) / price_range
                price_bar = [f'--{current_price:.2f}--' if i == int(scale * progress) else '-' for i in range(scale)]
                price_bar.insert(0, f"SL:{stop_loss_price:.2f}")
                price_bar.append(f"TP:{executor.take_profit_order.order.price:.2f}")
                lines.extend(["\nPosition progress:\n"])
                lines.extend(["".join(price_bar)])

        warning_lines.extend(self.balance_warning(self.get_market_trading_pair_tuples()))
        if len(warning_lines) > 0:
            lines.extend(["", "*** WARNINGS ***"] + warning_lines)
        return "\n".join(lines)
