import datetime
import random
import time
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pandas_ta as ta  # noqa: F401
from pydantic import BaseModel

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.connector.derivative.position import PositionSide
from hummingbot.connector.markets_recorder import MarketsRecorder
from hummingbot.core.data_type.common import OrderType, PositionAction
from hummingbot.core.data_type.in_flight_order import InFlightOrder
from hummingbot.core.event.events import (
    BuyOrderCompletedEvent,
    BuyOrderCreatedEvent,
    OrderCancelledEvent,
    OrderFilledEvent,
    SellOrderCompletedEvent,
    SellOrderCreatedEvent,
)
from hummingbot.data_feed.candles_data_feed.candles_data_feed import BinanceCandlesFeed
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class PositionConfig(BaseModel):
    timestamp: float
    trading_pair: str
    exchange: str
    order_type: OrderType
    side: PositionSide
    entry_price: Optional[Decimal] = None
    amount: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    time_limit: int


class Signal(BaseModel):
    id: int
    value: float
    position_config: PositionConfig


class BotProfile(BaseModel):
    balance_limit: Decimal
    max_order_amount: Decimal
    long_threshold: float
    short_threshold: float
    leverage: int


class PositionExecutorStatus(Enum):
    NOT_STARTED = 1
    ORDER_PLACED = 2
    CANCELED_BY_TIME_LIMIT = 3
    ACTIVE_POSITION = 4
    CLOSE_PLACED = 5
    CLOSED_BY_TIME_LIMIT = 6
    CLOSED_BY_STOP_LOSS = 7
    CLOSED_BY_TAKE_PROFIT = 8


class TrackedOrder:
    def __init__(self, order_id: Optional[str] = None):
        self._order_id = order_id
        self._order = None

    @property
    def order_id(self):
        return self._order_id

    @order_id.setter
    def order_id(self, order_id: str):
        self._order_id = order_id

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, order: InFlightOrder):
        self._order = order


class PositionExecutor:
    def __init__(self, signal_value: float, position_config: PositionConfig, strategy: ScriptStrategyBase):
        self._position_config = position_config
        self._strategy = strategy
        self._status: PositionExecutorStatus = PositionExecutorStatus.NOT_STARTED
        self._open_order: TrackedOrder = TrackedOrder()
        self._take_profit_order: TrackedOrder = TrackedOrder()
        self._time_limit_order: TrackedOrder = TrackedOrder()
        self._stop_loss_order: TrackedOrder = TrackedOrder()
        self.signal_value = signal_value
        self._close_timestamp = None

    @property
    def position_config(self):
        return self._position_config

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status: PositionExecutorStatus):
        self._status = status

    @property
    def close_timestamp(self):
        return self._status

    @close_timestamp.setter
    def close_timestamp(self, close_timestamp: float):
        self._close_timestamp = close_timestamp

    @property
    def connector(self) -> ConnectorBase:
        return self._strategy.connectors[self._position_config.exchange]

    @property
    def exchange(self):
        return self.position_config.exchange

    @property
    def trading_pair(self):
        return self.position_config.trading_pair

    @property
    def amount(self):
        return self.position_config.amount

    @property
    def entry_price(self):
        if self.status in [PositionExecutorStatus.NOT_STARTED,
                           PositionExecutorStatus.ORDER_PLACED,
                           PositionExecutorStatus.CANCELED_BY_TIME_LIMIT]:
            entry_price = self.position_config.entry_price
            price = entry_price if entry_price else self.connector.get_mid_price(self.trading_pair)
        else:
            price = self.open_order.order.average_executed_price
        return price

    @property
    def close_price(self):
        if self.status == PositionExecutorStatus.CLOSED_BY_STOP_LOSS:
            return self.stop_loss_order.order.average_executed_price
        elif self.status == PositionExecutorStatus.CLOSED_BY_TAKE_PROFIT:
            return self.take_profit_order.order.average_executed_price
        elif self.status == PositionExecutorStatus.CLOSED_BY_TIME_LIMIT:
            return self.time_limit_order.order.average_executed_price
        else:
            return None

    @property
    def pnl(self):
        if self.status in [PositionExecutorStatus.CLOSED_BY_TIME_LIMIT,
                           PositionExecutorStatus.CLOSED_BY_STOP_LOSS,
                           PositionExecutorStatus.CLOSED_BY_TAKE_PROFIT]:
            if self.side == PositionSide.LONG:
                return (self.close_price - self.entry_price) / self.entry_price
            else:
                return (self.entry_price - self.close_price) / self.entry_price
        elif self.status == PositionExecutorStatus.ACTIVE_POSITION:
            current_price = self.connector.get_mid_price(self.trading_pair)
            if self.side == PositionSide.LONG:
                return (current_price - self.entry_price) / self.entry_price
            else:
                return (self.entry_price - current_price) / self.entry_price
        else:
            return 0

    @property
    def timestamp(self):
        return self.position_config.timestamp

    @property
    def time_limit(self):
        return self.position_config.time_limit

    @property
    def end_time(self):
        return self.timestamp + self.time_limit

    @property
    def side(self):
        return self.position_config.side

    @property
    def open_order_type(self):
        return self.position_config.order_type

    @property
    def stop_loss_price(self):
        stop_loss_price = self.entry_price * (
            1 - self._position_config.stop_loss) if self.side == PositionSide.LONG else self.entry_price * (
            1 + self._position_config.stop_loss)
        return stop_loss_price

    @property
    def take_profit_price(self):
        take_profit_price = self.entry_price * (
            1 + self._position_config.take_profit) if self.side == PositionSide.LONG else self.entry_price * (
            1 - self._position_config.take_profit)
        return take_profit_price

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
        if self.status == PositionExecutorStatus.NOT_STARTED:
            self.control_open_order()
        elif self.status == PositionExecutorStatus.ORDER_PLACED:
            self.control_cancel_order_by_time_limit()
        elif self.status == PositionExecutorStatus.ACTIVE_POSITION:
            self.control_take_profit()
            self.control_stop_loss()
            self.control_time_limit()
        elif self.status == PositionExecutorStatus.CLOSE_PLACED:
            pass

    def clean_executor(self):
        if self.status in [PositionExecutorStatus.CLOSED_BY_TIME_LIMIT,
                           PositionExecutorStatus.CLOSED_BY_STOP_LOSS]:
            if self.take_profit_order.order and (
                    self.take_profit_order.order.is_cancelled or
                    self.take_profit_order.order.is_pending_cancel_confirmation or
                    self.take_profit_order.order.is_failure
            ):
                pass
            else:
                self._strategy.logger().info(f"Take profit order status: {self.take_profit_order.order.current_state}")
                self.remove_take_profit()

    def remove_take_profit(self):
        self._strategy.cancel(
            connector_name=self.exchange,
            trading_pair=self.trading_pair,
            order_id=self._take_profit_order.order_id
        )
        self._strategy.logger().info("Removing take profit since the position is not longer available")

    def control_open_order(self):
        if not self.open_order.order_id:
            order_id = self._strategy.place_order(
                connector_name=self.exchange,
                trading_pair=self.trading_pair,
                amount=self.amount,
                price=self.entry_price,
                order_type=self.open_order_type,
                position_action=PositionAction.OPEN,
                position_side=self.side
            )
            self._open_order.order_id = order_id

    def control_cancel_order_by_time_limit(self):
        if self.timestamp / 1000 + self.time_limit >= self._strategy.current_timestamp:
            self._strategy.cancel(
                connector_name=self.exchange,
                trading_pair=self.trading_pair,
                order_id=self._open_order.order_id
            )

    def control_take_profit(self):
        if not self.take_profit_order.order_id:
            order_id = self._strategy.place_order(
                connector_name=self._position_config.exchange,
                trading_pair=self._position_config.trading_pair,
                amount=self.open_order.order.executed_amount_base,
                price=self.take_profit_price,
                order_type=OrderType.LIMIT,
                position_action=PositionAction.CLOSE,
                position_side=PositionSide.LONG if self.side == PositionSide.SHORT else PositionSide.SHORT
            )
            self._take_profit_order.order_id = order_id

    def control_stop_loss(self):
        current_price = self.connector.get_mid_price(self.trading_pair)
        trigger_stop_loss = False
        if self.side == PositionSide.LONG and current_price <= self.stop_loss_price:
            trigger_stop_loss = True
        elif self.side == PositionSide.SHORT and current_price >= self.stop_loss_price:
            trigger_stop_loss = True

        if trigger_stop_loss:
            if not self.stop_loss_order.order_id:
                order_id = self._strategy.place_order(
                    connector_name=self.exchange,
                    trading_pair=self.trading_pair,
                    amount=self.open_order.order.executed_amount_base,
                    price=current_price,
                    order_type=OrderType.MARKET,
                    position_action=PositionAction.CLOSE,
                    position_side=PositionSide.LONG if self.side == PositionSide.SHORT else PositionSide.SHORT
                )
                self._stop_loss_order.order_id = order_id
                self._status = PositionExecutorStatus.CLOSE_PLACED

    def control_time_limit(self):
        position_expired = self.end_time < self._strategy.current_timestamp
        if position_expired:
            if not self._time_limit_order.order_id:
                price = self.connector.get_mid_price(self.trading_pair)
                order_id = self._strategy.place_order(
                    connector_name=self.exchange,
                    trading_pair=self.trading_pair,
                    amount=self.open_order.order.executed_amount_base,
                    price=price,
                    order_type=OrderType.MARKET,
                    position_action=PositionAction.CLOSE,
                    position_side=PositionSide.LONG if self.side == PositionSide.SHORT else PositionSide.SHORT
                )
                self._time_limit_order.order_id = order_id
                self._status = PositionExecutorStatus.CLOSE_PLACED

    def process_order_completed_event(self, event: Union[BuyOrderCompletedEvent, SellOrderCompletedEvent]):
        if self.open_order.order_id == event.order_id:
            self.status = PositionExecutorStatus.ACTIVE_POSITION
        elif self.stop_loss_order.order_id == event.order_id:
            self._strategy.logger().info("Closed by Stop loss")
            self.remove_take_profit()
            self.status = PositionExecutorStatus.CLOSED_BY_STOP_LOSS
            self.close_timestamp = time.time()
        elif self.time_limit_order.order_id == event.order_id:
            self._strategy.logger().info("Closed by Time Limit")
            self.remove_take_profit()
            self.status = PositionExecutorStatus.CLOSED_BY_TIME_LIMIT
            self.close_timestamp = time.time()
        elif self.take_profit_order.order_id == event.order_id:
            self._strategy.logger().info("Closed by Take Profit")
            self.status = PositionExecutorStatus.CLOSED_BY_TAKE_PROFIT
            self.close_timestamp = time.time()

    def process_order_created_event(self, event: Union[BuyOrderCreatedEvent, SellOrderCreatedEvent]):
        if self.open_order.order_id == event.order_id:
            self.open_order.order = self.get_order(event.order_id)
            self.status = PositionExecutorStatus.ORDER_PLACED
        elif self.take_profit_order.order_id == event.order_id:
            self.take_profit_order.order = self.get_order(event.order_id)
            self._strategy.logger().info("Take profit Created")
        elif self.stop_loss_order.order_id == event.order_id:
            self._strategy.logger().info("Stop loss Created")
            self.stop_loss_order.order = self.get_order(event.order_id)
        elif self.time_limit_order.order_id == event.order_id:
            self._strategy.logger().info("Time Limit Created")
            self.time_limit_order.order = self.get_order(event.order_id)

    def process_order_canceled_event(self, event: OrderCancelledEvent):
        if self.open_order.order_id == event.order_id:
            self.status = PositionExecutorStatus.CANCELED_BY_TIME_LIMIT
            self.close_timestamp = time.time()

    def process_order_filled_event(self, event: OrderFilledEvent):
        if self.open_order.order_id == event.order_id:
            if self.status == PositionExecutorStatus.ACTIVE_POSITION:
                self._strategy.logger().info("Position incremented, updating take profit.")
            else:
                self.status = PositionExecutorStatus.ACTIVE_POSITION
                self._strategy.logger().info("Position taken, placing take profit next tick.")


class SignalFactory:
    def __init__(self, max_records: int, connectors: Dict[str, ConnectorBase], interval: str = "1m"):
        self.connectors = connectors
        self.candles = {
            connector_name: {trading_pair: BinanceCandlesFeed(trading_pair=trading_pair, interval=interval,
                                                              max_records=max_records)
                             for trading_pair in connector.trading_pairs} for
            connector_name, connector in self.connectors.items()}
        for connector_name, trading_pairs_candles in self.candles.items():
            for candles in trading_pairs_candles.values():
                candles.start()

    @property
    def all_data_sources_ready(self):
        return all(np.array([[candles.is_ready for trading_pair, candles in trading_pairs_candles.items()]
                             for connector_name, trading_pairs_candles in self.candles.items()]).flatten())

    def candles_df(self):
        return {connector_name: {trading_pair: candles.candles for trading_pair, candles in
                trading_pairs_candles.items()}
                for connector_name, trading_pairs_candles in self.candles.items()}

    def features_df(self):
        candles_df = self.candles_df().copy()
        for connector_name, trading_pairs_candles in candles_df.items():
            for trading_pair, candles in trading_pairs_candles.items():
                candles.ta.rsi(length=14, append=True)
        return candles_df

    def current_features(self):
        return {connector_name: {trading_pair: features.iloc[-1, :].to_dict() for trading_pair, features in
                                 trading_pairs_features.items()}
                for connector_name, trading_pairs_features in self.features_df().items()}

    def get_signals(self):
        signals = self.current_features().copy()
        for connector_name, trading_pairs_features in signals.items():
            for trading_pair, features in trading_pairs_features.items():
                value = (features["RSI_14"] - 50) / 50
                signal = Signal(
                    id=str(random.randint(1, 1e10)),
                    value=value,
                    position_config=PositionConfig(
                        timestamp=datetime.datetime.now().timestamp(),
                        stop_loss=Decimal(0.01),
                        take_profit=Decimal(0.004),
                        time_limit=10,
                        order_type=OrderType.MARKET,
                        amount=Decimal(1),
                        side=PositionSide.LONG if value < 0 else PositionSide.SHORT,
                        trading_pair=trading_pair,
                        exchange=connector_name,
                    ),
                )
                signals[connector_name][trading_pair] = signal
        return signals


class DirectionalStrategyPerpetuals(ScriptStrategyBase):
    bot_profile = BotProfile(
        balance_limit=Decimal(1000),
        max_order_amount=Decimal(30),
        long_threshold=0.5,
        short_threshold=-0.5,
        leverage=10,
    )
    max_executors_by_connector_trading_pair = 1
    trading_pairs = ["ETH-USDT", "BTC-USDT"]
    exchange = "binance_perpetual_testnet"
    set_leverage_flag = None
    signal_executors: Dict[str, PositionExecutor] = {}
    stored_executors: List[str] = []
    markets = {exchange: set(trading_pairs)}

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        self.signal_factory = None

    def get_active_executors(self):
        return {signal: executor for signal, executor in self.signal_executors.items() if executor.status not in
                [PositionExecutorStatus.CLOSED_BY_TIME_LIMIT,
                 PositionExecutorStatus.CLOSED_BY_TAKE_PROFIT,
                 PositionExecutorStatus.CLOSED_BY_STOP_LOSS,
                 PositionExecutorStatus.CANCELED_BY_TIME_LIMIT]
                }

    def get_active_executors_by_connector_trading_pair(self, connector_name, trading_pair):
        return {signal: executor for signal, executor in self.signal_executors.items() if executor.status not in
                [PositionExecutorStatus.CLOSED_BY_TIME_LIMIT,
                 PositionExecutorStatus.CLOSED_BY_TAKE_PROFIT,
                 PositionExecutorStatus.CLOSED_BY_STOP_LOSS,
                 PositionExecutorStatus.CANCELED_BY_TIME_LIMIT] and executor.exchange == connector_name
                and executor.trading_pair == trading_pair
                }

    def get_closed_executors(self):
        return {signal: executor for signal, executor in self.signal_executors.items() if executor.status in
                [PositionExecutorStatus.CLOSED_BY_TIME_LIMIT,
                 PositionExecutorStatus.CLOSED_BY_TAKE_PROFIT,
                 PositionExecutorStatus.CLOSED_BY_STOP_LOSS,
                 PositionExecutorStatus.CANCELED_BY_TIME_LIMIT]
                }

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
        # TODO: fix bug in binance perpetuals to set leverage
        # if not self.set_leverage_flag:
        #     for connector in self.connectors.values():
        #         for trading_pair in connector.trading_pairs:
        #             connector.set_leverage(trading_pair=trading_pair, leverage=self.bot_profile.leverage)
        if not self.signal_factory:
            self.signal_factory = SignalFactory(max_records=500, connectors=self.connectors, interval="1s")
        if self.signal_factory.all_data_sources_ready:
            # TODO: Order the dictionary by highest abs signal values
            for connector_name, trading_pair_signals in self.signal_factory.get_signals().items():
                for trading_pair, signal in trading_pair_signals.items():
                    if len(self.get_active_executors_by_connector_trading_pair(connector_name, trading_pair).keys()) < self.max_executors_by_connector_trading_pair:
                        if signal.value > self.bot_profile.long_threshold or signal.value < self.bot_profile.short_threshold:
                            position_config = signal.position_config
                            price = self.connectors[position_config.exchange].get_mid_price(position_config.trading_pair)
                            position_config.amount = (self.bot_profile.max_order_amount / price) * position_config.amount
                            self.signal_executors[signal.id] = PositionExecutor(
                                signal_value=signal.value,
                                position_config=position_config,
                                strategy=self
                            )
        else:
            self.logger().info("Waiting until all the data sources are ready.")
        for executor in self.get_active_executors().values():
            executor.control_position()
        for executor in self.get_closed_executors().values():
            executor.clean_executor()
        self.store_executors()

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
        for executor in self.get_active_executors().values():
            executor.process_order_completed_event(event)

    def did_create_buy_order(self, event: BuyOrderCreatedEvent):
        self.did_create_order(event)

    def did_create_sell_order(self, event: SellOrderCreatedEvent):
        self.did_create_order(event)

    def did_create_order(self, event: Union[BuyOrderCreatedEvent, SellOrderCreatedEvent]):
        for executor in self.get_active_executors().values():
            executor.process_order_created_event(event)

    def did_fill_order(self, event: OrderFilledEvent):
        for executor in self.get_active_executors().values():
            executor.process_order_filled_event(event)

    def did_cancel_order(self, event: OrderCancelledEvent):
        for executor in self.get_active_executors().values():
            executor.process_order_canceled_event(event)

    def store_executors(self):
        executors_to_store = {signal_id: executor for signal_id, executor in self.get_closed_executors().items()
                              if signal_id not in self.stored_executors}
        for signal_id, executor in executors_to_store.items():
            signal = {
                "id": signal_id,
                "timestamp": int(executor.timestamp),
                "close_timestamp": int(executor.close_timestamp),
                "value": executor.signal_value,
                "sl": executor.position_config.stop_loss,
                "tp": executor.position_config.take_profit,
                "tl": executor.position_config.time_limit,
                "exchange": executor.exchange,
                "trading_pair": executor.trading_pair,
                "side": executor.side.name,
                "last_status": executor.status.name,
                "order_type": executor.position_config.order_type.name,
                "amount": executor.amount,
                "entry_price": executor.entry_price,
                "close_price": executor.close_price,
                "pnl": executor.pnl,
                "leverage": self.bot_profile.leverage,
            }
            MarketsRecorder.get_instance().add_closed_signal(signal)
            self.stored_executors.append(signal_id)

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

        if len(self.get_closed_executors().keys()) > 0:
            lines.extend(["\n########################################## Closed Executors ##########################################"])

        for signal_id, executor in self.get_closed_executors().items():
            lines.extend([f"""
| Signal: {signal_id}
| Trading Pair: {executor.trading_pair} | Exchange: {executor.exchange} | Side: {executor.side} | Amount: {executor.amount:.4f}
| Status: {executor.status}
| Entry price: {executor.entry_price}  | Close price: {executor.close_price} --> PNL: {executor.pnl * 100:.2f}%
"""])
            lines.extend(["-----------------------------------------------------------------------------------------------------------"])

        if len(self.get_active_executors().keys()) > 0:
            lines.extend(["\n########################################## Active Executors ##########################################"])

        for signal_id, executor in self.get_active_executors().items():
            current_price = self.connectors[executor.position_config.exchange].get_mid_price(
                executor.position_config.trading_pair)
            lines.extend([f"""
| Signal: {signal_id}
| Trading Pair: {executor.trading_pair} | Exchange: {executor.exchange} | Side: {executor.side} | Amount: {executor.amount:.4f}
| Entry price: {executor.entry_price}  | Current price: {current_price} --> PNL: {executor.pnl * 100:.2f}%

"""])
            time_scale = 67
            price_scale = 47

            start_time = datetime.datetime.fromtimestamp(executor.position_config.timestamp)
            duration = datetime.timedelta(seconds=executor.position_config.time_limit)
            end_time = start_time + duration
            current_timestamp = datetime.datetime.fromtimestamp(self.current_timestamp)
            seconds_remaining = (end_time - current_timestamp)

            time_progress = (duration.seconds - seconds_remaining.seconds) / duration.seconds
            time_bar = "".join(['*' if i < time_scale * time_progress else '-' for i in range(time_scale)])
            lines.extend([f"Time limit: {time_bar}"])

            if executor.stop_loss_price != 0 and executor.take_profit_order and executor.take_profit_order.order:
                progress = 0
                stop_loss_price = executor.stop_loss_price
                if executor.side == PositionSide.LONG:
                    price_range = executor.take_profit_order.order.price - stop_loss_price
                    progress = (current_price - stop_loss_price) / price_range
                elif executor.side == PositionSide.SHORT:
                    price_range = stop_loss_price - executor.take_profit_order.order.price
                    progress = (stop_loss_price - current_price) / price_range
                price_bar = [f'--{current_price:.2f}--' if i == int(price_scale * progress) else '-' for i in range(price_scale)]
                price_bar.insert(0, f"SL:{stop_loss_price:.2f}")
                price_bar.append(f"TP:{executor.take_profit_order.order.price:.2f}")
                lines.extend(["".join(price_bar), "\n"])
            lines.extend(["-----------------------------------------------------------------------------------------------------------"])

        if self.signal_factory and self.signal_factory.all_data_sources_ready:
            lines.extend(["\n############################################ Market Data ############################################"])
            candles_df = self.signal_factory.features_df()
            for connector_name, trading_pair_signal in self.signal_factory.get_signals().items():
                for trading_pair, signal in trading_pair_signal.items():
                    df = candles_df[self.exchange][trading_pair]
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    lines.extend([f"""

| Trading Pair: {trading_pair} | Exchange: {connector_name}
| Signal: {signal.value:.2f}

"""])
                    lines.extend(["    " + line for line in df.tail().to_string(index=False).split("\n")])
                    lines.extend(["\n-----------------------------------------------------------------------------------------------------------"])

        else:
            lines.extend(["", "  No data collected."])

        warning_lines.extend(self.balance_warning(self.get_market_trading_pair_tuples()))
        if len(warning_lines) > 0:
            lines.extend(["", "*** WARNINGS ***"] + warning_lines)
        return "\n".join(lines)
