import asyncio
import logging
from typing import Any, Dict, Optional, Union

import aiohttp
import numpy as np
import pandas as pd

from hummingbot.core.api_throttler.async_throttler import AsyncThrottler
from hummingbot.core.api_throttler.data_types import LinkedLimitWeightPair, RateLimit
from hummingbot.core.network_base import NetworkBase
from hummingbot.core.network_iterator import NetworkStatus
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.core.web_assistant.connections.data_types import WSJSONRequest
from hummingbot.core.web_assistant.web_assistants_factory import WebAssistantsFactory
from hummingbot.core.web_assistant.ws_assistant import WSAssistant
from hummingbot.logger import HummingbotLogger


class BinanceCandlesFeed(NetworkBase):
    _bcf_logger: Optional[HummingbotLogger] = None
    _binance_candles_shared_instance: "BinanceCandlesFeed" = None
    base_url = "https://api.binance.com"
    health_check_endpoint = "/api/v3/ping"
    candles_endpoint = "/api/v3/klines"
    wss_url = "wss://stream.binance.com:9443/ws"
    # TODO: abstract logic of intervals
    intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
    rate_limits = [
        RateLimit("raw", limit=1200, time_interval=60),
        RateLimit(candles_endpoint, limit=1200, time_interval=60, linked_limits=[LinkedLimitWeightPair("raw", 1)]),
        RateLimit(health_check_endpoint, limit=1200, time_interval=60, linked_limits=[LinkedLimitWeightPair("raw", 1)])]
    columns = ["timestamp", "open", "low", "high", "close", "volume", "close_time",
               "quote_asset_volume", "n_trades", "taker_buy_base_volume", "taker_buy_quote_volume"]

    @classmethod
    def logger(cls) -> HummingbotLogger:
        if cls._bcf_logger is None:
            cls._bcf_logger = logging.getLogger(__name__)
        return cls._bcf_logger

    @classmethod
    def get_instance(cls) -> "BinanceCandlesFeed":
        if cls._binance_candles_shared_instance is None:
            cls._binance_candles_shared_instance = BinanceCandlesFeed()
        return cls._binance_candles_shared_instance

    def __init__(self, exchange: str, trading_pair: str, interval: str = "1m", update_interval: float = 60.0):
        super().__init__()
        self._ready_event = asyncio.Event()
        self._shared_client: Optional[aiohttp.ClientSession] = None
        async_throttler = AsyncThrottler(
            rate_limits=self.rate_limits)
        self._api_factory = WebAssistantsFactory(throttler=async_throttler)

        self._exchange = exchange
        self._trading_pair = trading_pair
        self._ex_trading_pair = trading_pair.strip("-")
        self._interval = interval
        self._check_network_interval = update_interval

        # TODO: check to remove
        self._ev_loop = asyncio.get_event_loop()
        self._candles_array: Union[np.array(), None] = None
        self._update_interval: float = update_interval
        self._fetch_candles_task: Optional[asyncio.Task] = None
        self._listen_candles_task: Optional[asyncio.Task] = None

    @property
    def name(self):
        return "binance_candles_api"

    @property
    def health_check_url(self):
        return self.base_url + self.health_check_endpoint

    @property
    def candles_url(self):
        return self.base_url + self.candles_endpoint

    async def check_network(self) -> NetworkStatus:
        rest_assistant = await self._api_factory.get_rest_assistant()
        await rest_assistant.execute_request(url=self.health_check_url,
                                             throttler_limit_id=self.health_check_endpoint)
        return NetworkStatus.CONNECTED

    @property
    def candles(self) -> pd.DataFrame:
        return pd.DataFrame(self._candles_array, columns=self.columns)

    async def fetch_candles_loop(self):
        while True:
            try:
                await self.fetch_candles()
                # TODO: check where to wait this event
                self._ready_event.set()
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger().network(f"Error fetching a new candles from {self.candles_url}.", exc_info=True,
                                      app_warning_msg="Couldn't fetch newest candles from CustomAPI. "
                                                      "Check network connection.")
                await self._ready_event.wait()
            await asyncio.sleep(self._update_interval)

    async def fetch_candles(self):
        rest_assistant = await self._api_factory.get_rest_assistant()
        candles = await rest_assistant.execute_request(url=self.candles_url,
                                                       throttler_limit_id=self.candles_endpoint,
                                                       params={"symbol": self._ex_trading_pair,
                                                               "interval": self._interval})

        self._candles_array = np.array(candles)[:, :-1].astype(np.float)

    async def start_network(self):
        await self.stop_network()
        self._fetch_candles_task = safe_ensure_future(self.fetch_candles_loop())
        self._listen_candles_task = safe_ensure_future(self.listen_for_subscriptions())

    async def stop_network(self):
        if self._fetch_candles_task is not None:
            self._fetch_candles_task.cancel()
            self._fetch_candles_task = None
        if self._listen_candles_task is not None:
            self._listen_candles_task.cancel()
            self._listen_candles_task = None

    def start(self):
        NetworkBase.start(self)

    def stop(self):
        NetworkBase.stop(self)

    async def listen_for_subscriptions(self):
        """
        Connects to the trade events and order diffs websocket endpoints and listens to the messages sent by the
        exchange. Each message is stored in its own queue.
        """
        ws: Optional[WSAssistant] = None
        while True:
            try:
                ws: WSAssistant = await self._connected_websocket_assistant()
                await self._subscribe_channels(ws)
                await self._process_websocket_messages(websocket_assistant=ws)
            except asyncio.CancelledError:
                raise
            except ConnectionError as connection_exception:
                self.logger().warning(f"The websocket connection was closed ({connection_exception})")
            except Exception:
                self.logger().exception(
                    "Unexpected error occurred when listening to order book streams. Retrying in 5 seconds...",
                )
                await self._sleep(1.0)
            finally:
                await self._on_order_stream_interruption(websocket_assistant=ws)

    async def _connected_websocket_assistant(self) -> WSAssistant:
        ws: WSAssistant = await self._api_factory.get_ws_assistant()
        await ws.connect(ws_url=self.wss_url,
                         ping_timeout=30)
        return ws

    async def _subscribe_channels(self, ws: WSAssistant):
        """
        Subscribes to the candles events through the provided websocket connection.
        :param ws: the websocket assistant used to connect to the exchange
        """
        try:
            candle_params = []
            candle_params.append(f"{self._ex_trading_pair.lower()}@kline_{self._interval}")
            payload = {
                "method": "SUBSCRIBE",
                "params": candle_params,
                "id": 1
            }
            subscribe_candles_request: WSJSONRequest = WSJSONRequest(payload=payload)

            await ws.send(subscribe_candles_request)

            self.logger().info("Subscribed to public klines...")
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger().error(
                "Unexpected error occurred subscribing to order book trading and delta streams...",
                exc_info=True
            )
            raise

    async def _process_websocket_messages(self, websocket_assistant: WSAssistant):
        async for ws_response in websocket_assistant.iter_messages():
            data: Dict[str, Any] = ws_response.data
            if data is not None:  # data will be None when the websocket is disconnected
                # timestamp = data["k"]["t"]
                # open = data["k"]["o"]
                # low = data["k"]["l"]
                # high = data["k"]["h"]
                # close = data["k"]["c"]
                # volume = data["k"]["v"]
                # close_ts = data["k"]["T"]
                # quote_asset_volume = data["k"]["q"]
                # n_trades = data["k"]["n"]
                # taker_buy_base_volume = data["k"]["V"]
                # taker_buy_quote_volume = data["k"]["V"]
                self.logger().info(data)

    async def _sleep(self, delay):
        """
        Function added only to facilitate patching the sleep in unit tests without affecting the asyncio module
        """
        await asyncio.sleep(delay)

    async def _on_order_stream_interruption(self, websocket_assistant: Optional[WSAssistant] = None):
        websocket_assistant and await websocket_assistant.disconnect()
