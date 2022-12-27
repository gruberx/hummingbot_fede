import asyncio
import logging
from typing import Optional, Union

import aiohttp
import pandas as pd

from hummingbot.core.api_throttler.async_throttler import AsyncThrottler
from hummingbot.core.api_throttler.data_types import LinkedLimitWeightPair, RateLimit
from hummingbot.core.network_base import NetworkBase
from hummingbot.core.network_iterator import NetworkStatus
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.core.web_assistant.web_assistants_factory import WebAssistantsFactory
from hummingbot.core.web_assistant.ws_assistant import WSAssistant
from hummingbot.logger import HummingbotLogger


class CandlesDataFeed(NetworkBase):
    _cadf_logger: Optional[HummingbotLogger] = None
    _candles_df_shared_instance: "CandlesDataFeed" = None
    candles_api_configuration = {
        "binance": {
            "base_url": "https://api.binance.com",
            "health_check_endpoint": "/api/v3/ping",
            "historical_candles": {
                "endpoint": "/api/v3/klines",
                # TODO: abstract logic of intervals
                "intervals": ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w",
                              "1M"],
            },
            "rate_limits": [
                RateLimit("raw", limit=1200, time_interval=60),
                RateLimit("/api/v3/klines", limit=1200, time_interval=60,
                          linked_limits=[LinkedLimitWeightPair("raw", 1)]),
                RateLimit("/api/v3/ping", limit=1200, time_interval=60,
                          linked_limits=[LinkedLimitWeightPair("raw", 1)])
            ]
        }}

    @classmethod
    def logger(cls) -> HummingbotLogger:
        if cls._cadf_logger is None:
            cls._cadf_logger = logging.getLogger(__name__)
        return cls._cadf_logger

    @classmethod
    def get_instance(cls) -> "CandlesDataFeed":
        if cls._candles_df_shared_instance is None:
            cls._candles_df_shared_instance = CandlesDataFeed()
        return cls._candles_df_shared_instance

    def __init__(self, exchange: str, trading_pair: str, interval: str = "1m", update_interval: float = 60.0):
        super().__init__()
        self._ready_event = asyncio.Event()
        self._shared_client: Optional[aiohttp.ClientSession] = None
        async_throttler = AsyncThrottler(
            rate_limits=self.candles_api_configuration[exchange]["rate_limits"])
        self._api_factory = WebAssistantsFactory(throttler=async_throttler)

        self._exchange = exchange
        self._trading_pair = trading_pair
        self._interval = interval
        self._check_network_interval = update_interval

        # TODO: check to remove
        self._ev_loop = asyncio.get_event_loop()
        self._candles: Union[pd.DataFrame(), None] = None
        self._update_interval: float = update_interval
        self._fetch_candles_task: Optional[asyncio.Task] = None

    @property
    def name(self):
        return "candles_api"

    @property
    def api_url(self):
        return self.candles_api_configuration[self._exchange]["base_url"]

    @property
    def health_check_endpoint(self):
        return self.candles_api_configuration[self._exchange]["health_check_endpoint"]

    @property
    def health_check_url(self):
        return self.api_url + self.health_check_endpoint

    @property
    def candles_endpoint(self):
        return self.candles_api_configuration[self._exchange]["historical_candles"]["endpoint"]

    @property
    def candles_url(self):
        return self.api_url + self.candles_endpoint

    async def check_network(self) -> NetworkStatus:
        rest_assistant = await self._api_factory.get_rest_assistant()
        try:
            await rest_assistant.execute_request(
                url=self.health_check_url,
                throttler_limit_id=self.health_check_endpoint)
        except IOError as error:
            self.logger().error(error)
            return NetworkStatus.NOT_CONNECTED
        return NetworkStatus.CONNECTED

    def get_candles(self) -> pd.DataFrame:
        return self._candles

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
            await asyncio.sleep(self._update_interval)

    async def fetch_candles(self):
        rest_assistant = await self._api_factory.get_rest_assistant()
        candles = await rest_assistant.execute_request(url=self.candles_url,
                                                       throttler_limit_id=self.candles_endpoint,
                                                       params={"symbol": self._trading_pair,
                                                               "interval": self._interval})

        self._candles = pd.DataFrame(candles, columns=[
            "timestamp", "open", "low", "high", "close", "volume", "close_time",
            "quote_asset_volume", "n_trades", "taker_buy_volume", "taker_sell_volume", "unused"
        ])
        self._candles.drop(columns=["unused"], inplace=True)

    async def start_network(self):
        await self.stop_network()
        self._fetch_candles_task = safe_ensure_future(self.fetch_candles_loop())

    async def stop_network(self):
        if self._fetch_candles_task is not None:
            self._fetch_candles_task.cancel()
            self._fetch_candles_task = None

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
        await ws.connect(ws_url="wss://data-stream.binance.com",
                         ping_timeout=30)
        return ws
