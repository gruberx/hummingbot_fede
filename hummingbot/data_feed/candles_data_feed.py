import asyncio
import logging
from typing import Optional, Union

import aiohttp
import pandas as pd

from hummingbot.core.network_base import NetworkBase
from hummingbot.core.network_iterator import NetworkStatus
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.logger import HummingbotLogger


class CandlesDataFeed(NetworkBase):
    cadf_logger: Optional[HummingbotLogger] = None
    candles_api_configuration = {
        "binance": {
            "base_url": "https://api.binance.com",
            "historical_candles_endpoint": "/api/v3/klines",
            # TODO: abstract logic of intervals
            "intervals": ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
        }
    }

    @classmethod
    def logger(cls) -> HummingbotLogger:
        if cls.cadf_logger is None:
            cls.cadf_logger = logging.getLogger(__name__)
        return cls.cadf_logger

    def __init__(self, exchange: str, trading_pair: str, interval: str = "1m", update_interval: float = 5.0):
        super().__init__()
        self._ready_event = asyncio.Event()
        self._shared_client: Optional[aiohttp.ClientSession] = None
        self._api_url = self.candles_api_configuration[exchange]["base_url"]
        self._historical_candles_endpoint = self.candles_api_configuration[exchange]["historical_candles_endpoint"]
        self._candles_url = self._api_url + self._historical_candles_endpoint
        self._trading_pair = trading_pair
        self._interval = interval
        self._check_network_interval = 30.0
        self._ev_loop = asyncio.get_event_loop()
        self._candles: Union[pd.DataFrame(), None] = None
        self._update_interval: float = update_interval
        self._fetch_candles_task: Optional[asyncio.Task] = None

    @property
    def name(self):
        return "custom_api"

    @property
    def health_check_endpoint(self):
        return self._api_url

    def _http_client(self) -> aiohttp.ClientSession:
        if self._shared_client is None:
            self._shared_client = aiohttp.ClientSession()
        return self._shared_client

    async def check_network(self) -> NetworkStatus:
        client = self._http_client()
        async with client.request("GET", self.health_check_endpoint) as resp:
            status_text = await resp.text()
            if resp.status != 200:
                raise Exception(f"Custom API Feed {self.name} server error: {status_text}")
        return NetworkStatus.CONNECTED

    def get_candles(self) -> pd.DataFrame:
        return self._candles

    async def fetch_candles_loop(self):
        while True:
            try:
                await self.fetch_candles()
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger().network(f"Error fetching a new candles from {self._api_url}.", exc_info=True,
                                      app_warning_msg="Couldn't fetch newest candles from CustomAPI. "
                                                      "Check network connection.")

            await asyncio.sleep(self._update_interval)

    async def fetch_candles(self):
        client = self._http_client()
        async with client.request("GET", self._candles_url, params={"symbol": self._trading_pair,
                                                                    "interval": self._interval}) as resp:
            json = await resp.json()
            if resp.status != 200:
                raise Exception(f"Custom API Feed {self.name} server error: {json}")
            self._candles = pd.DataFrame(json, columns=[
                "timestamp", "open", "low", "high", "close", "volume", "close_time",
                "quote_asset_volume", "n_trades", "taker_buy_volume", "taker_sell_volume", "unused"
            ])
            self._candles.drop(columns=["unused"], inplace=True)
        self._ready_event.set()

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
