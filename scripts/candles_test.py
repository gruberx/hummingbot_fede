from typing import Dict

from hummingbot.client.settings import ConnectorSetting
from hummingbot.core.clock import Clock
from hummingbot.data_feed.candles_data_feed.candles_data_feed import CandlesDataFeed
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class CandlesTest(ScriptStrategyBase):
    trading_pair = "BTC-USDT"
    exchange = "binance_paper_trade"

    markets = {exchange: {trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorSetting]) -> None:
        super().__init__(connectors)
        self.candles_feed = CandlesDataFeed(exchange="binance", trading_pair="BTCUSDT")
        self.candles_feed.start()

    def stop(self, clock: Clock) -> None:
        if self.candles_feed:
            self.candles_feed.stop()
        super().stop(clock)

    def format_status(self) -> str:
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        lines = []
        warning_lines = []
        warning_lines.extend(self.network_warning(self.get_market_trading_pair_tuples()))

        balance_df = self.get_balance_df()
        lines.extend(["", "  Balances:"] + ["    " + line for line in balance_df.to_string(index=False).split("\n")])

        candles = self.candles_feed.get_candles()
        lines.extend(["", "  OHLC:"] + ["    " + line for line in candles.to_string(index=False).split("\n")])

        warning_lines.extend(self.balance_warning(self.get_market_trading_pair_tuples()))
        if len(warning_lines) > 0:
            lines.extend(["", "*** WARNINGS ***"] + warning_lines)
        return "\n".join(lines)
