from sqlalchemy import BigInteger, Column, Index, Integer, Text

from hummingbot.model import HummingbotBase
from hummingbot.model.decimal_type_decorator import SqliteDecimal


class Signal(HummingbotBase):
    __tablename__ = "Signal"
    __table_args__ = (Index("o_exchange_trading_pair_index",
                            "exchange", "trading_pair"),)

    id = Column(Text, primary_key=True, nullable=False)
    timestamp = Column(BigInteger, nullable=False)
    close_timestamp = Column(BigInteger, nullable=False)
    value = Column(SqliteDecimal(6), nullable=False)
    sl = Column(SqliteDecimal(6), nullable=False)
    tp = Column(SqliteDecimal(6), nullable=False)
    tl = Column(SqliteDecimal(6), nullable=False)
    exchange = Column(Text, nullable=False)
    trading_pair = Column(Text, nullable=False)
    side = Column(Text, nullable=False)
    last_status = Column(Text, nullable=False)
    order_type = Column(Text, nullable=False)
    amount = Column(SqliteDecimal(6), nullable=False)
    entry_price = Column(SqliteDecimal(6), nullable=False)
    close_price = Column(SqliteDecimal(6), nullable=False)
    pnl = Column(SqliteDecimal(6), nullable=False)
    leverage = Column(Integer, nullable=False, default=1)
