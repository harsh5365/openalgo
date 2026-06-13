"""
NIFTY Multi-Target Strategy Module
Encapsulates all business logic for entry/exit conditions and order management.
Designed for both backtesting and live trading via OpenAlgo.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union
from datetime import datetime
import pandas as pd
from enum import Enum


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    PARTIAL_EXIT = "partial_exit"
    CLOSED = "closed"


@dataclass
class StrategyParams:
    """Strategy configuration parameters"""
    supertrend_period: int = 10
    supertrend_multiplier: float = 2
    entry_start_time: str = "09:15"
    entry_end_time: str = "15:30"
    require_daily_trend_alignment: bool = True
    skip_entry_weekdays: set = field(default_factory=lambda: {4})  # Friday
    vix_max: Optional[float] = 18
    event_risk_dates: set = field(default_factory=set)
    lot_qty: int = 65
    initial_lots: int = 3
    
    buy_params: Dict[str, float] = field(default_factory=lambda: {
        't1': 0.751 / 100,
        't2': 0.81 / 100,
        'sl': 0.88 / 100,
    })
    sell_params: Dict[str, float] = field(default_factory=lambda: {
        't1': 0.938 / 100,
        't2': 0.962 / 100,
        'sl': 0.782 / 100,
    })


@dataclass
class Order:
    """Represents a single trade order with multi-lot exit tracking"""
    name: str
    date: str
    entry_time: str
    entry_price: float
    buy_sell: OrderSide
    qty: int
    sl: float
    initial_sl: float
    t1_price: float
    t2_price: float
    t1_done: bool = False
    t2_done: bool = False
    remaining_lots: int = 0
    remaining_qty: int = 0
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    remark: Optional[str] = None
    traded: Optional[str] = "yes"
    entry_idx: Optional[Union[str, pd.Timestamp]] = None
    status: OrderStatus = OrderStatus.ACTIVE
    
    # Tracking data for analysis
    exits: List[Dict] = field(default_factory=list)
    highest_price: float = 0.0
    lowest_price: float = float('inf')


class NIFTYStrategy:
    """Core strategy logic: conditions, order building, and exit management"""
    
    def __init__(self, params: StrategyParams):
        self.params = params
        self.current_order: Optional[Order] = None
        self.exit_history: List[Dict] = []
        
    def build_order(
        self,
        datetimex: Union[str, pd.Timestamp],
        close_price: float,
        side: OrderSide,
    ) -> Order:
        """Create a new order with calculated targets and stop loss"""
        if isinstance(datetimex, pd.Timestamp):
            dt_label = datetimex.strftime("%Y-%m-%d %H:%M:%S")
            entry_idx = datetimex
        else:
            dt_label = datetimex
            entry_idx = datetimex

        params = (
            self.params.buy_params if side == OrderSide.BUY 
            else self.params.sell_params
        )
        entry_price = close_price
        
        if side == OrderSide.BUY:
            stop_loss = round(entry_price * (1 - params['sl']), 2)
            t1_price = round(entry_price * (1 + params['t1']), 2)
            t2_price = round(entry_price * (1 + params['t2']), 2)
        else:
            stop_loss = round(entry_price * (1 + params['sl']), 2)
            t1_price = round(entry_price * (1 - params['t1']), 2)
            t2_price = round(entry_price * (1 - params['t2']), 2)
        
        total_qty = self.params.lot_qty * self.params.initial_lots
        
        return Order(
            name='NIFTY',
            date=dt_label[:10],
            entry_time=dt_label[11:16],
            entry_price=entry_price,
            buy_sell=side,
            qty=total_qty,
            sl=stop_loss,
            initial_sl=stop_loss,
            t1_price=t1_price,
            t2_price=t2_price,
            remaining_lots=self.params.initial_lots,
            remaining_qty=total_qty,
            entry_idx=entry_idx,
        )
    
    def record_exit(
        self,
        order: Order,
        datetimex: Union[str, pd.Timestamp],
        exit_price: float,
        exit_lots: int,
        remark: str,
        highest_price: float,
        lowest_price: float,
    ) -> Dict:
        """Record a partial or full exit and return exit details"""
        bought = order.buy_sell == OrderSide.BUY
        qty = self.params.lot_qty * exit_lots
        
        if bought:
            pnl = round((exit_price - order.entry_price) * qty, 2)
        else:
            pnl = round((order.entry_price - exit_price) * qty, 2)

        if isinstance(datetimex, pd.Timestamp):
            exit_time = datetimex.strftime("%Y-%m-%d %H:%M")
        else:
            exit_time = datetimex[:16]
        
        exit_record = {
            'name': order.name,
            'date': order.date,
            'entry_time': order.entry_time,
            'entry_price': order.entry_price,
            'buy_sell': order.buy_sell.value,
            'qty': qty,
            'sl': order.sl,
            'initial_sl': order.initial_sl,
            'exit_time': exit_time,
            'exit_price': round(exit_price, 2),
            'exit_lots': exit_lots,
            'exit_qty': qty,
            'remaining_lots_after_exit': order.remaining_lots - exit_lots,
            'remaining_qty_after_exit': order.remaining_qty - qty,
            'pnl': pnl,
            'remark': remark,
            'highest_price': highest_price,
            'lowest_price': lowest_price,
        }
        
        # Update order state
        order.remaining_lots -= exit_lots
        order.remaining_qty -= qty
        order.exits.append(exit_record)
        
        if order.remaining_lots <= 0:
            order.status = OrderStatus.CLOSED
        else:
            order.status = OrderStatus.PARTIAL_EXIT
        
        self.exit_history.append(exit_record)
        return exit_record
    
    def check_entry_conditions(
        self,
        candle_time: str,
        candle_date: str,
        weekday: int,
        ema_short: float,
        ema_long: float,
        prev_ema_short: float,
        prev_ema_long: float,
        regime: str,
        prev_hr_st_col: Optional[str] = None,
        vix_close: Optional[float] = None,
    ) -> tuple[bool, bool]:
        """
        Check BUY and SELL conditions.
        Returns: (buy_condition, sell_condition)
        """
        # Time and date checks (common to both)
        time_check = (
            self.params.entry_start_time <= candle_time <= self.params.entry_end_time
            and weekday not in self.params.skip_entry_weekdays
            and candle_date not in self.params.event_risk_dates
        )
        
        # BUY conditions
        ema_buy = (ema_short > ema_long and prev_ema_short <= prev_ema_long)
        regime_buy = regime == 'Low ATR'
        if self.params.require_daily_trend_alignment:
            regime_buy = regime_buy and prev_hr_st_col == "up"
        if self.params.vix_max is not None and vix_close is not None:
            regime_buy = regime_buy and vix_close <= self.params.vix_max
        
        buy_condition = (
            ema_buy and 
            self.current_order is None and 
            time_check and 
            regime_buy
        )
        
        # SELL conditions
        ema_sell = (ema_short < ema_long and prev_ema_short >= prev_ema_long)
        regime_sell = regime == 'Low ATR'
        if self.params.require_daily_trend_alignment:
            regime_sell = regime_sell and prev_hr_st_col == "down"
        if self.params.vix_max is not None and vix_close is not None:
            regime_sell = regime_sell and vix_close <= self.params.vix_max
        
        sell_condition = (
            ema_sell and 
            self.current_order is None and 
            time_check and 
            regime_sell
        )
        
        return buy_condition, sell_condition
    
    def check_exit_conditions(
        self,
        candle_time: str,
        candle_date: str,
        weekday: int,
        high: float,
        low: float,
        close: float,
        highest_price: float,
        lowest_price: float,
    ) -> Optional[tuple[str, float, int]]:
        """
        Check exit conditions for active order.
        Returns: (remark, exit_price, exit_lots) or None
        """
        if self.current_order is None or self.current_order.status == OrderStatus.CLOSED:
            return None
        
        order = self.current_order
        bought = order.buy_sell == OrderSide.BUY
        sold = order.buy_sell == OrderSide.SELL
        is_friday = weekday == 4 and candle_time >= "15:15"
        
        params = (
            self.params.buy_params if bought 
            else self.params.sell_params
        )
        trail_amount = order.entry_price * params['sl']
        
        # BUY exits
        if bought:
            # T1 target
            if not order.t1_done and high >= order.t1_price:
                order.t1_done = True
                order.sl = max(order.sl, order.entry_price)
                return ("buy_t1_hit", order.t1_price, 1)
            
            # T2 target
            if not order.t2_done and high >= order.t2_price:
                order.t2_done = True
                order.sl = max(order.sl, order.t1_price)
                return ("buy_t2_hit", order.t2_price, 1)
            
            # Trailing stop loss for last lot
            if order.remaining_lots == 1:
                trail_sl = round(close - trail_amount, 2)
                order.sl = max(order.sl, trail_sl)
            
            # Stop loss hit
            if low <= order.sl:
                return ("buy_stop_loss_hit", order.sl, order.remaining_lots)
            
            # Friday close-out
            if is_friday:
                return ("friday_hits", close, order.remaining_lots)
        
        # SELL exits
        if sold:
            # T1 target
            if not order.t1_done and low <= order.t1_price:
                order.t1_done = True
                order.sl = min(order.sl, order.entry_price)
                return ("sell_t1_hit", order.t1_price, 1)
            
            # T2 target
            if not order.t2_done and low <= order.t2_price:
                order.t2_done = True
                order.sl = min(order.sl, order.t1_price)
                return ("sell_t2_hit", order.t2_price, 1)
            
            # Trailing stop loss for last lot
            if order.remaining_lots == 1:
                trail_sl = round(close + trail_amount, 2)
                order.sl = min(order.sl, trail_sl)
            
            # Stop loss hit
            if high >= order.sl:
                return ("sell_stop_loss_hit", order.sl, order.remaining_lots)
            
            # Friday close-out
            if is_friday:
                return ("friday_hits", close, order.remaining_lots)
        
        return None
    
    def reset_order(self):
        """Clear current order after closure"""
        self.current_order = None
    
    def get_exit_history_df(self) -> pd.DataFrame:
        """Return all exits as a DataFrame"""
        return pd.DataFrame(self.exit_history)