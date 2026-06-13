# Update codebase as well
import pandas as pd
from rich import print
import tradehull_backtesting_support as tbs
import datetime
import pdb

supertrend_period     = 10
supertrend_multiplier = 2
ENTRY_START_TIME = "09:15"
ENTRY_END_TIME = "15:30"
REQUIRE_DAILY_TREND_ALIGNMENT = True
SKIP_ENTRY_WEEKDAYS = {4}  # Friday entries were net negative in this backtest.
VIX_MAX = 18
EVENT_RISK_DATES = {
    # Add known high-risk event dates here, e.g. budget, RBI policy, election result,
    # major Fed/US CPI days when you do not want fresh positional trades.
    # "2024-06-04",
}
VERBOSE = False
LOT_QTY = 65
INITIAL_LOTS = 3
BUY_PARAMS = {
    't1': 0.751 / 100,
    't2': 0.81 / 100,
    'sl': 0.88 / 100,
}
SELL_PARAMS = {
    't1': 0.938 / 100,
    't2': 0.962 / 100,
    'sl': 0.782 / 100,
}

# df            = pd.read_csv('Historical Data/NIFTY 15 mins 2021-26.csv')
# hr_df         = pd.read_csv('Historical Data/NIFTY 60 mins 2021-26.csv')
df            = pd.read_csv('Historical Data/NIFTY 15 mins 2021-26v2.csv')
hr_df         = pd.read_csv('Historical Data/NIFTY 1D 2021-26.csv')
vix_data      = pd.read_csv('Historical Data/VIX 22-04-2021 to 2026.csv').set_index('Date ')
df            = df.set_index('timestamp')
hr_df         = hr_df.set_index('timestamp')
hr_df.index = pd.to_datetime(hr_df.index)
vix_data.index = pd.to_datetime(vix_data.index)
# df            = df.tail(500)


# this below condition to check data from specific dates
# df = df[df.index.str[:10].isin(['2024-10-03', '2024-10-04'])]

# Filter the DataFrame to keep only records from 2024-10-04 (04-10-2024 in dd-mm-yyyy)
# df            = df[:3000]
df            = tbs.supertrend(df, atr_period=supertrend_period, atr_multiplier=supertrend_multiplier)
hr_df         = tbs.supertrend(hr_df, atr_period=supertrend_period, atr_multiplier=supertrend_multiplier)
# df['slow_k'], df['slow_d'] = tbs.slow_stochastic(df, 144, 34) 
df            = tbs.emacrossover(df, 7, 21) # pass dataframe, lower days, and higher days
# df            = df['2021-02-01 09:15:00+05:30':]
# df            = tbs.calculate_rsi_using_talib(df, 21, 'close')
df            = tbs.atr_calculate(df, 14)
df            = tbs.atr_regime(df)

st_col        = f'STX_{supertrend_period}_{supertrend_multiplier}'
st_val        = f'ST_{supertrend_period}_{supertrend_multiplier}'
final_result  = []


def empty_order():
    return {
        'name': None,
        'date': None,
        'entry_time': None,
        'entry_price': None,
        'buy_sell': None,
        'qty': None,
        'sl': None,
        'exit_time': None,
        'exit_price': None,
        'pnl': None,
        'remark': None,
        'traded': None,
    }


def build_order(datetimex, candle_data, side):
    params = BUY_PARAMS if side == "BUY" else SELL_PARAMS
    entry_price = candle_data['close']
    if side == "BUY":
        stop_loss = round(entry_price * (1 - params['sl']), 2)
        t1_price = round(entry_price * (1 + params['t1']), 2)
        t2_price = round(entry_price * (1 + params['t2']), 2)
    else:
        stop_loss = round(entry_price * (1 + params['sl']), 2)
        t1_price = round(entry_price * (1 - params['t1']), 2)
        t2_price = round(entry_price * (1 - params['t2']), 2)

    return {
        'name': 'NIFTY',
        'date': datetimex[:10],
        'entry_time': datetimex[11:16],
        'entry_price': entry_price,
        'buy_sell': side,
        'qty': LOT_QTY * INITIAL_LOTS,
        'sl': stop_loss,
        'initial_sl': stop_loss,
        't1_price': t1_price,
        't2_price': t2_price,
        't1_done': False,
        't2_done': False,
        'remaining_lots': INITIAL_LOTS,
        'remaining_qty': LOT_QTY * INITIAL_LOTS,
        'exit_time': None,
        'exit_price': None,
        'pnl': None,
        'remark': None,
        'traded': "yes",
        'entry_idx': datetimex,
    }


def append_exit(order, datetimex, exit_price, exit_lots, remark, highest_price, lowest_price):
    qty = LOT_QTY * exit_lots
    if order['buy_sell'] == "BUY":
        pnl = round((exit_price - order['entry_price']) * qty, 2)
    else:
        pnl = round((order['entry_price'] - exit_price) * qty, 2)

    result = order.copy()
    result['exit_time'] = datetimex[:16]
    result['exit_price'] = round(exit_price, 2)
    result['exit_lots'] = exit_lots
    result['exit_qty'] = qty
    result['qty'] = qty
    result['remaining_lots_after_exit'] = order['remaining_lots'] - exit_lots
    result['remaining_qty_after_exit'] = order['remaining_qty'] - qty
    result['pnl'] = pnl
    result['remark'] = remark
    result['highest_price'] = highest_price
    result['lowest_price'] = lowest_price
    final_result.append(result)

    order['remaining_lots'] -= exit_lots
    order['remaining_qty'] -= qty


current_order = empty_order()


for datetimex, candle_data in df.iterrows():

    # Skip iteration if it's Saturday (5) or Sunday (6)
    candle_date = datetimex[:10]
    candle_time = datetimex[11:16]
    weekday = datetime.datetime.strptime(candle_date, "%Y-%m-%d").weekday()
    if weekday in [5, 6]:
        continue
    if candle_time < "09:15" or candle_time > "15:30":
        continue

    # ----------------------------------------- Entry Block -----------------------------------------
    # prev_k_below_20 = df['slow_k'].shift(1) < 20
    # bc1 = candle_data[st_col] == "up"
    # To avoid errors in the first few rows where shift(1) would result in NaN, we wrap the condition in a try-except block.
    bc1 = False
    current_idx = df.index.get_loc(datetimex)
    # Debugging for 9:15-9:45 candles on 4th Oct 2024
    # if datetimex[:10] == "2024-10-04" and datetimex[11:16] in ["09:15", "09:30", "09:45"]:
    #     print(f"Debug {datetimex}:")
    #     print(candle_data)
    pos = -1
    vix_pos = -1
    if current_idx > 0:
        hr_ts = hr_df.index.asof(pd.Timestamp(candle_date))
        vix_ts = vix_data.index.asof(pd.Timestamp(candle_date))
        if pd.notna(hr_ts):
            hr_loc = hr_df.index.get_loc(hr_ts)
            pos = tbs.index_loc_to_pos(hr_loc)
        if pd.notna(vix_ts):
            vix_loc = vix_data.index.get_loc(vix_ts)
            vix_pos = tbs.index_loc_to_pos(vix_loc)

    if current_idx > 0:
        prev_idx = current_idx - 1
        bc1 = (
            candle_data['ema_short'] > candle_data['ema_long'] and
            df.iloc[prev_idx]['ema_short'] <= df.iloc[prev_idx]['ema_long']
        )
    bc2 = current_order['traded'] is None
    bc3 = (
        ENTRY_START_TIME <= candle_time <= ENTRY_END_TIME and
        weekday not in SKIP_ENTRY_WEEKDAYS and
        candle_date not in EVENT_RISK_DATES
    )
    # this is the slow stochastic crossover
    # bc4 = prev_k_below_20 and tbs.crossed_above(candle_data['slow_k'], candle_data['slow_d'])
    # bc4 = df['ema_signal'].loc[datetimex] == 1
    prev_hr_st_col = None
    bc4 = candle_data['regime'] == 'Low ATR'
    if pos > 0:
        prev_hr = hr_df.iloc[pos - 1]
        prev_hr_st_col = prev_hr[st_col]
    if REQUIRE_DAILY_TREND_ALIGNMENT:
        bc4 = bc4 and prev_hr_st_col == "up"
    if vix_pos > 0 and VIX_MAX is not None:
        prev_vix = vix_data.iloc[vix_pos - 1]
        bc4 = bc4 and prev_vix['Close '] <= VIX_MAX
        

    # sc1 = candle_data[st_col] == "down"
    sc1 = False
    if current_idx > 0:
        prev_idx = current_idx - 1
        sc1 = (
            candle_data['ema_short'] < candle_data['ema_long'] and
            df.iloc[prev_idx]['ema_short'] >= df.iloc[prev_idx]['ema_long']
        )
    sc2 = current_order['traded'] is None
    sc3 = bc3
    sc4 = candle_data['regime'] == 'Low ATR'
    if REQUIRE_DAILY_TREND_ALIGNMENT:
        sc4 = sc4 and prev_hr_st_col == "down"
    if vix_pos > 0 and VIX_MAX is not None:
        prev_vix = vix_data.iloc[vix_pos - 1]
        sc4 = sc4 and prev_vix['Close '] <= VIX_MAX


    if bc1 and bc2 and bc3 and bc4:
        if VERBOSE:
            print(f"Buy condition hit for NIFTY on {datetimex}")
        current_order = build_order(datetimex, candle_data, "BUY")
        continue


    if sc1 and sc2 and sc3 and sc4:
        if VERBOSE:
            print(f"Sell condition hit for NIFTY on {datetimex}")
        current_order = build_order(datetimex, candle_data, "SELL")
        continue



    # ----------------------------------------- Exit Block -----------------------------------------
    if current_order['traded'] == "yes":

        bought = current_order['buy_sell'] == "BUY"
        sold    = current_order['buy_sell'] == "SELL"
        its_friday = weekday == 4 and candle_time >= "15:15"

        highest_price = df.loc[current_order['entry_idx']:datetimex, 'high'].max()
        lowest_price = df.loc[current_order['entry_idx']:datetimex, 'low'].min()
        entry_price = current_order['entry_price']
        params = BUY_PARAMS if bought else SELL_PARAMS
        trail_amount = entry_price * params['sl']

        if bought:
            if not current_order['t1_done'] and candle_data['high'] >= current_order['t1_price']:
                append_exit(
                    current_order,
                    datetimex,
                    current_order['t1_price'],
                    1,
                    "buy_t1_hit",
                    highest_price,
                    lowest_price,
                )
                current_order['t1_done'] = True
                current_order['sl'] = max(current_order['sl'], entry_price)

            if not current_order['t2_done'] and candle_data['high'] >= current_order['t2_price']:
                append_exit(
                    current_order,
                    datetimex,
                    current_order['t2_price'],
                    1,
                    "buy_t2_hit",
                    highest_price,
                    lowest_price,
                )
                current_order['t2_done'] = True
                current_order['sl'] = max(current_order['sl'], current_order['t1_price'])

            if current_order['remaining_lots'] == 1:
                trail_sl = round(candle_data['close'] - trail_amount, 2)
                current_order['sl'] = max(current_order['sl'], trail_sl)

            stoploss_hit = candle_data['low'] <= current_order['sl']
            if stoploss_hit:
                append_exit(
                    current_order,
                    datetimex,
                    current_order['sl'],
                    current_order['remaining_lots'],
                    "buy_stop_loss_hit",
                    highest_price,
                    lowest_price,
                )
                current_order = empty_order()
                continue

            if its_friday:
                append_exit(
                    current_order,
                    datetimex,
                    candle_data['close'],
                    current_order['remaining_lots'],
                    "friday_hits",
                    highest_price,
                    lowest_price,
                )
                current_order = empty_order()
                continue

        if sold:
            if not current_order['t1_done'] and candle_data['low'] <= current_order['t1_price']:
                append_exit(
                    current_order,
                    datetimex,
                    current_order['t1_price'],
                    1,
                    "sell_t1_hit",
                    highest_price,
                    lowest_price,
                )
                current_order['t1_done'] = True
                current_order['sl'] = min(current_order['sl'], entry_price)

            if not current_order['t2_done'] and candle_data['low'] <= current_order['t2_price']:
                append_exit(
                    current_order,
                    datetimex,
                    current_order['t2_price'],
                    1,
                    "sell_t2_hit",
                    highest_price,
                    lowest_price,
                )
                current_order['t2_done'] = True
                current_order['sl'] = min(current_order['sl'], current_order['t1_price'])

            if current_order['remaining_lots'] == 1:
                trail_sl = round(candle_data['close'] + trail_amount, 2)
                current_order['sl'] = min(current_order['sl'], trail_sl)

            stoploss_hit = candle_data['high'] >= current_order['sl']
            if stoploss_hit:
                append_exit(
                    current_order,
                    datetimex,
                    current_order['sl'],
                    current_order['remaining_lots'],
                    "sell_stop_loss_hit",
                    highest_price,
                    lowest_price,
                )
                current_order = empty_order()
                continue

            if its_friday:
                append_exit(
                    current_order,
                    datetimex,
                    candle_data['close'],
                    current_order['remaining_lots'],
                    "friday_hits",
                    highest_price,
                    lowest_price,
                )
                current_order = empty_order()
                continue



print()
final_result = pd.DataFrame(final_result)
# final_result.to_csv('Results/5years_st_ema_mtf.csv', index=False)
final_result.to_csv('Results/3may_3_targets.csv', index=False)
