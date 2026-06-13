"""
NIFTY Multi-Target EMA Crossover Backtest

Port of 3may_3_targets.py using OpenAlgo API and openalgo.ta instead of
tradehull_backtesting_support. Strategy logic lives in backtesting/nifty_strategy.py.

Data sources (DATA_SOURCE env):
  - openalgo (default): client.history() for NIFTY 15m, daily, and INDIAVIX
  - csv: local CSV files (INTRADAY_CSV, DAILY_CSV, VIX_CSV)
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from openalgo import api, ta
from rich import print

SCRIPT_DIR = Path(__file__).resolve().parent
BACKTESTING_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKTESTING_DIR))

from nifty_strategy import NIFTYStrategy, OrderSide, StrategyParams  # noqa: E402

load_dotenv(find_dotenv(), override=False)

DATA_SOURCE = os.getenv("DATA_SOURCE", "openalgo").strip().lower()
API_KEY = os.getenv("OPENALGO_API_KEY", os.getenv("API_KEY", ""))
API_HOST = os.getenv("OPENALGO_HOST") or os.getenv("HOST_SERVER", "http://127.0.0.1:5000")

SYMBOL = os.getenv("SYMBOL", "NIFTY")
EXCHANGE = os.getenv("EXCHANGE", "NSE_INDEX")
INTERVAL = os.getenv("INTERVAL", "15m")
DAILY_INTERVAL = os.getenv("DAILY_INTERVAL", "D")
HISTORY_SOURCE = os.getenv("HISTORY_SOURCE", "db")
START_DATE = os.getenv("START_DATE", (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d"))
END_DATE = os.getenv("END_DATE", datetime.now().strftime("%Y-%m-%d"))

VIX_SYMBOL = os.getenv("VIX_SYMBOL", "INDIAVIX")
VIX_EXCHANGE = os.getenv("VIX_EXCHANGE", "NSE_INDEX")
VIX_FILTER_ENABLED = os.getenv("VIX_FILTER_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}

INTRADAY_CSV = os.getenv("INTRADAY_CSV", "Historical Data/NIFTY 15 mins 2021-26v2.csv")
DAILY_CSV = os.getenv("DAILY_CSV", "Historical Data/NIFTY 1D 2021-26.csv")
VIX_CSV = os.getenv("VIX_CSV", "Historical Data/VIX 22-04-2021 to 2026.csv")
RESULT_CSV = os.getenv("RESULT_CSV", str(SCRIPT_DIR / "NIFTY_ema_crossover_backtest.csv"))


def create_client() -> api:
    if not API_KEY:
        raise RuntimeError(
            "OPENALGO_API_KEY is not set. Add it to .env or set DATA_SOURCE=csv for local CSV files."
        )
    return api(api_key=API_KEY, host=API_HOST)


def _history_error_message(response: dict[str, Any], label: str) -> str:
    message = response.get("message", "Unknown API error")
    error_type = response.get("error_type", "")

    hints: list[str] = []
    if error_type == "connection_error":
        hints.append(f"Start OpenAlgo at {API_HOST} (e.g. `uv run app.py`).")
    elif "local database" in message.lower() or error_type == "no_data":
        hints.append("Download NIFTY data in Historify, or set DATA_SOURCE=csv with local CSV paths.")
    elif error_type == "api_error" and "apikey" in message.lower():
        hints.append("Set OPENALGO_API_KEY in .env from the OpenAlgo API key page.")

    hint_text = f" {' '.join(hints)}" if hints else ""
    return f"Failed to fetch {label} data: {message}.{hint_text}"


def normalize_history_frame(response: Any, label: str) -> pd.DataFrame:
    if isinstance(response, pd.DataFrame):
        df = response.copy()
    elif isinstance(response, dict):
        if response.get("status") == "error":
            raise RuntimeError(_history_error_message(response, label))
        data = response.get("data")
        if data is None:
            raise RuntimeError(
                f"Failed to fetch {label} data: unexpected response keys {sorted(response)}"
            )
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame(response)

    if df.empty:
        raise ValueError(f"No historical data returned for {label}")

    df.columns = [str(column).strip().lower() for column in df.columns]
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(sorted(missing))}")

    for column in required | {"volume"}:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.dropna(subset=["open", "high", "low", "close"])


def load_csv_frame(path: str, label: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"{label} CSV not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    df.columns = [str(column).strip().lower() for column in df.columns]

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    else:
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col])
        df = df.set_index(first_col)

    df = df.sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    rename_map = {"close ": "close", "open ": "open", "high ": "high", "low ": "low"}
    df = df.rename(columns={key: value for key, value in rename_map.items() if key in df.columns})

    if label == "vix" and "close" not in df.columns:
        raise ValueError(f"{label} CSV must include a close column")

    if label == "vix":
        return df

    return normalize_history_frame(df.reset_index().rename(columns={"index": "timestamp"}), label)


def fetch_history(
    client: api,
    symbol: str,
    exchange: str,
    interval: str,
    start_date: str,
    end_date: str,
    label: str,
) -> pd.DataFrame:
    print(f"Fetching {label}: {symbol} {exchange} {interval} {start_date} to {end_date}")
    response = client.history(
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        source=HISTORY_SOURCE,
    )
    df = normalize_history_frame(response, label)
    print(f"{label}: {len(df)} candles from {df.index.min()} to {df.index.max()}")
    return df


def add_atr(df: pd.DataFrame, period: int) -> pd.DataFrame:
    df = df.copy()
    try:
        df["atr"] = pd.Series(ta.atr(df["high"], df["low"], df["close"], period=period), index=df.index)
    except Exception:
        previous_close = df["close"].shift(1)
        true_range = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - previous_close).abs(),
                (df["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr"] = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return df


def add_atr_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Match tradehull_backtesting_support.atr_regime quantile buckets."""
    df = df.copy()
    high = df["atr"].quantile(0.75)
    low = df["atr"].quantile(0.25)
    df["regime"] = np.where(
        df["atr"] > high,
        "High ATR",
        np.where(df["atr"] < low, "Low ATR", "Medium ATR"),
    )
    return df


def add_ema_crossover(df: pd.DataFrame, short_period: int, long_period: int) -> pd.DataFrame:
    df = df.copy()
    df["ema_short"] = pd.Series(ta.ema(df["close"], short_period), index=df.index)
    df["ema_long"] = pd.Series(ta.ema(df["close"], long_period), index=df.index)
    return df


def normalize_supertrend_direction(direction: pd.Series) -> pd.Series:
    direction = pd.Series(direction).copy()
    if pd.api.types.is_bool_dtype(direction):
        return direction.map({True: "up", False: "down"})
    if pd.api.types.is_numeric_dtype(direction):
        return direction.map(lambda value: "up" if value < 0 else "down")
    return direction.astype(str).str.lower().map(
        {
            "-1": "up",
            "true": "up",
            "up": "up",
            "bullish": "up",
            "1": "down",
            "false": "down",
            "down": "down",
            "bearish": "down",
        }
    )


def add_supertrend(df: pd.DataFrame, atr_period: int, atr_multiplier: float) -> pd.DataFrame:
    df = df.copy()
    st_value, st_direction = ta.supertrend(
        df["high"],
        df["low"],
        df["close"],
        period=atr_period,
        multiplier=atr_multiplier,
    )
    st_col = f"STX_{atr_period}_{atr_multiplier}"
    df[st_col] = normalize_supertrend_direction(pd.Series(st_direction, index=df.index))
    df[f"ST_{atr_period}_{atr_multiplier}"] = pd.Series(st_value, index=df.index)
    return df


def index_loc_to_pos(loc: int | slice | np.ndarray | list) -> int:
    if isinstance(loc, slice):
        return loc.start or 0
    if isinstance(loc, np.ndarray) and loc.dtype == bool:
        return int(np.flatnonzero(loc)[0])
    if isinstance(loc, (list, np.ndarray)):
        return int(loc[0])
    return int(loc)


def load_data(params: StrategyParams) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    if DATA_SOURCE == "csv":
        df = load_csv_frame(INTRADAY_CSV, "intraday")
        hr_df = load_csv_frame(DAILY_CSV, "daily")
        vix_data = load_csv_frame(VIX_CSV, "vix") if VIX_FILTER_ENABLED else None
        return df, hr_df, vix_data

    client = create_client()
    df = fetch_history(client, SYMBOL, EXCHANGE, INTERVAL, START_DATE, END_DATE, "intraday")
    hr_df = fetch_history(client, SYMBOL, EXCHANGE, DAILY_INTERVAL, START_DATE, END_DATE, "daily")

    vix_data = None
    if VIX_FILTER_ENABLED and params.vix_max is not None:
        try:
            vix_data = fetch_history(
                client, VIX_SYMBOL, VIX_EXCHANGE, DAILY_INTERVAL, START_DATE, END_DATE, "India VIX"
            )
        except Exception as exc:
            print(f"[yellow]Warning: VIX data unavailable, continuing without VIX filter: {exc}[/yellow]")

    return df, hr_df, vix_data


def calculate_indicators(
    df: pd.DataFrame,
    hr_df: pd.DataFrame,
    params: StrategyParams,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = add_supertrend(df, params.supertrend_period, params.supertrend_multiplier)
    hr_df = add_supertrend(hr_df, params.supertrend_period, params.supertrend_multiplier)
    df = add_ema_crossover(df, 7, 21)
    df = add_atr(df, 14)
    df = add_atr_regime(df)
    return df, hr_df


def backtest(
    df: pd.DataFrame,
    hr_df: pd.DataFrame,
    vix_data: pd.DataFrame | None,
    params: StrategyParams,
    verbose: bool = False,
) -> pd.DataFrame:
    strategy = NIFTYStrategy(params)
    st_col = f"STX_{params.supertrend_period}_{params.supertrend_multiplier}"

    print(f"[bold cyan]Starting backtest with {len(df)} candles[/bold cyan]")

    for datetimex, candle_data in df.iterrows():
        candle_date = datetimex.strftime("%Y-%m-%d")
        candle_time = datetimex.strftime("%H:%M")
        weekday = datetimex.weekday()

        if weekday in [5, 6] or candle_time < "09:15" or candle_time > "15:30":
            continue

        current_idx = df.index.get_loc(datetimex)
        pos = -1
        vix_pos = -1
        prev_hr_st_col = None
        vix_close = None

        if current_idx > 0:
            hr_ts = hr_df.index.asof(pd.Timestamp(candle_date))
            if pd.notna(hr_ts):
                hr_loc = hr_df.index.get_loc(hr_ts)
                pos = index_loc_to_pos(hr_loc)

            if vix_data is not None and params.vix_max is not None:
                vix_ts = vix_data.index.asof(pd.Timestamp(candle_date))
                if pd.notna(vix_ts):
                    vix_loc = vix_data.index.get_loc(vix_ts)
                    vix_pos = index_loc_to_pos(vix_loc)

            if pos > 0:
                prev_hr_st_col = hr_df.iloc[pos - 1][st_col]

            if vix_pos > 0:
                vix_close = float(vix_data.iloc[vix_pos - 1]["close"])

        close_price = float(candle_data["close"])
        high_price = float(candle_data["high"])
        low_price = float(candle_data["low"])
        regime = candle_data.get("regime", "Low ATR")
        ema_short = float(candle_data.get("ema_short", 0))
        ema_long = float(candle_data.get("ema_long", 0))

        prev_ema_short = 0.0
        prev_ema_long = 0.0
        if current_idx > 0:
            prev_candle = df.iloc[current_idx - 1]
            prev_ema_short = float(prev_candle.get("ema_short", 0))
            prev_ema_long = float(prev_candle.get("ema_long", 0))

        if strategy.current_order is None:
            buy_cond, sell_cond = strategy.check_entry_conditions(
                candle_time=candle_time,
                candle_date=candle_date,
                weekday=weekday,
                ema_short=ema_short,
                ema_long=ema_long,
                prev_ema_short=prev_ema_short,
                prev_ema_long=prev_ema_long,
                regime=regime,
                prev_hr_st_col=prev_hr_st_col,
                vix_close=vix_close,
            )

            if buy_cond:
                strategy.current_order = strategy.build_order(datetimex, close_price, OrderSide.BUY)
                if verbose:
                    print(f"[green]BUY[/green] entry at {datetimex} @ {close_price}")
                continue

            if sell_cond:
                strategy.current_order = strategy.build_order(datetimex, close_price, OrderSide.SELL)
                if verbose:
                    print(f"[red]SELL[/red] entry at {datetimex} @ {close_price}")
                continue

        if strategy.current_order is not None:
            highest = float(df.loc[strategy.current_order.entry_idx : datetimex, "high"].max())
            lowest = float(df.loc[strategy.current_order.entry_idx : datetimex, "low"].min())

            exit_result = strategy.check_exit_conditions(
                candle_time=candle_time,
                candle_date=candle_date,
                weekday=weekday,
                high=high_price,
                low=low_price,
                close=close_price,
                highest_price=highest,
                lowest_price=lowest,
            )

            if exit_result:
                remark, exit_price, exit_lots = exit_result
                strategy.record_exit(
                    strategy.current_order,
                    datetimex,
                    exit_price,
                    exit_lots,
                    remark,
                    highest,
                    lowest,
                )

                if verbose:
                    side = "BUY" if strategy.current_order.buy_sell == OrderSide.BUY else "SELL"
                    print(f"[yellow]{side} EXIT[/yellow]: {remark} @ {exit_price}")

                if strategy.current_order.remaining_lots <= 0:
                    strategy.reset_order()

    print(f"[bold cyan]Backtest complete. Total exits: {len(strategy.exit_history)}[/bold cyan]")
    return strategy.get_exit_history_df()


def print_summary(results_df: pd.DataFrame) -> None:
    if len(results_df) == 0:
        print("[red]No trades executed[/red]")
        return

    trades = results_df.groupby(["date", "entry_time", "buy_sell"], dropna=False)["pnl"].sum()
    total_pnl = results_df["pnl"].sum()
    win_count = int((trades > 0).sum())
    loss_count = int((trades <= 0).sum())
    avg_win = trades[trades > 0].mean() if win_count > 0 else 0
    avg_loss = trades[trades <= 0].mean() if loss_count > 0 else 0

    print("\n[bold cyan]=== BACKTEST SUMMARY ===[/bold cyan]")
    print(f"Exit rows: {len(results_df)}")
    print(f"Completed trades: {len(trades)}")
    print(f"Total P&L: [bold]{total_pnl:.2f}[/bold]")
    print(f"Wins: {win_count} | Losses: {loss_count}")
    print(f"Win Rate: {(win_count / len(trades) * 100):.2f}%")
    print(f"Avg Win: {avg_win:.2f} | Avg Loss: {avg_loss:.2f}")
    if loss_count > 0 and avg_loss != 0:
        print(f"Profit Factor: {abs(avg_win * win_count / (avg_loss * loss_count)):.2f}")
    print("\nExit reason breakdown")
    print(results_df.groupby("remark")["pnl"].agg(["count", "sum"]).to_string())


if __name__ == "__main__":
    params = StrategyParams(
        supertrend_period=10,
        supertrend_multiplier=2,
        entry_start_time="09:15",
        entry_end_time="15:30",
        require_daily_trend_alignment=True,
        skip_entry_weekdays={4},
        vix_max=18,
        event_risk_dates=set(),
        lot_qty=65,
        initial_lots=3,
    )

    df, hr_df, vix_data = load_data(params)
    df, hr_df = calculate_indicators(df, hr_df, params)
    results = backtest(df, hr_df, vix_data, params, verbose=False)

    result_path = Path(RESULT_CSV)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(result_path, index=False)
    print_summary(results)
    print(f"\nResults saved to: {result_path.resolve()}")
