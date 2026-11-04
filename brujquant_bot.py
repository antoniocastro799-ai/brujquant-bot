import os
import sys
import time
import json
import sqlite3
import tempfile
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError:
    ccxt = None

# Configure logging with cross-platform temp directory
_temp_dir = tempfile.gettempdir()
_log_file = os.path.join(_temp_dir, "brujquant_binance.log")
_db_default_path = os.path.join(_temp_dir, "brujquant_binance.sqlite")  # BUG FIX 1: variable name was mangled

logger = logging.getLogger("BrujQuantStandalone")
logger.setLevel(logging.INFO)
_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setFormatter(_formatter)
logger.addHandler(_stdout_handler)

try:
    _file_handler = RotatingFileHandler(_log_file, maxBytes=5 * 1024 * 1024, backupCount=3)  # BUG FIX 2: missing variable name
    _file_handler.setFormatter(_formatter)
    logger.addHandler(_file_handler)
except OSError:
    pass


@dataclass(frozen=True)
class EntrySignal:
    strategy_id: str
    strategy_name: str
    symbol: str
    timeframe: str
    candle_timestamp: int
    side: str
    entry_price: float
    raw_stop_price: float
    confluence_score: int
    confluence_reasons: List[str]
    is_micro_dca: bool = False
    dca_grid: Optional[List[float]] = None


@dataclass(frozen=True)
class BracketPlan:
    entry: EntrySignal
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_distance_pct: float
    reward_distance_pct: float
    rr_ratio: float


@dataclass(frozen=True)
class TimeMachineVerdict:
    simulated_runs: int
    prob_target_first: float
    prob_stop_first: float
    expected_value_r: float
    passed: bool
    verdict_reason: str


@dataclass(frozen=True)
class PositionBudget:
    phase: str
    leverage: int
    notional_usdt: float
    contracts: float
    risk_usdt: float
    expected_reward_usdt: float


@dataclass(frozen=True)
class RiskVerdict:
    passed: bool
    reason: str


class RiskGuardianDB:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.getenv("BRUJQUANT_DB_PATH", _db_default_path)
        self.conn = sqlite3.connect(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    opened_at TEXT,
                    closed_at TEXT,
                    symbol TEXT,
                    strategy_id TEXT,
                    side TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    notional_usdt REAL,
                    pnl_usdt REAL,
                    fee_usdt REAL,
                    balance_after REAL,
                    status TEXT
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_candles (
                    symbol TEXT,
                    timeframe TEXT,
                    candle_ts INTEGER,
                    strategy_id TEXT,
                    created_at TEXT,
                    PRIMARY KEY (symbol, timeframe, candle_ts)
                )
            """)

    def is_candle_processed(self, symbol: str, timeframe: str, candle_ts: int) -> bool:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT 1 FROM processed_candles WHERE symbol = ? AND timeframe = ? AND candle_ts = ?",
            (symbol, timeframe, candle_ts),
        )
        return cur.fetchone() is not None

    def mark_candle_processed(self, symbol: str, timeframe: str, candle_ts: int, strategy_id: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO processed_candles (symbol, timeframe, candle_ts, strategy_id, created_at) VALUES (?, ?, ?, ?, ?)",
                (symbol, timeframe, candle_ts, strategy_id, datetime.now(timezone.utc).isoformat()),
            )

    def check_preflight_risk(self, symbol: str, timeframe: str, candle_ts: int,
                             rr_ratio: float, now_utc: Optional[datetime] = None) -> RiskVerdict:
        now_utc = now_utc or datetime.now(timezone.utc)
        weekday = now_utc.weekday()
        if weekday == 5 or (weekday == 6 and now_utc.hour < 22) or (weekday == 4 and now_utc.hour >= 22):
            return RiskVerdict(passed=False, reason="WEEKEND_LOW_LIQUIDITY_PAUSED")

        if self.is_candle_processed(symbol, timeframe, candle_ts):
            return RiskVerdict(passed=False, reason="DUPLICATE_CANDLE_IDEMPOTENCY_SKIP")

        if rr_ratio < 1.8:
            return RiskVerdict(passed=False, reason=f"RR_TOO_LOW ({rr_ratio:.2f} < 1.80)")

        cur = self.conn.cursor()
        cur.execute("SELECT pnl_usdt, closed_at FROM trades WHERE status = 'CLOSED' ORDER BY id DESC LIMIT 2")
        rows = cur.fetchall()
        if len(rows) == 2 and all(r[0] < 0 for r in rows):
            elapsed_hours = (now_utc - datetime.fromisoformat(rows[0][1])).total_seconds() / 3600.0
            if elapsed_hours < 48.0:
                remaining_hours = round(48.0 - elapsed_hours, 1)
                return RiskVerdict(passed=False, reason=f"COOLDOWN_CIRCUIT_BREAKER ({remaining_hours}h remaining)")

        return RiskVerdict(passed=True, reason="PASS_ALL_PREFLIGHT_CHECKS")

    def record_trade(self, symbol: str, strategy_id: str, side: str, entry: float,
                     exit_p: float, notional: float, pnl: float, fee: float, balance_after: float) -> None:
        now_str = datetime.now(timezone.utc).isoformat()
        with self.conn:
            self.conn.execute("""
                INSERT INTO trades (opened_at, closed_at, symbol, strategy_id, side,
                                    entry_price, exit_price, notional_usdt, pnl_usdt,
                                    fee_usdt, balance_after, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'CLOSED')
            """, (now_str, now_str, symbol, strategy_id, side, entry, exit_p, notional, pnl, fee, balance_after))


class DayTraderEntryTimer:
    @staticmethod
    def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
        df["rsi"] = 100.0 - (100.0 / (1.0 + (avg_gain / (avg_loss + 1e-9))))

        for period in (21, 50, 100, 200):
            df[f"sma{period}"] = df["close"].rolling(period).mean()

        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        df["vol_ma20"] = df["volume"].rolling(20).mean()
        return df

    @staticmethod
    def find_local_pivots(df_closed: pd.DataFrame, lookback: int = 80) -> Tuple[List[Tuple[int, float, float]], List[Tuple[int, float, float]]]:
        window = df_closed.tail(lookback).reset_index(drop=True)
        closes = window["close"].values
        rsis = window["rsi"].values
        low_pivots, high_pivots = [], []

        for i in range(2, len(window) - 2):
            if closes[i] <= min(closes[i - 2:i + 3]):
                low_pivots.append((i, float(closes[i]), float(rsis[i])))
            if closes[i] >= max(closes[i - 2:i + 3]):
                high_pivots.append((i, float(closes[i]), float(rsis[i])))

        return low_pivots, high_pivots

    @classmethod
    def detect_three_touch_levels(cls, df_closed: pd.DataFrame, lookback: int = 90, tolerance_pct: float = 0.0035) -> Dict[str, List[float]]:
        low_pivots, high_pivots = cls.find_local_pivots(df_closed, lookback=lookback)

        def cluster(pivots: List[Tuple[int, float, float]]) -> List[float]:
            prices = [p[1] for p in pivots]
            confirmed = []
            for p in prices:
                touches = [x for x in prices if abs(x - p) / p <= tolerance_pct]
                if len(touches) >= 3:
                    avg_lvl = float(np.mean(touches))
                    if not any(abs(avg_lvl - c) / c <= tolerance_pct for c in confirmed):
                        confirmed.append(round(avg_lvl, 4))
            return confirmed

        return {"supports": cluster(low_pivots), "resistances": cluster(high_pivots)}

    @classmethod
    def scan_for_entry(cls, df: pd.DataFrame, balance_usdt: float,
                       symbol: str, timeframe: str) -> Optional[EntrySignal]:
        df = cls.compute_indicators(df)
        if len(df) < 105:
            return None

        df_closed = df.iloc[:-1].copy()
        closed_candle = df_closed.iloc[-1]
        candle_ts = int(closed_candle["timestamp"]) if "timestamp" in closed_candle else int(time.time())

        price = float(closed_candle["close"])
        rsi = float(closed_candle["rsi"])
        macd_val = float(closed_candle["macd"])
        macd_hist = float(closed_candle["macd_hist"])
        prev_macd_hist = float(df_closed.iloc[-2]["macd_hist"])

        low_pivots, high_pivots = cls.find_local_pivots(df_closed, lookback=80)
        levels = cls.detect_three_touch_levels(df_closed, lookback=90)

        if len(low_pivots) >= 2:
            _, p1, r1 = low_pivots[-2]
            _, p2, r2 = low_pivots[-1]
            reg_bull = (p2 < p1 * 0.999) and (r2 > r1 + 2.0) and (r1 < 36.0)
            hid_bull = (p2 > p1 * 1.001) and (r2 < r1 - 2.0) and (r2 < 40.0)
            micro_res = float(df_closed["close"].iloc[-6:-1].max())

            if (reg_bull or hid_bull) and price > micro_res and macd_val < 0 and macd_hist >= prev_macd_hist:
                sl_raw = min(p2 * 0.992, price * 0.990)
                div_name = "Regular Alcista" if reg_bull else "Oculta Alcista"
                return EntrySignal(
                    strategy_id="STRAT_02",
                    strategy_name=f"Divergencia RSI ({div_name})",
                    symbol=symbol,
                    timeframe=timeframe,
                    candle_timestamp=candle_ts,
                    side="buy",
                    entry_price=round(price, 4),
                    raw_stop_price=round(sl_raw, 4),
                    confluence_score=3,
                    confluence_reasons=[
                        f"RSI {div_name} ({r1:.1f} -> {r2:.1f})",
                        f"Micro-Breakout ({micro_res:.4f})",
                        "MACD < 0 ascending",
                    ],
                )

        if len(high_pivots) >= 2:
            _, p1, r1 = high_pivots[-2]
            _, p2, r2 = high_pivots[-1]
            reg_bear = (p2 > p1 * 1.001) and (r2 < r1 - 2.0) and (r1 > 64.0)
            hid_bear = (p2 < p1 * 0.999) and (r2 > r1 + 2.0) and (r2 > 60.0)
            micro_sup = float(df_closed["close"].iloc[-6:-1].min())

            if (reg_bear or hid_bear) and price < micro_sup and macd_val > 0 and macd_hist <= prev_macd_hist:
                sl_raw = max(p2 * 1.008, price * 1.010)
                div_name = "Regular Bajista" if reg_bear else "Oculta Bajista"
                return EntrySignal(
                    strategy_id="STRAT_02",
                    strategy_name=f"Divergencia RSI ({div_name})",
                    symbol=symbol,
                    timeframe=timeframe,
                    candle_timestamp=candle_ts,
                    side="sell",
                    entry_price=round(price, 4),
                    raw_stop_price=round(sl_raw, 4),
                    confluence_score=3,
                    confluence_reasons=[
                        f"RSI {div_name} ({r1:.1f} -> {r2:.1f})",
                        f"Micro-Breakdown ({micro_sup:.4f})",
                        "MACD > 0 descending",
                    ],
                )

        recent = df_closed.iloc[-80:-1]
        swing_high = float(recent["high"].max())
        swing_low = float(recent["low"].min())
        fib_range = swing_high - swing_low

        if fib_range > 0:
            gp_bull_top = swing_high - fib_range * 0.618
            gp_bull_bot = swing_high - fib_range * 0.660
            fib_bull_inv = swing_high - fib_range * 0.786
            trend_up = price > float(closed_candle["sma200"]) or float(closed_candle["sma50"]) > float(closed_candle["sma200"])

            if (gp_bull_bot <= price <= gp_bull_top * 1.002) and trend_up and rsi < 44.0:
                sl_raw = min(fib_bull_inv * 0.998, price * 0.989)
                return EntrySignal(
                    strategy_id="STRAT_03",
                    strategy_name="Golden Pocket Fibonacci (LONG)",
                    symbol=symbol,
                    timeframe=timeframe,
                    candle_timestamp=candle_ts,
                    side="buy",
                    entry_price=round(price, 4),
                    raw_stop_price=round(sl_raw, 4),
                    confluence_score=3,
                    confluence_reasons=[
                        f"Golden Pocket ({gp_bull_bot:.4f}-{gp_bull_top:.4f})",
                        f"Invalidation < 0.786 ({fib_bull_inv:.4f})",
                        f"SMA200 Bullish + RSI ({rsi:.1f})",
                    ],
                )

            gp_bear_bot = swing_low + fib_range * 0.618
            gp_bear_top = swing_low + fib_range * 0.660
            fib_bear_inv = swing_low + fib_range * 0.786
            trend_down = price < float(closed_candle["sma200"]) and float(closed_candle["sma50"]) < float(closed_candle["sma200"])

            if (gp_bear_bot * 0.998 <= price <= gp_bear_top) and trend_down and rsi > 56.0:
                sl_raw = max(fib_bear_inv * 1.002, price * 1.011)
                return EntrySignal(
                    strategy_id="STRAT_03",
                    strategy_name="Golden Pocket Fibonacci (SHORT)",
                    symbol=symbol,
                    timeframe=timeframe,
                    candle_timestamp=candle_ts,
                    side="sell",
                    entry_price=round(price, 4),
                    raw_stop_price=round(sl_raw, 4),
                    confluence_score=3,
                    confluence_reasons=[
                        f"Golden Pocket ({gp_bear_bot:.4f}-{gp_bear_top:.4f})",
                        f"Invalidation > 0.786 ({fib_bear_inv:.4f})",
                        f"SMA200 Bearish + RSI ({rsi:.1f})",
                    ],
                )

        for res_lvl in levels["resistances"]:
            was_above = float(df_closed["close"].iloc[-5:-1].max()) > res_lvl * 1.004
            retesting_support = abs(price - res_lvl) / res_lvl <= 0.0035 and price >= res_lvl * 0.998
            if was_above and retesting_support and float(closed_candle["volume"]) >= float(closed_candle["vol_ma20"]) * 0.9:
                return EntrySignal(
                    strategy_id="STRAT_01",
                    strategy_name="S/R Flip 3-Touches (LONG)",
                    symbol=symbol,
                    timeframe=timeframe,
                    candle_timestamp=candle_ts,
                    side="buy",
                    entry_price=round(res_lvl, 4),
                    raw_stop_price=round(res_lvl * 0.991, 4),
                    confluence_score=3,
                    confluence_reasons=[
                        f"3-Touch Support Flip ({res_lvl:.4f})",
                        "Post-Only Maker Limit Retest",
                        "Volume > 0.9x MA20",
                    ],
                )

        for sup_lvl in levels["supports"]:
            was_below = float(df_closed["close"].iloc[-5:-1].min()) < sup_lvl * 0.996
            retesting_resistance = abs(price - sup_lvl) / sup_lvl <= 0.0035 and price <= sup_lvl * 1.002
            if was_below and retesting_resistance and float(closed_candle["volume"]) >= float(closed_candle["vol_ma20"]) * 0.9:
                return EntrySignal(
                    strategy_id="STRAT_01",
                    strategy_name="S/R Flip 3-Touches (SHORT)",
                    symbol=symbol,
                    timeframe=timeframe,
                    candle_timestamp=candle_ts,
                    side="sell",
                    entry_price=round(sup_lvl, 4),
                    raw_stop_price=round(sup_lvl * 1.009, 4),
                    confluence_score=3,
                    confluence_reasons=[
                        f"3-Touch Resistance Flip ({sup_lvl:.4f})",
                        "Post-Only Maker Limit Retest",
                        "Volume > 0.9x MA20",
                    ],
                )

        if balance_usdt >= 12.0 and levels["resistances"] and levels["supports"]:
            top_res = max(levels["resistances"])
            if abs(price - top_res) / top_res <= 0.003 and rsi > 62.0:
                grid = [round(top_res, 4), round(top_res * 1.004, 4), round(top_res * 1.008, 4)]
                avg_entry = float(np.mean(grid))
                return EntrySignal(
                    strategy_id="STRAT_10",
                    strategy_name="Range Micro-DCA (Phase 2)",
                    symbol=symbol,
                    timeframe=timeframe,
                    candle_timestamp=candle_ts,
                    side="sell",
                    entry_price=round(avg_entry, 4),
                    raw_stop_price=round(avg_entry * 1.012, 4),
                    confluence_score=3,
                    confluence_reasons=[
                        f"Balance >= 12 USDT ({balance_usdt:.2f})",
                        f"DCA Grid {grid}",
                        "Resistance Range Scalp",
                    ],
                    is_micro_dca=True,
                    dca_grid=grid,
                )

        return None


class DayTraderExitPlanner:
    @staticmethod
    def adjust_psychological_price(target_price: float, side: str) -> float:
        for step in (1000.0, 100.0, 10.0, 1.0):
            if target_price >= step:
                nearest_round = round(target_price / step) * step
                if abs(target_price - nearest_round) / nearest_round <= 0.005:
                    return round(nearest_round * (0.9975 if side == "buy" else 1.0025), 4)
        return round(target_price, 4)

    @classmethod
    def plan_bracket_exit(cls, entry_sig: EntrySignal) -> Optional[BracketPlan]:
        entry_p = entry_sig.entry_price
        raw_sl = entry_sig.raw_stop_price

        if entry_sig.side == "buy":
            if raw_sl >= entry_p:
                return None
            risk_dist = entry_p - raw_sl
            if (risk_dist / entry_p) < 0.004:
                raw_sl = entry_p * 0.994
                risk_dist = entry_p - raw_sl
            tp1 = round(entry_p + risk_dist * 1.5, 4)
            tp2_raw = entry_p + risk_dist * (1.6 if entry_sig.is_micro_dca else 2.5)
            tp2 = cls.adjust_psychological_price(tp2_raw, "buy")
            reward_dist = tp2 - entry_p
        else:
            if raw_sl <= entry_p:
                return None
            risk_dist = raw_sl - entry_p
            if (risk_dist / entry_p) < 0.004:
                raw_sl = entry_p * 1.006
                risk_dist = raw_sl - entry_p
            tp1 = round(entry_p - risk_dist * 1.5, 4)
            tp2_raw = entry_p - risk_dist * (1.6 if entry_sig.is_micro_dca else 2.5)
            tp2 = cls.adjust_psychological_price(tp2_raw, "sell")
            reward_dist = entry_p - tp2

        rr_ratio = reward_dist / max(risk_dist, 1e-6)
        if rr_ratio < 1.5:
            return None

        return BracketPlan(
            entry=entry_sig,
            stop_loss=round(raw_sl, 4),
            take_profit_1=tp1,
            take_profit_2=tp2,
            risk_distance_pct=round(risk_dist / entry_p, 4),
            reward_distance_pct=round(reward_dist / entry_p, 4),
            rr_ratio=round(rr_ratio, 2),
        )


class DayTraderTimeMachine:
    @staticmethod
    def simulate_trade_odds(plan: BracketPlan, df: pd.DataFrame, n_paths: int = 1000) -> TimeMachineVerdict:
        returns = df["close"].pct_change().dropna().tail(80).values
        if len(returns) < 30:
            returns = np.random.normal(0, 0.003, 80)

        std_dev = float(np.std(returns)) or 0.003
        mean_ret = float(np.mean(returns))
        entry_p = plan.entry.entry_price
        sl_p = plan.stop_loss
        tp_p = plan.take_profit_2
        side = plan.entry.side

        drift = abs(mean_ret) * 0.5 if side == "buy" else -abs(mean_ret) * 0.5
        sim_noise = np.random.normal(drift, std_dev, size=(n_paths, 48))

        target_hits = 0
        stop_hits = 0

        for i in range(n_paths):
            current_p = entry_p
            outcome = 0
            for step in range(48):
                current_p *= (1.0 + sim_noise[i, step])
                if side == "buy":
                    if current_p >= tp_p:
                        outcome = 1
                        break
                    if current_p <= sl_p:
                        outcome = -1
                        break
                else:
                    if current_p <= tp_p:
                        outcome = 1
                        break
                    if current_p >= sl_p:
                        outcome = -1
                        break

            if outcome == 1:
                target_hits += 1
            elif outcome == -1:
                stop_hits += 1

        prob_target = target_hits / n_paths
        prob_stop = stop_hits / n_paths
        ev_r = (prob_target * plan.rr_ratio) - (prob_stop * 1.0)

        breakeven_prob = 1.0 / (1.0 + plan.rr_ratio)
        passed = (ev_r >= 0.05) and (prob_target > prob_stop)

        verdict_reason = f"EV={ev_r:.2f}R P(Target)={prob_target*100:.1f}% P(Stop)={prob_stop*100:.1f}% BE_Req={breakeven_prob*100:.1f}%"
        return TimeMachineVerdict(
            simulated_runs=n_paths,
            prob_target_first=round(prob_target, 3),
            prob_stop_first=round(prob_stop, 3),
            expected_value_r=round(ev_r, 2),
            passed=passed,
            verdict_reason=verdict_reason,
        )


class DayTraderBudgetManager:
    @staticmethod
    def size_position(balance_usdt: float, plan: BracketPlan) -> PositionBudget:
        if balance_usdt < 12.0:
            phase, risk_usdt, leverage, min_notional = "PHASE_1", max(0.25, balance_usdt * 0.05), 5, 5.5
        elif balance_usdt < 30.0:
            phase, risk_usdt, leverage, min_notional = "PHASE_2", balance_usdt * 0.04, 4, 12.0
        else:
            phase, risk_usdt, leverage, min_notional = "PHASE_3", balance_usdt * 0.03, 3, 20.0

        raw_notional = risk_usdt / max(plan.risk_distance_pct, 0.005)
        safe_max_notional = balance_usdt * leverage * 0.95
        notional_usdt = max(min_notional, min(raw_notional, safe_max_notional))
        contracts = notional_usdt / plan.entry.entry_price
        actual_risk = notional_usdt * plan.risk_distance_pct
        expected_reward = actual_risk * plan.rr_ratio

        return PositionBudget(
            phase=phase,
            leverage=leverage,
            notional_usdt=round(notional_usdt, 2),
            contracts=round(contracts, 4),
            risk_usdt=round(actual_risk, 2),
            expected_reward_usdt=round(expected_reward, 2),
        )


class DayTraderBrokerExecutor:
    MICRO_PAIRS = ("XRP/USDT:USDT", "DOGE/USDT:USDT", "SOL/USDT:USDT")

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.exchange = None
        if ccxt is not None and os.getenv("BINANCE_API_KEY"):
            self.exchange = ccxt.binanceusdm({
                "apiKey": os.getenv("BINANCE_API_KEY", ""),
                "secret": os.getenv("BINANCE_API_SECRET", ""),
                "enableRateLimit": True,
                "options": {"defaultType": "future", "adjustForTimeDifference": True},
            })

    def fetch_live_balance(self) -> float:
        if not self.exchange or self.dry_run:
            return 6.55
        try:
            bal_info = self.exchange.fetch_balance()
            return float(bal_info["USDT"]["free"])
        except Exception as e:
            logger.warning("fetch_balance_failed err=%s", e)
            return 6.55

    def fetch_live_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 120) -> Optional[pd.DataFrame]:
        if not self.exchange:
            return None
        try:
            raw_candles = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(raw_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            return df
        except Exception as e:
            logger.warning("fetch_ohlcv_failed symbol=%s err=%s", symbol, e)
            return None

    def dispatch_bracket_order(self, plan: BracketPlan, budget: PositionBudget,
                               tm: TimeMachineVerdict, guardian: RiskGuardianDB) -> Dict:
        signal = plan.entry
        symbol = signal.symbol

        if self.dry_run or self.exchange is None or not os.getenv("BINANCE_API_KEY"):
            guardian.mark_candle_processed(symbol, signal.timeframe, signal.candle_timestamp, signal.strategy_id)
            return {
                "mode": "DRY_RUN",
                "status": "ORDER_APPROVED_AND_SIMULATED",
                "symbol": symbol,
                "side": signal.side,
                "strategy": signal.strategy_name,
                "entry_price": signal.entry_price,
                "stop_loss": plan.stop_loss,
                "take_profit_1_breakeven": plan.take_profit_1,
                "take_profit_2_target": plan.take_profit_2,
                "rr_ratio": plan.rr_ratio,
                "budget": asdict(budget),
                "timemachine": asdict(tm),
            }

        for attempt in range(1, 4):
            try:
                self.exchange.load_markets()
                try:
                    self.exchange.set_margin_mode("ISOLATED", symbol)
                except Exception:
                    pass
                self.exchange.set_leverage(budget.leverage, symbol)

                qty = float(self.exchange.amount_to_precision(symbol, budget.contracts))
                entry_p = float(self.exchange.price_to_precision(symbol, signal.entry_price))
                sl_p = float(self.exchange.price_to_precision(symbol, plan.stop_loss))
                tp_p = float(self.exchange.price_to_precision(symbol, plan.take_profit_2))
                exit_side = "sell" if signal.side == "buy" else "buy"

                main_order = self.exchange.create_order(
                    symbol=symbol,
                    type="limit",
                    side=signal.side,
                    amount=qty,
                    price=entry_p,
                    params={"timeInForce": "GTX"},
                )
                sl_order = self.exchange.create_order(
                    symbol=symbol,
                    type="STOP_MARKET",
                    side=exit_side,
                    amount=qty,
                    params={"stopPrice": sl_p, "closePosition": True, "workingType": "MARK_PRICE"},
                )
                tp_order = self.exchange.create_order(
                    symbol=symbol,
                    type="TAKE_PROFIT_MARKET",
                    side=exit_side,
                    amount=qty,
                    params={"stopPrice": tp_p, "closePosition": True, "workingType": "MARK_PRICE"},
                )

                guardian.mark_candle_processed(symbol, signal.timeframe, signal.candle_timestamp, signal.strategy_id)
                return {
                    "mode": "LIVE_BINANCE",
                    "status": "BRACKET_DISPATCHED",
                    "main_order_id": main_order.get("id"),
                    "sl_order_id": sl_order.get("id"),
                    "tp_order_id": tp_order.get("id"),
                    "breakeven_trigger_price": plan.take_profit_1,
                    "budget": asdict(budget),
                }
            except Exception as err:
                logger.warning("retry=%d/3 error=%s", attempt, err)
                time.sleep(2 ** attempt)

        return {"status": "ERROR_MAX_RETRIES"}


class BrujQuantDayTraderAgent:
    def __init__(self, db_path: Optional[str] = None, dry_run: bool = True):
        self.guardian = RiskGuardianDB(db_path)
        self.broker = DayTraderBrokerExecutor(dry_run=dry_run)

    def evaluate_and_trade(self, df: pd.DataFrame, balance_usdt: float = 6.55,
                           symbol: str = "XRP/USDT:USDT", timeframe: str = "15m") -> Dict:
        entry_sig = DayTraderEntryTimer.scan_for_entry(df, balance_usdt, symbol, timeframe)
        if not entry_sig:
            return {"status": "WAITING_FOR_SETUP"}

        plan = DayTraderExitPlanner.plan_bracket_exit(entry_sig)
        if not plan:
            return {"status": "REJECTED_EXIT_PLAN"}

        preflight = self.guardian.check_preflight_risk(symbol, timeframe, entry_sig.candle_timestamp, plan.rr_ratio)
        if not preflight.passed:
            return {"status": "BLOCKED_BY_RISK_MANAGER", "reason": preflight.reason}

        tm_verdict = DayTraderTimeMachine.simulate_trade_odds(plan, df)
        if not tm_verdict.passed:
            return {"status": "BLOCKED_BY_TIME_MACHINE", "reason": tm_verdict.verdict_reason}

        budget = DayTraderBudgetManager.size_position(balance_usdt, plan)
        return self.broker.dispatch_bracket_order(plan, budget, tm_verdict, self.guardian)

    def run_live_cycle(self, symbol: str = "XRP/USDT:USDT", timeframe: str = "15m") -> Dict:
        bal = self.broker.fetch_live_balance()
        df = self.broker.fetch_live_ohlcv(symbol, timeframe=timeframe, limit=120)
        if df is None or len(df) < 105:
            return {"status": "FETCH_FAILED_OR_INSUFFICIENT_BARS"}
        return self.evaluate_and_trade(df, balance_usdt=bal, symbol=symbol, timeframe=timeframe)


def main():
    is_live = os.getenv("DRY_RUN", "true").lower() == "false"
    symbol = os.getenv("TRADING_SYMBOL", "XRP/USDT:USDT")
    timeframe = os.getenv("TRADING_TIMEFRAME", "15m")
    check_interval = int(os.getenv("CHECK_INTERVAL_SEC", "60"))

    logger.info("Starting BrujQuant DayTrader Bot. Mode=%s Symbol=%s Timeframe=%s",
                "LIVE_BINANCE" if is_live else "DRY_RUN", symbol, timeframe)

    agent = BrujQuantDayTraderAgent(dry_run=not is_live)

    # Initial probe cycle
    res = agent.run_live_cycle(symbol=symbol, timeframe=timeframe)
    logger.info("Probe cycle output: %s", json.dumps(res))

    if os.getenv("SINGLE_RUN", "false").lower() == "true":
        return

    logger.info("Daemon active. Listening every %d seconds on completed candles...", check_interval)
    while True:
        try:
            cycle_res = agent.run_live_cycle(symbol=symbol, timeframe=timeframe)
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            logger.info("Cycle @ %s: %s", now_str, json.dumps(cycle_res))
        except Exception as err:
            logger.error("Daemon exception: %s", err)
        time.sleep(check_interval)


if __name__ == "__main__":
    main()
