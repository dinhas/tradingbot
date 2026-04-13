import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class Labeler:
    def __init__(
        self,
        sl_mult: float = 2.0,
        tp_mult: float = 4.0,
        time_barrier: int = 35,
        stride: int = 10,
        warmup: int = 60,
        cusum_threshold_mult: float = 0.5,
        adx_trend_threshold: float = 20.0,
        use_dynamic_barriers: bool = True,
    ):
        """
        Dual-trade labeling: At each timestep, simulate both a BUY and SELL trade
        with the same SL/TP structure used in live execution.
        
        Args:
            sl_mult: Stop-loss distance in ATR multiples (applied symmetrically)
            tp_mult: Take-profit distance in ATR multiples (applied symmetrically)
            time_barrier: Maximum bars to hold before timeout
            stride: Step size between label samples
            warmup: Skip first N bars (rolling normalization unstable)
            cusum_threshold_mult: CUSUM event threshold multiplier on ATR
            adx_trend_threshold: ADX threshold below which direction is forced to Hold
            use_dynamic_barriers: Use dynamic realized-volatility barriers when True
        """
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult
        self.time_barrier = time_barrier
        self.stride = stride
        self.warmup = warmup
        self.cusum_threshold_mult = cusum_threshold_mult
        self.adx_trend_threshold = adx_trend_threshold
        self.use_dynamic_barriers = use_dynamic_barriers

    @staticmethod
    def _cusum_filter(close_prices: np.ndarray, atrs: np.ndarray, threshold_mult: float) -> List[int]:
        """Return event indices where CUSUM deviation exceeds ATR-scaled threshold."""
        if len(close_prices) < 2:
            return []

        events = []
        s_pos = 0.0
        s_neg = 0.0
        last_price = close_prices[0]

        for i in range(1, len(close_prices)):
            atr = atrs[i]
            if np.isnan(atr) or atr == 0:
                continue

            threshold = threshold_mult * atr
            diff = close_prices[i] - last_price
            s_pos = max(0.0, s_pos + diff)
            s_neg = min(0.0, s_neg + diff)

            if s_pos >= threshold or s_neg <= -threshold:
                events.append(i)
                s_pos = 0.0
                s_neg = 0.0
                last_price = close_prices[i]

        return events

    def label_data(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        """
        Dual-Trade Labeling:
        
        At each timestep, simulate:
          - BUY trade:  SL = entry - sl_mult*ATR,  TP = entry + tp_mult*ATR
          - SELL trade: SL = entry + sl_mult*ATR,  TP = entry - tp_mult*ATR
        
        Direction:
          +1 if BUY TP hits before BUY SL (within time_barrier)
          -1 if SELL TP hits before SELL SL (within time_barrier)
          0  if neither TP hits before its SL within the time_barrier
          If both win, take whichever TP hit first.
        
        Returns DataFrame with [direction, quality, meta, entry_time, exit_time, barrier_hit]
        """
        close_col = f"{asset}_close"
        atr_col = f"{asset}_atr_14"
        high_col = f"{asset}_high"
        low_col = f"{asset}_low"
        adx_col_candidates = [f"{asset}_adx_14", f"{asset}_adx14"]
        adx_col = next((col for col in adx_col_candidates if col in df.columns), None)

        if close_col not in df.columns or atr_col not in df.columns:
            raise ValueError(f"Required columns {close_col} or {atr_col} missing.")

        labels = []
        indices = []

        n = len(df)
        prices_close = df[close_col].values
        prices_high = df[high_col].values
        prices_low = df[low_col].values
        atrs = df[atr_col].values
        adx_values = df[adx_col].values if adx_col else None
        timestamps = df.index

        close_returns = pd.Series(prices_close, index=df.index).pct_change()
        realized_vol_20 = close_returns.rolling(window=20, min_periods=20).std().values
        annualized_price_vol = realized_vol_20 * prices_close * np.sqrt(288.0)
        barrier_scale = np.sqrt(self.time_barrier / 288.0)
        dynamic_base = annualized_price_vol * barrier_scale

        cusum_events = self._cusum_filter(prices_close, atrs, self.cusum_threshold_mult)
        candidate_indices = [
            idx for idx in cusum_events
            if self.warmup <= idx < (n - self.time_barrier)
        ]
        n_cusum_events = len(candidate_indices)

        if n_cusum_events < 1000:
            logger.warning(
                f"{asset}: only {n_cusum_events} CUSUM events (<1000). Falling back to stride={self.stride} sampling."
            )
            candidate_indices = list(range(self.warmup, n - self.time_barrier, self.stride))

        n_trending_regime = 0
        n_directional = 0

        for curr_idx in candidate_indices:
            entry_price = prices_close[curr_idx]
            atr = atrs[curr_idx]

            if np.isnan(atr) or atr == 0:
                continue

            # ── Define trade levels ──
            if self.use_dynamic_barriers:
                base_vol = dynamic_base[curr_idx]
                if np.isnan(base_vol) or base_vol <= 0:
                    continue
                tp_dist = self.tp_mult * base_vol
                sl_dist = self.sl_mult * base_vol
            else:
                tp_dist = self.tp_mult * atr
                sl_dist = self.sl_mult * atr

            buy_tp = entry_price + tp_dist   # Long TP (above)
            buy_sl = entry_price - sl_dist   # Long SL (below)
            sell_tp = entry_price - tp_dist   # Short TP (below)
            sell_sl = entry_price + sl_dist   # Short SL (above)

            # ── Look-ahead window ──
            window_highs = prices_high[curr_idx + 1 : curr_idx + self.time_barrier + 1]
            window_lows  = prices_low[curr_idx + 1 : curr_idx + self.time_barrier + 1]

            # ── BUY trade outcome ──
            # TP hit when high reaches buy_tp, SL hit when low reaches buy_sl
            buy_tp_bars = np.where(window_highs >= buy_tp)[0]
            buy_sl_bars = np.where(window_lows <= buy_sl)[0]
            buy_tp_bar = buy_tp_bars[0] if len(buy_tp_bars) > 0 else self.time_barrier + 1
            buy_sl_bar = buy_sl_bars[0] if len(buy_sl_bars) > 0 else self.time_barrier + 1
            buy_wins = (buy_tp_bar < buy_sl_bar) and (buy_tp_bar < self.time_barrier)

            # ── SELL trade outcome ──
            # TP hit when low reaches sell_tp, SL hit when high reaches sell_sl
            sell_tp_bars = np.where(window_lows <= sell_tp)[0]
            sell_sl_bars = np.where(window_highs >= sell_sl)[0]
            sell_tp_bar = sell_tp_bars[0] if len(sell_tp_bars) > 0 else self.time_barrier + 1
            sell_sl_bar = sell_sl_bars[0] if len(sell_sl_bars) > 0 else self.time_barrier + 1
            sell_wins = (sell_tp_bar < sell_sl_bar) and (sell_tp_bar < self.time_barrier)

            # ── Determine direction ──
            if buy_wins and sell_wins:
                # Both trades win — take whichever TP hit first
                direction = 1 if buy_tp_bar <= sell_tp_bar else -1
                exit_bar = min(buy_tp_bar, sell_tp_bar)
                barrier_hit = direction
            elif buy_wins:
                direction = 1
                exit_bar = buy_tp_bar
                barrier_hit = 1
            elif sell_wins:
                direction = -1
                exit_bar = sell_tp_bar
                barrier_hit = -1
            else:
                # No TP winner; infer directional move from which stop was hit first.
                # This keeps labels informative without admitting low-volatility noise.
                buy_sl_hit = buy_sl_bar < self.time_barrier
                sell_sl_hit = sell_sl_bar < self.time_barrier

                if buy_sl_hit and (not sell_sl_hit or buy_sl_bar <= sell_sl_bar):
                    direction = -1
                    exit_bar = buy_sl_bar
                    barrier_hit = -1
                elif sell_sl_hit and (not buy_sl_hit or sell_sl_bar < buy_sl_bar):
                    direction = 1
                    exit_bar = sell_sl_bar
                    barrier_hit = 1
                else:
                    direction = 0
                    exit_bar = self.time_barrier
                    barrier_hit = 0

            if adx_values is not None:
                current_adx = adx_values[curr_idx]
                if not np.isnan(current_adx):
                    if current_adx >= self.adx_trend_threshold:
                        n_trending_regime += 1
                    else:
                        n_skipped_ranging += 1
                        continue

            if adx_values is not None:
                current_adx = adx_values[curr_idx]
                if not np.isnan(current_adx):
                    if current_adx >= self.adx_trend_threshold:
                        n_trending_regime += 1
                    else:
                        direction = 0
                        barrier_hit = 0

            exit_idx = curr_idx + 1 + exit_bar
            exit_idx = min(exit_idx, n - 1)

            # ── Meta label ──
            meta = 1 if direction != 0 else 0
            if meta == 1:
                n_directional += 1

            # ── Quality score ──
            # Based on the winning trade's price action cleanliness
            slice_highs = prices_high[curr_idx + 1 : exit_idx + 1]
            slice_lows  = prices_low[curr_idx + 1 : exit_idx + 1]

            if slice_highs.size == 0 or slice_lows.size == 0:
                continue

            window_max = np.max(slice_highs)
            window_min = np.min(slice_lows)

            if direction == 1:  # Winning BUY trade
                mfe = window_max - entry_price   # How far price went in our favor
                mae = entry_price - window_min    # How far price went against us
            elif direction == -1:  # Winning SELL trade
                mfe = entry_price - window_min
                mae = window_max - entry_price
            else:  # No winner — measure best potential
                mfe_buy = window_max - entry_price
                mae_buy = entry_price - window_min
                mfe_sell = entry_price - window_min
                mae_sell = window_max - entry_price
                # Use whichever side was closer to winning
                if mfe_buy / (self.tp_mult * atr) > mfe_sell / (self.tp_mult * atr):
                    mfe, mae = mfe_buy, mae_buy
                else:
                    mfe, mae = mfe_sell, mae_sell

            raw_quality = (mfe - mae) / (atr + 1e-6)
            clipped_quality = np.clip(raw_quality, -2, 2)
            normalized_quality = (clipped_quality + 2) / 4  # Maps to [0, 1]

            labels.append({
                'direction': direction,
                'quality': normalized_quality,
                'meta': meta,
                'entry_time': timestamps[curr_idx],
                'exit_time': timestamps[exit_idx],
                'barrier_hit': barrier_hit
            })
            indices.append(timestamps[curr_idx])

        pct_dir = (100.0 * n_directional / len(labels)) if labels else 0.0
        pct_trend = (100.0 * n_trending_regime / len(labels)) if labels else 0.0
        logger.info(
            f"{asset}: {n_cusum_events} CUSUM events, "
            f"{n_directional} directional labels ({pct_dir:.1f}%), "
            f"{n_trending_regime} in trending regime ({pct_trend:.1f}%)"
        )

        return pd.DataFrame(labels, index=indices)

if __name__ == "__main__":
    # Test with dummy data or real data if available
    pass
