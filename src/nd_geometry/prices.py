"""Share price reference from Yahoo Finance's public chart endpoint (no key needed)."""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, datetime, timezone

from .filings import Fetch, cached_get, http_get

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range}&interval={interval}&events=splits"
BROWSER_UA = "Mozilla/5.0"


@dataclass(frozen=True)
class PriceHistory:
    symbol: str
    currency: str
    last_price: float
    last_date: date
    dates: tuple[date, ...]
    closes: tuple[float, ...]  # split-adjusted to today's share count
    splits: tuple[tuple[date, float], ...]  # (date, new shares per old share)

    def price_on(self, when: date) -> float | None:
        """Close on the last trading day on or before `when`."""
        i = bisect_right(self.dates, when) - 1
        return self.closes[i] if i >= 0 else None

    def split_factor_after(self, when: date) -> float:
        """Multiply a share count reported on `when` by this to express it in today's shares."""
        factor = 1.0
        for day, ratio in self.splits:
            if day > when:
                factor *= ratio
        return factor


def _utc_date(stamp: int) -> date:
    return datetime.fromtimestamp(stamp, tz=timezone.utc).date()


def parse_chart(payload: bytes | str, symbol: str = "") -> PriceHistory:
    result = json.loads(payload)["chart"]["result"][0]
    meta = result["meta"]
    stamps = result.get("timestamp") or []
    closes = result["indicators"]["quote"][0]["close"]
    pairs = [(_utc_date(t), float(c)) for t, c in zip(stamps, closes) if c is not None]
    splits = tuple(sorted((_utc_date(s["date"]), s["numerator"] / s["denominator"])
                          for s in (result.get("events", {}).get("splits") or {}).values()))
    last = float(meta.get("regularMarketPrice", pairs[-1][1]))
    last_day = _utc_date(meta["regularMarketTime"]) if "regularMarketTime" in meta else pairs[-1][0]
    return PriceHistory(symbol or meta.get("symbol", ""), meta.get("currency", "USD"), last, last_day,
                        tuple(d for d, _ in pairs), tuple(c for _, c in pairs), splits)


def fetch_prices(symbol: str, range_: str = "max", interval: str = "1d", fetch: Fetch | None = None,
                 cache_dir=None) -> PriceHistory:
    getter = fetch or (lambda url: http_get(url, user_agent=BROWSER_UA))
    url = CHART_URL.format(symbol=symbol.upper().replace(".", "-"), range=range_, interval=interval)
    return parse_chart(cached_get(url, getter, cache_dir, 6), symbol.upper())