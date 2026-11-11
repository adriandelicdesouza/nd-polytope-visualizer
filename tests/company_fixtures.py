"""Synthetic SEC company facts for tests: a steady grower, nine fiscal years, no network."""

from __future__ import annotations

from datetime import date

FIRST_YEAR = 2016
LAST_YEAR = 2024


def _flow(year: int, value: float) -> dict:
    return {"start": f"{year}-01-01", "end": f"{year}-12-31", "val": value, "form": "10-K",
            "filed": f"{year + 1}-02-15", "fy": year, "fp": "FY"}


def _instant(year: int, value: float, form: str = "10-K") -> dict:
    return {"end": f"{year}-12-31", "val": value, "form": form, "filed": f"{year + 1}-02-15"}


def make_raw(growth: float = 0.08, margin: float = 0.20, tax: float = 0.25, capex: float = 0.05,
             da: float = 0.045, cfo_gap: float = 0.02, base: float = 1_000e6, growths: dict | None = None) -> dict:
    """Facts for revenue growing at `growth` (or at `growths[year]` where given), with ratios exactly as given."""
    rev = {FIRST_YEAR: base}
    for y in range(FIRST_YEAR + 1, LAST_YEAR + 1):
        rev[y] = rev[y - 1] * (1 + (growths or {}).get(y, growth))
    nopat = lambda y: rev[y] * margin * (1 - tax)
    tags = {
        "Revenues": [_flow(y, rev[y]) for y in rev],
        "OperatingIncomeLoss": [_flow(y, rev[y] * margin) for y in rev],
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest":
            [_flow(y, rev[y] * margin) for y in rev],
        "IncomeTaxExpenseBenefit": [_flow(y, rev[y] * margin * tax) for y in rev],
        "PaymentsToAcquirePropertyPlantAndEquipment": [_flow(y, rev[y] * capex) for y in rev],
        "DepreciationDepletionAndAmortization": [_flow(y, rev[y] * da) for y in rev],
        "NetCashProvidedByUsedInOperatingActivities":
            [_flow(y, nopat(y) + rev[y] * da - rev[y] * (cfo_gap + 0.10 * growth)) for y in rev],
        "CashAndCashEquivalentsAtCarryingValue": [_instant(LAST_YEAR, 300e6)],
        "ShortTermInvestments": [_instant(LAST_YEAR, 100e6)],
        "LongTermDebtNoncurrent": [_instant(LAST_YEAR, 900e6)],
        "ShortTermBorrowings": [_instant(LAST_YEAR, 100e6)],
    }
    facts = {t: {"units": {"USD": rows}} for t, rows in tags.items()}
    facts["WeightedAverageNumberOfDilutedSharesOutstanding"] = {
        "units": {"shares": [_flow(y, 100e6) for y in rev]}}
    dei = {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
        {"end": f"{LAST_YEAR + 1}-01-31", "val": 98e6, "form": "10-K", "filed": f"{LAST_YEAR + 1}-02-15"}]}}}
    return {"entityName": "Steady Growth Corp", "facts": {"us-gaap": facts, "dei": dei}}


def fiscal_end(year: int) -> date:
    return date(year, 12, 31)