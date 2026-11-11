"""Turn NFLX_Revenue_Build_Model.xlsx into a SegmentBuild spec (examples/specs/nflx_revenue_build.json).

Needs openpyxl (not a project dependency). Usage: python extract_nflx_build.py workbook.xlsx out.json

Mapping from the workbook's Assumptions sheet to build factors:
  subscription volume   rows 12 (base path), B13 / B14 (bull / bear spread)   -> "add" shift in [bear, bull]
  subscription pricing  rows 18, B19 / B20                                   -> "add" shift in [bear, bull]
  subscription FX       row 26 (bear path; base and bull are 0)              -> "scale" in [0, 1]; 0 = base or bull, 1 = bear
  advertising growth    row 36, B37 / B38 (bull / bear multiplier)           -> "scale" in [bear, bull]
FY2026E base revenue comes from Historical Financials (F24 subscription residual, F23 advertising guidance).
The workbook's consolidated revenue (Geographic Build row 13) and its bull and bear totals (Scenarios rows 24, 35)
are stored as checks under "workbook_total" and "workbook_scenarios".
"""

import json
import sys

import openpyxl
from openpyxl.utils import get_column_letter

FIRST_COL = 7  # column G = FY2027E


def path(ws, row):
    values, col = [], FIRST_COL
    while isinstance(ws.cell(row, col).value, (int, float)):
        values.append(float(ws.cell(row, col).value))
        col += 1
    return values, f"{ws.title}!{get_column_letter(FIRST_COL)}{row}:{get_column_letter(col - 1)}{row}"


def main(xlsx: str, out: str) -> None:
    wb = openpyxl.load_workbook(xlsx, data_only=True)
    a, h, g = wb["Assumptions"], wb["Historical Financials"], wb["Geographic Build"]
    cell = lambda ws, ref: float(ws[ref].value)
    vol, vol_src = path(a, 12)
    price, price_src = path(a, 18)
    fx_bear, fx_src = path(a, 26)
    ads, ads_src = path(a, 36)
    total, total_src = path(g, 13)
    sc = wb["Scenarios"]
    bull, bull_src = path(sc, 24)
    bear, bear_src = path(sc, 35)
    spec = {
        "name": "Netflix revenue build (from NFLX_Revenue_Build_Model.xlsx)",
        "base_year": "FY2026E",
        "first_year": 2027,
        "streams": [
            {"name": "subscription", "base_revenue": cell(h, "F24"), "source": "Historical Financials!F24",
             "factors": [
                 {"name": "volume", "mode": "add", "path": vol,
                  "range": {"low": cell(a, "B14"), "base": 0.0, "high": cell(a, "B13")},
                  "source": f"{vol_src}; spreads Assumptions!B14 (bear), B13 (bull)"},
                 {"name": "pricing", "mode": "add", "path": price,
                  "range": {"low": cell(a, "B20"), "base": 0.0, "high": cell(a, "B19")},
                  "source": f"{price_src}; spreads Assumptions!B20 (bear), B19 (bull)"},
                 {"name": "fx", "mode": "scale", "path": fx_bear,
                  "range": {"low": 0.0, "base": 0.0, "high": 1.0},
                  "source": f"{fx_src} is the bear FX path; base and bull FX are zero"}]},
            {"name": "advertising", "base_revenue": cell(h, "F23"), "source": "Historical Financials!F23",
             "factors": [
                 {"name": "growth", "mode": "scale", "path": ads,
                  "range": {"low": cell(a, "B38"), "base": 1.0, "high": cell(a, "B37")},
                  "source": f"{ads_src}; multipliers Assumptions!B38 (bear), B37 (bull)"}]},
        ],
        "workbook_total": total,
        "workbook_total_source": total_src,
        "workbook_scenarios": {"bull": bull, "bear": bear, "source": f"{bull_src}, {bear_src}"},
    }
    with open(out, "w") as fh:
        json.dump(spec, fh, indent=2)
    print(f"wrote {out}: {len(vol)} forecast years, FY2026E base {spec['streams'][0]['base_revenue'] + spec['streams'][1]['base_revenue']:,.1f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])