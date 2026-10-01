"""Scarcity indicators, as they were published, for the calendar-spread rule.

The large losses of the sell-front / buy-next rule are scarcity shocks: the
front contract spikes when supply is short (MEMORY.md, loss diagnosis). Those
are visible first in fundamentals, not in prices. Each indicator here is read
from the report AS PUBLISHED and is usable only from its release date, so a
backtest never sees a revised number or a report before it came out.

* Natural gas: EIA weekly working gas in storage vs the same week of earlier
  years (released the Thursday after the week ending Friday).
* Rough rice: WASDE U.S. rice projected stocks-to-use vs the two prior years in
  the same table.
* Feeder cattle: WASDE projected U.S. beef production, next year vs this year
  (falling production = a short cattle herd).

WASDE files come from the USDA ESMIS archive (text since 2017 and in 2010; the
Excel version, flattened to text, for 2011-2016).
"""

from __future__ import annotations

import bisect
import os
import re
from datetime import date
from typing import Dict, List, Optional, Tuple

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _numbers(line: str) -> List[float]:
    return [float(x) for x in _NUM.findall(line)]


def parse_rice(text: str) -> Optional[Tuple[float, float]]:
    """(projected stocks-to-use, mean stocks-to-use of the two earlier years).

    Columns of the TOTAL RICE block: earlier year, last year (estimate), then the
    projection as of last month and as of this month. The last number on a row
    is this month's projection.
    """
    start = text.find("U.S. Rice Supply and Use")
    if start < 0:
        return None
    block = text[start:start + 6000]
    total = block.find("TOTAL RICE")
    block = block[total:] if total >= 0 else block
    use = ending = None
    for line in block.splitlines():
        s = line.strip()
        if use is None and s.startswith("Use, Total"):
            use = _numbers(s.replace("Use, Total", ""))
        elif ending is None and s.startswith("Ending Stocks"):
            ending = _numbers(s.replace("Ending Stocks", ""))
        if use and ending:
            break
    if not use or not ending or len(use) < 3 or len(ending) < 3:
        return None
    proj = ending[-1] / use[-1]
    past = (ending[0] / use[0] + ending[1] / use[1]) / 2
    return proj, past


def parse_beef(text: str) -> Optional[float]:
    """Projected U.S. beef production, next year vs current year, as a fraction.

    In 'U.S. Quarterly Animal Product Production' each year block ends with
    '<Mon>Proj.' rows (or an 'Annual' row once the year is complete); the beef
    figure is the first number. The last row of each block is the latest view.
    """
    start = text.find("U.S. Quarterly Animal Product Production")
    if start < 0:
        return None
    block = text[start:start + 4000]
    end = block.find("U.S. Quarterly Prices")
    block = block[:end] if end > 0 else block
    years: List[Tuple[int, float]] = []
    current_year: Optional[int] = None
    latest: Optional[float] = None
    for line in block.splitlines():
        s = line.strip()
        # A year opens a block: alone on its line in the text reports
        # ("2018"), or followed by the first quarter's row in the Excel ones
        # flattened to text ("2013.0  IV  6423.0 ...").
        m = re.match(r"((?:19|20)\d{2})(?:\.0)?(?:\s|$)", s)
        if m:
            if current_year is not None and latest is not None:
                years.append((current_year, latest))
            current_year, latest = int(m.group(1)), None
            continue
        if current_year is None:
            continue
        if re.match(r"(Annual|[A-Z][a-z]{2}\s*Proj\.?|Proj\.)", s):
            nums = _numbers(re.sub(r"^(Annual|[A-Z][a-z]{2}\s*Proj\.?|Proj\.)", "", s))
            if nums:
                latest = nums[0]
    if current_year is not None and latest is not None:
        years.append((current_year, latest))
    if len(years) < 2:
        return None
    (_, this_year), (_, next_year) = years[-2], years[-1]
    return next_year / this_year - 1 if this_year else None


def load_wasde(folder: str) -> Dict[date, str]:
    out = {}
    for name in os.listdir(folder):
        m = re.fullmatch(r"wasde_(\d{4}-\d{2}-\d{2})\.txt", name)
        if m:
            with open(os.path.join(folder, name), encoding="utf-8", errors="ignore") as fh:
                out[date.fromisoformat(m.group(1))] = fh.read()
    return out


class Published:
    """A series keyed by release date; `as_of(d)` is the last value released
    strictly before day d (a report released on the entry day is not used)."""

    def __init__(self, values: Dict[date, float]):
        self.days = sorted(values)
        self.values = values

    def as_of(self, d: date) -> Optional[float]:
        i = bisect.bisect_left(self.days, d) - 1
        return self.values[self.days[i]] if i >= 0 else None


#: EIA weekly working gas, Lower 48 (Bcf). Free, no key, history from 2010.
EIA_STORAGE_URL = "https://www.eia.gov/dnav/ng/hist_xls/NW2_EPG0_SWO_R48_BCFw.xls"
#: The week ends on Friday; the report comes out the following Thursday.
EIA_RELEASE_LAG_DAYS = 6


def parse_eia_storage(xls_bytes: bytes) -> Dict[date, float]:
    """{week-ending date: working gas in Bcf} from the EIA history workbook."""
    import xlrd  # only needed for this one file format
    from datetime import timedelta
    sheet = xlrd.open_workbook(file_contents=xls_bytes).sheet_by_name("Data 1")
    out = {}
    for i in range(3, sheet.nrows):
        serial, value = sheet.row_values(i)[:2]
        if value != "":
            out[date(1899, 12, 30) + timedelta(days=int(serial))] = float(value)
    return out


def storage_deficit(storage: Dict[date, float], min_years: int = 3) -> Dict[date, float]:
    """{release date: storage / mean of the same week in up to 5 earlier years - 1}.

    Keyed by RELEASE date (week end + 6 days), so `Published.as_of` never hands
    a backtest or a live signal a number before the market had it.
    """
    from datetime import timedelta
    weeks = sorted(storage)

    def near(d: date) -> Optional[float]:
        i = bisect.bisect_left(weeks, d)
        cands = [weeks[j] for j in (i - 1, i) if 0 <= j < len(weeks)]
        w = min(cands, key=lambda x: abs((x - d).days)) if cands else None
        return storage[w] if w and abs((w - d).days) <= 4 else None

    out = {}
    for w in weeks:
        prior = [v for k in range(1, 6) if (v := near(w - timedelta(days=364 * k))) is not None]
        if len(prior) >= min_years:
            out[w + timedelta(days=EIA_RELEASE_LAG_DAYS)] = storage[w] / (sum(prior) / len(prior)) - 1
    return out
