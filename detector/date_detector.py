"""
detector/date_detector.py
--------------------------
Rule-based checks for date and datetime columns.

Checks performed:
  - Unparseable / invalid date strings
  - Mixed date formats within the same column
  - Future dates where they should not exist
  - Dates outside a plausible historical range (before 1900)
  - Impossible cross-column date logic (end before start, etc.)
"""

import re
import pandas as pd
from datetime import datetime

# Today's date — used as the ceiling for "no future dates" checks
TODAY = pd.Timestamp.now().normalize()

# Common date format patterns to detect mixing
FORMAT_PATTERNS = [
    (r"^\d{4}-\d{2}-\d{2}$",              "YYYY-MM-DD"),
    (r"^\d{1,2}/\d{1,2}/\d{4}$",          "MM/DD/YYYY"),
    (r"^\d{1,2}-\d{1,2}-\d{4}$",          "DD-MM-YYYY"),
    (r"^\d{1,2}\.\d{1,2}\.\d{4}$",        "DD.MM.YYYY"),
    (r"^[A-Za-z]+ \d{1,2},? \d{4}$",      "Month D, YYYY"),
    (r"^\d{1,2} [A-Za-z]+ \d{4}$",        "D Month YYYY"),
]


def _flag(row_num, col, value, issue_type, method, reason):
    return {
        "row": row_num,
        "column": col,
        "value": value,
        "issue_type": issue_type,
        "method": method,
        "reason": reason,
    }


def _detect_format(value: str) -> str:
    """Return a format label for a date string, or 'unknown'."""
    for pattern, label in FORMAT_PATTERNS:
        if re.match(pattern, str(value).strip()):
            return label
    return "other"


def check_unparseable_dates(df: pd.DataFrame, date_cols: list) -> list[dict]:
    """
    Flag values in date columns that cannot be parsed as a date by pandas.
    These represent corrupted, mistyped, or non-date values in a date field.
    """
    issues = []
    for col in date_cols:
        for idx in df.index:
            val = df.at[idx, col]
            if pd.isna(val) or str(val).strip() == "":
                continue
            try:
                pd.to_datetime(str(val), dayfirst=True)
            except Exception:
                issues.append(_flag(
                    df.at[idx, "_row_number"], col, val,
                    "Unparseable Date", "Rule: Date Parse Check",
                    f"'{val}' could not be parsed as a valid date."
                ))
    return issues


def check_mixed_formats(df: pd.DataFrame, date_cols: list) -> list[dict]:
    """
    Detect columns where date strings use inconsistent formats (e.g. some rows
    use YYYY-MM-DD while others use MM/DD/YYYY).  Mixed formats cause silent
    parsing errors and incorrect sorting.
    """
    issues = []
    for col in date_cols:
        format_map = {}  # format_label → list of row numbers
        for idx in df.index:
            val = df.at[idx, col]
            if pd.isna(val):
                continue
            fmt = _detect_format(str(val))
            format_map.setdefault(fmt, []).append(idx)

        if len(format_map) <= 1:
            continue  # all the same format — fine

        dominant_fmt = max(format_map, key=lambda k: len(format_map[k]))
        for fmt, indices in format_map.items():
            if fmt == dominant_fmt:
                continue
            for idx in indices:
                val = df.at[idx, col]
                issues.append(_flag(
                    df.at[idx, "_row_number"], col, val,
                    "Mixed Date Format", "Rule: Date Format Check",
                    f"Format '{fmt}' differs from the dominant format '{dominant_fmt}' in this column."
                ))
    return issues


def check_future_dates(df: pd.DataFrame, date_cols: list,
                       allowed_future_cols: list = None) -> list[dict]:
    """
    Flag dates that are in the future.  Some columns legitimately store future
    dates (e.g. scheduled_date); pass those column names in allowed_future_cols
    to skip them.
    """
    issues = []
    skip = set(allowed_future_cols or [])
    for col in date_cols:
        if col in skip:
            continue
        for idx in df.index:
            val = df.at[idx, col]
            if pd.isna(val):
                continue
            try:
                parsed = pd.to_datetime(str(val), dayfirst=True)
                if parsed > TODAY:
                    issues.append(_flag(
                        df.at[idx, "_row_number"], col, val,
                        "Future Date", "Rule: Date Range Check",
                        f"Date '{val}' is in the future (after {TODAY.date()})."
                    ))
            except Exception:
                pass  # already caught by unparseable check
    return issues


def check_historical_range(df: pd.DataFrame, date_cols: list,
                            min_year: int = 1900) -> list[dict]:
    """
    Flag dates before min_year (default 1900) as implausible for most
    business datasets.
    """
    issues = []
    for col in date_cols:
        for idx in df.index:
            val = df.at[idx, col]
            if pd.isna(val):
                continue
            try:
                parsed = pd.to_datetime(str(val), dayfirst=True)
                if parsed.year < min_year:
                    issues.append(_flag(
                        df.at[idx, "_row_number"], col, val,
                        "Implausible Historical Date", "Rule: Date Range Check",
                        f"Date '{val}' is before {min_year}, which is unlikely for this dataset."
                    ))
            except Exception:
                pass
    return issues


def check_cross_column_logic(df: pd.DataFrame, date_col_pairs: list) -> list[dict]:
    """
    For pairs of date columns where one should come before the other
    (e.g. start_date / end_date), flag rows where the order is violated.

    date_col_pairs: list of (earlier_col, later_col) tuples.
    The function auto-generates common pairs from the available date columns
    if none are provided.
    """
    issues = []
    for (start_col, end_col) in date_col_pairs:
        for idx in df.index:
            start_val = df.at[idx, start_col]
            end_val = df.at[idx, end_col]
            if pd.isna(start_val) or pd.isna(end_val):
                continue
            try:
                start_dt = pd.to_datetime(str(start_val), dayfirst=True)
                end_dt = pd.to_datetime(str(end_val), dayfirst=True)
                if end_dt < start_dt:
                    issues.append(_flag(
                        df.at[idx, "_row_number"],
                        f"{start_col}, {end_col}",
                        f"{start_val} → {end_val}",
                        "Impossible Date Logic", "Rule: Cross-Column Date Check",
                        f"'{end_col}' ({end_val}) is earlier than '{start_col}' ({start_val})."
                    ))
            except Exception:
                pass
    return issues


def _auto_pair_date_cols(date_cols: list) -> list:
    """
    Automatically pair start/end and similar date columns by name.
    Matches whole tokens (the column name split on non-alphanumeric
    characters) rather than raw substrings, so a column such as
    'weekend_flag' is not treated as matching 'end' just because those
    letters happen to appear inside a longer word. Duplicate pairs,
    which can arise when more than one keyword rule matches the same
    two columns, are removed.
    """
    pairs = []
    keywords = {
        "start": "end",
        "begin": "end",
        "order": "delivery",
        "created": "updated",
        "open": "close",
        "from": "to",
    }

    def tokens(name: str) -> set:
        return set(re.split(r"[^a-z0-9]+", name.lower())) - {""}

    token_map = {c: tokens(c) for c in date_cols}

    for start_kw, end_kw in keywords.items():
        for col, col_tokens in token_map.items():
            if start_kw in col_tokens:
                for other, other_tokens in token_map.items():
                    if other != col and end_kw in other_tokens:
                        pairs.append((col, other))

    # Remove duplicate pairs without changing their order
    return list(dict.fromkeys(pairs))
    return pairs


def run_all(df: pd.DataFrame, date_cols: list, allowed_future_cols: list = None) -> pd.DataFrame:
    """
    Run all date checks and return a combined DataFrame of issues.
    `allowed_future_cols` is passed through to check_future_dates to
    exempt columns where a future date is legitimate (e.g. a scheduled
    delivery date); defaults to no exemptions if not provided.
    """
    if not date_cols:
        print("  No date columns to analyse.")
        return pd.DataFrame()

    print(f"\n[Date Detection] Analysing {len(date_cols)} date column(s)...")

    all_issues = []

    unparseable = check_unparseable_dates(df, date_cols)
    print(f"  Unparseable dates            → {len(unparseable)} issues")
    all_issues.extend(unparseable)

    mixed = check_mixed_formats(df, date_cols)
    print(f"  Mixed date formats           → {len(mixed)} issues")
    all_issues.extend(mixed)

    future = check_future_dates(df, date_cols, allowed_future_cols=allowed_future_cols)
    print(f"  Future dates                 → {len(future)} issues")
    all_issues.extend(future)

    historical = check_historical_range(df, date_cols)
    print(f"  Implausible historical dates → {len(historical)} issues")
    all_issues.extend(historical)

    pairs = _auto_pair_date_cols(date_cols)
    cross = check_cross_column_logic(df, pairs)
    print(f"  Cross-column date logic      → {len(cross)} issues")
    all_issues.extend(cross)

    print(f"  Total date issues found: {len(all_issues)}")
    return pd.DataFrame(all_issues)
