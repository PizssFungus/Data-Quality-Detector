"""
detector/text_detector.py
--------------------------
Rule-based checks for text and categorical columns.
Each check targets a specific class of data quality problem and records
the row number, column name, offending value, and a plain-English reason.

Checks performed:
  - Missing / null values
  - Sentinel placeholders  (N/A, NULL, ?, -999, UNKNOWN, ERROR …)
  - Whitespace-only values
  - Inconsistent casing within the same column
  - Extra leading / trailing whitespace
  - Values that look like noise or corrupted data (@@, ???, <tags>)
  - Rare category values (appear in < 1 % of rows)
  - Numbers inside text-only fields
  - Embedded special characters within otherwise ordinary text
  - Duplicate exact rows
"""

import re
import pandas as pd


# Sentinels that represent "no value" but are stored as strings
SENTINEL_PATTERNS = {
    "N/A", "NA", "NULL", "NONE", "NIL", "UNKNOWN", "ERROR",
    "?", "??", "-", "--", "N.A.", "N.A", "TBD", "TBC", "MISSING",
}

NOISE_RE = re.compile(r"^[^a-zA-Z0-9\s]{3,}$")          # e.g. "??@@##"
HTML_TAG_RE = re.compile(r"<[^>]+>")                       # e.g. "<br>"
NUMBER_ONLY_RE = re.compile(r"^\s*-?\d+(\.\d+)?\s*$")     # pure number in text col


def _flag(row_num, col, value, issue_type, method, reason):
    return {
        "row": row_num,
        "column": col,
        "value": value,
        "issue_type": issue_type,
        "method": method,
        "reason": reason,
    }


def check_missing(df: pd.DataFrame, text_cols: list) -> list[dict]:
    """Flag null / NaN values in text columns."""
    issues = []
    for col in text_cols:
        mask = df[col].isna()
        for idx in df.index[mask]:
            issues.append(_flag(
                df.at[idx, "_row_number"], col, None,
                "Missing Value", "Rule: Null Check",
                f"Column '{col}' is empty (null/NaN)."
            ))
    return issues


def check_sentinels(df: pd.DataFrame, text_cols: list) -> list[dict]:
    """Flag values that are placeholders rather than real data (N/A, NULL, ?, etc.)."""
    issues = []
    for col in text_cols:
        for idx in df.index:
            val = df.at[idx, col]
            if pd.isna(val):
                continue
            if str(val).strip().upper() in SENTINEL_PATTERNS:
                issues.append(_flag(
                    df.at[idx, "_row_number"], col, val,
                    "Sentinel Placeholder", "Rule: Sentinel Check",
                    f"Value '{val}' is a known placeholder rather than real data."
                ))
    return issues


def check_whitespace_only(df: pd.DataFrame, text_cols: list) -> list[dict]:
    """Flag cells that contain only spaces — they look filled but carry no data."""
    issues = []
    for col in text_cols:
        for idx in df.index:
            val = df.at[idx, col]
            if pd.notna(val) and str(val).strip() == "" and str(val) != "":
                issues.append(_flag(
                    df.at[idx, "_row_number"], col, repr(val),
                    "Whitespace-Only Value", "Rule: Whitespace Check",
                    f"Column '{col}' appears filled but contains only whitespace."
                ))
    return issues


def check_leading_trailing_spaces(df: pd.DataFrame, text_cols: list) -> list[dict]:
    """Flag values with leading or trailing whitespace."""
    issues = []
    for col in text_cols:
        for idx in df.index:
            val = df.at[idx, col]
            if pd.notna(val) and isinstance(val, str):
                if val != val.strip():
                    issues.append(_flag(
                        df.at[idx, "_row_number"], col, repr(val),
                        "Leading/Trailing Whitespace", "Rule: Whitespace Check",
                        f"Value has extra whitespace: {repr(val)}"
                    ))
    return issues


def check_inconsistent_casing(df: pd.DataFrame, text_cols: list) -> list[dict]:
    """
    For columns with low cardinality (likely categorical), flag values whose
    casing differs from the most common form of that value.
    e.g. 'engineering' and 'SALES' when others use title case.
    """
    issues = []
    for col in text_cols:
        series = df[col].dropna().astype(str).str.strip()
        unique_count = series.nunique()
        # Only worth checking on low-cardinality columns (categorical-like)
        if unique_count > 200:
            continue
        if len(series) > 0 and unique_count / len(series) > 0.5:
            continue

        # Group by lowercased value; find the dominant casing for each group
        grouped = series.groupby(series.str.lower())
        dominant = {}
        for key, group in grouped:
            dominant[key] = group.mode().iloc[0]  # most common casing

        for idx in df.index:
            val = df.at[idx, col]
            if pd.isna(val):
                continue
            val_str = str(val).strip()
            key = val_str.lower()
            if key in dominant and val_str != dominant[key]:
                issues.append(_flag(
                    df.at[idx, "_row_number"], col, val_str,
                    "Inconsistent Casing", "Rule: Casing Check",
                    f"Expected '{dominant[key]}' but found '{val_str}'."
                ))
    return issues


def check_noise_values(df: pd.DataFrame, text_cols: list) -> list[dict]:
    """Flag values that look like corrupted or meaningless data (@@, ???, <tags>)."""
    issues = []
    for col in text_cols:
        for idx in df.index:
            val = df.at[idx, col]
            if pd.isna(val):
                continue
            val_str = str(val).strip()
            if NOISE_RE.match(val_str):
                issues.append(_flag(
                    df.at[idx, "_row_number"], col, val_str,
                    "Noise / Corrupted Value", "Rule: Noise Check",
                    f"Value '{val_str}' looks like corrupted or meaningless data."
                ))
            elif HTML_TAG_RE.search(val_str):
                issues.append(_flag(
                    df.at[idx, "_row_number"], col, val_str,
                    "HTML Tag in Value", "Rule: Noise Check",
                    f"Value '{val_str}' contains an HTML/system tag."
                ))
    return issues


def check_rare_categories(df: pd.DataFrame, text_cols: list,
                           threshold: float = 0.01,
                           max_unique_ratio: float = 0.5) -> list[dict]:
    """
    Flag values in categorical columns that appear in fewer than `threshold`
    fraction of rows (default 1 %).  Rare categories can indicate typos,
    miscoded entries, or data from a different source.
    """
    issues = []
    n = len(df)
    for col in text_cols:
        series = df[col].dropna().astype(str).str.strip()
        if len(series) == 0:
            continue
        if series.nunique() > 200:
            continue  # skip free-text columns
        if series.nunique() / len(series) > max_unique_ratio:
            continue  # skip identifier-like columns (mostly-unique values)
        freq = series.value_counts(normalize=True)
        rare_vals = freq[freq < threshold].index.tolist()

        if not rare_vals:
            continue
        for idx in df.index:
            val = df.at[idx, col]
            if pd.isna(val):
                continue
            if str(val).strip() in rare_vals:
                pct = round(freq[str(val).strip()] * 100, 2)
                issues.append(_flag(
                    df.at[idx, "_row_number"], col, val,
                    "Rare Category Value", "Rule: Frequency Check",
                    f"Value '{val}' appears in only {pct}% of rows — possible typo or miscoding."
                ))
    return issues


def check_numbers_in_text(df: pd.DataFrame, text_cols: list) -> list[dict]:
    """
    Flag columns where most values are non-numeric strings but some cells
    contain pure numbers — likely a data entry error or mixed-type column.
    """
    issues = []
    for col in text_cols:
        series = df[col].dropna().astype(str)
        numeric_count = series.apply(lambda v: bool(NUMBER_ONLY_RE.match(v))).sum()
        total = len(series)
        if total == 0:
            continue
        # Only flag if the column is mostly text but some values are numbers
        if 0 < numeric_count < total * 0.5:
            for idx in df.index:
                val = df.at[idx, col]
                if pd.notna(val) and NUMBER_ONLY_RE.match(str(val).strip()):
                    issues.append(_flag(
                        df.at[idx, "_row_number"], col, val,
                        "Number in Text Field", "Rule: Type Consistency Check",
                        f"Column '{col}' is primarily text, but this cell contains a numeric value."
                    ))
    return issues


def check_duplicates(df: pd.DataFrame) -> list[dict]:
    """
    Flag exact duplicate rows (excluding the _row_number helper column).
    Only the duplicate copies are flagged, not the original.
    """
    issues = []
    data_cols = [c for c in df.columns if c != "_row_number"]
    dupe_mask = df.duplicated(subset=data_cols, keep="first")
    for idx in df.index[dupe_mask]:
        issues.append(_flag(
            df.at[idx, "_row_number"], "ALL COLUMNS", "—",
            "Duplicate Row", "Rule: Duplicate Check",
            "This row is an exact duplicate of an earlier record."
        ))
    return issues



# Characters that are legitimate in most text fields
CLEAN_TEXT_RE = re.compile(r"^[\w\s\-\'\.\_,&()]+$", re.UNICODE)
# Detects special chars embedded inside otherwise normal text
EMBEDDED_SPECIAL_RE = re.compile(r"[^\w\s\-\'\.\_,&()]", re.UNICODE)


def check_embedded_special_chars(df: pd.DataFrame, text_cols: list) -> list[dict]:
    """
    Flags values that contain special characters embedded in text
    (e.g. A$if, Ivan@Ivanov, #Marko, Nina@@Kovacs).
    Unlike the noise check which requires the ENTIRE value to be special
    characters, this catches fields that are mostly normal text but have
    one or more unexpected symbols mixed in.
    Skips columns where special chars are expected (email, url, code fields).
    """
    skip_keywords = {"email", "url", "link", "code", "id", "ref", "hash", "key", "phone"}
    issues = []
    for col in text_cols:
        col_lower = col.lower()
        if any(kw in col_lower for kw in skip_keywords):
            continue
        for idx in df.index:
            val = df.at[idx, col]
            if pd.isna(val) or str(val).strip() == "":
                continue
            val_str = str(val).strip()
            # Skip pure noise (caught by noise check already)
            if NOISE_RE.match(val_str):
                continue
            if not CLEAN_TEXT_RE.match(val_str):
                special = EMBEDDED_SPECIAL_RE.findall(val_str)
                issues.append(_flag(
                    df.at[idx, "_row_number"], col, val_str,
                    "Special Character in Text", "Rule: Special Char Check",
                    f"Value contains unexpected character(s) {special}: '{val_str}'"
                ))
    return issues

def run_all(df: pd.DataFrame, text_cols: list) -> pd.DataFrame:
    """
    Run all text/categorical checks and return a combined DataFrame of issues.
    """
    if not text_cols:
        print("  No text columns to analyse.")
        return pd.DataFrame()

    print(f"\n[Text Detection] Analysing {len(text_cols)} text/categorical columns...")

    all_issues = []
    checks = [
        ("Missing values",           check_missing(df, text_cols)),
        ("Sentinel placeholders",    check_sentinels(df, text_cols)),
        ("Whitespace-only values",   check_whitespace_only(df, text_cols)),
        ("Leading/trailing spaces",  check_leading_trailing_spaces(df, text_cols)),
        ("Inconsistent casing",      check_inconsistent_casing(df, text_cols)),
        ("Noise/corrupted values",   check_noise_values(df, text_cols)),
        ("Rare categories",          check_rare_categories(df, text_cols)),
        ("Numbers in text fields",   check_numbers_in_text(df, text_cols)),
        ("Duplicate rows",           check_duplicates(df)),
        ("Special chars in text",    check_embedded_special_chars(df, text_cols)),
    ]

    for label, result in checks:
        print(f"  {label:<30} → {len(result)} issues")
        all_issues.extend(result)

    print(f"  Total text issues found: {len(all_issues)}")
    return pd.DataFrame(all_issues)
