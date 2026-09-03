"""
detector/loader.py
------------------
Handles loading the CSV dataset and basic preprocessing.
Separates columns into numeric, text, and potential date categories
so each can be passed to the right detection module downstream.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.
    Tries common encodings in order — falls back to latin-1 which
    accepts any byte sequence, avoiding UnicodeDecodeError on messy files.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    for encoding in ("utf-8", "utf-8-sig", "windows-1252", "latin-1"):
        try:
            df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
            print(f"Loaded '{path.name}' ({len(df)} rows, {len(df.columns)} columns) "
                  f"using {encoding} encoding.")
            return df
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not decode {filepath} with any supported encoding.")


def classify_columns(df: pd.DataFrame) -> dict:
    """
    Splits DataFrame columns into three groups:
    - numeric:  columns pandas already recognises as int/float
    - date:     object columns whose name hints at a date (date, time, year…)
                or whose first non-null value looks like a date string
    - text:     everything else (categorical strings, free text, etc.)

    Returns a dict with keys 'numeric', 'date', 'text', each a list of column names.
    """
    date_keywords = {"date", "time", "year", "month", "day", "timestamp", "created", "updated"}

    numeric_cols, date_cols, text_cols = [], [], []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # A numeric column whose name is (or contains, as a whole word)
            # "year", and whose values plausibly look like calendar years,
            # is better treated as a date-like column than a raw numeric
            # one: an implausible year is a validity problem, not a
            # statistical anomaly. Matching on the whole word "year"
            # specifically (not "years") avoids misclassifying genuinely
            # numeric duration columns such as "tenure_years".
            col_tokens = set(re.split(r"[^a-z0-9]+", col.lower())) - {""}
            if "year" in col_tokens:
                non_null = df[col].dropna()
                if len(non_null) > 0 and non_null.between(1900, 2100).mean() >= 0.9:
                    date_cols.append(col)
                    continue
            numeric_cols.append(col)
        else:
            col_lower = col.lower()
            # Check column name for date-related keywords
            if any(kw in col_lower for kw in date_keywords):
                date_cols.append(col)
            else:
                # Peek at the first non-null value and try parsing it as a date
                sample = df[col].dropna().head(5)
                looks_like_date = False
                for val in sample:
                    try:
                        pd.to_datetime(str(val), dayfirst=True)
                        looks_like_date = True
                        break
                    except Exception:
                        pass
                if looks_like_date:
                    date_cols.append(col)
                else:
                    # Might be a numeric column contaminated by a handful of
                    # bad values (a typo like '4a', or 'abd' in a price field)
                    # rather than genuinely categorical or free-text data.
                    non_null = df[col].dropna()
                    numeric_share = (
                        pd.to_numeric(non_null, errors="coerce").notna().mean()
                        if len(non_null) > 0 else 0
                    )
                    if numeric_share >= 0.8:
                        numeric_cols.append(col)
                    else:
                        text_cols.append(col)

    print(f"\nColumn classification:")
    print(f"  Numeric : {numeric_cols}")
    print(f"  Date    : {date_cols}")
    print(f"  Text    : {text_cols}")

    return {"numeric": numeric_cols, "date": date_cols, "text": text_cols}


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light preprocessing before analysis:
    - Strip leading/trailing whitespace from column names
    - Keep a copy of the original index so flagged rows can be traced back
      to their position in the original file (1-based for readability).
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Store original 1-based row numbers as a column for the output report
    df.insert(0, "_row_number", range(1, len(df) + 1))

    return df