"""
detector/exporter.py
---------------------
Exports the flagged issues and the original rows that contain them
to CSV files in the output directory.

Output files:
  - flagged_records.csv   : all detected issues with row/column/reason
  - flagged_rows.csv      : the original data rows that were flagged
                            (joined back to the source dataset)
"""

import os
import pandas as pd


def export_issues(issues_df: pd.DataFrame, output_dir: str) -> str:
    """
    Save the full issues table (one row per detected problem) to CSV.
    Columns: row, column, value, issue_type, method, reason
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "flagged_records.csv")
    issues_df.to_csv(path, index=False, encoding="utf-8")
    print(f"  Saved: {path}  ({len(issues_df)} issue records)")
    return path


def export_flagged_rows(original_df: pd.DataFrame, issues_df: pd.DataFrame,
                        output_dir: str) -> str:
    """
    Save the original rows (from the source dataset) that were flagged
    for at least one issue.  Includes a 'flagged_issue_types' column
    summarising what was detected on each row.
    """
    os.makedirs(output_dir, exist_ok=True)

    if issues_df.empty:
        print("  No flagged rows to export.")
        return ""

    # Gather issue type summaries per row number
    summary = (
        issues_df.groupby("row")["issue_type"]
        .apply(lambda x: " | ".join(sorted(set(x))))
        .reset_index()
        .rename(columns={"issue_type": "flagged_issue_types"})
    )

    # Join back to original data on _row_number
    flagged_rows = original_df[original_df["_row_number"].isin(summary["row"])].copy()
    flagged_rows = flagged_rows.merge(summary, left_on="_row_number", right_on="row", how="left")
    flagged_rows = flagged_rows.drop(columns=["row"], errors="ignore")

    # Move helper columns to front
    cols = ["_row_number", "flagged_issue_types"] + \
           [c for c in flagged_rows.columns if c not in ("_row_number", "flagged_issue_types")]
    flagged_rows = flagged_rows[cols]

    path = os.path.join(output_dir, "flagged_rows.csv")
    flagged_rows.to_csv(path, index=False, encoding="utf-8")
    print(f"  Saved: {path}  ({len(flagged_rows)} original rows flagged)")
    return path


def run_all(original_df: pd.DataFrame, issues_df: pd.DataFrame, output_dir: str):
    """Export both issue records and the original flagged rows."""
    print(f"\n[Exporter] Writing output files...")
    export_issues(issues_df, output_dir)
    export_flagged_rows(original_df, issues_df, output_dir)