"""
main.py
--------
Entry point for the Automatic Data Quality Detector.

Usage:
    python main.py <path_to_csv> [output_directory]

Examples:
    python main.py sample_dirty_data.csv
    python main.py sample_dirty_data.csv my_results/

If no output directory is provided, results are saved to ./output/
"""

import sys
import os
import pandas as pd

from detector.loader import load_dataset, classify_columns, preprocess
from detector import numeric_detector, text_detector, date_detector, reporter, exporter


def main(csv_path: str, output_dir: str = "output"):
    print("\n" + "=" * 65)
    print("   AUTOMATIC DATA QUALITY DETECTOR")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Load and preprocess
    # ------------------------------------------------------------------
    df_raw = load_dataset(csv_path)
    df = preprocess(df_raw)
    total_rows = len(df)

    column_groups = classify_columns(df)
    numeric_cols = column_groups["numeric"]
    date_cols    = column_groups["date"]
    text_cols    = column_groups["text"]

    # Remove the helper column from numeric list if it ended up there
    numeric_cols = [c for c in numeric_cols if c != "_row_number"]

    # ------------------------------------------------------------------
    # 2. Run detectors
    # ------------------------------------------------------------------
    numeric_issues = numeric_detector.run_all(df, numeric_cols)
    text_issues    = text_detector.run_all(df, text_cols)

    # Columns where a future date is expected and should not be flagged,
    # e.g. a scheduled delivery or renewal date. Empty by default; add
    # column names here if your dataset has one.
    allowed_future_cols = []
    date_issues    = date_detector.run_all(df, date_cols, allowed_future_cols=allowed_future_cols)

    # ------------------------------------------------------------------
    # 3. Combine all issues
    # ------------------------------------------------------------------
    all_parts = [p for p in [numeric_issues, text_issues, date_issues] if not p.empty]
    if all_parts:
        issues_df = pd.concat(all_parts, ignore_index=True)
    else:
        issues_df = pd.DataFrame(columns=["row", "column", "value",
                                           "issue_type", "method", "reason"])

    # ------------------------------------------------------------------
    # 4. Report & export
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    reporter.run_all(issues_df, total_rows, output_dir)
    exporter.run_all(df, issues_df, output_dir)

    # ------------------------------------------------------------------
    # 5. Final summary
    # ------------------------------------------------------------------
    flagged = issues_df["row"].nunique() if not issues_df.empty else 0
    pct = round(flagged / total_rows * 100, 1) if total_rows else 0

    print("\n" + "=" * 65)
    print("   DETECTION COMPLETE")
    print("=" * 65)
    print(f"  Dataset rows        : {total_rows}")
    print(f"  Rows with issues    : {flagged}  ({pct}%)")
    print(f"  Total issues found  : {len(issues_df)}")
    print(f"  Results saved to    : {os.path.abspath(output_dir)}/")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_csv> [output_dir]")
        print("Example: python main.py sample_dirty_data.csv output/")
        sys.exit(1)

    csv_file = sys.argv[1]
    out_dir  = sys.argv[2] if len(sys.argv) > 2 else "output"

    main(csv_file, out_dir)
