"""
detector/reporter.py
---------------------
Generates visualisations and a plain-text summary report from the
combined issues DataFrame produced by the detector pipeline.

Visualisations produced:
  1. Bar chart  — issue count by type
  2. Bar chart  — issue count by detection method
  3. Bar chart  — top most problematic columns
  4. Heatmap    — issue density across columns and issue types
  5. Histogram  — distribution of flagged row numbers (where in the file do problems cluster?)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Consistent colour palette
PALETTE = "Set2"
FIG_DPI = 120


def _save(fig, output_dir: str, filename: str):
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

def _single_column_issues(issues_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns only the issues attributable to exactly one column. Excludes
    multivariate numeric anomalies (which involve several columns jointly)
    and duplicate-row issues (the "ALL COLUMNS" placeholder), since
    crediting a joint event to any individual column, whether by full
    duplication or by a fractional split, either inflates a column's
    count or invents a non-integer "issue count" that corresponds to
    nothing real. These joint-detection issues remain fully visible in
    the issue-type and detection-method breakdowns; they are simply not
    forced into a per-column ranking they do not meaningfully belong in.
    """
    is_multi_column = issues_df["column"].str.contains(", ", regex=False)
    is_all_columns = issues_df["column"] == "ALL COLUMNS"
    return issues_df[~is_multi_column & ~is_all_columns]

def plot_issues_by_type(issues_df: pd.DataFrame, output_dir: str):
    """Bar chart: how many issues were found per issue type."""
    counts = issues_df["issue_type"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=counts.values, y=counts.index, hue=counts.index,
                palette=PALETTE, legend=False, ax=ax)
    ax.set_title("Issues Detected by Type", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Issues")
    ax.set_ylabel("Issue Type")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    for i, v in enumerate(counts.values):
        ax.text(v + 0.2, i, str(v), va="center", fontsize=9)
    fig.tight_layout()
    _save(fig, output_dir, "1_issues_by_type.png")


def plot_issues_by_method(issues_df: pd.DataFrame, output_dir: str):
    """Bar chart: how many issues were flagged by each detection method."""
    counts = issues_df["method"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=counts.values, y=counts.index, hue=counts.index,
                palette=PALETTE, legend=False, ax=ax)
    ax.set_title("Issues Detected by Method", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Issues")
    ax.set_ylabel("Detection Method")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    for i, v in enumerate(counts.values):
        ax.text(v + 0.2, i, str(v), va="center", fontsize=9)
    fig.tight_layout()
    _save(fig, output_dir, "2_issues_by_method.png")


def plot_issues_by_column(issues_df: pd.DataFrame, output_dir: str, top_n: int = 15):
    """
    Bar chart: the columns with the most flagged issues, counting only
    issues attributable to a single column. Multivariate numeric
    anomalies and duplicate-row issues are excluded here since they are
    not "about" any one column; see _single_column_issues.
    """
    single = _single_column_issues(issues_df)
    counts = single["column"].value_counts().head(top_n)
    if counts.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=counts.values, y=counts.index, hue=counts.index,
                palette=PALETTE, legend=False, ax=ax)
    ax.set_title("Problematic Columns", fontsize=14, fontweight="bold")    
    ax.set_xlabel("Number of Issues")
    ax.set_ylabel("Column")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    for i, v in enumerate(counts.values):
        ax.text(v + 0.2, i, str(v), va="center", fontsize=9)
    fig.tight_layout()
    _save(fig, output_dir, "3_issues_by_column.png")

def plot_heatmap(issues_df: pd.DataFrame, output_dir: str):
    """
    Heatmap: columns (x) vs issue types (y), cell = count. Only issues
    attributable to a single column are included; see
    _single_column_issues for why multivariate anomalies and duplicate
    rows are excluded rather than split or duplicated across columns.
    """
    single = _single_column_issues(issues_df)
    pivot = single.pivot_table(index="issue_type", columns="column",
                                values="row", aggfunc="count", fill_value=0)

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.9), max(5, len(pivot) * 0.7)))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.5,
                cbar_kws={"label": "Issue Count"}, ax=ax)
    ax.set_title("Issue Type × Column Heatmap", fontsize=14, fontweight="bold")
    ax.set_xlabel("Column")
    ax.set_ylabel("Issue Type")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    _save(fig, output_dir, "4_heatmap_type_vs_column.png")


def plot_row_distribution(issues_df: pd.DataFrame, output_dir: str):
    """
    Bar chart: number of issues on each individual flagged row, positioned
    at that row's real number in the original file. Figure width scales
    with the row range so bars remain visible rather than washing out
    into sub-pixel width on datasets with hundreds of rows; no white
    edges, which erase most of a bar's colour once bars get that thin.
    Note that no static chart can label every individual row once there
    are hundreds of them; for pinpointing an exact row, flagged_records.csv
    is the right tool, this chart is for spotting where problems cluster.
    """
    rows = pd.to_numeric(issues_df["row"], errors="coerce").dropna()
    if rows.empty:
        return

    counts = rows.value_counts().sort_index()
    row_range = max(1, counts.index.max() - counts.index.min())
    fig_width = max(10, min(30, row_range / 40))

    fig, ax = plt.subplots(figsize=(fig_width, 4))
    ax.bar(counts.index, counts.values, width=max(1.0, row_range / 400),
           color=sns.color_palette(PALETTE)[0])
    ax.set_title("Issues per Flagged Row", fontsize=14, fontweight="bold")
    ax.set_xlabel("Row Number (in original file)")
    ax.set_ylabel("Number of Issues")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Label every individual flagged row when there are few enough to
    # read; with hundreds of flagged rows, individual labels would just
    # overlap into an unreadable smear, so fall back to a clean,
    # evenly-spaced integer axis instead.
    if len(counts) <= 50:
        ax.set_xticks(counts.index)
        ax.set_xticklabels(counts.index, rotation=90 if len(counts) > 20 else 0)
    else:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=20))

    fig.tight_layout()
    _save(fig, output_dir, "5_row_distribution.png")


def generate_text_report(issues_df: pd.DataFrame, total_rows: int,
                         output_dir: str) -> str:
    """
    Writes a plain-text summary report and returns its file path.
    """
    path = os.path.join(output_dir, "data_quality_report.txt")
    flagged_rows = issues_df["row"].nunique() if not issues_df.empty else 0

    lines = [
        "=" * 65,
        "        AUTOMATIC DATA QUALITY DETECTION REPORT",
        "=" * 65,
        f"  Total rows in dataset : {total_rows}",
        f"  Rows with issues      : {flagged_rows}  "
        f"({round(flagged_rows / total_rows * 100, 1) if total_rows else 0}%)",
        f"  Total issues found    : {len(issues_df)}",
        "",
    ]

    if issues_df.empty:
        lines.append("  No issues detected.")
    else:
        # --- Summary by type ---
        lines.append("-" * 65)
        lines.append("  ISSUES BY TYPE")
        lines.append("-" * 65)
        for issue_type, count in issues_df["issue_type"].value_counts().items():
            lines.append(f"  {issue_type:<40} {count:>5} issues")

        # --- Summary by method ---
        lines.append("")
        lines.append("-" * 65)
        lines.append("  ISSUES BY DETECTION METHOD")
        lines.append("-" * 65)
        for method, count in issues_df["method"].value_counts().items():
            lines.append(f"  {method:<40} {count:>5} issues")

        # --- Top problematic columns ---
        lines.append("")
        lines.append("-" * 65)
        lines.append("  PROBLEMATIC COLUMNS (single-column issues only; joint")        
        lines.append("  multivariate anomalies are not attributable to one column")
        lines.append("  and are counted under ISSUES BY TYPE instead)")
        lines.append("-" * 65)
        col_counts = _single_column_issues(issues_df)["column"].value_counts().head(10)
        for col, count in col_counts.items():
            lines.append(f"  {col:<40} {count:>5} issues")

        # --- Sample flagged records ---
        lines.append("")
        lines.append("-" * 65)
        lines.append("  SAMPLE FLAGGED RECORDS (first 20)")
        lines.append("-" * 65)
        sample = issues_df.head(20)
        for _, row in sample.iterrows():
            lines.append(
                f"  Row {str(row['row']).rjust(4)} | {str(row['column'])[:20]:<20} | "
                f"{str(row['issue_type'])[:25]:<25} | {str(row['reason'])[:60]}"
            )

    lines.append("")
    lines.append("=" * 65)
    lines.append("  See flagged_records.csv for the full list of flagged rows.")
    lines.append("  See plots/ for visualisations.")
    lines.append("=" * 65)

    report_text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"  Saved: {path}")
    return path


def run_all(issues_df: pd.DataFrame, total_rows: int, output_dir: str):
    """Generate all visualisations and the text report."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if issues_df.empty:
        print("  No issues to visualise.")
        generate_text_report(issues_df, total_rows, output_dir)
        return

    print(f"\n[Reporter] Generating visualisations...")
    plot_issues_by_type(issues_df, plots_dir)
    plot_issues_by_method(issues_df, plots_dir)
    plot_issues_by_column(issues_df, plots_dir)
    plot_heatmap(issues_df, plots_dir)
    plot_row_distribution(issues_df, plots_dir)

    print(f"\n[Reporter] Writing text report...")
    generate_text_report(issues_df, total_rows, output_dir)
