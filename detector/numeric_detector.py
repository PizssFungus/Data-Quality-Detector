"""
detector/numeric_detector.py
-----------------------------
Applies five anomaly/outlier detection methods to the numeric columns
of the dataset. Each method flags suspicious rows and records WHY that
row was flagged, which column was involved, and which model caught it.

Methods used:
  1. Isolation Forest   — tree-based; efficient on high-dimensional data
  2. Local Outlier Factor (LOF) — density-based; catches local outliers
  3. KMeans clustering  — distance from cluster centre as anomaly proxy
  4. DBSCAN clustering  — density-based; labels low-density points as noise
  5. Z-score (statistical) — flags values more than 3 standard deviations from the mean
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler


def _build_issues(rows: pd.Index, df: pd.DataFrame, columns: list,
                  method: str, reason_template: str) -> list[dict]:
    """Helper: turn a set of flagged row indices into issue records."""
    issues = []
    for idx in rows:
        row_num = df.at[idx, "_row_number"]
        for col in columns:
            issues.append({
                "row": row_num,
                "column": col,
                "value": df.at[idx, col],
                "issue_type": "Numeric Anomaly",
                "method": method,
                "reason": reason_template,
            })
    return issues


def run_isolation_forest(df: pd.DataFrame, numeric_cols: list,
                         contamination: float = 0.05) -> list[dict]:
    """
    Isolation Forest — randomly partitions the feature space using trees.
    Anomalies are isolated in fewer splits than normal points, so they
    end up with shorter average path lengths.  contamination sets the
    expected fraction of outliers (default 5 %).
    Returns -1 for anomalies, 1 for normal points.
    """
    X = df[numeric_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean())

    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(X)

    flagged = df.index[preds == -1]
    print(f"  Isolation Forest   → {len(flagged)} outlier rows detected")

    issues = []
    for idx in flagged:
        row_num = df.at[idx, "_row_number"]
        issues.append({
            "row": row_num,
            "column": ", ".join(numeric_cols),
            "value": str(X.loc[idx].to_dict()),
            "issue_type": "Numeric Anomaly",
            "method": "Isolation Forest",
            "reason": (
                "Row isolated quickly by random trees, suggesting its combination "
                "of numeric values is unusual compared to the rest of the dataset."
            ),
        })
    return issues


def run_lof(df: pd.DataFrame, numeric_cols: list,
            contamination: float = 0.05, n_neighbors: int = 20) -> list[dict]:
    """
    Local Outlier Factor (LOF) — compares the local density of a point
    to the density of its k nearest neighbours.  A point with a much
    lower density than its neighbours gets a high LOF score and is
    labelled an outlier.  Good at finding outliers that are only unusual
    in their local neighbourhood, not globally.
    """
    X = df[numeric_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean())

    # LOF's density calculation is Euclidean-distance-based, exactly like
    # KMeans and DBSCAN, so it needs the same standardization they already
    # get; without it, columns on a larger raw scale (e.g. salary) dominate
    # the distance calculation and can hide genuine outliers in
    # smaller-scale columns entirely.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # A fixed n_neighbors of 20 is appropriate for a "local" neighbourhood
    # on datasets with hundreds of rows, but on small datasets it either
    # exceeds the sample size (sklearn silently falls back to n_samples-1)
    # or represents such a large fraction of the data that the comparison
    # is no longer meaningfully local at all. Scaling it down for small
    # datasets keeps LOF measuring local density rather than effectively
    # comparing every row to the whole dataset.
    effective_n_neighbors = max(2, min(n_neighbors, len(df) // 5))
    model = LocalOutlierFactor(n_neighbors=effective_n_neighbors, contamination=contamination)
    preds = model.fit_predict(X_scaled)

    flagged = df.index[preds == -1]
    print(f"  LOF                → {len(flagged)} outlier rows detected")

    issues = []
    for idx in flagged:
        row_num = df.at[idx, "_row_number"]
        issues.append({
            "row": row_num,
            "column": ", ".join(numeric_cols),
            "value": str(X.loc[idx].to_dict()),
            "issue_type": "Numeric Anomaly",
            "method": "Local Outlier Factor (LOF)",
            "reason": (
                "Row's local density is significantly lower than its neighbours, "
                "meaning its numeric values deviate from the surrounding data points."
            ),
        })
    return issues


def run_kmeans(df: pd.DataFrame, numeric_cols: list,
               n_clusters: int = 3, percentile: float = 95) -> list[dict]:
    """
    KMeans clustering — groups rows into n_clusters clusters by minimising
    within-cluster distance.  After fitting, we compute each row's distance
    to its assigned cluster centre.  Rows in the top percentile of distances
    are treated as anomalies — they belong to a cluster but sit far from its
    centre, suggesting they are unusual for that group.
    """
    X = df[numeric_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    centres = model.cluster_centers_[labels]
    distances = np.linalg.norm(X_scaled - centres, axis=1)
    threshold = np.percentile(distances, percentile)

    flagged = df.index[distances > threshold]
    print(f"  KMeans             → {len(flagged)} outlier rows detected (top {100-percentile:.0f}% distance)")

    issues = []
    for idx in flagged:
        row_num = df.at[idx, "_row_number"]
        issues.append({
            "row": row_num,
            "column": ", ".join(numeric_cols),
            "value": str(X.loc[idx].to_dict()),
            "issue_type": "Numeric Anomaly",
            "method": "KMeans Clustering",
            "reason": (
                f"Row is in the top {100-percentile:.0f}% of distances from its cluster centre, "
                "indicating its numeric values are atypical even within its assigned group."
            ),
        })
    return issues


def run_dbscan(df: pd.DataFrame, numeric_cols: list,
               eps: float = 1.5, min_samples: int = 5) -> list[dict]:
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) —
    groups points that are closely packed together and marks points in
    low-density regions as noise (label -1).  Unlike KMeans it does not
    require specifying the number of clusters and naturally identifies
    outliers as points that do not belong to any cluster.
    eps controls the neighbourhood radius; min_samples is the minimum
    number of points required to form a dense region.
    """
    X = df[numeric_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)

    flagged = df.index[labels == -1]
    print(f"  DBSCAN             → {len(flagged)} noise/outlier rows detected")

    issues = []
    for idx in flagged:
        row_num = df.at[idx, "_row_number"]
        issues.append({
            "row": row_num,
            "column": ", ".join(numeric_cols),
            "value": str(X.loc[idx].to_dict()),
            "issue_type": "Numeric Anomaly",
            "method": "DBSCAN",
            "reason": (
                "Row could not be assigned to any dense cluster — its numeric "
                "values sit in a sparse region of the data, classifying it as noise."
            ),
        })
    return issues


def run_zscore(df: pd.DataFrame, numeric_cols: list, threshold: float = 3.0) -> list[dict]:
    """
    Z-score (statistical deviation detection) — for each numeric column,
    computes how many standard deviations each value is from the column mean.
    Values with |z| > threshold (default 3) are flagged as statistical outliers.
    This is a per-column check, so it can pinpoint exactly which column
    contains the extreme value, unlike the multi-variate ML methods above.
    """
    issues = []
    total = 0

    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        mean = series.mean()
        std = series.std()

        if std == 0 or pd.isna(std):
            continue  # constant column — z-score undefined

        z_scores = (series - mean).abs() / std
        flagged = df.index[z_scores > threshold]
        total += len(flagged)

        for idx in flagged:
            row_num = df.at[idx, "_row_number"]
            z = round(z_scores[idx], 2)
            issues.append({
                "row": row_num,
                "column": col,
                "value": df.at[idx, col],
                "issue_type": "Statistical Outlier",
                "method": "Z-score",
                "reason": (
                    f"Value is {z}σ from the column mean "
                    f"(mean={round(mean, 2)}, std={round(std, 2)}), "
                    f"exceeding the {threshold}σ threshold."
                ),
            })

    print(f"  Z-score            → {total} outlier values detected across numeric columns")
    return issues


def run_all(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """
    Runs all numeric detection methods and combines their results.
    Missing value check runs first so nulls are recorded before
    being silently imputed for the ML models.
    """
    if not numeric_cols:
        print("  No numeric columns to analyse.")
        return pd.DataFrame()

    print(f"\n[Numeric Detection] Analysing {len(numeric_cols)} numeric columns...")

    all_issues = []

    missing = check_missing_numeric(df, numeric_cols)
    print(f"  Missing values (numeric)   → {len(missing)} issues")
    all_issues.extend(missing)

    all_issues.extend(run_isolation_forest(df, numeric_cols))
    all_issues.extend(run_lof(df, numeric_cols))
    all_issues.extend(run_kmeans(df, numeric_cols))
    all_issues.extend(run_dbscan(df, numeric_cols))
    all_issues.extend(run_zscore(df, numeric_cols))

    print(f"  Total numeric issues found: {len(all_issues)}")
    return pd.DataFrame(all_issues)


def check_missing_numeric(df, numeric_cols):
    """
    Flags null / NaN values in numeric columns.
    Runs before the ML models so missing values are clearly identified
    rather than being silently imputed and potentially triggering
    false positives in the anomaly detectors.
    """
    issues = []
    for col in numeric_cols:
        coerced = pd.to_numeric(df[col], errors="coerce")
        true_null_mask = df[col].isna()
        invalid_mask = coerced.isna() & df[col].notna()

        for idx in df.index[true_null_mask]:
            issues.append({
                "row": df.at[idx, "_row_number"],
                "column": col,
                "value": df.at[idx, col],
                "issue_type": "Missing Value",
                "method": "Rule: Null Check",
                "reason": f"Column '{col}' is empty (null/NaN).",
            })

        for idx in df.index[invalid_mask]:
            issues.append({
                "row": df.at[idx, "_row_number"],
                "column": col,
                "value": df.at[idx, col],
                "issue_type": "Invalid Numeric Value",
                "method": "Rule: Type Consistency Check",
                "reason": f"Column '{col}' should be numeric but contains '{df.at[idx, col]}'.",
            })
    return issues
