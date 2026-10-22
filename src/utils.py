"""Hàm tiện ích cho khám phá và kiểm tra chất lượng dữ liệu Parkinson."""
from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd


def quality_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo hồ sơ chất lượng cơ bản cho từng cột."""
    n = len(df)
    result = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "unique_count": df.nunique(dropna=True),
        "all_missing": df.isna().all(),
    })
    result.index.name = "column"
    result.attrs["row_count"] = n
    result.attrs["duplicate_rows"] = int(df.duplicated().sum())
    return result.reset_index()


def _numeric_features(df: pd.DataFrame, exclude_cols: Iterable[str] = ()) -> list[str]:
    excluded = set(exclude_cols)
    return [c for c in df.select_dtypes(include=np.number).columns if c not in excluded]


def iqr_outlier_report(
    df: pd.DataFrame,
    exclude_cols: Iterable[str] = (),
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """Báo cáo số điểm ngoài [Q1 - k*IQR, Q3 + k*IQR] theo từng cột số."""
    rows = []
    for col in _numeric_features(df, exclude_cols):
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        flagged = (values < lower) | (values > upper)
        rows.append({
            "feature": col, "q1": q1, "q3": q3, "iqr": iqr,
            "lower_bound": lower, "upper_bound": upper,
            "outlier_count": int(flagged.sum()),
            "outlier_pct": round(float(flagged.mean() * 100), 2),
        })
    return pd.DataFrame(rows, columns=[
        "feature", "q1", "q3", "iqr", "lower_bound", "upper_bound",
        "outlier_count", "outlier_pct",
    ]).sort_values("outlier_count", ascending=False, ignore_index=True)


def mad_outlier_report(
    df: pd.DataFrame,
    exclude_cols: Iterable[str] = (),
    threshold: float = 3.5,
) -> pd.DataFrame:
    """Báo cáo ngoại lai theo robust z-score sử dụng MAD.

    MAD = median(|x - median(x)|). Cột có MAD bằng 0 sẽ được báo riêng và không
    đánh dấu ngoại lai bằng công thức robust z-score để tránh chia cho 0.
    """
    rows = []
    for col in _numeric_features(df, exclude_cols):
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        median = float(values.median())
        mad = float((values - median).abs().median())
        if mad == 0:
            count = 0
            pct = 0.0
            note = "MAD = 0; không tính robust z-score"
        else:
            robust_z = 0.6745 * (values - median) / mad
            flagged = robust_z.abs() > threshold
            count = int(flagged.sum())
            pct = round(float(flagged.mean() * 100), 2)
            note = ""
        rows.append({
            "feature": col, "median": median, "mad": mad,
            "threshold": threshold, "outlier_count": count,
            "outlier_pct": pct, "note": note,
        })
    return pd.DataFrame(rows, columns=[
        "feature", "median", "mad", "threshold", "outlier_count",
        "outlier_pct", "note",
    ]).sort_values("outlier_count", ascending=False, ignore_index=True)


def find_target_column(columns: Iterable[str]) -> str | None:
    """Tự nhận diện cột nhãn thường gặp; trả về None nếu không thấy."""
    normalized = {str(c).strip().lower(): str(c) for c in columns}
    for candidate in ("status", "target", "label"):
        if candidate in normalized:
            return normalized[candidate]
    return None


def find_identifier_columns(columns: Iterable[str]) -> list[str]:
    """Nhận diện các cột định danh phổ biến, không nên dùng làm đặc trưng."""
    identifiers = {"name", "id", "record_id", "patient_id", "subject_id"}
    return [str(c) for c in columns if str(c).strip().lower() in identifiers]
