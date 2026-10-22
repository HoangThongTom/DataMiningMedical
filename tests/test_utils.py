"""Unit tests nhỏ cho các hàm tiện ích (không chạy tự động khi mở dự án)."""
import pandas as pd
from src.utils import quality_profile, iqr_outlier_report, mad_outlier_report, find_target_column


def test_quality_profile_reports_missing_values():
    df = pd.DataFrame({"x": [1, None, 3], "status": [0, 1, 1]})
    profile = quality_profile(df).set_index("column")
    assert profile.loc["x", "missing_count"] == 1
    assert profile.loc["status", "missing_count"] == 0


def test_iqr_report_detects_extreme_value():
    df = pd.DataFrame({"x": [1, 1, 2, 2, 3, 100], "status": [0, 0, 0, 1, 1, 1]})
    report = iqr_outlier_report(df, exclude_cols=["status"])
    assert report.loc[0, "feature"] == "x"
    assert report.loc[0, "outlier_count"] >= 1


def test_mad_report_handles_zero_mad():
    df = pd.DataFrame({"x": [5, 5, 5, 5], "status": [0, 0, 1, 1]})
    report = mad_outlier_report(df, exclude_cols=["status"])
    assert report.loc[0, "mad"] == 0
    assert "MAD = 0" in report.loc[0, "note"]


def test_find_target_column_is_case_insensitive():
    assert find_target_column(["MDVP:Fo(Hz)", "Status"]) == "Status"
