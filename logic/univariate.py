import pandas as pd
import streamlit as st
import datetime


def _is_effectively_datetime(series: pd.Series, sample_size: int = 100) -> bool:
    sample = series.dropna().head(sample_size)
    return not sample.empty and all(isinstance(x, (pd.Timestamp, datetime.datetime)) for x in sample)

@st.cache_data
def get_metrical_columns(df: pd.DataFrame, min_unique: int = 20) -> list[str]:
    """Return columns that are numeric or datetime-like and have enough unique values."""
    result = []
    for col in df.columns:
        series = df[col]
        if (
            pd.api.types.is_numeric_dtype(series) or _is_effectively_datetime(series)
        ) and series.nunique(dropna=True) >= min_unique:
            result.append(col)
    return result

@st.cache_data
def get_categorical_columns(df: pd.DataFrame, max_unique: int = 20) -> list[str]:
    """Return columns that are object or low-cardinality numeric, excluding datetime-like."""
    result = []
    for col in df.columns:
        series = df[col]
        unique_count = series.nunique(dropna=True)
        if _is_effectively_datetime(series):
            continue
        if series.dtype == "object" or (pd.api.types.is_numeric_dtype(series) and unique_count < max_unique):
            result.append(col)
    return result