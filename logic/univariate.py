import pandas as pd
import streamlit as st
from data.constants import UNIQUE_THRESH


def get_datetime_columns(df: pd.DataFrame):
    """Return columns that are datetime like."""
    return [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]

def get_metrical_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that are numeric and have enough unique values."""
    result = []
    for col in df.columns:
        series = df[col]
        if (
            pd.api.types.is_numeric_dtype(series)
            and series.nunique(dropna=True) >= UNIQUE_THRESH
        ):
            result.append(col)
    return result

def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that are object or low-cardinality numeric, excluding datetime-like."""
    result = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        
        unique_count = series.nunique(dropna=True)
        if series.dtype == "object" or (pd.api.types.is_numeric_dtype(series) and unique_count < UNIQUE_THRESH):
            result.append(col)
    return result