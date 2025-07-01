import pandas as pd
import streamlit as st


def _looks_like_timestamp(series: pd.Series, sample_size: int = 20, threshold: float = 0.9) -> bool:
    if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
        return False

    sample = series.dropna().sample(n=min(sample_size, len(series)), random_state=42)

    parsed = pd.to_datetime(sample, errors="coerce", utc=True)
    success_rate = parsed.notna().mean()
    
    return success_rate >= threshold

@st.cache_data
def get_metrical_columns(df: pd.DataFrame, min_unique: int = 20) -> list[str]:
    """Return columns that are numeric or datetime-like and have enough unique values."""
    result = []
    for col in df.columns:
        series = df[col]
        if (
            pd.api.types.is_numeric_dtype(series) or _looks_like_timestamp(series)
        ) and series.nunique(dropna=True) >= min_unique:
            result.append(col)
    return result

@st.cache_data
def get_categorical_columns(df: pd.DataFrame, max_unique: int = 20) -> list[str]:
    """Return columns that are object or low-cardinality numeric, excluding datetime-like."""
    result = []
    for col in df.columns:
        series = df[col]
        if _looks_like_timestamp(series):
            continue
        
        unique_count = series.nunique(dropna=True)
        if series.dtype == "object" or (pd.api.types.is_numeric_dtype(series) and unique_count < max_unique):
            result.append(col)
    return result