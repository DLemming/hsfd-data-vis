import streamlit as st
import pandas as pd
from data.constants import CUSTOM_CHECKS

@st.cache_data
def compute_health(df: pd.DataFrame, columns: list[str]=None) -> dict:
    total = len(df)
    checks = []

    if columns is None:
        columns = df.columns

    for col in columns:
        if CUSTOM_CHECKS and col in CUSTOM_CHECKS:
            checks.append(CUSTOM_CHECKS[col](df[col]))
        else:
            checks.append(df[col].notnull())

    if not checks:
        healthy_mask = pd.Series([False] * total, index=df.index)
    else:
        healthy_mask = checks[0]
        for mask in checks[1:]:
            healthy_mask &= mask

    healthy_count = healthy_mask.sum()
    unhealthy_count = total - healthy_count
    healthy_percent = round(100 * healthy_count / total, 2) if total else 0

    return {
        "healthy_mask": healthy_mask,
        "healthy_count": healthy_count,
        "unhealthy_count": unhealthy_count,
        "healthy_percent": healthy_percent,
        "total": total
    }

def get_healthy_mask(df: pd.DataFrame) -> pd.Series:
    result = compute_health(df)
    return result["healthy_mask"]