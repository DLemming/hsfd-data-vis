import streamlit as st
from data.constants import (
    HEALTHY_ONLY,
    FULL_DATASET
)

def sidebar():
    st.sidebar.title("Filter Options")

    use_full_data = st.sidebar.checkbox(
        "Use full dataset",
        value=FULL_DATASET
    )

    healthy_only = st.sidebar.checkbox(
        "Only use healthy data",
        value=HEALTHY_ONLY
    )

    # Add more filters later as needed (date range, passenger count, etc.)

    return {
        "use_full_data": use_full_data,
        "healthy_only": healthy_only,
    }