import streamlit as st
from data.constants import (
    HEALTHY_ONLY,
    USE_SUBSET,
    MIN_SAMPLES,
    MAX_SAMPLES,
    DEFAULT_SAMPLES
)

def sidebar():
    st.sidebar.title("Filter Options")

    use_subset = st.sidebar.checkbox(
        "Use Subset (faster)",
        value=USE_SUBSET
    )
    if use_subset:
        num_samples = st.sidebar.slider(
            label="How many samples",
            value=DEFAULT_SAMPLES,
            min_value=MIN_SAMPLES,
            max_value=MAX_SAMPLES,
            step=1000
        )
    else:
        num_samples = None

    healthy_only = st.sidebar.checkbox("Only use Healthy Data", value=HEALTHY_ONLY)

    # Add more filters later as needed (date range, passenger count, etc.)

    return {
        "num_samples": num_samples,
        "healthy_only": healthy_only,
    }