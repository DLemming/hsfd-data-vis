import streamlit as st
from datetime import datetime
from data.constants import (
    HEALTHY_ONLY,
    USE_SUBSET,
    MIN_SAMPLES,
    MAX_SAMPLES,
    DEFAULT_SAMPLES
)

def sidebar():
    st.sidebar.title("Filter Options")

    # Teilmenge oder gesamte Datenmenge
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

    # Nur valide Fahrten (z. B. mit gültiger Geo-Koordinate)
    healthy_only = st.sidebar.checkbox("Only use Healthy Data", value=HEALTHY_ONLY)

    # Optionaler Datumsbereich
    enable_date_filter = st.sidebar.checkbox("Filter by Date Range")
    if enable_date_filter:
        date_range = st.sidebar.date_input(
            "Pickup Date Range",
            value=(datetime(2015, 1, 1), datetime(2015, 1, 31)),
            min_value=datetime(2015, 1, 1),
            max_value=datetime(2015, 12, 31)
        )
        pickup_date_start, pickup_date_end = date_range
    else:
        pickup_date_start, pickup_date_end = None, None

    # Optionaler Filter für Passagieranzahl
    enable_passenger_filter = st.sidebar.checkbox("Filter by Passenger Count")
    if enable_passenger_filter:
        passenger_count = st.sidebar.slider(
            "Passenger Count (min - max)",
            min_value=1,
            max_value=8,
            value=(1, 6)
        )
        passenger_count_min, passenger_count_max = passenger_count
    else:
        passenger_count_min, passenger_count_max = None, None

    return {
        "use_full_data": not use_subset,
        "num_samples": num_samples,
        "healthy_only": healthy_only,
        "pickup_date_start": pickup_date_start,
        "pickup_date_end": pickup_date_end,
        "passenger_count_min": passenger_count_min,
        "passenger_count_max": passenger_count_max,
    }
