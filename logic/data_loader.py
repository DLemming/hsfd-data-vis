import pandas as pd
import streamlit as st
from logic.health import get_healthy_mask


@st.cache_data
def load_taxi_data(filters: dict):
    """
    Loads the taxi data from a csv file.
    Optionally filters down to a subset and/or selects only healthy data.
    """
    num_samples = filters["num_samples"]
    healthy_only = filters["healthy_only"]

    df = pd.read_csv(
        "data/data-small.csv",
        parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"]
    )
    df = df.drop(columns=["RatecodeID", "store_and_fwd_flag"])

    if num_samples is not None:
        n = min(num_samples, len(df))
        df = df.sample(n=n, random_state=42)

    if healthy_only:
        mask = get_healthy_mask(df)
        df = df[mask]

    # NEU: Passenger Count Filter anwenden
    passenger_count_min = filters.get("passenger_count_min")
    passenger_count_max = filters.get("passenger_count_max")
    if passenger_count_min is not None and passenger_count_max is not None:
        df = df[
            (df["passenger_count"] >= passenger_count_min) &
            (df["passenger_count"] <= passenger_count_max)
        ]

    if len(df) == 0:
        st.warning("Warning: no rows match your filter criteria")
    return df
