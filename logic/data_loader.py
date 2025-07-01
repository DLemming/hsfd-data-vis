import pandas as pd
import streamlit as st
from logic.health import get_healthy_mask
from data.constants import SAMPLE_SIZE


@st.cache_data
def load_taxi_data(use_full_data: bool, healthy_only: bool):
    """
    Loads the taxi data from a csv file.
    Optionally filters down to a subset and/or selects only healthy data.
    """

    df = pd.read_csv(
        "data/data-small.csv",
        parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"]
    )
    df = df.drop(columns=["RatecodeID", "store_and_fwd_flag"])

    if not use_full_data:
        df = df.sample(n=SAMPLE_SIZE, random_state=42)

    if healthy_only:
        mask = get_healthy_mask(df)
        df = df[mask]

    return df