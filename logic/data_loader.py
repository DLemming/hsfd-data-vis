import pandas as pd
import streamlit as st

@st.cache_data
def load_taxi_data():
    """Lädt die Taxi-Daten aus einer CSV-Datei und gibt die ersten nrows zurück."""
    return pd.read_csv(r"data/data-small.csv")

@st.cache_data
def filter_valid(df: pd.DataFrame):
    """Entfernt alle korrupten Datenpunkte"""
    df = df.drop(columns=['RatecodeID'])
    df = df.dropna()
    df = df[
        df['pickup_latitude'].between(40.5, 41.0) &
        df['pickup_longitude'].between(-74.3, -73.6) &
        df['dropoff_latitude'].between(40.5, 41.0) &
        df['dropoff_longitude'].between(-74.3, -73.6)
    ]
    df = df[df['passenger_count'] > 0]
    df = df[df['fare_amount'] > 0]
    return df