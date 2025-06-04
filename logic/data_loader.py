import pandas as pd
import streamlit as st

@st.cache_data
def load_taxi_data():
    """Lädt die Taxi-Daten aus einer CSV-Datei und gibt die ersten nrows zurück."""
    return pd.read_csv(r"data/data-small.csv")
