import streamlit as st
from components.geo_visualization import plot_geo_visualization
from logic.data_loader import load_taxi_data

df = load_taxi_data()

st.title('Geo visualization over time')

plot_geo_visualization(df)
