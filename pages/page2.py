import streamlit as st
from components.bivariate import plot_corr_heatmap
from components.sidebar import sidebar
from logic.data_loader import load_taxi_data

filters = sidebar()
df = load_taxi_data(filters["use_full_data"], filters["healthy_only"])

plot_corr_heatmap(df)

st.title('Prediction of Fare Amount using Regression Model')