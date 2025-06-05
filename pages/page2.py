import streamlit as st
from components.bivariate import plot_corr_heatmap
from logic.data_loader import load_taxi_data

df = load_taxi_data()

plot_corr_heatmap(df)

st.title('Prediction of Fare Amount using Regression Model')