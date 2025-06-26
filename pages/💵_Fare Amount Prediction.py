import streamlit as st
from logic.data_loader import load_taxi_data, filter_valid
from logic.regression import fit_regression_model
from components.regression import (
    show_regression_inputs,
    plot_regression,
    plot_all_vars,
    plot_3d_regression_surface
)

st.set_page_config(layout="wide")

df = load_taxi_data()
df = filter_valid(df)

# Show simple pickup map
model, df = fit_regression_model(df)

show_regression_inputs(model, df)


st.subheader("Regression Model Analysis")
plot_regression(model, df)

plot_all_vars(model, df)

plot_3d_regression_surface(model, df)