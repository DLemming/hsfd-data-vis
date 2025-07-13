import streamlit as st

from logic.data_loader import load_taxi_data

from components.sidebar import sidebar
from components.health import render_health_component
from components.univariate import (
    render_bar_chart,
    render_boxplot,
    render_histogram,
    render_pie_chart
)
from components.bivariate import (
    plot_corr_heatmap,
    plot_bivariate_scatter
)


st.set_page_config(page_title="NYC Taxi Trips", page_icon="🚖", layout="wide")

df = load_taxi_data(sidebar())


st.title("Univariate Analysis")

# Top row
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Data Integrity")
    render_health_component(df)
with col2:
    st.subheader("Category Counts")
    render_bar_chart(df)
with col3:
    st.subheader("Category Proportions")
    render_pie_chart(df)

# Bottom row
col4, col5 = st.columns([2, 1])

with col4:
    st.subheader("Value Distribution")
    render_histogram(df)
with col5:
    st.subheader("Value Variability")
    render_boxplot(df)



st.markdown("---")
st.title("Bivariate Analysis")

col6, col7 = st.columns(2)

with col6:
    st.subheader("Correlation Heatmap")
    # st.markdown("<br><br><br>", unsafe_allow_html=True)
    selected_cols = plot_corr_heatmap(df)
with col7:
    st.subheader("Scatterplot (Metrical only)")
    plot_bivariate_scatter(df)

