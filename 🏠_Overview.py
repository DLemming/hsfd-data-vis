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

filters = sidebar()
df = load_taxi_data(filters["use_full_data"], filters["healthy_only"])

# Top row
col1, col2 = st.columns([1,2])
with col1:
    st.markdown("### Data Integrity")
    render_health_component(df)
with col2:
    st.markdown("### Histogram")
    render_histogram(df)

# Bottom row
col3, col4, col5 = st.columns(3)
with col3:
    st.markdown("### Bar Chart")
    render_bar_chart(df)
with col4:
    st.markdown("### Pie Chart")
    render_pie_chart(df)
with col5:
    st.markdown("### Box Plot")
    render_boxplot(df)



st.markdown("---")
st.title("Bivariate Analysis")

col6, col7 = st.columns(2)

with col6:
    st.subheader("Correlation Heatmap")
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    plot_corr_heatmap(df)
with col7:
    st.subheader("Scatterplot")
    plot_bivariate_scatter(df)

