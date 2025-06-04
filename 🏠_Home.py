import streamlit as st

from logic.data_loader import load_taxi_data

from components.passenger_count import show_passenger_count_bar_chart
from components.health import render_health_component
from components.univariate import (
    render_bar_chart,
    render_boxplot,
    render_histogram,
    render_pie_chart
)

st.set_page_config(page_title="NYC Taxi Trips", page_icon="🚖", layout="wide")
#st.title('NYC Yellow Taxi Trip Data')

df = load_taxi_data()

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

#show_passenger_count_bar_chart(df)
