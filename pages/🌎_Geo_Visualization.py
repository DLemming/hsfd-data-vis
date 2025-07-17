import streamlit as st
from components.geo_visualization import (
    plot_geo_visualization,
    plot_trip_animation,
    plot_tip_heatmap,
    plot_anomaly_trips,
    plot_traffic_congestion,
    plot_direction_rose,
    plot_zone_density_heatmap,
    plot_taxi_sinkholes
)
from components.sidebar import sidebar
from logic.data_loader import load_taxi_data

# Load sidebar filters
df = load_taxi_data(sidebar())

st.title('🗺️ Geo Visualization of NYC Taxi Data')

tab_selection = st.radio(
    "🔍 Select a visualization:",
    options=[
        "🌍 Geo Map",
        "🎞️ Trip Animation",
        "💰 Tip Heatmap",
        "🚨 Anomalies",
        "🚦 Traffic",
        "🧭 Direction",
        "📊 Zone Density",
        "🕳️ Taxi Sinkholes"
    ],
    key="selected_tab",
    horizontal=True
)

# Display according to selected tab
if tab_selection == "🌍 Geo Map":
    st.subheader("🌍 Interactive Geo Visualization")
    plot_geo_visualization(df)

elif tab_selection == "🎞️ Trip Animation":
    st.subheader("🎞️ Trip Progress Over Time")
    plot_trip_animation(df)

elif tab_selection == "💰 Tip Heatmap":
    st.subheader("💰 Heatmap of Tip Amounts")
    plot_tip_heatmap(df)

elif tab_selection == "🚨 Anomalies":
    st.subheader("🚨 Suspicious Trips")
    plot_anomaly_trips(df)

elif tab_selection == "🚦 Traffic":
    st.subheader("🚦 Traffic Density / Congestion")
    plot_traffic_congestion(df)

elif tab_selection == "🧭 Direction":
    st.subheader("🧭 Directional Analysis (Rose Plot)")
    plot_direction_rose(df)

elif tab_selection == "📊 Zone Density":
    st.subheader("📊 Heatmap by Taxi Zones")
    plot_zone_density_heatmap(df)

elif tab_selection == "🕳️ Taxi Sinkholes":
    st.subheader("Taxi Sinkholes 🕳️ Net Gain/Loss by Region")
    plot_taxi_sinkholes(df)
