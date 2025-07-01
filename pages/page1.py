import streamlit as st
from components.geo_visualization import plot_geo_visualization
from components.geo_visualization import plot_trip_animation
from components.geo_visualization import plot_tip_heatmap
from components.geo_visualization import plot_anomaly_trips
from components.geo_visualization import  plot_traffic_congestion
from components.geo_visualization import  plot_direction_rose
from components.geo_visualization import plot_zone_density_heatmap
from components.sidebar import sidebar
from logic.data_loader import load_taxi_data

filters = sidebar()
df = load_taxi_data(filters["use_full_data"], filters["healthy_only"])

st.title('Geo visualization over time')

plot_geo_visualization(df)

plot_trip_animation(df)

plot_tip_heatmap(df)

plot_anomaly_trips(df)

plot_traffic_congestion(df)

plot_direction_rose(df)

plot_zone_density_heatmap(df)