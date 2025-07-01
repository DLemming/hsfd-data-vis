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

# Sidebar mit Filter
filters = sidebar()
df = load_taxi_data(filters["use_full_data"], filters["healthy_only"])

st.title('🗺️ Geo-Visualisierung NYC Taxi Daten')

# Tabs definieren
tabs = st.tabs([
    "🌍 Geo Map",
    "🎞️ Fahrt-Animation",
    "💰 Tip Heatmap",
    "🚨 Anomalien",
    "🚦 Verkehr",
    "🧭 Fahrtrichtung",
    "📊 Zonen-Dichte",
    "Test"
])

with tabs[0]:
    st.subheader("🌍 Interaktive Geo-Visualisierung")
    plot_geo_visualization(df)

with tabs[1]:
    st.subheader("🎞️ Fahrtverlauf über Zeit")
    plot_trip_animation(df)

with tabs[2]:
    st.subheader("💰 Heatmap der Tip-Beträge")
    plot_tip_heatmap(df)

with tabs[3]:
    st.subheader("🚨 Auffällige Fahrten")
    plot_anomaly_trips(df)

with tabs[4]:
    st.subheader("🚦 Verkehrsdichte / Stau")
    plot_traffic_congestion(df)

with tabs[5]:
    st.subheader("🧭 Richtungsanalyse (Rose Plot)")
    plot_direction_rose(df)

with tabs[6]:
    st.subheader("📊 Heatmap nach Taxi-Zonen")
    plot_zone_density_heatmap(df)

with tabs[7]:
    st.subheader("🧪 Test – Pickup Tower")
    st.info("Dies ist der Test-Tab. Sollte immer sichtbar sein – unabhängig von der Datenlage.")
    plot_taxi_sinkholes(df)

