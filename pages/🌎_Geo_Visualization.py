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

# Sidebar-Filter laden
df = load_taxi_data(sidebar())

st.title('🗺️ Geo-Visualisierung NYC Taxi Daten')

tab_selection = st.radio(
    "🔍 Wähle eine Visualisierung:",
    options=[
        "🌍 Geo Map",
        "🎞️ Fahrt-Animation",
        "💰 Tip Heatmap",
        "🚨 Anomalien",
        "🚦 Verkehr",
        "🧭 Fahrtrichtung",
        "📊 Zonen-Dichte",
        "🧪 Test"
    ],
    key="selected_tab",
    horizontal=True
)


# Entsprechend der Auswahl anzeigen
if tab_selection == "🌍 Geo Map":
    st.subheader("🌍 Interaktive Geo-Visualisierung")
    plot_geo_visualization(df)

elif tab_selection == "🎞️ Fahrt-Animation":
    st.subheader("🎞️ Fahrtverlauf über Zeit")
    plot_trip_animation(df)

elif tab_selection == "💰 Tip Heatmap":
    st.subheader("💰 Heatmap der Tip-Beträge")
    plot_tip_heatmap(df)

elif tab_selection == "🚨 Anomalien":
    st.subheader("🚨 Auffällige Fahrten")
    plot_anomaly_trips(df)

elif tab_selection == "🚦 Verkehr":
    st.subheader("🚦 Verkehrsdichte / Stau")
    plot_traffic_congestion(df)

elif tab_selection == "🧭 Fahrtrichtung":
    st.subheader("🧭 Richtungsanalyse (Rose Plot)")
    plot_direction_rose(df)

elif tab_selection == "📊 Zonen-Dichte":
    st.subheader("📊 Heatmap nach Taxi-Zonen")
    plot_zone_density_heatmap(df)

elif tab_selection == "🧪 Test":
    st.subheader("🧪 Test – Pickup Tower")
    st.info("Dies ist der Test-Tab. Sollte immer sichtbar sein – unabhängig von der Datenlage.")
    plot_taxi_sinkholes(df)
