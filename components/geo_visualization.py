import streamlit as st
import pandas as pd
import pydeck as pdk

def plot_geo_visualization(df):
    required = {'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'}
    if not required.issubset(df.columns):
        st.error("Es fehlen erforderliche Spalten für die Geo-Visualisierung.")
        return

    # (0,0)-Werte filtern
    df_pickup = df[(df['pickup_longitude'] != 0) & (df['pickup_latitude'] != 0)].copy()
    df_dropoff = df[(df['dropoff_longitude'] != 0) & (df['dropoff_latitude'] != 0)].copy()

    # Umbenennen für Pydeck
    df_pickup = df_pickup.rename(columns={'pickup_latitude': 'lat', 'pickup_longitude': 'lon'})[['lat', 'lon']]
    df_dropoff = df_dropoff.rename(columns={'dropoff_latitude': 'lat', 'dropoff_longitude': 'lon'})[['lat', 'lon']]

    option = st.radio("Welche Punkte sollen angezeigt werden?", ("Abholorte", "Zielorte", "Beides"))

    layers = []

    if option in ("Abholorte", "Beides"):
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=df_pickup,
            get_position='[lon, lat]',
            get_fill_color='[0, 102, 255, 160]',  # Blau mit Transparenz
            get_radius=15,
            pickable=False
        ))

    if option in ("Zielorte", "Beides"):
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=df_dropoff,
            get_position='[lon, lat]',
            get_fill_color='[255, 0, 0, 160]',  # Rot mit Transparenz
            get_radius=15,
            pickable=False
        ))

    # Mittelwert-Zentrum für Kamera
    center_lat = pd.concat([df_pickup['lat'], df_dropoff['lat']]).mean()
    center_lon = pd.concat([df_pickup['lon'], df_dropoff['lon']]).mean()

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v10",
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=11,
            pitch=0
        ),
        layers=layers
    ))