import streamlit as st
import pandas as pd
import pydeck as pdk

def plot_geo_visualization(df):
    required = {
        'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude',
        'tpep_pickup_datetime'
    }
    if not required.issubset(df.columns):
        st.error("CSV muss die Spalten 'pickup/dropoff_xx' und 'tpep_pickup_datetime' enthalten.")
        return

    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df = df.dropna(subset=['tpep_pickup_datetime'])

    # Tagesfilter
    min_date = df['tpep_pickup_datetime'].dt.date.min()
    max_date = df['tpep_pickup_datetime'].dt.date.max()
    selected_date = st.slider("Datum auswählen (24h-Zeitraum)", min_value=min_date, max_value=max_date, value=min_date)

    start = pd.Timestamp(selected_date)
    end = start + pd.Timedelta(days=1)
    df_day = df[(df['tpep_pickup_datetime'] >= start) & (df['tpep_pickup_datetime'] < end)].copy()

    # (0,0)-Filter
    df_day = df_day[
        (df_day['pickup_longitude'] != 0) & (df_day['pickup_latitude'] != 0) &
        (df_day['dropoff_longitude'] != 0) & (df_day['dropoff_latitude'] != 0)
    ].reset_index(drop=True)

    # Toggle für Modus
    show_all = st.checkbox("Alle Punkte anzeigen (Pickup + Dropoff)", value=True)

    layers = []

    if show_all:
        # Alle Pickups
        df_pickup = df_day.rename(columns={
            'pickup_latitude': 'lat',
            'pickup_longitude': 'lon'
        })[['lat', 'lon']]

        df_dropoff = df_day.rename(columns={
            'dropoff_latitude': 'lat',
            'dropoff_longitude': 'lon'
        })[['lat', 'lon']]

        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=df_pickup,
            get_position='[lon, lat]',
            get_fill_color='[0, 102, 255, 160]',
            get_radius=10
        ))

        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=df_dropoff,
            get_position='[lon, lat]',
            get_fill_color='[255, 0, 0, 160]',
            get_radius=10
        ))

        center_lat = pd.concat([df_pickup['lat'], df_dropoff['lat']]).mean()
        center_lon = pd.concat([df_pickup['lon'], df_dropoff['lon']]).mean()
    else:
        # Fahrt-Auswahl
        trip_idx = st.selectbox(
            "Fahrt auswählen (Pickup → Dropoff)",
            options=df_day.index,
            format_func=lambda i: f"{df_day['tpep_pickup_datetime'][i]} | Distanz: {df_day['trip_distance'][i]} mi"
        )

        pickup = df_day.loc[trip_idx, ['pickup_latitude', 'pickup_longitude']]
        dropoff = df_day.loc[trip_idx, ['dropoff_latitude', 'dropoff_longitude']]

        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame([{"lat": pickup['pickup_latitude'], "lon": pickup['pickup_longitude']}]),
            get_position='[lon, lat]',
            get_fill_color='[0, 102, 255, 200]',
            get_radius=10
        ))

        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame([{"lat": dropoff['dropoff_latitude'], "lon": dropoff['dropoff_longitude']}]),
            get_position='[lon, lat]',
            get_fill_color='[255, 0, 0, 200]',
            get_radius=10
        ))

        layers.append(pdk.Layer(
            "ArcLayer",
            data=pd.DataFrame([{
                "from_lat": pickup['pickup_latitude'],
                "from_lon": pickup['pickup_longitude'],
                "to_lat": dropoff['dropoff_latitude'],
                "to_lon": dropoff['dropoff_longitude']
            }]),
            get_source_position='[from_lon, from_lat]',
            get_target_position='[to_lon, to_lat]',
            get_width=4,
            get_source_color='[0, 150, 255]',
            get_target_color='[255, 0, 0]',
            pickable=False
        ))

        center_lat = (pickup['pickup_latitude'] + dropoff['dropoff_latitude']) / 2
        center_lon = (pickup['pickup_longitude'] + dropoff['dropoff_longitude']) / 2

    # Karte anzeigen
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v10",
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=12,
            pitch=0
        ),
        layers=layers
    ))

    