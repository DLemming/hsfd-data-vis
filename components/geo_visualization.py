import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy
import time

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
        df_pickup = df_day.rename(columns={
            'pickup_latitude': 'lat',
            'pickup_longitude': 'lon'
        })[['lat', 'lon']]

        df_dropoff = df_day.rename(columns={
            'dropoff_latitude': 'lat',
            'dropoff_longitude': 'lon'
        })[['lat', 'lon']]

        use_heatmap = st.checkbox("Heatmap anzeigen", value=False)

        if use_heatmap:
            df_all = pd.concat([df_pickup, df_dropoff])
            layers.append(pdk.Layer(
                "HeatmapLayer",
                data=df_all,
                get_position='[lon, lat]',
                aggregation='"MEAN"',
                get_weight=1,
                radiusPixels=15,
                intensity=0.4,
                threshold=0.03
)
)
        else:
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

def plot_trip_animation(df):
    required = {
        'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude',
        'tpep_pickup_datetime', 'tpep_dropoff_datetime'
    }

    if not required.issubset(df.columns):
        st.error("CSV muss folgende Spalten enthalten: pickup/dropoff Koordinaten + Zeiten.")
        return

    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
    df = df.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])

    df['date'] = df['tpep_pickup_datetime'].dt.date
    unique_dates = sorted(df['date'].unique())
    selected_date = st.selectbox("Datum auswählen", unique_dates)

    df_day = df[df['date'] == selected_date].copy()
    df_day = df_day[
        (df_day['pickup_latitude'] != 0) &
        (df_day['pickup_longitude'] != 0) &
        (df_day['dropoff_latitude'] != 0) &
        (df_day['dropoff_longitude'] != 0)
    ]

    day_start = pd.Timestamp(f"{selected_date}")
    df_day['start_ts'] = (df_day['tpep_pickup_datetime'] - day_start).dt.total_seconds()
    df_day['end_ts'] = (df_day['tpep_dropoff_datetime'] - day_start).dt.total_seconds()

    df_day['path'] = df_day.apply(lambda row: [
        [row['pickup_longitude'], row['pickup_latitude']],
        [row['dropoff_longitude'], row['dropoff_latitude']]
    ], axis=1)
    df_day['timestamps'] = df_day.apply(lambda row: [row['start_ts'], row['end_ts']], axis=1)

    df_viz = df_day[['path', 'timestamps']]

    # Session-State initialisieren
    if 'current_time' not in st.session_state:
        st.session_state.current_time = 0
    if 'playing' not in st.session_state:
        st.session_state.playing = False

    max_time = int(df_day['end_ts'].max())

    # Layout: Spalten für Steuerung
    col1, col2 = st.columns([1, 5])
    if col1.button("▶️ Start" if not st.session_state.playing else "⏸️ Pause"):
        st.session_state.playing = not st.session_state.playing

    # Slider manuell steuerbar, aber auch während Autoplay sichtbar
    st.session_state.current_time = col2.slider(
        "Aktuelle Zeit (Sekunden seit Mitternacht)", 
        min_value=0, 
        max_value=max_time, 
        value=st.session_state.current_time,
        step=60
    )

    # Kartenmittelpunkt berechnen
    center_lat = df_day['pickup_latitude'].mean()
    center_lon = df_day['pickup_longitude'].mean()

    trips_layer = pdk.Layer(
        "TripsLayer",
        data=df_viz,
        get_path="path",
        get_timestamps="timestamps",
        get_color=[253, 128, 93],
        opacity=0.8,
        width_min_pixels=2,
        rounded=True,
        trail_length=600,
        current_time=st.session_state.current_time
    )

    # Karte rendern
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/dark-v10',
        layers=[trips_layer],
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=12,
            pitch=45
        )
    ))

    # Autoplay-Loop
    if st.session_state.playing:
        time.sleep(0.1)  # leichtes Delay für realistische Wiedergabe
        st.session_state.current_time += 60  # 60 Sekunden pro Frame
        if st.session_state.current_time > max_time:
            st.session_state.current_time = 0
        st.rerun() 
import streamlit as st
import pandas as pd
import pydeck as pdk

def plot_tip_heatmap(df):
    required = {
        'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude',
        'tip_amount'
    }

    if not required.issubset(df.columns):
        st.error("CSV muss 'pickup/dropoff_longitude', 'pickup/dropoff_latitude' und 'tip_amount' enthalten.")
        return

    st.subheader("💵 Trinkgeld-Heatmap")

    # Auswahl: Pickup oder Dropoff
    location_type = st.radio(
        "Wähle Standortbasis für Heatmap:",
        options=["Pickup", "Dropoff"],
        index=0
    )

    # Optional: Normierung
    normalize = st.checkbox("Normieren nach Fahrtpreis (Tip %)", value=False)

    if normalize and 'fare_amount' in df.columns:
        df = df[df['fare_amount'] > 0]
        df['tip_ratio'] = df['tip_amount'] / df['fare_amount']
        df = df[df['tip_ratio'] <= 1.0]  # unrealistische Ausreißer filtern
        df['weight'] = df['tip_ratio']
    else:
        df = df[df['tip_amount'] >= 0]
        df['weight'] = df['tip_amount']

    # Koordinaten auswählen
    if location_type == "Pickup":
        df = df[
            (df['pickup_longitude'] != 0) & (df['pickup_latitude'] != 0)
        ].copy()
        df['lon'] = df['pickup_longitude']
        df['lat'] = df['pickup_latitude']
    else:
        df = df[
            (df['dropoff_longitude'] != 0) & (df['dropoff_latitude'] != 0)
        ].copy()
        df['lon'] = df['dropoff_longitude']
        df['lat'] = df['dropoff_latitude']

    # Zentrum bestimmen
    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()

    # HeatmapLayer bauen
    layer = pdk.Layer(
        "HeatmapLayer",
        data=df,
        get_position='[lon, lat]',
        get_weight="weight",
        radiusPixels=40,
        intensity=0.6,
        threshold=0.1
    )

    view = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=11,
        pitch=40
    )

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v10",
        initial_view_state=view,
        layers=[layer],
        tooltip={"text": "Trinkgeld"}
    ))
