import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes
from sklearn.cluster import KMeans

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
            ))
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

        # Sicherstellen, dass session_state initialisiert ist
    if 'animation_running' not in st.session_state:
        st.session_state.animation_running = False
    if 'current_time' not in st.session_state:
        st.session_state.current_time = 0
    if 'playing' not in st.session_state:
        st.session_state.playing = False

    col1, col2 = st.columns([1, 5])
    if col1.button("▶️ Start" if not st.session_state.playing else "⏸️ Pause", key="trip_toggle_button"):
        st.session_state.playing = not st.session_state.playing

    max_time = int(df_day['end_ts'].max())

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

def plot_tip_heatmap(df):
    required = {
        'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude',
        'tip_amount', 'tpep_pickup_datetime'
    }

    if not required.issubset(df.columns):
        st.error("CSV muss 'pickup/dropoff_longitude', 'pickup/dropoff_latitude', 'tip_amount' und 'tpep_pickup_datetime' enthalten.")
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

    df['fare_amount'] = pd.to_numeric(df['fare_amount'], errors='coerce')
    df['tip_amount'] = pd.to_numeric(df['tip_amount'], errors='coerce')

    if normalize and 'fare_amount' in df.columns:
        df = df[df['fare_amount'] > 0]
        df['tip_ratio'] = df['tip_amount'] / df['fare_amount']
        df = df[df['tip_ratio'] <= 1.0]  # unrealistische Ausreißer filtern
        df['weight'] = df['tip_ratio']
    else:
        df = df[df['tip_amount'] >= 0]
        df['weight'] = df['tip_amount']

    # Zeitfilterung (optional)
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df = df.dropna(subset=['tpep_pickup_datetime'])
    filter_by_time = st.checkbox("Nach Uhrzeit filtern?", value=False)
    
    if filter_by_time:
        selected_hour = st.slider("Uhrzeit auswählen", 0, 23, 8, key="congestion_static_hour")
        df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
        df = df[df['pickup_hour'] == selected_hour]

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

    if df.empty:
        st.warning("Keine Daten nach den aktuellen Filtern.")
        return

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

def plot_anomaly_trips(df):
    required = {
        'pickup_latitude', 'pickup_longitude',
        'dropoff_latitude', 'dropoff_longitude',
        'trip_distance', 'tpep_pickup_datetime', 'tpep_dropoff_datetime'
    }

    if not required.issubset(df.columns):
        st.error("CSV muss pickup/dropoff-Koordinaten, trip_distance und Zeitstempel enthalten.")
        return

    st.subheader("🚨 Verdächtig langsame Fahrten")

    df = df.copy()
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')

    # Dauer und Geschwindigkeit berechnen
    df['duration_hr'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 3600
    df['speed_kmh'] = df['trip_distance'] / df['duration_hr']

    # Nur gültige Werte behalten
    df = df[
        (df['duration_hr'] > 0) &
        (df['trip_distance'] > 0) &
        (df['speed_kmh'] > 0)
    ]

    # Schwellenwert für langsame Fahrten
    speed_limit = st.slider("Grenze für Stau (km/h)", 1, 30, 10, key="stau_speed_limit")


    df_anomaly = df[df['speed_kmh'] < speed_limit].copy()

    if df_anomaly.empty:
        st.warning("Keine verdächtigen Fahrten mit diesen Kriterien gefunden.")
        return

    st.write(f"🚩 {len(df_anomaly)} verdächtige Fahrten bei Geschwindigkeit < {speed_limit} km/h")

    # Spalten für ArcLayer vorbereiten
    df_anomaly['from_lat'] = df_anomaly['pickup_latitude']
    df_anomaly['from_lon'] = df_anomaly['pickup_longitude']
    df_anomaly['to_lat'] = df_anomaly['dropoff_latitude']
    df_anomaly['to_lon'] = df_anomaly['dropoff_longitude']
    df_anomaly['pickup_time'] = df_anomaly['tpep_pickup_datetime'].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Layer: Pickup-Punkte
    point_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_anomaly,
        get_position='[from_lon, from_lat]',
        get_radius=40,
        get_fill_color='[255, 0, 0, 160]',
        pickable=True
    )

    # Layer: Fahrtverlauf
    arc_layer = pdk.Layer(
        "ArcLayer",
        data=df_anomaly,
        get_source_position='[from_lon, from_lat]',
        get_target_position='[to_lon, to_lat]',
        get_source_color='[255, 0, 0]',
        get_target_color='[255, 0, 0]',
        get_width=3,
        pickable=True
    )

    # Karten-Zentrum
    center_lat = df_anomaly['from_lat'].mean()
    center_lon = df_anomaly['from_lon'].mean()

    # Karte anzeigen mit Tooltip
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v10",
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=11,
            pitch=40
        ),
        layers=[arc_layer, point_layer],
        tooltip={
            "html": """
                <b>Pickup:</b> {pickup_time}<br/>
                <b>Speed:</b> {speed_kmh} km/h<br/>
                <b>Distanz:</b> {trip_distance} mi
            """,
            "style": {
                "backgroundColor": "rgba(255, 0, 0, 0.8)",
                "color": "white"
            }
        }
    ))

def plot_traffic_congestion(df):
    required = {
        'tpep_pickup_datetime', 'tpep_dropoff_datetime',
        'trip_distance', 'pickup_latitude', 'pickup_longitude'
    }

    if not required.issubset(df.columns):
        st.error("CSV muss 'pickup_lat/lon', 'tpep_pickup/dropoff_datetime' und 'trip_distance' enthalten.")
        return

    st.subheader("🚦 Verkehrsstaus erkennen (Heatmap nach Uhrzeit)")

    # Umwandlung in Timestamps
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
    df = df.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])

    # Dauer & Geschwindigkeit berechnen
    df['duration_hr'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 3600
    df = df[
        (df['duration_hr'] > 0) &
        (df['trip_distance'] > 0)
    ].copy()

    df['speed_kmh'] = df['trip_distance'] / df['duration_hr']
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour

    # Uhrzeitfilter
    selected_hour = st.slider("Uhrzeit auswählen", 0, 23, 8)
    df_hour = df[df['pickup_hour'] == selected_hour]

    # Geschwindigkeitsgrenze für "Stau"
    speed_threshold = st.slider("Grenze für Stau (km/h)", 1, 30, 10)
    df_congestion = df_hour[df_hour['speed_kmh'] <= speed_threshold].copy()

    if df_congestion.empty:
        st.warning("Keine Staufahrten für diese Uhrzeit und Schwelle gefunden.")
        return

    # Zentrum der Karte
    center_lat = df_congestion['pickup_latitude'].mean()
    center_lon = df_congestion['pickup_longitude'].mean()

    # HeatmapLayer auf Pickup-Punkte
    congestion_layer = pdk.Layer(
        "HeatmapLayer",
        data=df_congestion,
        get_position='[pickup_longitude, pickup_latitude]',
        get_weight=1,
        radiusPixels=50,
        intensity=0.7,
        threshold=0.05
    )

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v10",
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=11,
            pitch=30
        ),
        layers=[congestion_layer],
        tooltip={"text": "Staubereich: Geschwindigkeit < {} km/h".format(speed_threshold)}
    ))

def plot_direction_rose(df):
    required = {
        'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude',
        'tpep_pickup_datetime'
    }

    if not required.issubset(df.columns):
        st.error("CSV muss 'pickup/dropoff_xx' und 'tpep_pickup_datetime' enthalten.")
        return

    df = df.copy()
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df = df.dropna(subset=['tpep_pickup_datetime'])

    # Uhrzeitfilter
    selected_hour = st.slider("Uhrzeit auswählen (Pickup)", 0, 23, 8)
    df['hour'] = df['tpep_pickup_datetime'].dt.hour
    df = df[df['hour'] == selected_hour]

    # Filter (0,0)
    df = df[
        (df['pickup_latitude'] != 0) & (df['pickup_longitude'] != 0) &
        (df['dropoff_latitude'] != 0) & (df['dropoff_longitude'] != 0)
    ]

    if df.empty:
        st.warning("Keine gültigen Fahrten für diese Stunde.")
        return

    # Richtungswinkel berechnen (in Grad)
    dx = df['dropoff_longitude'] - df['pickup_longitude']
    dy = df['dropoff_latitude'] - df['pickup_latitude']
    angles = np.degrees(np.arctan2(dy, dx))
    angles = (angles + 360) % 360  # in [0, 360)

    # Windrose: Histogramm der Richtungen
    bins = np.arange(0, 360 + 22.5, 22.5)  # 16 Sektoren
    counts, _ = np.histogram(angles, bins=bins)
    angles_mid = np.radians(bins[:-1] + 11.25)  # Mitte der Sektoren

    # Plot
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw={'projection': 'polar'})
    ax.bar(angles_mid, counts, width=np.radians(22.5), bottom=0.0, color='dodgerblue', alpha=0.75, edgecolor='black')

    ax.set_theta_zero_location('N')  # Norden oben
    ax.set_theta_direction(-1)       # Uhrzeigersinn
    ax.set_title(f"🧭 Fahrtrichtung um {selected_hour}:00 Uhr", va='bottom')
    st.pyplot(fig)


def plot_zone_density_heatmap(df):
    required = {
        'pickup_latitude', 'pickup_longitude',
        'dropoff_latitude', 'dropoff_longitude',
        'tpep_pickup_datetime'
    }

    if not required.issubset(df.columns):
        st.error("CSV muss pickup/dropoff Koordinaten und 'tpep_pickup_datetime' enthalten.")
        return

    st.subheader("🔥 Zonen mit den meisten Fahrten (Heatmap, geclustert)")

    df = df.copy()
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df = df.dropna(subset=['tpep_pickup_datetime'])

    # Tagesfilter
    min_date = df['tpep_pickup_datetime'].dt.date.min()
    max_date = df['tpep_pickup_datetime'].dt.date.max()
    if min_date == max_date:
        selected_date = min_date
        st.info(f"Nur ein Datum vorhanden: {min_date}")
    else:
        selected_date = st.slider("Datum auswählen (24h)", min_value=min_date, max_value=max_date, value=min_date)

    start = pd.Timestamp(selected_date)
    end = start + pd.Timedelta(days=1)
    df = df[(df['tpep_pickup_datetime'] >= start) & (df['tpep_pickup_datetime'] < end)]

    # Auswahl: Pickup vs Dropoff
    mode = st.radio("Welche Orte zeigen?", options=["Pickup", "Dropoff"], horizontal=True)

    if mode == "Pickup":
        coords = df[['pickup_latitude', 'pickup_longitude']].dropna()
        coords.columns = ['lat', 'lon']
    else:
        coords = df[['dropoff_latitude', 'dropoff_longitude']].dropna()
        coords.columns = ['lat', 'lon']

    # Nur gültige Koordinaten
    coords = coords[(coords['lat'] != 0) & (coords['lon'] != 0)]

    if coords.empty:
        st.warning("Keine gültigen Koordinaten.")
        return

    # Slider für Anzahl Cluster
    max_possible_clusters = min(10, len(coords))  # Safety limit
    n_clusters = st.slider("Anzahl Cluster", min_value=5, max_value=max_possible_clusters, value=100)

    # Clustering mit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    coords['cluster'] = kmeans.fit_predict(coords[['lat', 'lon']])
    coords['cluster_lat'] = kmeans.cluster_centers_[coords['cluster'], 0]
    coords['cluster_lon'] = kmeans.cluster_centers_[coords['cluster'], 1]

    # Cluster-Zentren und Häufigkeit
    zone_counts = coords.groupby(['cluster_lat', 'cluster_lon']).size().reset_index(name='weight')
    zone_counts.rename(columns={'cluster_lat': 'lat', 'cluster_lon': 'lon'}, inplace=True)

    center_lat = zone_counts['lat'].mean()
    center_lon = zone_counts['lon'].mean()

    layer = pdk.Layer(
        "HeatmapLayer",
        data=zone_counts,
        get_position='[lon, lat]',
        get_weight="weight",
        radiusPixels=40,
        intensity=0.6,
        threshold=0.05
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
        tooltip={"text": "Fahrten: {weight}"}
    ))

def plot_taxi_sinkholes(df):
    required = {
        'pickup_latitude', 'pickup_longitude',
        'dropoff_latitude', 'dropoff_longitude'
    }

    if not required.issubset(df.columns):
        st.error("CSV muss pickup/dropoff Koordinaten enthalten.")
        return

    st.subheader("🕳️ Taxi Sinkholes – Netto-Gewinn/Verlust nach Region")

    df = df.copy()
    df = df[
        (df['pickup_latitude'] != 0) & (df['pickup_longitude'] != 0) &
        (df['dropoff_latitude'] != 0) & (df['dropoff_longitude'] != 0)
    ]

    # Runden auf Gitterzellen
    precision = 3  # ca. ~100m Raster
    df['pickup_lat'] = df['pickup_latitude'].round(precision)
    df['pickup_lon'] = df['pickup_longitude'].round(precision)
    df['dropoff_lat'] = df['dropoff_latitude'].round(precision)
    df['dropoff_lon'] = df['dropoff_longitude'].round(precision)

    # Zählung pro Zelle
    pickup_counts = df.groupby(['pickup_lat', 'pickup_lon']).size().reset_index(name='pickups')
    dropoff_counts = df.groupby(['dropoff_lat', 'dropoff_lon']).size().reset_index(name='dropoffs')

    pickup_counts = pickup_counts.rename(columns={'pickup_lat': 'lat', 'pickup_lon': 'lon'})
    dropoff_counts = dropoff_counts.rename(columns={'dropoff_lat': 'lat', 'dropoff_lon': 'lon'})

    # Merge pickup und dropoff counts
    merged = pd.merge(dropoff_counts, pickup_counts, on=['lat', 'lon'], how='outer').fillna(0)
    merged['net_flow'] = merged['dropoffs'] - merged['pickups']  # positive = Quelle, negativ = Senke

    if merged.empty:
        st.warning("Keine gültigen Daten nach Gruppierung.")
        return

    center_lat = merged['lat'].mean()
    center_lon = merged['lon'].mean()

    # Color Mapping: grün (Quelle) → rot (Senke)
    max_abs_flow = np.abs(merged['net_flow']).max()
    merged['color'] = merged['net_flow'].apply(lambda x: [0, 255, 0, 180] if x > 0 else [255, 0, 0, 180])

    layer = pdk.Layer(
        "ColumnLayer",
        data=merged,
        get_position='[lon, lat]',
        get_elevation='net_flow',
        elevation_scale=10,
        radius=70,
        get_fill_color='color',
        pickable=True,
        auto_highlight=True,
    )

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=11,
        pitch=50
    )

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v10",
        initial_view_state=view_state,
        layers=[layer],
        tooltip={"text": "📍 Netto: {net_flow} (Dropoffs - Pickups)"}
    ))