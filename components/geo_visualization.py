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
        'dropoff_longitude', 'dropoff_latitude'
    }
    if not required.issubset(df.columns):
        st.error("CSV must contain the columns 'pickup/dropoff_xx'.")
        return

    # Filter out (0,0) coordinates
    df = df[
        (df['pickup_longitude'] != 0) & (df['pickup_latitude'] != 0) &
        (df['dropoff_longitude'] != 0) & (df['dropoff_latitude'] != 0)
    ].reset_index(drop=True)

    # Heatmap toggle
    use_heatmap = st.checkbox("Show Heatmap", value=False)

    # Prepare data
    df_pickup = df.rename(columns={
        'pickup_latitude': 'lat',
        'pickup_longitude': 'lon'
    })[['lat', 'lon']]

    df_dropoff = df.rename(columns={
        'dropoff_latitude': 'lat',
        'dropoff_longitude': 'lon'
    })[['lat', 'lon']]

    layers = []

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
            get_radius=10,
            get_position='[lon, lat]'
        ))

    # Calculate map center
    center_lat = pd.concat([df_pickup['lat'], df_dropoff['lat']]).mean()
    center_lon = pd.concat([df_pickup['lon'], df_dropoff['lon']]).mean()

    # Show map
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
        st.error("CSV must contain the following columns: pickup/dropoff coordinates + timestamps.")
        return

    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
    df = df.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])

    df['date'] = df['tpep_pickup_datetime'].dt.date
    unique_dates = sorted(df['date'].unique())
    selected_date = st.selectbox("Select date", unique_dates)

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

    # Ensure session_state is initialized
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
        "Current Time (Seconds since midnight)", 
        min_value=0, 
        max_value=max_time, 
        value=st.session_state.current_time,
        step=60
    )

    # Calculate map center
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

    # Render map
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

    # Autoplay loop
    if st.session_state.playing:
        time.sleep(0.1)  # small delay for realistic playback
        st.session_state.current_time += 60  # 60 seconds per frame
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
        st.error("CSV must contain 'pickup/dropoff_longitude', 'pickup/dropoff_latitude', 'tip_amount', and 'tpep_pickup_datetime'.")
        return


    # Select: Pickup or Dropoff
    location_type = st.radio(
        "Select location basis for heatmap:",
        options=["Pickup", "Dropoff"],
        index=0
    )

    # Optional: Normalize
    normalize = st.checkbox("Normalize by fare amount (Tip %)", value=False)

    df['fare_amount'] = pd.to_numeric(df['fare_amount'], errors='coerce')
    df['tip_amount'] = pd.to_numeric(df['tip_amount'], errors='coerce')

    if normalize and 'fare_amount' in df.columns:
        df = df[df['fare_amount'] > 0]
        df['tip_ratio'] = df['tip_amount'] / df['fare_amount']
        df = df[df['tip_ratio'] <= 1.0]  # filter unrealistic outliers
        df['weight'] = df['tip_ratio']
    else:
        df = df[df['tip_amount'] >= 0]
        df['weight'] = df['tip_amount']

    # Optional: Filter by time
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df = df.dropna(subset=['tpep_pickup_datetime'])
    filter_by_time = st.checkbox("Filter by time of day?", value=False)

    if filter_by_time:
        selected_hour = st.slider("Select hour", 0, 23, 8, key="congestion_static_hour")
        df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
        df = df[df['pickup_hour'] == selected_hour]

    # Select coordinates
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
        st.warning("No data after applying the current filters.")
        return

    # Determine map center
    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()

    # Build HeatmapLayer
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
        tooltip={"text": "Tip"}
    ))

def plot_anomaly_trips(df):
    required = {
        'pickup_latitude', 'pickup_longitude',
        'dropoff_latitude', 'dropoff_longitude',
        'trip_distance', 'tpep_pickup_datetime', 'tpep_dropoff_datetime'
    }

    if not required.issubset(df.columns):
        st.error("CSV must contain pickup/dropoff coordinates, trip_distance, and timestamps.")
        return


    df = df.copy()
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')

    # Calculate duration and speed
    df['duration_hr'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 3600
    df['speed_kmh'] = df['trip_distance'] / df['duration_hr']

    # Keep only valid values
    df = df[
        (df['duration_hr'] > 0) &
        (df['trip_distance'] > 0) &
        (df['speed_kmh'] > 0)
    ]

    # Threshold for slow trips
    speed_limit = st.slider("Traffic threshold (km/h)", 1, 30, 10, key="stau_speed_limit")

    df_anomaly = df[df['speed_kmh'] < speed_limit].copy()

    if df_anomaly.empty:
        st.warning("No suspicious trips found with these criteria.")
        return

    st.write(f"🚩 {len(df_anomaly)} suspicious trips at speeds < {speed_limit} km/h")

    # Prepare columns for ArcLayer
    df_anomaly['from_lat'] = df_anomaly['pickup_latitude']
    df_anomaly['from_lon'] = df_anomaly['pickup_longitude']
    df_anomaly['to_lat'] = df_anomaly['dropoff_latitude']
    df_anomaly['to_lon'] = df_anomaly['dropoff_longitude']
    df_anomaly['pickup_time'] = df_anomaly['tpep_pickup_datetime'].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Layer: Pickup points
    point_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_anomaly,
        get_position='[from_lon, from_lat]',
        get_radius=40,
        get_fill_color='[255, 0, 0, 160]',
        pickable=True
    )

    # Layer: Trip paths
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

    # Map center
    center_lat = df_anomaly['from_lat'].mean()
    center_lon = df_anomaly['from_lon'].mean()

    # Show map with tooltip
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
                <b>Distance:</b> {trip_distance} mi
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
        st.error("CSV must contain 'pickup_lat/lon', 'tpep_pickup/dropoff_datetime', and 'trip_distance'.")
        return

    # Convert to timestamps
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
    df = df.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])

    # Calculate duration and speed
    df['duration_hr'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 3600
    df = df[
        (df['duration_hr'] > 0) &
        (df['trip_distance'] > 0)
    ].copy()

    df['speed_kmh'] = df['trip_distance'] / df['duration_hr']
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour

    # Hour filter
    selected_hour = st.slider("Select hour of day", 0, 23, 8)
    df_hour = df[df['pickup_hour'] == selected_hour]

    # Speed threshold for congestion
    speed_threshold = st.slider("Congestion threshold (km/h)", 1, 30, 10)
    df_congestion = df_hour[df_hour['speed_kmh'] <= speed_threshold].copy()

    if df_congestion.empty:
        st.warning("No congested trips found for this hour and threshold.")
        return

    # Map center
    center_lat = df_congestion['pickup_latitude'].mean()
    center_lon = df_congestion['pickup_longitude'].mean()

    # Heatmap layer for pickup points
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
        tooltip={"text": "Congestion Zone: Speed < {} km/h".format(speed_threshold)}
    ))

def plot_direction_rose(df):
    required = {
        'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude',
        'tpep_pickup_datetime'
    }

    if not required.issubset(df.columns):
        st.error("CSV must contain 'pickup/dropoff_xx' and 'tpep_pickup_datetime'.")
        return

    df = df.copy()
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df = df.dropna(subset=['tpep_pickup_datetime'])

    # Hour filter
    selected_hour = st.slider("Select hour (Pickup)", 0, 23, 8)
    df['hour'] = df['tpep_pickup_datetime'].dt.hour
    df = df[df['hour'] == selected_hour]

    # Filter out (0,0) coordinates
    df = df[
        (df['pickup_latitude'] != 0) & (df['pickup_longitude'] != 0) &
        (df['dropoff_latitude'] != 0) & (df['dropoff_longitude'] != 0)
    ]

    if df.empty:
        st.warning("No valid trips for this hour.")
        return

    # Calculate direction angle (in degrees)
    dx = df['dropoff_longitude'] - df['pickup_longitude']
    dy = df['dropoff_latitude'] - df['pickup_latitude']
    angles = np.degrees(np.arctan2(dy, dx))
    angles = (angles + 360) % 360  # Normalize to [0, 360)

    # Wind rose: histogram of directions
    bins = np.arange(0, 360 + 22.5, 22.5)  # 16 sectors
    counts, _ = np.histogram(angles, bins=bins)
    angles_mid = np.radians(bins[:-1] + 11.25)  # Midpoints of sectors

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    ax.bar(angles_mid, counts, width=np.radians(22.5), bottom=0.0,
           color='dodgerblue', alpha=0.75, edgecolor='black')

    ax.set_theta_zero_location('N')  # North at top
    ax.set_theta_direction(-1)       # Clockwise
    ax.set_title(f"🧭 Trip Directions Around {selected_hour}:00", va='bottom')
    st.pyplot(fig)


def plot_zone_density_heatmap(df):
    required = {
        'pickup_latitude', 'pickup_longitude',
        'dropoff_latitude', 'dropoff_longitude',
        'tpep_pickup_datetime'
    }

    if not required.issubset(df.columns):
        st.error("CSV must contain pickup/dropoff coordinates and 'tpep_pickup_datetime'.")
        return


    df = df.copy()
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df = df.dropna(subset=['tpep_pickup_datetime'])

    # Date filter
    min_date = df['tpep_pickup_datetime'].dt.date.min()
    max_date = df['tpep_pickup_datetime'].dt.date.max()
    if min_date == max_date:
        selected_date = min_date
        st.info(f"Only one date available: {min_date}")
    else:
        selected_date = st.slider("Select date (24h)", min_value=min_date, max_value=max_date, value=min_date)

    start = pd.Timestamp(selected_date)
    end = start + pd.Timedelta(days=1)
    df = df[(df['tpep_pickup_datetime'] >= start) & (df['tpep_pickup_datetime'] < end)]

    # Selection: Pickup vs Dropoff
    mode = st.radio("Which locations to show?", options=["Pickup", "Dropoff"], horizontal=True)

    if mode == "Pickup":
        coords = df[['pickup_latitude', 'pickup_longitude']].dropna()
        coords.columns = ['lat', 'lon']
    else:
        coords = df[['dropoff_latitude', 'dropoff_longitude']].dropna()
        coords.columns = ['lat', 'lon']

    # Only valid coordinates
    coords = coords[(coords['lat'] != 0) & (coords['lon'] != 0)]

    if coords.empty:
        st.warning("No valid coordinates.")
        return

    # Safety: do not allow more clusters than data points
    max_possible_clusters = min(300, len(coords))  # Safety limit
    min_clusters = 2

    if max_possible_clusters < min_clusters:
        st.warning(f"Not enough data points for clustering (at least {min_clusters} required).")
        return

    n_clusters = st.slider(
        "Number of clusters",
        min_value=min_clusters,
        max_value=max_possible_clusters,
        value=min(min_clusters + 3, max_possible_clusters)
    )

    # Clustering with KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    coords['cluster'] = kmeans.fit_predict(coords[['lat', 'lon']])
    coords['cluster_lat'] = kmeans.cluster_centers_[coords['cluster'], 0]
    coords['cluster_lon'] = kmeans.cluster_centers_[coords['cluster'], 1]

    # Cluster centers and frequencies
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
        tooltip={"text": "Trips: {weight}"}
    ))

def plot_taxi_sinkholes(df):
    required = {
        'pickup_latitude', 'pickup_longitude',
        'dropoff_latitude', 'dropoff_longitude'
    }

    if not required.issubset(df.columns):
        st.error("CSV must contain pickup/dropoff coordinates.")
        return

    df = df.copy()
    df = df[
        (df['pickup_latitude'] != 0) & (df['pickup_longitude'] != 0) &
        (df['dropoff_latitude'] != 0) & (df['dropoff_longitude'] != 0)
    ]

    # Round to grid cells
    precision = 3  # approx. ~100m grid
    df['pickup_lat'] = df['pickup_latitude'].round(precision)
    df['pickup_lon'] = df['pickup_longitude'].round(precision)
    df['dropoff_lat'] = df['dropoff_latitude'].round(precision)
    df['dropoff_lon'] = df['dropoff_longitude'].round(precision)

    # Count per cell
    pickup_counts = df.groupby(['pickup_lat', 'pickup_lon']).size().reset_index(name='pickups')
    dropoff_counts = df.groupby(['dropoff_lat', 'dropoff_lon']).size().reset_index(name='dropoffs')

    pickup_counts = pickup_counts.rename(columns={'pickup_lat': 'lat', 'pickup_lon': 'lon'})
    dropoff_counts = dropoff_counts.rename(columns={'dropoff_lat': 'lat', 'dropoff_lon': 'lon'})

    # Merge pickup and dropoff counts
    merged = pd.merge(dropoff_counts, pickup_counts, on=['lat', 'lon'], how='outer').fillna(0)
    merged['net_flow'] = merged['dropoffs'] - merged['pickups']  # positive = source, negative = sink

    if merged.empty:
        st.warning("No valid data after grouping.")
        return

    center_lat = merged['lat'].mean()
    center_lon = merged['lon'].mean()

    # Color mapping: green (source) → red (sink)
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
        tooltip={"text": "📍 Net: {net_flow} (Dropoffs - Pickups)"}
    ))