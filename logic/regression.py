from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import streamlit as st


def haversine(lat1, lon1, lat2, lon2):
    """Calculate the Haversine distance between two points."""
    R = 6371  # Radius of Earth in kilometers
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c  # Return distance in kilometers

@st.cache_data
def fit_regression_model(df):
    df = df.copy()
    
    # Compute distance
    df['distance'] = haversine(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    
    # Compute trip duration in minutes
    df['pickup_dt'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['dropoff_dt'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['trip_duration'] = (df['dropoff_dt'] - df['pickup_dt']).dt.total_seconds() / 60.0
    
    # Filter out garbage
    df = df[
        (df['distance'] > 0) &
        (df['distance'] < 500) &
        (df['trip_duration'] > 0) &
        (df['trip_duration'] < 120)
    ]

    # Prepare data
    X = df[['distance', 'trip_duration']].to_numpy()
    y = df['total_amount'].to_numpy()

    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    return model, df


def get_average_trip_duration(df, distance, km_thresh=1.0):
    """
    Given pickup and dropoff coordinates, find similar trips based on Haversine distance and return the average trip duration.
    """
    # Filter trips that are within the threshold distance of the selected trip's distance
    similar_trips = df[np.abs(df['distance'] - distance) <= km_thresh]

    # If there are similar trips, return the average trip duration
    if not similar_trips.empty:
        return similar_trips['trip_duration'].mean()
    else:
        return 30  # Default value if no similar trips found