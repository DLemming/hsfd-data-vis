from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import streamlit as st

def haversine(lat1, lon1, lat2, lon2):
    # Earth radius in kilometers
    R = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    distance = R * c
    return distance

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
