import streamlit as st
import pydeck as pdk
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from logic.regression import (
    haversine,
    get_average_trip_duration
)
from data.constants import NYC_LOCATIONS


def show_regression_inputs(model, df):

    # Create layout: map on left, sliders on right
    col1, col2 = st.columns([3, 1])

    with col2:
        # All available location names
        location_names = list(NYC_LOCATIONS.keys())

        st.markdown("**Choose Route**")

        # Pickup dropdown
        pickup_location = st.selectbox("Pickup (green)", location_names, index=1)

        # Dropoff dropdown with pickup excluded
        dropoff_options = [loc for loc in location_names if loc != pickup_location]
        dropoff_location = st.selectbox("Destination (blue)", dropoff_options, index=3)
        st.markdown("</br>", unsafe_allow_html=True)

    # Get coordinates
    pickup_lat, pickup_lon = NYC_LOCATIONS[pickup_location]
    dropoff_lat, dropoff_lon = NYC_LOCATIONS[dropoff_location]

    # Calculate distance (Haversine or any other method)
    distance = haversine(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)

    # Get average trip duration based on similar trips
    avg_trip_duration = get_average_trip_duration(df, distance)

    with col2:
        # Set the average trip duration as the default in the slider
        st.markdown("**Trip Duration**")
        trip_duration = st.slider("Select Duration (minutes)", 1, 120, int(avg_trip_duration))
        st.markdown("</br></br>", unsafe_allow_html=True)

    # Map layer
    layers = [
        pdk.Layer(
            "HeatmapLayer",
            data=df,
            get_position='[pickup_longitude, pickup_latitude]',
            opacity=0.9,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=[
                {"lat": pickup_lat, "lon": pickup_lon},
                {"lat": dropoff_lat, "lon": dropoff_lon},
            ],
            get_position='[lon, lat]',
            get_color='[255, 0, 0, 160]',
            get_radius=100,
        ),
        pdk.Layer(
            "ArcLayer",
            data=[{
                "start": [pickup_lon, pickup_lat],
                "end": [dropoff_lon, dropoff_lat],
            }],
            get_source_position="start",
            get_target_position="end",
            get_source_color=[0, 255, 0, 160],
            get_target_color=[0, 0, 255, 160],
            width_scale=0.0001,
            width_min_pixels=3,
            width_max_pixels=10,
        )
    ]

    with col1:
        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=(pickup_lat + dropoff_lat) / 2,
                longitude=(pickup_lon + dropoff_lon) / 2,
                zoom=11,
                pitch=50,
            ),
            layers=layers
        ))

    # Predict fare
    X = np.array([[distance, trip_duration]])
    fare_pred = model.predict(X)[0]

    with col2:
        st.success(f"Predicted fare: ${fare_pred:.2f}")

def plot_regression(model, df):
    # Sample a subset of the data for clearer visualization
    df_sample = df.sample(100)  # Or pick a smaller subset for clarity

    # Calculate the predicted fares
    X_sample = df_sample[['distance', 'trip_duration']].to_numpy()
    y_sample = df_sample['total_amount'].to_numpy()
    y_pred = model.predict(X_sample)

    # Create the figure for the first plot (Distance vs Fare)
    fig1 = go.Figure()

    # Scatter plot of actual data (distance vs. fare amount)
    fig1.add_trace(go.Scatter(
        x=df_sample['distance'], 
        y=y_sample, 
        mode='markers', 
        name='Actual Data', 
        marker=dict(color='blue', opacity=0.6)
    ))

    # Scatter plot of predicted data (fitted regression line)
    fig1.add_trace(go.Scatter(
        x=df_sample['distance'], 
        y=y_pred, 
        mode='markers', 
        name='Predictions', 
        marker=dict(color='red', opacity=0.6)
    ))

    # Update layout for the first plot
    fig1.update_layout(
        title="Distance vs. Fare Amount",
        xaxis_title="Distance (km)",
        yaxis_title="Fare Amount ($)"
    )

    # Create the figure for the second plot (Trip Duration vs Fare)
    fig2 = go.Figure()

    # Scatter plot of actual data (trip duration vs. fare amount)
    fig2.add_trace(go.Scatter(
        x=df_sample['trip_duration'], 
        y=y_sample, 
        mode='markers', 
        name='Actual Data', 
        marker=dict(color='blue', opacity=0.6)
    ))

    # Scatter plot of predicted data (fitted regression line)
    fig2.add_trace(go.Scatter(
        x=df_sample['trip_duration'], 
        y=y_pred, 
        mode='markers', 
        name='Predictions', 
        marker=dict(color='red', opacity=0.6)
    ))

    # Update layout for the second plot
    fig2.update_layout(
        title="Trip Duration vs. Fare Amount",
        xaxis_title="Trip Duration (min)",
        yaxis_title="Fare Amount ($)",
    )

    # Create two columns in Streamlit
    col1, col2 = st.columns(2)

    # Plot the first figure in the first column
    with col1:
        st.plotly_chart(fig1)

    # Plot the second figure in the second column
    with col2:
        st.plotly_chart(fig2)

def plot_all_vars(model, df):
    # Ensure 'distance' and 'trip_duration' columns exist
    if 'distance' not in df.columns:
        df['distance'] = haversine(
            df['pickup_latitude'], df['pickup_longitude'],
            df['dropoff_latitude'], df['dropoff_longitude']
        )
    if 'trip_duration' not in df.columns:
        df['trip_duration'] = (df['dropoff_dt'] - df['pickup_dt']).dt.total_seconds() / 60.0

    # Calculate predicted fares
    X = df[['distance', 'trip_duration']].to_numpy()
    y_pred = model.predict(X)

    # Add predicted fare to the DataFrame
    df['predicted_fare'] = y_pred

    # Create the scatter plot for Distance vs Trip Duration
    fig1 = px.scatter(
        df, 
        x='distance', 
        y='trip_duration', 
        color='predicted_fare', 
        color_continuous_scale='Viridis',
        title="Distance and Trip Duration vs Predicted Fare",
        labels={'distance': 'Distance (km)', 'trip_duration': 'Trip Duration (min)', 'predicted_fare': 'Predicted Fare ($)'},
        opacity=0.7
    )

    # Calculate feature contributions (for simplicity, let's take a simple linear regression-based approach)
    # Since you already have a linear regression model, we can break down the prediction into its parts:
    # model = a linear regression model, X = [distance, trip_duration], coefficients = [coeff_distance, coeff_duration]

    coefficients = model.coef_  # This will give you the weight for each feature
    feature_names = ['Distance', 'Trip Duration']
    
    # We need to calculate the contribution for each feature
    # Contributions are simply the coefficient * feature value (but we will average it across the sample for simplicity)
    df['distance_contribution'] = df['distance'] * coefficients[0]
    df['duration_contribution'] = df['trip_duration'] * coefficients[1]

    # Average contributions for each feature
    avg_distance_contribution = df['distance_contribution'].mean()
    avg_duration_contribution = df['duration_contribution'].mean()

    # Create the contribution bar chart
    fig2 = go.Figure()

    fig2.add_trace(go.Bar(
        x=feature_names,
        y=[avg_distance_contribution, avg_duration_contribution],
        name="Feature Contributions",
        marker=dict(color=['blue', 'green'])
    ))

    fig2.update_layout(
        title="Feature Contribution to Predicted Fare",
        xaxis_title="Features",
        yaxis_title="Average Contribution to Fare ($)",
    )

    # Create two columns in Streamlit
    col1, col2 = st.columns([3,1])

    # Plot the scatter plot in the first column
    with col1:
        st.plotly_chart(fig1)

    # Plot the bar chart in the second column
    with col2:
        st.plotly_chart(fig2)

def plot_3d_regression_surface(model, df):
    # Generate a grid of values for 'distance' and 'trip_duration'
    x_range = np.linspace(df['distance'].min(), df['distance'].max(), 30)
    y_range = np.linspace(df['trip_duration'].min(), df['trip_duration'].max(), 30)
    
    # Create meshgrid for the features
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    
    # Use the linear regression model to calculate the predicted fare (z) for the grid
    z_grid = model.intercept_ + model.coef_[0] * x_grid + model.coef_[1] * y_grid
    
    # Create the 3D scatter plot
    fig = go.Figure()

    # Scatter plot of the actual data points (Distance, Trip Duration, Actual Fare)
    fig.add_trace(go.Scatter3d(
        x=df['distance'], 
        y=df['trip_duration'], 
        z=df['total_amount'], 
        mode='markers',
        marker=dict(color='blue', opacity=0.6),
        name='Actual Data',
    ))

    # Plot the regression surface
    fig.add_trace(go.Surface(
        x=x_grid, 
        y=y_grid, 
        z=z_grid, 
        colorscale='reds', 
        opacity=0.3,
        name='Fitted Surface'
    ))

    # Update layout
    fig.update_layout(
        title="3D Regression Surface Plot",
        scene=dict(
            xaxis_title='Distance (km)',
            yaxis_title='Trip Duration (min)',
            zaxis_title='Fare Amount ($)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.1)  # Adjust camera position (you can tweak this for different perspectives)
            )
        ),
        height=800  # Increase the height for more vertical space
    )

    # Show the plot in Streamlit
    st.plotly_chart(fig)