import streamlit as st
import pandas as pd
import plotly.express as px

def show_passenger_count_bar_chart(df):

    # Passenger Count zusammenfassen
    count_data = df['passenger_count'].value_counts().sort_index().reset_index()
    count_data.columns = ['Passenger Count', 'Number of Trips']

    # Plotly Bar Chart
    fig = px.bar(
        count_data,
        x='Passenger Count',
        y='Number of Trips',
        title='Distribution of Passenger Count',
        labels={'Number of Trips': 'Anzahl Fahrten', 'Passenger Count': 'Passagiere'}
    )

    st.plotly_chart(fig)
