import streamlit as st
import pandas as pd
import plotly.express as px

def show_passenger_count_bar_chart(df):
    # Passenger Count zusammenfassen
    count_data = df['passenger_count'].value_counts().sort_index().reset_index()
    count_data.columns = ['Passenger Count', 'Number of Trips']

    # Kompaktes Plotly-Bar-Chart (ohne feste Größe)
    fig = px.bar(
        count_data,
        x='Passenger Count',
        y='Number of Trips',
        title='Distribution of Passenger Count',
        labels={'Number of Trips': 'Anzahl Fahrten', 'Passenger Count': 'Passagiere'},
        color_discrete_sequence=['#f7b731'],
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),    # enge Margins
        font=dict(size=11),
        title_font=dict(size=13),
        bargap=0.2,  # Balkenabstand reduziert (optional)
        
    )
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))

    st.plotly_chart(fig, use_container_width=True)
