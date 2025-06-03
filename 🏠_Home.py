import plotly.express as px
import streamlit as st
from logic.data_loader import load_taxi_data
from components.passenger_count import show_passenger_count_bar_chart

st.set_page_config(page_title="NYC Taxi Trips", page_icon="🚖", layout="wide")
st.title('NYC Yellow Taxi Trip Data')

# Daten laden 
df = load_taxi_data()

show_passenger_count_bar_chart(df)