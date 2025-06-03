import streamlit as st

st.set_page_config(page_title="NYC Taxi Trips", page_icon="🚖", layout="wide")
st.title('NYC Yellow Taxi Trip Data')


# Streamlit pages should only contain minimal code
# to glue different components together.
# All logic, plotting, and non-streamlit related functions
# should live in /components or /logic.