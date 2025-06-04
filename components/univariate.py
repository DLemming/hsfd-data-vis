import streamlit as st
import pandas as pd
import plotly.express as px
from logic.univariate import get_metrical_columns, get_categorical_columns

def render_boxplot(df: pd.DataFrame):
    numeric = get_metrical_columns(df)
    if numeric:
        col = st.selectbox(
            "Boxplot Column",
            numeric,
            label_visibility="collapsed",
            index=8 # total_amount
        )
        __boxplot(df, col)

def render_histogram(df: pd.DataFrame):
    numeric = get_metrical_columns(df)
    if numeric:
        col = st.selectbox("Histogram Column", numeric, label_visibility="collapsed")
        __histogram(df, col)

def render_bar_chart(df: pd.DataFrame):
    categorical = get_categorical_columns(df)
    if categorical:
        col = st.selectbox("Bar Chart Column", categorical, label_visibility="collapsed", index=3)
        __bar_chart(df, col)

def render_pie_chart(df: pd.DataFrame):
    categorical = get_categorical_columns(df)
    if categorical:
        col = st.selectbox("Pie Chart Column", categorical, label_visibility="collapsed", index=0)
        __pie_chart(df, col)



### Private Methods ###
def __boxplot(df: pd.DataFrame, column: str):
    fig = px.box(df, y=column, points="all")
    st.plotly_chart(fig, use_container_width=True)

def __histogram(df: pd.DataFrame, column: str):
    fig = px.histogram(df, x=column)
    st.plotly_chart(fig, use_container_width=True)

def __bar_chart(df: pd.DataFrame, column: str):
    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = [column, "count"]
    fig = px.bar(value_counts, x=column, y="count")
    st.plotly_chart(fig, use_container_width=True)

def __pie_chart(df: pd.DataFrame, column: str):
    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = [column, "count"]
    fig = px.pie(value_counts, names=column, values="count", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)