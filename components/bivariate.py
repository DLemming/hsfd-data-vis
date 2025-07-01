import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from logic.univariate import get_metrical_columns

def plot_corr_heatmap(df: pd.DataFrame):
    """
    Plot a correlation matrix using Plotly with 0s fully transparent,
    and a diverging red-blue color scale.
    """
    columns = get_metrical_columns(df)
    numeric_df = df[columns]

    corr = numeric_df.corr(method="pearson")
    if corr.isnull().all().all():
        st.warning("Keine gültige Korrelationsmatrix – evtl. nur konstante Werte.")
        return

    # Replace exact zeros with NaN for transparency
    z = corr.mask(corr == 0).values
    labels = corr.columns.tolist()

    # Custom red–transparent–blue RGBA color scale
    colorscale = [
        [0.0, "rgba(134, 202, 253, 1)"],      # Blue with full opacity
        [0.25, "rgba(134, 202, 253, 0.5)"],   # Blue with 3/4 opacity
        [0.5, "rgba(0, 0, 0, 0)"],           # Transparent
        [0.75, "rgba(229, 73, 65, 0.5)"],   # Red with 3/4 opacity
        [1.0, "rgba(229, 73, 65, 1)"]          # Red with full opacity
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale=colorscale,
        zmin=-1,
        zmax=1,
    ))

    fig.update_layout(
        xaxis=dict(side='top'        ),
        yaxis=dict(autorange='reversed'),
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_bivariate_scatter(df: pd.DataFrame):
    """
    Display an interactive scatterplot for any two user-selected columns (x/y).
    """
    columns = get_metrical_columns(df)

    if len(columns) < 2:
        st.warning("At least to distinct columns with numeric values required.")
        return

    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("X-Axis", columns, key="scatter_x", index=5)
    with col2:
        y_col = st.selectbox("Y-Axis", columns, key="scatter_y")

    # Only plot if x != y
    if x_col == y_col:
        st.info("Please choose two different columns.")
        return

    # Drop rows where x or y is missing
    plot_df = df[[x_col, y_col]].dropna()

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        opacity=0.7,
        labels={x_col: x_col, y_col: y_col}
    )

    fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
    st.plotly_chart(fig, use_container_width=True)
