import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from logic.health import compute_health

ALLOWED_COLS = [
    "VendorID",
    "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "pickup_longitude", "pickup_latitude",
    "dropoff_longitude","dropoff_latitude",
    "payment_type","fare_amount","tip_amount","tolls_amount","improvement_surcharge","total_amount"
]


def render_health_component(df: pd.DataFrame):
    filtered_cols = [col for col in ALLOWED_COLS if col in df.columns]

    with st.expander("Select columns for health check"):
        selected_cols = st.multiselect(
            "Select columnse", options=filtered_cols,
            default=filtered_cols,
            label_visibility="collapsed"
        )
    if selected_cols:
        stats = compute_health(df, selected_cols)
        __render_health_donut(
            stats["healthy_count"],
            stats["unhealthy_count"],
            stats["healthy_percent"]
        )
    else:
        st.warning("Please select at least one column.")


### Private Functions ###
def __render_health_donut(healthy_count: int, unhealthy_count: int, healthy_percent: float):
    fig = go.Figure(data=[go.Pie(
        labels=["Healthy", "Unhealthy"],
        values=[healthy_count, unhealthy_count],
        hole=0.6,
        marker=dict(colors=["#2ECC71", "#E74C3C"]),
        textinfo="none"
    )])

    fig.add_annotation(dict(
        text=f"{healthy_percent}%",
        x=0.5, y=0.5,
        font_size=40,
        showarrow=False
    ))

    fig.update_layout(margin=dict(t=20, b=20, l=0, r=0), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)