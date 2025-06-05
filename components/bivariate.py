import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_corr_heatmap(df):
    # RatecodeID rauswerfen, wenn vorhanden
    df = df.drop(columns=["RatecodeID"], errors="ignore")

    # Nur numerische Spalten extrahieren
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    if numeric_df.shape[1] < 2:
        st.warning("Mindestens zwei numerische Spalten erforderlich für eine Korrelationsmatrix.")
        st.write("Gefundene numerische Spalten:", list(numeric_df.columns))
        return

    # Korrelation berechnen
    corr = numeric_df.corr()

    if corr.isnull().all().all():
        st.warning("Konnte keine gültige Korrelationsmatrix berechnen – möglicherweise nur konstante Werte.")
        return

    # Plot erstellen
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax, cbar=True)
    ax.set_title("Korrelationsmatrix (ohne RatecodeID)")

    st.pyplot(fig)