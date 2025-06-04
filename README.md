# Data Visualization Streamlit App

A simple interactive dashboard for visualizing New York City taxi trip data. Explore trip durations, distances, fares, and more using intuitive controls and real-time plots.

## Setup

### 1. Clone the repository:
```sh
git clone https://github.com/DLemming/hsfd-data-vis.git
cd hsfd-data-vis
```

### 2. Install dependencies:
We use [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management (faster than pip). To install all dependencies and activate venv, run:
```sh
uv sync
source .venv/bin/activate
```

## Usage

Make sure you're in the project root. Run the Streamlit app with:
```sh
streamlit run 🏠_Home.py
```

## TODO
- Plot Missing Values (Data Quality Metrics) - David
- Univariate Analysis (alle Dropdown mit Spaltenauswahl): - David
    - Boxplot
    - Histogram (Kontinuierlich)
    - Bar Chart (Categorical)
    - PieChart (Anteile von kategorischen Variablen)
- Bivariate Analysis: - Jan
    - Correlation Heatmap
    - Line Chart (flexibel kontinuierliche Variablen auswählen)
    - Scatter Plot ()
- Advanced Methods
    - Geovisualiation over Time - Jan
    - Fare amount predicotor (Regression) - David

## Preview

## Dataset

The data used is from the [NYC Yellow Taxi Trip Data](https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data), uploaded to Kaggle by the user ELEMENTO.