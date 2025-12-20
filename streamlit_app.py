import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import xarray as xr
import numpy as np



st.set_page_config(page_title="Florida Tide Station Dashboard", layout="wide")

# ---------------------------
# USER SETTINGS
# ---------------------------
STATIONS = {
    "Key West, FL": "8724580",
    "Virginia Key, Miami FL": "8723214",
    "Naples, FL": "8725110",
    "Fernandina Beach, FL": "8720030"
}

# ---------------------------
# PAGE TITLE
# ---------------------------
st.title("🌊 Florida Tide Gauge Dashboard & Flooding Risk Viewer")

station_name = st.selectbox("Choose a Florida Tide Gauge Station", list(STATIONS.keys()))
station_id = STATIONS[station_name]

st.subheader(f"Tide Station Selected: {station_name} (ID {station_id})")

st.header("📈 Historical Sea Level Trend")

@st.cache_data
def get_historical_trend(station_id):
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

    # ---- 1) FIRST TRY monthly_mean ----
    params_monthly = {
        "product": "monthly_mean",
        "datum": "MLLW",
        "station": station_id,
        "time_zone": "gmt",
        "units": "metric",
        "format": "json",
        "begin_date": "19000101",
        "end_date": datetime.now().strftime("%Y%m%d")
    }

    try:
        r = requests.get(url, params=params_monthly, timeout=10)
        data = r.json()
    except:
        data = {}

    if "data" in data:
        df = pd.DataFrame(data["data"])

        # Expected columns: year, month, MSL (or similar)
        if all(col in df.columns for col in ["year", "month"]):
            df["t"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-01")
            value_col = [c for c in df.columns if c not in ["year", "month", "t"]][0]
            df["v"] = pd.to_numeric(df[value_col], errors="coerce")
            df = df[["t", "v"]]
            df.dropna(inplace=True)
            return df

    # ---- 2) FALLBACK: hourly_height ----
    params_hourly = {
        "product": "hourly_height",
        "datum": "MLLW",
        "station": station_id,
        "time_zone": "gmt",
        "units": "metric",
        "format": "json",
        "begin_date": "20000101",  # NOAA restricts large queries
        "end_date": datetime.now().strftime("%Y%m%d")
    }

    try:
        r = requests.get(url, params=params_hourly, timeout=10)
        data = r.json()
    except:
        st.error("NOAA did not return valid JSON.")
        st.text(r.text)
        return pd.DataFrame()

    if "data" not in data:
        st.error("No usable data returned for this station.")
        st.json(data)
        return pd.DataFrame()

    df = pd.DataFrame(data["data"])
    # Expected columns: t, v
    if "t" in df.columns and "v" in df.columns:
        df["t"] = pd.to_datetime(df["t"], errors="coerce")
        df["v"] = pd.to_numeric(df["v"], errors="coerce")
        df.dropna(subset=["t", "v"], inplace=True)
        return df

    # If NOAA gives something odd
    st.error("NOAA response shape unrecognized:")
    st.json(data)
    return pd.DataFrame()

import pandas as pd
import requests
from datetime import datetime, timedelta
import streamlit as st


def get_monthly_predictions(station_id, year, month):
    """Fetch predictions for a single month (NOAA max allowed)."""
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

    # Compute begin/end dates
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end = datetime(year, month + 1, 1) - timedelta(days=1)

    params = {
        "product": "predictions",
        "application": "web_services",
        "begin_date": start.strftime("%Y%m%d"),
        "end_date": end.strftime("%Y%m%d"),
        "station": station_id,
        "datum": "MLLW",
        "time_zone": "gmt",
        "units": "metric",
        "interval": "h",    # hourly predictions
        "format": "json"
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
    except Exception:
        return pd.DataFrame()  # NOAA errors are silent for monthly calls

    try:
        data = r.json()
    except:
        return pd.DataFrame()

    if "predictions" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["predictions"])
    df["t"] = pd.to_datetime(df["t"])
    df["v"] = pd.to_numeric(df["v"])
    return df


@st.cache_data(show_spinner=True)
def get_historical_predictions(station_id, start_year=2000):
    """Downloads predictions month-by-month and concatenates."""
    all_data = []

    current_year = datetime.now().year
    current_month = datetime.now().month

    with st.spinner("Downloading predicted tide history (month by month)..."):
        for year in range(start_year, current_year + 1):
            max_month = 12 if year < current_year else current_month

            for month in range(1, max_month + 1):
                df = get_monthly_predictions(station_id, year, month)
                if not df.empty:
                    all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    full_df = pd.concat(all_data)
    full_df = full_df.sort_values("t")
    return full_df.reset_index(drop=True)

df_hist = get_historical_trend(station_id)

if df_hist.empty:
    st.warning("No historical data available for this station.")
else:
    fig_hist = px.line(
        df_hist,
        x="t",
        y="v",
        title=f"Historical Sea Level at {station_name}",
        labels={"t": "Time", "v": "Sea Level (m)"}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

df_pred = get_historical_predictions(station_id, start_year=2000)

#if df_pred.empty:
#    st.error("No prediction data available.")
#else:
#    st.success(f"Loaded {len(df_pred)} prediction records.")
#    st.line_chart(df_pred.set_index("t")["v"])


def plot_predictions(df_pred, station_name):
    if df_pred.empty:
        st.error("No prediction data to plot.")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_pred["t"],
        y=df_pred["v"],
        mode="lines",
        name="Predicted Tide",
        line=dict(width=1.5)
    ))

    fig.update_layout(
        title=f"Predicted Tide Levels at {station_name}",
        xaxis_title="Date",
        yaxis_title="Predicted Sea Level (m)",
        template="plotly_white",
        height=500,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=80, b=40)
    )

    fig.update_xaxes(showgrid=True, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridcolor="lightgray")

    st.plotly_chart(fig, use_container_width=True)

plot_predictions(df_pred, station_name)



st.header("🗓 Surge Analysis Window")

days_back = st.slider(
    "Days to analyze for residual surge",
    min_value=7,
    max_value=60,
    value=30
)

end_date = datetime.utcnow().date()
start_date = end_date - timedelta(days=days_back)

def get_observed_hourly_window(station_id, start_date, end_date):
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

    params = {
        "product": "hourly_height",
        "datum": "MLLW",
        "station": station_id,
        "time_zone": "gmt",
        "units": "metric",
        "format": "json",
        "begin_date": start_date.strftime("%Y%m%d"),
        "end_date": end_date.strftime("%Y%m%d")
    }

    r = requests.get(url, params=params, timeout=10)
    data = r.json()

    if "data" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["data"])
    df["t"] = pd.to_datetime(df["t"])
    df["obs"] = pd.to_numeric(df["v"], errors="coerce")
    return df[["t", "obs"]].dropna()

def get_predicted_hourly_window(station_id, start_date, end_date):
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

    params = {
        "product": "predictions",
        "datum": "MLLW",
        "station": station_id,
        "time_zone": "gmt",
        "units": "metric",
        "interval": "h",
        "format": "json",
        "begin_date": start_date.strftime("%Y%m%d"),
        "end_date": end_date.strftime("%Y%m%d")
    }

    r = requests.get(url, params=params, timeout=10)
    data = r.json()

    if "predictions" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["predictions"])
    df["t"] = pd.to_datetime(df["t"])
    df["pred"] = pd.to_numeric(df["v"], errors="coerce")
    return df[["t", "pred"]].dropna()

def plot_residual_surge(df, station_name):
    if df.empty:
        st.warning("No overlapping observed/predicted data for surge.")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["t"],
        y=df["surge"],
        mode="lines",
        name="Residual Surge",
        line=dict(color="crimson", width=1.5)
    ))

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="No Surge",
        annotation_position="top left"
    )

    fig.update_layout(
        title=f"Residual Surge at {station_name}",
        xaxis_title="Date",
        yaxis_title="Residual Surge (m)",
        template="plotly_white",
        height=500,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=80, b=40)
    )

    fig.update_xaxes(showgrid=True, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridcolor="lightgray")

    st.plotly_chart(fig, use_container_width=True)

df_obs = get_observed_hourly_window(station_id, start_date, end_date)
df_pred = get_predicted_hourly_window(station_id, start_date, end_date)

df_surge = pd.merge(df_obs, df_pred, on="t", how="inner")
df_surge["surge"] = df_surge["obs"] - df_surge["pred"]

st.header("🌪 Residual Surge (Observed − Predicted)")
plot_residual_surge(df_surge, station_name)
