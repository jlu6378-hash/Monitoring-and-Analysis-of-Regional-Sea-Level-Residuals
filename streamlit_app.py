import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import xarray as xr
import numpy as np

st.set_page_config(page_title="Florida Tide Station Dashboard", layout="wide")

STATIONS = {
    "Key West, FL": "8724580",
    "Virginia Key, Miami FL": "8723214",
    "Naples Bay (North), FL": "8725114",
    "Fernandina Beach, FL": "8720030"
}

DATUMS = {
    "MHHW (Mean Higher-High Water)": "MHHW",
    "MHW (Mean High Water)": "MHW",
    "MTL (Mean Tide Level)": "MTL",
    "MSL (Mean Sea Level)": "MSL",
    "DTL (Mean Diurnal Tide Level)": "DTL",
    "MLW (Mean Low Water)": "MLW",
    "MLLW (Mean Lower-Low Water)": "MLLW",
    "STND (Station Datum)": "STND",
}


def get_observed_hourly_window(station_id, start_date, end_date, datum):
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

    params = {
        "product": "water_level",     # <-- changed here
        "datum": datum,
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
        return pd.DataFrame(columns=["t", "obs"])

    df = pd.DataFrame(data["data"])
    if not {"t", "v"}.issubset(df.columns):
        return pd.DataFrame(columns=["t", "obs"])

    df["t"] = pd.to_datetime(df["t"], errors="coerce")
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    df = df.dropna(subset=["t", "v"]).sort_values("t")

    # Resample to hourly to match NOAA hourly predictions
    hourly = (
        df.set_index("t")["v"]
          .resample("H")
          .mean()
          .dropna()
          .reset_index()
          .rename(columns={"v": "obs"})
    )
    return hourly

def get_predicted_hourly_window(station_id, start_date, end_date, datum):
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

    params = {
        "product": "predictions",
        "datum": datum,
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

def _date_chunks(start_date, end_date, chunk_days=30):
    chunks = []
    cur = start_date
    while cur <= end_date:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end_date)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return chunks

def get_observed_hourly_range(station_id, start_date, end_date, datum):
    parts = []
    for a, b in _date_chunks(start_date, end_date, chunk_days=30):
        df = get_observed_hourly_window(station_id, a, b, datum)
        if not df.empty:
            parts.append(df)

    if not parts:
        return pd.DataFrame(columns=["t", "obs"])

    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["t"]).sort_values("t").reset_index(drop=True)
    return out

def get_predicted_hourly_range(station_id, start_date, end_date, datum):
    parts = []
    for a, b in _date_chunks(start_date, end_date, chunk_days=30):
        df = get_predicted_hourly_window(station_id, a, b, datum)
        if not df.empty:
            parts.append(df)

    if not parts:
        return pd.DataFrame(columns=["t", "pred"])

    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["t"]).sort_values("t").reset_index(drop=True)
    return out



st.title("Florida Tide Gauge Dashboard & Flooding Risk Viewer")

c_station, c_datum = st.columns([2, 1])

with c_station:
    station_name = st.selectbox("Choose a Florida Tide Gauge Station", list(STATIONS.keys()))
    station_id = STATIONS[station_name]

with c_datum:
    datum_label = st.selectbox(
        "Tide Datum",
        list(DATUMS.keys()),
        index=list(DATUMS.values()).index("MLLW")  # default to MLLW
    )
    datum = DATUMS[datum_label]

st.subheader(f"Tide Station Selected: {station_name} (ID {station_id}) | Datum: {datum}")

st.header("Current Monitoring Window (Observed vs Predicted)")

overlay_days = st.slider("Days to display for monitoring", 7, 365, 60)
overlay_end = datetime.utcnow().date()
overlay_start = overlay_end - timedelta(days=overlay_days - 1)

# df_pred = get_historical_predictions(station_id, start_year=2000)

#if df_pred.empty:
#    st.error("No prediction data available.")
#else:
#    st.success(f"Loaded {len(df_pred)} prediction records.")
#    st.line_chart(df_pred.set_index("t")["v"])


def plot_obs_vs_pred(df, station_name):
    if df.empty:
        st.error("No overlapping observed/predicted data to plot.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["t"], y=df["obs"],
        mode="lines", name="Observed",
        line=dict(width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=df["t"], y=df["pred"],
        mode="lines", name="NOAA Tide Prediction (astronomical)",
        line=dict(width=1.2)
    ))

    fig.update_layout(
        title=f"Observed vs NOAA Tide Prediction at {station_name}",
        xaxis_title="Date",
        yaxis_title="Water Level (m)",
        template="plotly_white",
        height=500,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=80, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

# plot_predictions(df_pred, station_name)
df_obs_overlay = get_observed_hourly_range(station_id, overlay_start, overlay_end, datum)
df_pred_overlay = get_predicted_hourly_range(station_id, overlay_start, overlay_end, datum)

# st.write("OBS overlay rows:", len(df_obs_overlay))
# st.write("PRED overlay rows:", len(df_pred_overlay))

# if not df_obs_overlay.empty:
#     st.write("OBS time range:", df_obs_overlay["t"].min(), "→", df_obs_overlay["t"].max())
# if not df_pred_overlay.empty:
#     st.write("PRED time range:", df_pred_overlay["t"].min(), "→", df_pred_overlay["t"].max())

if df_obs_overlay.empty or df_pred_overlay.empty:
    st.warning("Observed and predicted data did not overlap for the selected window (possible station gaps or API truncation). Try a different window or datum.")

    df_overlay = pd.DataFrame(columns=["t", "obs", "pred"])
else:
    df_overlay = pd.merge(df_obs_overlay, df_pred_overlay, on="t", how="inner").sort_values("t")

plot_obs_vs_pred(df_overlay, station_name)

if not df_overlay.empty:
    latest_obs = df_overlay["obs"].iloc[-1]
    latest_pred = df_overlay["pred"].iloc[-1]
    latest_resid = latest_obs - latest_pred
    latest_time = df_overlay["t"].iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Observed", f"{latest_obs:.3f} m")
    c2.metric("Latest Predicted Tide", f"{latest_pred:.3f} m")
    c3.metric("Latest Residual (Obs − Pred)", f"{latest_resid:.3f} m")
    c4.metric("Last Timestamp", latest_time.strftime("%Y-%m-%d %H:%M UTC"))


debug_noaa = st.checkbox("Debug NOAA responses", value=False)

if debug_noaa:
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    params_test = {
        "product": "hourly_height",
        "datum": datum,
        "station": station_id,
        "time_zone": "gmt",
        "units": "metric",
        "format": "json",
        "begin_date": (datetime.utcnow().date() - timedelta(days=2)).strftime("%Y%m%d"),
        "end_date": datetime.utcnow().date().strftime("%Y%m%d"),
    }
    r = requests.get(url, params=params_test, timeout=10)
    st.write("HTTP status:", r.status_code)
    st.write("Raw keys:", list(r.json().keys()))
    st.json(r.json())


st.header("Discrepancy Diagnostics: Non-tidal Residual")

days_back = st.slider(
    "Days to analyze for residual surge",
    min_value=7,
    max_value=60,
    value=30
)

end_date = datetime.utcnow().date()
start_date = end_date - timedelta(days=days_back - 1)


def plot_residual_surge(df, station_name):
    if df.empty:
        st.warning("No overlapping observed/predicted data for residual.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["t"],
        y=df["surge"],
        mode="lines",
        name="Non-tidal Residual (Obs − Pred)",
        line=dict(width=1.5)
    ))

    # Zero baseline
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
    )

    fig.update_layout(
        title=f"Non-tidal Residual (Observed − NOAA Tide Prediction) at {station_name}",
        xaxis_title="Date",
        yaxis_title="Residual (m)",
        template="plotly_white",
        height=500,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=80, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

df_obs = get_observed_hourly_range(station_id, start_date, end_date, datum)
df_pred = get_predicted_hourly_range(station_id, start_date, end_date, datum)

# st.write("OBS residual rows:", len(df_obs))
# st.write("PRED residual rows:", len(df_pred))

# if not df_obs.empty:
#     st.write("OBS residual range:", df_obs["t"].min(), "→", df_obs["t"].max())
# if not df_pred.empty:
#     st.write("PRED residual range:", df_pred["t"].min(), "→", df_pred["t"].max())

if df_obs.empty or df_pred.empty:
    st.warning("Observed and predicted data did not overlap for the selected window (possible station gaps or API truncation). Try a different window or datum.")
    df_surge = pd.DataFrame(columns=["t", "obs", "pred", "surge"])
else:
    df_surge = pd.merge(df_obs, df_pred, on="t", how="inner").sort_values("t")
    df_surge["surge"] = df_surge["obs"] - df_surge["pred"]

if not df_surge.empty:
    thr = st.number_input("Residual alert threshold (m)", value=0.30, step=0.05)
    s = df_surge["surge"]

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Mean Residual", f"{s.mean():.3f} m")
    s2.metric("Max Residual", f"{s.max():.3f} m")
    s3.metric("95th Percentile", f"{s.quantile(0.95):.3f} m")
    s4.metric(f"Hours > {thr:.2f} m", int((s > thr).sum()))


st.header("Non-tidal Residual (Observed − NOAA Tide Prediction)")
plot_residual_surge(df_surge, station_name)

with st.expander("Show residual data table"):
    df_show = df_surge.copy()
    df_show = df_show.sort_values("t")

    df_show = df_show.rename(columns={
        "obs": "observed_m",
        "pred": "predicted_m",
        "surge": "residual_m"
    })

    st.dataframe(df_show.tail(200), use_container_width=True)



st.header("Historical Sea Level Trend")
@st.cache_data
def get_historical_trend(station_id, datum):
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

    # FIRST TRY monthly_mean
    params_monthly = {
        "product": "monthly_mean",
        "datum": datum,
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

    # FALLBACK: hourly_height
    params_hourly = {
        "product": "hourly_height",
        "datum": datum,
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

    st.error("NOAA response shape unrecognized:")
    st.json(data)
    return pd.DataFrame()

df_hist = get_historical_trend(station_id, datum)

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



st.header("Flooding Map (Vicinity Context)")
st.write("Use NOAA’s Sea Level Rise Viewer:")
st.link_button("Open NOAA Sea Level Rise Viewer", "https://coast.noaa.gov/digitalcoast/tools/slr.html")

st.header("Sea-level Rise Projections (Physics-based)")
st.write("Long-term sea-level rise projections (IPCC-aligned), different from tide prediction:")
st.link_button("Open IPCC Sea Level Projection Tool", "https://sealevel.nasa.gov/data_tools/17/")



# old unused functions

# def get_monthly_predictions(station_id, year, month):
#     """Fetch predictions for a single month (NOAA max allowed)."""
#     url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

#     # Compute begin/end dates
#     start = datetime(year, month, 1)
#     if month == 12:
#         end = datetime(year + 1, 1, 1) - timedelta(days=1)
#     else:
#         end = datetime(year, month + 1, 1) - timedelta(days=1)

#     params = {
#         "product": "predictions",
#         "application": "web_services",
#         "begin_date": start.strftime("%Y%m%d"),
#         "end_date": end.strftime("%Y%m%d"),
#         "station": station_id,
#         "datum": datum,
#         "time_zone": "gmt",
#         "units": "metric",
#         "interval": "h",    # hourly predictions
#         "format": "json"
#     }

#     try:
#         r = requests.get(url, params=params, timeout=10)
#         r.raise_for_status()
#     except Exception:
#         return pd.DataFrame()  # NOAA errors are silent for monthly calls

#     try:
#         data = r.json()
#     except:
#         return pd.DataFrame()

#     if "predictions" not in data:
#         return pd.DataFrame()

#     df = pd.DataFrame(data["predictions"])
#     df["t"] = pd.to_datetime(df["t"])
#     df["v"] = pd.to_numeric(df["v"])
#     return df


# @st.cache_data(show_spinner=True)
# def get_historical_predictions(station_id, start_year=2000):
#     """Downloads predictions month-by-month and concatenates."""
#     all_data = []

#     current_year = datetime.now().year
#     current_month = datetime.now().month

#     with st.spinner("Downloading predicted tide history (month by month)..."):
#         for year in range(start_year, current_year + 1):
#             max_month = 12 if year < current_year else current_month

#             for month in range(1, max_month + 1):
#                 df = get_monthly_predictions(station_id, year, month)
#                 if not df.empty:
#                     all_data.append(df)

#     if not all_data:
#         return pd.DataFrame()

#     full_df = pd.concat(all_data)
#     full_df = full_df.sort_values("t")
#     return full_df.reset_index(drop=True)