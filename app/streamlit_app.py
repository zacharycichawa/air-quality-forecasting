"""Polish Air Quality: interactive demo of the project's data, analysis and forecasts.

Reads the compact snapshot in app/data/ (written by `airquality export-snapshot`),
so it runs anywhere without access to the project's PostgreSQL database.
Run locally:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

DATA = Path(__file__).parent / "data"
REPO = "https://github.com/zacharycichawa/air-quality-forecasting"
BLUE, GREY, INK = "#2a78d6", "#8f8e88", "#0b0b0b"
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # light -> dark
WHO_DAILY_PM25 = 15

st.set_page_config(page_title="Polish Air Quality", page_icon="🌫️", layout="wide")


# ------------------------------------------------------------------ data
@st.cache_data
def load() -> dict:
    d = {p.stem: pd.read_parquet(p) for p in DATA.glob("*.parquet")}
    d["metrics"] = {p.stem: json.loads(p.read_text()) for p in DATA.glob("h*.json")}
    d["meta"] = json.loads((DATA / "snapshot_meta.json").read_text())
    st_ = d["stations"]
    st_["label"] = st_["city"] + " · " + st_["station_name"].str.slice(0, 40)
    for name in ("daily_pm25",):
        d[name]["date"] = pd.to_datetime(d[name]["date"])
    d["monthly"]["month"] = pd.to_datetime(d["monthly"]["month"])
    return d


def color_for(values: pd.Series, vmin: float, vmax: float) -> list[list[int]]:
    """Map values to the sequential blue ramp as RGBA lists for pydeck."""
    ramp = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in SEQ])
    pos = np.clip((values.to_numpy() - vmin) / max(vmax - vmin, 1e-9), 0, 1) * (len(SEQ) - 1)
    lo, frac = np.floor(pos).astype(int), (pos % 1)[:, None]
    hi = np.minimum(lo + 1, len(SEQ) - 1)
    rgb = ramp[lo] * (1 - frac) + ramp[hi] * frac
    return [[*map(int, c), 230] for c in rgb]


MAP_SCALE = (5, 35)  # µg/m³: fixed so colours mean the same thing on every map


def station_map(df: pd.DataFrame, value: str, label: str, fmt: str = "{:.1f}") -> None:
    df = df.dropna(subset=[value, "latitude", "longitude"]).copy()
    df["color"] = color_for(df[value], *MAP_SCALE)
    df["value_txt"] = df[value].map(fmt.format)
    layer = pdk.Layer("ScatterplotLayer", df, get_position="[longitude, latitude]",
                      get_fill_color="color", get_radius=9000, radius_min_pixels=5,
                      pickable=True, stroked=True, get_line_color=[255, 255, 255],
                      line_width_min_pixels=1.5)
    st.pydeck_chart(pdk.Deck(
        layers=[layer], map_style=None,
        initial_view_state=pdk.ViewState(latitude=52.0, longitude=19.3, zoom=5.1),
        tooltip={"text": f"{{city}} ({{station_code}})\n{label}: {{value_txt}}"}))


d = load()
stations = d["stations"]

# ------------------------------------------------------------------ header
st.title("Polish Air Quality: data, analysis and PM2.5 forecasts")
st.caption(
    f"Hourly measurements from GIOŚ (official monitoring network) and ERA5 weather, "
    f"2019 onwards. Snapshot exported {d['meta']['exported_at'][:10]}. "
    f"[Source code and methodology]({REPO})")

tab_overview, tab_station, tab_forecast, tab_model = st.tabs(
    ["Overview", "Station explorer", "Forecast", "Model performance"])

# ------------------------------------------------------------------ overview
with tab_overview:
    m = d["monthly"][d["monthly"]["pollutant"] == "PM2.5"]
    yearly = (m.assign(year=m["month"].dt.year)
                .groupby("year")
                .apply(lambda g: pd.Series({
                    "mean": np.average(g["mean_ug_m3"], weights=g["valid_days"]),
                    "share_over_who": g["days_over_who"].sum() / g["valid_days"].sum()}),
                    include_groups=False))
    full_years = yearly.loc[:2025]
    last, first = full_years.index.max(), full_years.index.min()
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Mean PM2.5, {last}", f"{full_years.loc[last, 'mean']:.1f} µg/m³",
              f"{full_years.loc[last, 'mean'] / full_years.loc[first, 'mean'] - 1:+.0%} vs {first}",
              delta_color="inverse")
    c2.metric(f"Station-days above WHO guideline, {last}",
              f"{full_years.loc[last, 'share_over_who']:.0%}",
              help="Share of valid station-days with a daily mean above 15 µg/m³ "
                   "(WHO 2021 daily guideline).")
    c3.metric("WHO 2021 annual guideline", "5 µg/m³",
              f"{full_years.loc[last, 'mean'] / 5:.1f}× exceeded", delta_color="off")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Monthly mean PM2.5 across stations")
        trend = (m.groupby("month")
                  .apply(lambda g: np.average(g["mean_ug_m3"], weights=g["valid_days"]),
                         include_groups=False)
                  .rename("pm25").reset_index())
        line = alt.Chart(trend).mark_line(color=BLUE, strokeWidth=2).encode(
            x=alt.X("month:T", title=None),
            y=alt.Y("pm25:Q", title="µg/m³"),
            tooltip=[alt.Tooltip("month:T", format="%b %Y", title="Month"),
                     alt.Tooltip("pm25:Q", format=".1f", title="PM2.5 µg/m³")])
        who = alt.Chart(pd.DataFrame({"y": [WHO_DAILY_PM25]})).mark_rule(
            color="#d03b3b", strokeDash=[4, 3]).encode(y="y:Q")
        st.altair_chart((line + who).properties(height=330), width="stretch")
        st.caption("Dashed line: WHO daily guideline (15 µg/m³). The winter peaks are the "
                   "heating season.")
    with right:
        st.subheader(f"Mean PM2.5 by station, {last}")
        per_station = (m[m["month"].dt.year == last]
                       .groupby("station_code")
                       .apply(lambda g: np.average(g["mean_ug_m3"], weights=g["valid_days"]),
                              include_groups=False).rename("pm25").reset_index()
                       .merge(stations, on="station_code"))
        station_map(per_station, "pm25", "PM2.5 µg/m³")
        st.caption("Colour scale from light (≤ 5 µg/m³) to dark (≥ 35 µg/m³). "
                   "Hover a station for details.")

# ------------------------------------------------------------------ station explorer
with tab_station:
    opts = stations.sort_values("label")
    default = opts.index[opts["city"].eq("Łódź")].min() if opts["city"].eq("Łódź").any() else 0
    choice = st.selectbox("Station", opts["label"], index=int(opts.index.get_loc(default)))
    code = opts.loc[opts["label"] == choice, "station_code"].iloc[0]
    daily = d["daily_pm25"][d["daily_pm25"]["station_code"] == code].sort_values("date")
    years = sorted(daily["date"].dt.year.unique())
    full = [y for y in years if y < pd.Timestamp.now().year]  # default: last complete year
    yr = (st.select_slider("Year", options=years, value=(full or years)[-1])
          if len(years) > 1 else years[0])
    sel = daily[daily["date"].dt.year == yr]

    a, b, c = st.columns(3)
    a.metric("Mean PM2.5", f"{sel['pm25'].mean():.1f} µg/m³")
    b.metric("Days above WHO guideline", f"{(sel['pm25'] > WHO_DAILY_PM25).sum()} "
                                          f"of {len(sel)}")
    c.metric("Worst day", f"{sel['pm25'].max():.0f} µg/m³",
             sel.loc[sel["pm25"].idxmax(), "date"].strftime("%d %b") if len(sel) else None,
             delta_color="off")

    bars = alt.Chart(sel).mark_bar(width=2).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("pm25:Q", title="daily mean PM2.5, µg/m³"),
        color=alt.condition(f"datum.pm25 > {WHO_DAILY_PM25}", alt.value("#256abf"),
                            alt.value("#b7d3f6")),
        tooltip=[alt.Tooltip("date:T", format="%a %d %b %Y", title="Date"),
                 alt.Tooltip("pm25:Q", format=".1f", title="PM2.5 µg/m³")])
    who = alt.Chart(pd.DataFrame({"y": [WHO_DAILY_PM25]})).mark_rule(
        color="#d03b3b", strokeDash=[4, 3]).encode(y="y:Q")
    st.altair_chart((bars + who).properties(height=320), width="stretch")
    st.caption("Dark bars: days above the WHO daily guideline (dashed line).")

    prof = (daily.assign(month=daily["date"].dt.month)
                 .groupby("month")["pm25"].mean().reset_index())
    prof["month_name"] = pd.to_datetime(prof["month"].astype(str), format="%m").dt.strftime("%b")
    st.subheader("Typical month (all years)")
    st.altair_chart(alt.Chart(prof).mark_bar(color=BLUE, cornerRadiusTopLeft=3,
                                             cornerRadiusTopRight=3).encode(
        x=alt.X("month_name:N", sort=list(prof["month_name"]), title=None),
        y=alt.Y("pm25:Q", title="µg/m³"),
        tooltip=[alt.Tooltip("month_name:N", title="Month"),
                 alt.Tooltip("pm25:Q", format=".1f", title="Mean PM2.5")]
    ).properties(height=240), width="stretch")

# ------------------------------------------------------------------ forecast
with tab_forecast:
    live = d.get("forecasts_live", pd.DataFrame())
    if live.empty:
        st.info("No live forecasts in this snapshot yet.")
    else:
        live = live.merge(stations, on="station_code")
        horizon = st.radio("Horizon", sorted(live["horizon_h"].unique()), horizontal=True,
                           format_func=lambda h: f"{h} hours ahead")
        lv = live[live["horizon_h"] == horizon]
        st.caption(
            f"Issued from data up to {pd.to_datetime(lv['issue_local']).max():%d %b %Y %H:%M} "
            f"(Polish time) for {len(lv)} stations. "
            + ("Model uses weather forecasts." if lv["uses_weather"].any()
               else "This model uses pollution history and the calendar only (no weather)."))
        left, right = st.columns([2, 3])
        with left:
            station_map(lv, "predicted", f"PM2.5 forecast ({horizon} h)")
            st.caption("Colour scale from light (≤ 5 µg/m³) to dark (≥ 35 µg/m³).")
        with right:
            show = lv.sort_values("p_episode", ascending=False)[
                ["city", "station_name", "target_local", "predicted", "p_episode"]]
            st.dataframe(
                show, hide_index=True, width="stretch", height=420,
                column_config={
                    "city": "City", "station_name": "Station",
                    "target_local": st.column_config.DatetimeColumn("For (local time)",
                                                                    format="ddd D MMM, HH:mm"),
                    "predicted": st.column_config.NumberColumn("PM2.5 µg/m³", format="%.1f"),
                    "p_episode": st.column_config.ProgressColumn(
                        "Chance > 50 µg/m³", format="percent", min_value=0, max_value=1),
                })

# ------------------------------------------------------------------ model performance
MODEL_NAMES = {"persistence": "Persistence", "mean_24h": "24 h mean",
               "climatology": "Climatology", "lightgbm": "LightGBM (history only)",
               "lightgbm_weather": "LightGBM + weather forecast"}
LINE_COLORS = {"LightGBM + weather forecast": BLUE, "LightGBM (history only)": "#86b6ef",
               "Persistence": GREY}


def comparison_table(metrics: dict) -> pd.DataFrame:
    """Test-year MAE per model and horizon, from the saved experiment reports."""
    rows = []
    for h in (24, 48):
        base, real = metrics.get(f"h{h}_noweather"), metrics.get(f"h{h}_realistic")
        oracle = metrics.get(f"h{h}_weather")
        if not base:
            continue
        recs = {r["model"]: r for r in base["metrics"]["test"]}
        for key in ("persistence", "mean_24h", "climatology", "lightgbm"):
            rows.append({"horizon": f"{h} h", "model": key, "mae": recs[key]["mae"],
                         "skill": recs[key].get("skill_vs_persistence", 0.0)})
        if real:
            r = {x["model"]: x for x in real["metrics"]}["lightgbm_forecast_weather"]
            rows.append({"horizon": f"{h} h", "model": "lightgbm_weather", "mae": r["mae"],
                         "skill": r["skill_vs_persistence"]})
        if oracle:
            r = {x["model"]: x for x in oracle["metrics"]["test"]}["lightgbm"]
            rows.append({"horizon": f"{h} h", "model": "perfect_weather", "mae": r["mae"],
                         "skill": r["skill_vs_persistence"]})
    out = pd.DataFrame(rows)
    out["model_name"] = out["model"].map(
        {**MODEL_NAMES, "perfect_weather": "Perfect weather (upper bound)"})
    return out


with tab_model:
    st.markdown(
        "LightGBM forecasts hourly PM2.5 at every station, 24 and 48 hours ahead. "
        "Trained on 2019–2023, tuned on 2024, **tested once on 2025**, and compared with "
        "simple baselines it must beat, e.g. *persistence* (\"same as now\"). The weather "
        "model is scored with **real archived weather forecasts** issued 1–2 days earlier, "
        "exactly what would have been available at the time.")
    tbl = comparison_table(d["metrics"])
    order = list(tbl["model_name"].drop_duplicates())
    colors = {"LightGBM + weather forecast": BLUE, "LightGBM (history only)": "#86b6ef",
              "Perfect weather (upper bound)": "#e6effa"}
    base = alt.Chart().encode(
        y=alt.Y("model_name:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("mae:Q", title="MAE, µg/m³ (lower is better)"),
        tooltip=[alt.Tooltip("model_name:N", title="Model"),
                 alt.Tooltip("mae:Q", format=".2f", title="MAE"),
                 alt.Tooltip("skill:Q", format="+.0%", title="Better than persistence by")])
    bars = base.mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3, size=22,
                         stroke="#256abf", strokeWidth=0).encode(
        color=alt.Color("model_name:N", legend=None, scale=alt.Scale(
            domain=order, range=[colors.get(m, "#c3c2b7") for m in order])),
        strokeWidth=alt.condition("datum.model == 'perfect_weather'", alt.value(1),
                                  alt.value(0)))
    labels = base.mark_text(dx=5, align="left", fontSize=12).encode(
        text=alt.Text("mae:Q", format=".2f"))
    st.altair_chart(alt.layer(bars, labels, data=tbl).properties(width=300, height=230)
                    .facet(column=alt.Column("horizon:N", title=None, header=alt.Header(
                        labelFontSize=13, labelFontWeight="bold"))))
    st.caption("“Perfect weather” feeds the model the weather that actually happened "
               "(ERA5 reanalysis). No real forecast is that good, so it's only an upper bound.")

    mm = d["backtest_monthly_mae"]
    h = st.radio("Horizon ", sorted(mm["horizon_h"].unique()), horizontal=True,
                 format_func=lambda x: f"{x} h")
    mm = mm[mm["horizon_h"] == h].assign(model_name=lambda x: x["model"].map(MODEL_NAMES))
    st.subheader("Error by month (test year 2025)")
    st.altair_chart(alt.Chart(mm).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("month:O", title="month"),
        y=alt.Y("mae:Q", title="MAE, µg/m³"),
        color=alt.Color("model_name:N", title=None, legend=alt.Legend(labelLimit=260),
                        scale=alt.Scale(
            domain=list(LINE_COLORS), range=list(LINE_COLORS.values()))),
        tooltip=["model_name:N", "month:O", alt.Tooltip("mae:Q", format=".2f")]
    ).properties(height=280), width="stretch")

    st.subheader("Winter 2025, hour by hour")
    bt = d["backtest_winter"]
    bt = bt[bt["horizon_h"] == h].merge(stations[["station_code", "label"]], on="station_code")
    worst = bt.groupby("label")["actual"].max().sort_values(ascending=False)
    pick = st.selectbox("Station ", worst.index, index=0,
                        help="Sorted by highest measured value in Jan–Feb 2025.")
    s = bt[bt["label"] == pick]
    wide = s.pivot_table(index="target_local", columns="model", values="predicted").reset_index()
    wide = wide.merge(s[["target_local", "actual"]].drop_duplicates(), on="target_local")
    long = wide.melt("target_local", var_name="series", value_name="pm25")
    long["series"] = long["series"].map({"actual": "Measured", **MODEL_NAMES})
    domain = ["Measured", "LightGBM + weather forecast", "LightGBM (history only)", "Persistence"]
    st.altair_chart(alt.Chart(long.dropna(subset=["series"])).mark_line(strokeWidth=1.6).encode(
        x=alt.X("target_local:T", title=None),
        y=alt.Y("pm25:Q", title="PM2.5 µg/m³"),
        color=alt.Color("series:N", title=None, legend=alt.Legend(labelLimit=260),
                        scale=alt.Scale(
            domain=domain, range=[INK, BLUE, "#86b6ef", GREY])),
        strokeDash=alt.StrokeDash("series:N", legend=None, scale=alt.Scale(
            domain=domain, range=[[1, 0], [1, 0], [1, 0], [4, 3]])),
        tooltip=[alt.Tooltip("target_local:T", format="%d %b %H:%M", title="Time"),
                 "series:N", alt.Tooltip("pm25:Q", format=".1f")]
    ).properties(height=320).interactive(bind_y=False), width="stretch")
    st.caption("Drag to pan and scroll to zoom. Persistence is the measured curve shifted by "
               "the horizon, so it is always late. Weather forecasts help the model anticipate "
               "when calm, cold air will trap pollution, although sudden peaks are still "
               "under-predicted.")
