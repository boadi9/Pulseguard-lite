import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone
import streamlit as st

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="PulseGuard Lite", layout="wide")
SOURCE_INBOUND = "twcs_inbound_with_roberta.csv"     # uses RoBERTa labels
SOURCE_ALERTS  = "twcs_alerts_roberta.csv"           # alerts built on RoBERTa
REFRESH_SECONDS = 30                                  # auto refresh interval

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def load_data(inbound_path, alerts_path):
    df = pd.read_csv(inbound_path)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    alerts = pd.read_csv(alerts_path)
    if "created_at" in alerts.columns:
        alerts["created_at"] = pd.to_datetime(alerts["created_at"], errors="coerce", utc=True)
    return df, alerts

def slice_window(df, minutes=15):
    now_utc = datetime.now(timezone.utc)
    start = now_utc - timedelta(minutes=minutes)
    return df[(df["created_at"] >= start) & (df["created_at"] <= now_utc)], start, now_utc

def compute_kpis(df_recent):
    total = len(df_recent)
    neg = (df_recent["sentiment_roberta"] == "Negative").sum()
    rate = (neg / total) if total > 0 else 0.0
    return total, neg, rate

def rolling_sentiment(df, window="15T"):
    ts = df.set_index("created_at").sort_index()
    senti_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
    ts["senti_val"] = ts["sentiment_roberta"].map(senti_map).fillna(0)
    vol = ts["senti_val"].resample(window).count().rename("mentions")
    neg = (ts["sentiment_roberta"] == "Negative").resample(window).sum().rename("negatives")
    pos = (ts["sentiment_roberta"] == "Positive").resample(window).sum().rename("positives")
    rate = (neg / vol.replace(0, np.nan)).rename("neg_rate")
    idx = ts["senti_val"].rolling("1H").mean().resample(window).last().rename("rolling_sentiment_1h")
    out = pd.concat([vol, neg, pos, rate, idx], axis=1).dropna(how="all")
    return out.reset_index()

# -----------------------------
# UI
# -----------------------------
st.title("PulseGuard Lite: Real-Time Support Sentiment")
st.caption("Mentions per 15m, % negative, rolling sentiment, and last alerts. Powered by RoBERTa. Developed by Fog Analytics (https://foganalytics.org).")

with st.sidebar:
    # --- Branding (optional logo file in same folder) ---
    # st.image("fog_logo.png", use_column_width=True)
    st.markdown("**Fog Analytics Dashboard**")

    st.header("Controls")
    minutes = st.slider("Window (minutes)", 5, 120, 15, step=5)
    brand_filter = st.text_input("Filter brand (contains)", value="")
    st.markdown(f"Auto-refresh every {REFRESH_SECONDS}s")
    st.divider()
    st.markdown("Files expected:")
    st.code(f"{SOURCE_INBOUND}\n{SOURCE_ALERTS}")
    st.divider()
    st.markdown("Account and help:")
    st.markdown("- Home: https://foganalytics.org/")
    st.markdown("- Documentation: https://foganalytics.org/docs")
    st.markdown("- Support: support@foganalytics.org")

inbound, alerts = load_data(SOURCE_INBOUND, SOURCE_ALERTS)
if brand_filter.strip() != "":
    mask = inbound["author_id_brand"].fillna("").str.contains(brand_filter, case=False, na=False)
    inbound = inbound[mask]
    if "author_id_brand" in alerts.columns:
        alerts = alerts[alerts["author_id_brand"].fillna("").str.contains(brand_filter, case=False, na=False)]

recent, start, now_utc = slice_window(inbound, minutes=minutes)
total, neg, rate = compute_kpis(recent)

col1, col2, col3, col4 = st.columns(4)
col1.metric(f"Mentions (last {minutes}m)", value=total)
col2.metric(f"% Negative (last {minutes}m)", value=f"{round(rate * 100, 2)}%", delta=None)
col3.metric(f"Negatives (last {minutes}m)", value=neg)
latest_ts = recent["created_at"].max()
col4.metric("Last event time (UTC)", value=str(latest_ts) if pd.notna(latest_ts) else "n/a")

ts = rolling_sentiment(inbound, window="15T")

with st.container():
    st.subheader("Trend: Mentions and Negative Rate")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.line_chart(ts.set_index("created_at")[["mentions"]])
    with c2:
        st.line_chart(ts.set_index("created_at")[["neg_rate"]])

with st.container():
    st.subheader("Rolling Sentiment Index (1h mean, resampled 15m)")
    st.line_chart(ts.set_index("created_at")[["rolling_sentiment_1h"]])

with st.container():
    st.subheader("Last Alerts")
    if not alerts.empty:
        last_alerts = alerts.sort_values("created_at", ascending=False).head(50)
        show_cols = [c for c in ["created_at","author_id_brand","tweet_id","sentiment_roberta","confidence_roberta","response_time_min","sla_threshold_min","text_clean2"] if c in last_alerts.columns]
        st.dataframe(last_alerts[show_cols], use_container_width=True, height=400)
    else:
        st.info("No alerts available yet.")

with st.container():
    st.subheader("Recent Mentions (sample)")
    if not recent.empty:
        sample = recent.sort_values("created_at", ascending=False).head(50)
        show_cols = [c for c in ["created_at","author_id_brand","tweet_id","sentiment_roberta","confidence_roberta","text_clean2"] if c in sample.columns]
        st.dataframe(sample[show_cols], use_container_width=True, height=400)
    else:
        st.info("No recent mentions in selected window.")

st.caption(f"Live view. Refreshes every {REFRESH_SECONDS} seconds.")
