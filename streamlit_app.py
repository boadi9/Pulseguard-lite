#!/usr/bin/env python3
import streamlit as st
import pandas as pd
from pathlib import Path

# ---------- paths ----------
APP_DIR   = Path(__file__).resolve().parent
BASE_DIR  = APP_DIR.parent
OUT_DIR   = BASE_DIR / "outputs"
DATA_DIR  = BASE_DIR / "data"

STREAM_BUF = OUT_DIR / "twitter_stream_buffer.csv"   # live buffer from streamer (if running)
ALERT_LOG  = OUT_DIR / "alerts_log.csv"              # spike alerts (if any)
STATIC_CSV = DATA_DIR / "twcs_inbound_with_roberta.csv"  # fallback for quick demo

# ---------- ui ----------
st.set_page_config(page_title="PulseGuard Lite", layout="wide")
st.title("PulseGuard Lite — Real-Time Support Sentiment")

tabs = st.tabs(["Live stream", "Alerts", "Static snapshot"])

# ---------- live stream ----------
with tabs[0]:
    if STREAM_BUF.exists():
        df = pd.read_csv(STREAM_BUF, parse_dates=["t"])
        df = df.sort_values("t")
        st.metric("Total streamed", len(df))
        st.line_chart(df.set_index("t")["sentiment"].tail(500), height=240)
        st.subheader("Latest messages")
        st.dataframe(df.tail(25), use_container_width=True)
    else:
        st.info("No live buffer found at `outputs/twitter_stream_buffer.csv`. Start the streamer or use the Static snapshot tab.")

# ---------- alerts ----------
with tabs[1]:
    if ALERT_LOG.exists():
        al = pd.read_csv(ALERT_LOG, parse_dates=["ts"]).sort_values("ts", ascending=False)
        st.metric("Alerts (24h)", (al["ts"] > (pd.Timestamp.utcnow() - pd.Timedelta("1D"))).sum())
        st.dataframe(al.head(50), use_container_width=True)
    else:
        st.info("No alerts logged yet.")

# ---------- static snapshot ----------
with tabs[2]:
    if STATIC_CSV.exists():
        sdf = pd.read_csv(STATIC_CSV)
        # try to parse timestamp if present
        ts_col = next((c for c in sdf.columns if c.lower() in ("created_at","date","datetime","timestamp")), None)
        if ts_col:
            sdf[ts_col] = pd.to_datetime(sdf[ts_col], errors="coerce", utc=True)
        # try to show sentiment if already present
        sent_col = next((c for c in sdf.columns if "sentiment" in c.lower()), None)
        left, right = st.columns([2,1])
        with left:
            st.write("Preview (first 200 rows)")
            st.dataframe(sdf.head(200), use_container_width=True)
        with right:
            st.write("Columns")
            st.json(list(sdf.columns))
        if sent_col:
            st.subheader("Sentiment distribution (static)")
            st.bar_chart(sdf[sent_col].value_counts())
    else:
        st.warning("Static file not found at `data/twcs_inbound_with_roberta.csv`.")
