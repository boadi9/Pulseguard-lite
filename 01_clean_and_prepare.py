
# Purpose: Clean and prepare inbound tweet data for downstream analytics.
# Why: Reliable analytics start with consistent timestamps, cleaned text, and stable IDs.
# What it does:
# - Loads sample CSVs if present
# - Normalizes timestamps to UTC
# - Performs light text cleaning
# - Saves a prepared file for modeling

#!/usr/bin/env python3
# Purpose: Load inbound TWCS, normalize timestamps to UTC, clean text, and save prepared outputs.

from pathlib import Path
import os
import re
import pandas as pd

# ---------- paths ----------
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
OUT_DIR    = BASE_DIR / "outputs"; OUT_DIR.mkdir(parents=True, exist_ok=True)

IN_FILE    = Path(os.getenv("TWCS_INBOUND", DATA_DIR / "twcs_inbound_with_roberta.csv"))
OUT_CLEAN  = OUT_DIR / "twcs_prepared.csv"
OUT_SAMPLE = OUT_DIR / "sample_preview.csv"

# ---------- helpers ----------
_url  = re.compile(r"https?://\S+")
_emo  = re.compile(r"[^\w\s\.\,\!\?\-\'\"]", flags=re.UNICODE)

def clean_text(s: str) -> str:
    s = str(s).lower()
    s = _url.sub(" ", s)
    s = _emo.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

def pick(colnames, candidates):
    lc = {c.lower(): c for c in colnames}
    for cand in candidates:
        if cand in lc: return lc[cand]
    raise ValueError(f"Missing any of required columns: {candidates}")

# ---------- load ----------
print(f"[load] {IN_FILE}")
df = pd.read_csv(IN_FILE)

# pick required columns, tolerant to different schemas
text_col   = pick(df.columns, ["text","body","message","content"])
time_col   = pick(df.columns, ["created_at","created_at_utc","date","datetime","timestamp"])
id_col     = pick(df.columns, ["tweet_id","id","status_id","message_id"])

# ---------- transform ----------
print("[transform] normalize timestamps → UTC, clean text")
df["created_at"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
df["tweet_id"]   = df[id_col]
df["text_clean"] = df[text_col].map(clean_text)

df = df.dropna(subset=["created_at","text_clean"]).loc[:, ["tweet_id","created_at","text_clean"]]

# ---------- save ----------
df.to_csv(OUT_CLEAN, index=False)
df.head(5).to_csv(OUT_SAMPLE, index=False)

print(f"[done] rows={len(df):,}  clean={OUT_CLEAN}  sample={OUT_SAMPLE}")
