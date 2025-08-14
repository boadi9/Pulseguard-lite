
# Purpose: Compute response-time distribution and weekly percentiles for top brands.
# Why: Quantify customer support responsiveness across time.

import pandas as pd

INPUT = 'data/twcs_inbound_with_roberta.csv'
OUT_RT = 'data/twcs_first_reply_times.csv'
OUT_WK = 'data/twcs_weekly_rt_stats.csv'

print('Loading ' + INPUT)
df = pd.read_csv(INPUT)

# Assume response_time_min already present from earlier matching of mentions with first replies
rt_pairs = df[['author_id_brand','response_time_min']].dropna().copy()
rt_pairs.to_csv(OUT_RT, index=False)
print('Saved ' + OUT_RT)

# Weekly percentiles by brand
if 'created_at' in df.columns:
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True)
    df['week'] = df['created_at'].dt.to_period('W').dt.start_time
    agg = df.dropna(subset=['response_time_min']).groupby(['author_id_brand','week'])['response_time_min'].quantile([0.5,0.9,0.95]).unstack(level=-1)
    agg = agg.rename(columns={0.5:'p50_min',0.9:'p90_min',0.95:'p95_min'}).reset_index()
    agg.to_csv(OUT_WK, index=False)
    print('Saved ' + OUT_WK)
else:
    print('created_at missing; skipping weekly stats')
