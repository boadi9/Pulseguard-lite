
# Purpose: Plot sentiment trend and volumes with 15-minute bins.
# Why: Track sentiment direction and volume changes over time.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT = 'data/twcs_inbound_with_roberta.csv'

print('Loading ' + INPUT)
df = pd.read_csv(INPUT)
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

sns.set(style='whitegrid')

def enc(s):
    if s == 'Positive':
        return 1
    if s == 'Negative':
        return -1
    return 0

df = df.sort_values('created_at')
df['senti_val'] = df['sentiment_roberta'].apply(enc)
trend = df.set_index('created_at').resample('15T')['senti_val'].mean().rename('avg_sentiment_15m').reset_index()
counts = df.set_index('created_at').groupby('sentiment_roberta').resample('15T').size().unstack(0).fillna(0).reset_index().rename(columns={'created_at':'time'})

import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.plot(trend['created_at'], trend['avg_sentiment_15m'], color='purple', label='Avg sentiment (15m)')
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.title('Sentiment Trend Over Time (15m)')
plt.xlabel('Time')
plt.ylabel('Avg sentiment index (-1..1)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
for col in ['Negative','Neutral','Positive']:
    if col in counts.columns:
        plt.plot(counts['time'], counts[col], label=col)
plt.title('Volume by Sentiment (15m)')
plt.xlabel('Time')
plt.ylabel('Mentions')
plt.legend()
plt.tight_layout()
plt.show()

print('Plotted sentiment trend and volumes.')
