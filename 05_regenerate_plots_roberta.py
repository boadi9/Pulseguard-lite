
# Purpose: Regenerate key plots using RoBERTa labels.
# Why: Visuals help stakeholders grasp trends and outliers quickly.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

rt_pairs = pd.read_csv('data/twcs_first_reply_times.csv')
weekly_stats = pd.read_csv('data/twcs_weekly_rt_stats.csv')
inbound_rb = pd.read_csv('data/twcs_inbound_with_roberta.csv')
alerts_rb = pd.read_csv('data/twcs_alerts_roberta.csv')

print('Loaded data for plotting')

plt.figure(figsize=(8,5))
import numpy as np
sns.histplot(rt_pairs['response_time_min'], bins=40, kde=True, color='steelblue')
plt.title('Overall First Response Time (minutes)')
plt.xlabel('Minutes to First Reply')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

brand_counts = rt_pairs['author_id_brand'].value_counts().head(6).index.tolist()
weekly_top = weekly_stats[weekly_stats['author_id_brand'].isin(brand_counts)].copy()
plt.figure(figsize=(10,6))
for metric, color in [('p50_min','green'),('p90_min','orange'),('p95_min','red')]:
    tmp = weekly_top.sort_values('week')
    for b in brand_counts:
        sub = tmp[tmp['author_id_brand'] == b]
        plt.plot(sub['week'], sub[metric], marker='o', linewidth=1, label=b + ' ' + metric)
plt.xticks(rotation=45, ha='right')
plt.title('Weekly Response Time Percentiles (Top Brands)')
plt.xlabel('Week')
plt.ylabel('Minutes')
handles, labels = plt.gca().get_legend_handles_labels()
unique = dict(zip(labels, handles))
plt.legend(unique.values(), unique.keys(), bbox_to_anchor=(1.05,1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.show()

inbound_rb['created_at'] = pd.to_datetime(inbound_rb['created_at'], errors='coerce')
inbound_rb['date'] = inbound_rb['created_at'].dt.date
neg_daily = inbound_rb[inbound_rb['sentiment_roberta'] == 'Negative'].groupby('date').size().reset_index(name='neg_count')
alerts_rb['created_at'] = pd.to_datetime(alerts_rb['created_at'], errors='coerce')
alerts_rb['date'] = alerts_rb['created_at'].dt.date
alerts_daily = alerts_rb.groupby('date').size().reset_index(name='alerts_count')
plt.figure(figsize=(10,5))
plt.plot(neg_daily['date'], neg_daily['neg_count'], marker='o', color='crimson', label='Negative inbound (RoBERTa)')
plt.plot(alerts_daily['date'], alerts_daily['alerts_count'], marker='s', color='black', label='Alerts (RoBERTa)')
plt.title('Daily Negative Tweets and Alerts (RoBERTa)')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.show()

print('Finished plotting.')
