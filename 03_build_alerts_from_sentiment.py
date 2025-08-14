
# Purpose: Build SLA alerts based on negative sentiment and response-time breaches.
# Why: Focus agent attention on urgent, negative mentions not answered in time.

import pandas as pd

INPUT = 'data/twcs_inbound_with_roberta.csv'
OUTPUT = 'data/twcs_alerts_roberta.csv'

print('Loading ' + INPUT)
df = pd.read_csv(INPUT)

# Expected columns: sentiment_roberta, response_time_min, author_id_brand
thresholds = [30, 60, 120]
alerts = []
for t in thresholds:
    dfa = df[(df['sentiment_roberta'] == 'Negative') & ((df['response_time_min'].isna()) | (df['response_time_min'] > t))].copy()
    dfa['sla_threshold_min'] = t
    alerts.append(dfa)

alerts_df = pd.concat(alerts, ignore_index=True) if len(alerts) > 0 else pd.DataFrame()
print('Alerts rows: ' + str(len(alerts_df)))
alerts_df.to_csv(OUTPUT, index=False)
print('Saved ' + OUTPUT)
