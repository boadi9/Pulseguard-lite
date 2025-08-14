
# Purpose: Slack alert when last-10-minute negative rate >= 30% AND volume >= 50.
# Why: Alert ops when negativity spikes with sufficient volume.

import pandas as pd
from datetime import datetime, timezone, timedelta
import requests

INPUT = 'data/twcs_inbound_with_roberta.csv'
SLACK_WEBHOOK_URL = 'https://hooks.slack.com/services/REPLACE/ME/WEBHOOK'

print('Loading ' + INPUT)
df = pd.read_csv(INPUT)
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True)

now_utc = datetime.now(timezone.utc)
window_start = now_utc - timedelta(minutes=10)
recent = df[(df['created_at'] >= window_start) & (df['created_at'] <= now_utc)]

total = len(recent)
neg = (recent['sentiment_roberta'] == 'Negative').sum()
rate = (neg / total) if total > 0 else 0.0
print('Checked window ' + str(window_start) + ' to ' + str(now_utc))
print('Volume: ' + str(total))
print('Negatives: ' + str(neg))
print('Negative rate: ' + str(round(rate * 100, 2)) + '%')

if total >= 50 and rate >= 0.30:
    text = 'PulseGuard Alert: 10-min negative rate ' + str(round(rate * 100, 2)) + '% with volume ' + str(total)
    payload = {'text': text}
    try:
        resp = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=10)
        print('Slack status: ' + str(resp.status_code))
    except Exception as e:
        print('Slack send failed: ' + str(e))
else:
    print('Thresholds not met; no alert sent.')
