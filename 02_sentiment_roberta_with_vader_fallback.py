
# Purpose: Score sentiment using RoBERTa with VADER fallback.
# Why: Modern transformer sentiment is more accurate; VADER provides a lightweight backup.

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

INPUT = 'data/twcs_prepared.csv'
OUTPUT = 'data/twcs_inbound_with_roberta.csv'

print('Loading ' + INPUT)
df = pd.read_csv(INPUT)
texts = df['text_clean2'].fillna('').astype(str).tolist()

# Try RoBERTa
model_name = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
use_roberta = True
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    print('Loaded RoBERTa model')
except Exception as e:
    print('RoBERTa load failed: ' + str(e))
    use_roberta = False

# VADER
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

def score_roberta(texts, batch_size=32):
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = softmax(logits.numpy(), axis=1)
        idx = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        for j in range(len(batch)):
            preds.append((label_map[int(idx[j])], float(conf[j]), float(probs[j][2] - probs[j][0])))
    return preds

if use_roberta:
    try:
        preds = score_roberta(texts)
        df['sentiment_roberta'] = [p[0] for p in preds]
        df['confidence_roberta'] = [p[1] for p in preds]
        df['score_pos_minus_neg'] = [p[2] for p in preds]
    except Exception as e:
        print('RoBERTa inference failed: ' + str(e))
        use_roberta = False

if not use_roberta:
    print('Falling back to VADER')
    df['vader_compound'] = df['text_clean2'].apply(lambda x: sia.polarity_scores(x)['compound'])
    df['sentiment_roberta'] = np.where(df['vader_compound'] >= 0.05, 'Positive', np.where(df['vader_compound'] <= -0.05, 'Negative', 'Neutral'))
    df['confidence_roberta'] = df['vader_compound'].abs()
    df['score_pos_minus_neg'] = df['text_clean2'].apply(lambda x: sia.polarity_scores(x)['pos'] - sia.polarity_scores(x)['neg'])

print('Saving to ' + OUTPUT)
df.to_csv(OUTPUT, index=False)
print('Rows: ' + str(len(df)))
