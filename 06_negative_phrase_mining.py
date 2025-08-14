
# Purpose: Extract top uni/bi/tri-gram phrases from negative mentions.
# Why: Reveal drivers and themes behind negative spikes for actionability.

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re

INPUT = 'data/twcs_inbound_with_roberta.csv'

print('Loading ' + INPUT)
df = pd.read_csv(INPUT)
neg = df[df['sentiment_roberta'] == 'Negative'].copy()
texts = neg['text_clean2'].fillna('').astype(str).tolist()

stop_words = 'english'
vectorizers = {
    'uni': CountVectorizer(stop_words=stop_words, ngram_range=(1,1), min_df=2, max_df=0.9),
    'bi': CountVectorizer(stop_words=stop_words, ngram_range=(2,2), min_df=2, max_df=0.9),
    'tri': CountVectorizer(stop_words=stop_words, ngram_range=(3,3), min_df=2, max_df=0.9)
}

results = {}
for key, vec in vectorizers.items():
    if len(texts) == 0:
        results[key] = pd.DataFrame(columns=['phrase','count'])
        continue
    X = vec.fit_transform(texts)
    counts = np.asarray(X.sum(axis=0)).ravel()
    vocab = np.array(vec.get_feature_names_out())
    dfc = pd.DataFrame({'phrase': vocab, 'count': counts}).sort_values('count', ascending=False).head(50)
    results[key] = dfc
    out = 'data/negative_phrases_' + key + '.csv'
    dfc.to_csv(out, index=False)
    print('Saved ' + out + ' rows: ' + str(len(dfc)))

print('Done negative phrase mining.')
