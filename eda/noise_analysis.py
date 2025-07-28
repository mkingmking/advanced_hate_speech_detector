# eda/noise_analysis.py

import sys
from pathlib import Path
import re
from collections import Counter

import pandas as pd

# ─── PATH HACK ───
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from app.file_ingestion import ingest_and_validate_pandera

# 1. Load & validate
df = ingest_and_validate_pandera("../data/raw/train_raw.csv")

# 2. Define regex for noise
URL_RE     = re.compile(r"http\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
# Simple emoji range (supplement as needed)
EMOJI_RE   = re.compile(
    "[\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F680-\U0001F6FF]"   # transport & map symbols
)

# 3. Compute per-tweet noise features
def count_pattern(col, regex):
    return col.str.count(regex)

df["n_urls"]     = count_pattern(df["tweet"], URL_RE)
df["n_mentions"] = count_pattern(df["tweet"], MENTION_RE)
df["n_hashtags"] = count_pattern(df["tweet"], HASHTAG_RE)
df["n_emojis"]   = count_pattern(df["tweet"], EMOJI_RE)
df["length"]     = df["tweet"].str.len()

# 4. Summarize noise by label
noise_stats = df.groupby("label")[["n_urls","n_mentions","n_hashtags","n_emojis","length"]] \
                .agg(["mean","median","max"]) \
                .round(2)
print("\nNoise & length statistics by class:\n", noise_stats)

# 5. Extract top tokens & bi-grams per class
from sklearn.feature_extraction.text import CountVectorizer

def top_ngrams(corpus, ngram_range=(1,1), top_k=20):
    vec = CountVectorizer(ngram_range=ngram_range, token_pattern=r"\b\w+\b", min_df=5)
    X = vec.fit_transform(corpus)
    freqs = zip(vec.get_feature_names_out(), X.sum(axis=0).tolist()[0])
    return Counter(dict(freqs)).most_common(top_k)

for label in [0,1]:
    tweets = df[df["label"] == label]["tweet"]
    uni  = top_ngrams(tweets, (1,1), top_k=10)
    bi   = top_ngrams(tweets, (2,2), top_k=10)
    print(f"\nTop 10 unigrams for label={label}:\n", uni)
    print(f"Top 10 bigrams for label={label}:\n", bi)
