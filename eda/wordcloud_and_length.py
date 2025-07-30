# eda/wordcloud_and_length.py

import sys
from pathlib import Path
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# ─── PATH HACK ───
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from app.file_ingestion import ingest_and_validate_pandera

# 1. Load & validate
df = ingest_and_validate_pandera("data/raw/train_raw.csv")

# 2. Prepare text per class
texts = {
    0: " ".join(df[df["label"] == 0]["tweet"].tolist()),
    1: " ".join(df[df["label"] == 1]["tweet"].tolist()),
}

# 3. Generate and save word clouds
output_dir = PROJECT_ROOT / "eda"
output_dir.mkdir(exist_ok=True)

# Add "user" to the stopword set
custom_stopwords = STOPWORDS.union({"user"})

for label, text in texts.items():
    wc = WordCloud(
        width=800, height=400,
        stopwords=custom_stopwords,
        background_color="white",
        max_words=100
    ).generate(text)
    
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud (label={label})", fontsize=16)
    out_path = output_dir / f"wordcloud_label_{label}.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved word cloud for label={label} to {out_path}")

# 4. Tweet‐length histogram
df["length"] = df["tweet"].str.len()
plt.figure(figsize=(8,4))
plt.hist(df["length"], bins=50)
plt.title("Tweet Length Distribution")
plt.xlabel("Length (chars)")
plt.ylabel("Count")
plt.tight_layout()
hist_path = output_dir / "tweet_length_histogram.png"
plt.savefig(hist_path)
plt.show()
print(f"Saved length histogram to {hist_path}")
