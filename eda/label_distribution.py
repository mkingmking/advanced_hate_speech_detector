# eda/label_distribution.py

# ─── PATH HACK ───
import sys
from pathlib import Path

# If __file__ exists (script), use that; otherwise (notebook), fall back to cwd
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd()

sys.path.insert(0, str(PROJECT_ROOT))


from app.file_ingestion import ingest_and_validate_pandera
import matplotlib.pyplot as plt

# 1. Ingest & validate via your Pandera function
df = ingest_and_validate_pandera("../data/raw/train_raw.csv")

# 2. Compute counts
label_counts = df["label"].value_counts().sort_index()

# 3. Print raw numbers and proportions
total = label_counts.sum()
print("Label counts:\n", label_counts.to_dict())
print("\nLabel proportions:")
print((label_counts / total).round(3).to_dict())

# 4. Bar‐plot the distribution
plt.figure(figsize=(6,4))
label_counts.plot(kind="bar")
plt.title("Tweet Label Distribution")
plt.xlabel("Label (0 = Not Hate, 1 = Hate)")
plt.ylabel("Number of Tweets")
plt.tight_layout()
plt.savefig("visualisations/label_distribution.png")
plt.show()
