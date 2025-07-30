# app/tokenization.py

from transformers import AutoTokenizer
import pandas as pd
from pathlib import Path
import torch

# 1) BERTweet tokenizer (optimized for tweets)
def get_bertweet_tokenizer():
    # normalization=True applies basic tweet-specific pre-processing
    return AutoTokenizer.from_pretrained(
        "vinai/bertweet-base",
        use_fast=True,
        normalization=True
    )

# 2) BERT-base-uncased tokenizer (general English)
def get_bert_base_tokenizer():
    return AutoTokenizer.from_pretrained(
        "bert-base-uncased",
        use_fast=True
    )


def tokenize_processed_csv(file_path: str, tokenizer_type: str = "bertweet", max_length: int = 128):
    """
    Read a processed CSV with a 'cleaned_tweet' column and tokenize its text.

    Parameters:
    - file_path (str): Path to the processed CSV file.
    - tokenizer_type (str): 'bertweet' or 'bert-base' to choose the tokenizer.
    - max_length (int): Maximum sequence length for padding/truncation.

    Returns:
    - encodings (dict): Contains 'input_ids' and 'attention_mask' as PyTorch tensors.
    """
    # 1) Load processed data
    df = pd.read_csv(file_path)

    df["tweet"] = df["tweet"].fillna("").astype(str)
    texts = df["tweet"].tolist()

    # 2) Select tokenizer
    if tokenizer_type == "bertweet":
        tokenizer = get_bertweet_tokenizer()
    else:
        tokenizer = get_bert_base_tokenizer()

    # 4) Sanity‐check first few entries are strings
    sample = df["tweet"].tolist()[:5]
    assert all(isinstance(t, str) for t in sample), \
        f"Non-str detected in samples: {[type(t) for t in sample]}"

    # 3) Tokenize
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return encodings





    

if __name__ == "__main__":
    processed_file = Path("data/processed/train_raw_pandera_processed.csv")
    enc = tokenize_processed_csv(str(processed_file), tokenizer_type="bertweet", max_length=128)
    print("Tokenization complete:")
    print(" - input_ids shape:", enc["input_ids"].shape)
    print(" - attention_mask shape:", enc["attention_mask"].shape)
    


# Save just the tensors, not the full BatchEncoding
    to_save = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"]
    }
    output_path = processed_file.parent / "train_tokenized.pt"
    torch.save(to_save, output_path)
    print(f"→ Saved tokenized encodings to {output_path}")