# app/utils.py

import re
import ftfy
import emoji
import contractions
import torch
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

__all__ = [
    "normalize_repeated_chars",
    "remove_stopwords",
    "fix_encoding",
    "to_lower",
    "expand_contractions_text",
    "handle_emojis",
    "remove_urls",
    "normalize_mentions",
    "normalize_hashtags",
    "strip_punctuation",
    "collapse_spaces",
    "normalize_tweet",
    "select_device",
]

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
PUNCT_RE = re.compile(r"[^\w\s@]")

STOPWORDS = set(ENGLISH_STOP_WORDS)


def select_device() -> str:
    """Return the best available torch device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"



# any character (.) repeated 3+ times
REPEAT_RE = re.compile(r"(.)\1{2,}")

def normalize_repeated_chars(text: str) -> str:
    """Collapse 3+ repeated characters to two characters."""
    return REPEAT_RE.sub(r"\1\1", text)




def remove_stopwords(text: str) -> str:
    tokens = text.split()
    filtered = [t for t in tokens if t not in STOPWORDS]
    return " ".join(filtered)


def fix_encoding(text: str) -> str:
    return ftfy.fix_text(text)


def to_lower(text: str) -> str:
    return text.lower()


def expand_contractions_text(text: str) -> str:
    return contractions.fix(text)


def handle_emojis(text: str) -> str:
    text = emoji.demojize(text, delimiters=(" ", " "))
    return re.sub(r":([a-z0-9_]+):", r"\1", text)


def remove_urls(text: str) -> str:
    return URL_RE.sub("", text)


def normalize_mentions(text: str) -> str:
    return MENTION_RE.sub("@user", text)


def normalize_hashtags(text: str) -> str:
    return HASHTAG_RE.sub(r"\1", text)


def strip_punctuation(text: str) -> str:
    return PUNCT_RE.sub("", text)


def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()



def normalize_tweet(text: str) -> str:
    """
    1) Fix mojibake/encoding (ftfy)
    2) Lowercase
    3) Expand contractions
    4) Demojize: '😀' → ':grinning_face:'
    5) Convert ':grinning_face:' → 'grinning_face'
    6) Remove URLs
    7) Normalize mentions → '@user'
    8) Convert '#hashtag' → 'hashtag'
    9) Strip all remaining punctuation
   10) Collapse whitespace
    """
    text = fix_encoding(text)
    text = to_lower(text)
    text = expand_contractions_text(text)
    text = handle_emojis(text)
    text = remove_urls(text)
    text = normalize_mentions(text)
    text = normalize_hashtags(text)
    text = normalize_repeated_chars(text)
    text = strip_punctuation(text)
    text = remove_stopwords(text)
    text = collapse_spaces(text)

    return text
