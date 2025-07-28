# app/utils.py

import re
import ftfy
import emoji
import contractions
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

URL_RE     = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
# punctuation regex: keep word chars and spaces
PUNCT_RE   = re.compile(r"[^\w\s]")

STOPWORDS = set(ENGLISH_STOP_WORDS)



# any character (.) repeated 3+ times
REPEAT_RE = re.compile(r"(.)\1{2,}")

def normalize_repeated_chars(text: str) -> str:
    # collapse 4+ into 2: “soooo” → “soo”
    return REPEAT_RE.sub(r"\1\1", text)




def remove_stopwords(text: str) -> str:
    tokens = text.split()
    filtered = [t for t in tokens if t not in STOPWORDS]
    return " ".join(filtered)



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
    # 1) fix text
    text = ftfy.fix_text(text) #
    # 2) lowercase
    text = text.lower() # 
    # 3) expand contractions
    text = contractions.fix(text)
    # 4) demojize
    text = emoji.demojize(text, delimiters=(" ", " "))
    # 5) strip the colons around emoji names
    text = re.sub(r":([a-z0-9_]+):", r"\1", text)
    # 6) remove URLs
    text = URL_RE.sub("", text) # 
    # 7) normalize mentions
    text = MENTION_RE.sub("@user", text) #
    # 8) normalize hashtags
    text = HASHTAG_RE.sub(r"\1", text) # 

    text = normalize_repeated_chars(text) #
    # 9) remove remaining punctuation
    text = PUNCT_RE.sub("", text) #

    text = remove_stopwords(text) #
    # 10) collapse whitespace
    text = re.sub(r"\s+", " ", text).strip() #

    return text
