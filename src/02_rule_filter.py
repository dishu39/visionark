import re
import string
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Paths
INPUT_CSV = Path("../data/processed/flickr30k_pairs.csv")  # adjust if needed
OUTPUT_CSV = Path("../data/processed/flickr30k_rule_clean.csv")

print("Loading:", INPUT_CSV)
df = pd.read_csv(INPUT_CSV)
print("Total pairs before rule filtering:", len(df))

# Helper functions
url_pattern = re.compile(r"(http[s]?://|www\.|\.com\b|\.[a-z]{2,3}/)", re.IGNORECASE)

def has_url(text: str) -> bool:
    return bool(url_pattern.search(text))

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())

def word_len_ok(text: str, min_words: int = 3, max_words: int = 30) -> bool:
    return min_words <= len(text.split()) <= max_words

def punctuation_ratio(text: str) -> float:
    punct = sum(1 for c in text if c in string.punctuation)
    return punct / len(text) if len(text) > 0 else 1.0

def too_much_punctuation(text: str) -> bool:
    return punctuation_ratio(text) > 0.30

def non_alnum_ratio(text: str) -> float:
    non_alnum = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return non_alnum / len(text) if len(text) > 0 else 1.0

def too_many_non_alnum(text: str) -> bool:
    return non_alnum_ratio(text) > 0.50

def is_gibberish(text: str) -> bool:
    return len(text) <= 2 or (sum(c.isalpha() for c in text) / len(text)) < 0.3

# Apply filters
filtered = []
reasons = []

tqdm.pandas()

for caption in tqdm(df["caption"], desc="Filtering"):
    c = str(caption).lower().strip()

    if has_url(c):
        filtered.append(False); reasons.append("url"); continue
    if not word_len_ok(c):
        filtered.append(False); reasons.append("length"); continue
    if too_much_punctuation(c):
        filtered.append(False); reasons.append("punctuation"); continue
    if too_many_non_alnum(c):
        filtered.append(False); reasons.append("symbols"); continue
    if is_gibberish(c):
        filtered.append(False); reasons.append("gibberish"); continue

    filtered.append(True); reasons.append("ok")

df["keep_rule"] = filtered
df["reason"] = reasons

df_clean = df[df["keep_rule"]]

print(f"Before: {len(df)}")
print(f"After : {len(df_clean)}")
print(f"Removed: {len(df)-len(df_clean)}")

df_clean.to_csv(OUTPUT_CSV, index=False)
print("Saved to:", OUTPUT_CSV)
