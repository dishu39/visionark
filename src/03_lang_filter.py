from pathlib import Path
import pandas as pd
from tqdm import tqdm
from langdetect import detect_langs, DetectorFactory

# for reproducible results from langdetect
DetectorFactory.seed = 42

# -----------------------------
# Paths
# -----------------------------
# Prefer rule-cleaned file if it exists, else fall back to raw pairs
rule_clean_path = Path("../data/processed/flickr30k_rule_clean.csv")
raw_path = Path("../data/processed/flickr30k_pairs.csv")

if rule_clean_path.exists():
    INPUT_CSV = rule_clean_path
    print("Using rule-cleaned file as input:", INPUT_CSV)
else:
    INPUT_CSV = raw_path
    print("WARNING: rule-cleaned file not found, using raw pairs:", INPUT_CSV)

OUTPUT_CSV = Path("../data/processed/flickr30k_lang_clean.csv")

# -----------------------------
# Load data
# -----------------------------
print("Loading:", INPUT_CSV)
df = pd.read_csv(INPUT_CSV)
print("Total pairs before language filtering:", len(df))

# -----------------------------
# Language detection helper
# -----------------------------
def detect_language(text: str, min_conf: float = 0.80):
    """
    Returns:
      (lang_code, conf, keep_flag)
    where lang_code is 'en' etc, conf is probability,
    keep_flag is True if 'en' with conf >= min_conf.
    """
    if not isinstance(text, str) or not text.strip():
        return None, 0.0, False
    try:
        # detect_langs returns list of "lang:prob"
        langs = detect_langs(text)
        if not langs:
            return None, 0.0, False

        # pick best language
        best = max(langs, key=lambda l: l.prob)
        lang_code = best.lang
        conf = best.prob

        if lang_code == "en" and conf >= min_conf:
            return lang_code, conf, True
        else:
            return lang_code, conf, False

    except Exception:
        # langdetect sometimes fails on weird strings
        return None, 0.0, False

# -----------------------------
# Apply language filter
# -----------------------------
langs = []
confs = []
keep_flags = []

tqdm.pandas()

for txt in tqdm(df["caption"], desc="Detecting language"):
    lang_code, conf, keep = detect_language(txt, min_conf=0.80)
    langs.append(lang_code)
    confs.append(conf)
    keep_flags.append(keep)

df["lang"] = langs
df["lang_conf"] = confs
df["keep_lang"] = keep_flags

before = len(df)
df_en = df[df["keep_lang"]].copy()
after = len(df_en)

print(f"Pairs before: {before}")
print(f"Pairs after language filtering (English-only): {after}")
print(f"Removed: {before - after}  ({(before - after) / before * 100:.2f}%)")

# -----------------------------
# Save result
# -----------------------------
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df_en.to_csv(OUTPUT_CSV, index=False)
print("Saved English-only pairs to:", OUTPUT_CSV)
print(df_en.head())
