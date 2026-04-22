from pathlib import Path
import pandas as pd

# -------------------------------
# PATHS (using raw strings to avoid escape issues)
# -------------------------------
DATA_DIR = Path(r"D:\B.Tech\PROJECT\root\data")
csv_path = DATA_DIR / r"processed\flickr30k_clip_clean.csv"   # change if using scored version

print("📂 Loading CSV from:", csv_path)

# -------------------------------
# LOAD DATA
# -------------------------------
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print("❌ File not found. Check path again.")
    exit()
except Exception as e:
    print("❌ Error while reading file:", e)
    exit()

print("\n✅ File loaded successfully!")

# -------------------------------
# BASIC INFO
# -------------------------------
print("\n📌 COLUMNS:")
print(df.columns.tolist())

print("\n📌 FIRST 5 ROWS:")
print(df.head())

print("\n📌 TOTAL RECORDS:", len(df))

# -------------------------------
# STATISTICS
# -------------------------------
print("\n📊 CAPTION LENGTH ANALYSIS:")
df["word_len"] = df["caption"].astype(str).str.split().str.len()
print(df["word_len"].describe())

# -------------------------------
# OPTIONAL: CLIP SCORE STATS (if present)
# -------------------------------
if "clip_sim" in df.columns:
    print("\n📊 CLIP SIMILARITY SCORE ANALYSIS:")
    print(df["clip_sim"].describe())

    print("\n🔻 WORST 5 (Lowest similarity pairs):")
    print(df.sort_values("clip_sim").head()[["image_path", "caption", "clip_sim"]])

    print("\n🔺 BEST 5 (Highest similarity pairs):")
    print(df.sort_values("clip_sim", ascending=False).head()[["image_path", "caption", "clip_sim"]])

print("\n🟢 Analysis complete.")
